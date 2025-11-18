#!/usr/bin/env python
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from pettingzoo.butterfly import cooperative_pong_v5


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="이름",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="True면 cuda 사용",
    )
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="Weights & Biases 로깅 여부",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="cleanRL",
        help="wandb 프로젝트 이름",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="wandb 팀/엔티티 이름",
    )

    # 하이퍼파라미터
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2000000,
        help="전체 학습 타임스텝 수",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Adam learning rate",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=16,
        help="벡터 환경 개수",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=128,
        help="각 환경에서 rollout할 스텝 수 (한 업데이트당)",
    )
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="True면 학습 도중 lr 선형 감소",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor γ",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE의 람다값",
    )
    parser.add_argument(
        "--num-minibatches",
        type=int,
        default=4,
        help="미니배치 개수",
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=4,
        help="각 업데이트에서 PPO epoch 수",
    )
    parser.add_argument(
        "--norm-adv",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="True면 advantage 정규화",
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.1,
        help="PPO surrogate clipping 계수",
    )
    parser.add_argument(
        "--clip-vloss",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="True면 value function도 클리핑 적용",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="entropy 보너스 계수",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="value loss 계수",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="gradient clipping max norm",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="KL이 이 값 넘으면 epoch 일찍 종료",
    )

    args = parser.parse_args()
    # 전체 배치 크기 = env 개수 * rollout 길이
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


#  Network
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        obs_shape = envs.single_observation_space.shape
        in_channels = obs_shape[-1]

        self.backbone = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Feature size after CNN layers (for 84x84 input) = 64 * 7 * 7
        self.fc = nn.Sequential(
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )

        self.actor = layer_init(
            nn.Linear(512, envs.single_action_space.n), std=0.01
        )
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, H, W, C)  
        # 배치크기, 높이, 너비, 채널 순서로 들어옴
        x = x.clone()
        # 앞쪽 4 채널만 0~255 -> 0~1로 스케일링
        x[:, :, :, :4] = x[:, :, :, :4] / 255.0
        x = x.permute(0, 3, 1, 2) # 텐서 차원 순서 변경 → Conv2D가 요구하는 (B, C, H, W) 형식
        return x

    def get_value(self, x):
        x = self._preprocess(x)
        hidden = self.fc(self.backbone(x))
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None):
        x = self._preprocess(x)
        hidden = self.fc(self.backbone(x))
        logits = self.actor(hidden)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(hidden)
        return action, logprob, entropy, value


#  Main training loop
if __name__ == "__main__":
    args = parse_args()
    print(args)

    run_name = f"cooperative_pong_v5__{args.exp_name}__{args.seed}__{int(time.time())}"

    # Weights & Biases(optional)
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # TensorBoard writer
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()]),
    )

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # 디바이스 선택
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("Using device:", device)

    #  Env setup
    # cooperative_pong 은 항상 2 agents (2 paddles) :contentReference[oaicite:1]{index=1}
    n_agents = 2

    # ParallelEnv 생성
    base_env = cooperative_pong_v5.parallel_env()
    # Supersuit로 전처리: grayscale, resize, frame stack, agent indicator 등
    env = ss.color_reduction_v0(base_env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.agent_indicator_v0(env, type_only=False)

    # PettingZoo ParallelEnv -> Gymnasium VecEnv 래핑
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    # concat_vec_envs_v1: 하나의 VecEnv 안에 여러 병렬 env를 합침
    # cooperative_pong 한 env에 2 agents 있으므로, total slots = (num_envs // 2) * 2
    envs = ss.concat_vec_envs_v1(
        env,
        args.num_envs // n_agents, 
        #총 슬롯 개수 // 에이전트 개수 = 환경
        #8개의 환경 × 2 에이전트 = 총 16 슬롯
        num_cpus=0,
        base_class="gymnasium",
    )

    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True

    #  Agent & buffer
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Rollout buffer
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
        device=device,
    )
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape,
        device=device,
    )
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    terminations = torch.zeros((args.num_steps, args.num_envs), device=device)
    truncations = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    #  Training loop
    global_step = 0
    start_time = time.time()

    next_obs, info = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_termination = torch.zeros(args.num_envs, device=device)
    next_truncation = torch.zeros(args.num_envs, device=device)

    num_updates = args.total_timesteps // args.batch_size
    
    td_error_history = []
    
    episode_returns = []
    episode_lengths = []

    window_size = 256   # td error 관찰 일정주기만큼 묶어서


    for update in range(1, num_updates + 1):
        # learning rate annealing
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lr_now

        for step in range(args.num_steps):
            global_step += args.num_envs

            obs[step] = next_obs
            terminations[step] = next_termination
            truncations[step] = next_truncation

            # 정책에서 action sample
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.view(-1)

            actions[step] = action
            logprobs[step] = logprob

            # 환경 한 스텝
            next_obs_np, reward_np, termination_np, truncation_np, info = envs.step(
                action.cpu().numpy()
            )
            
    
            rewards[step] = torch.tensor(
                reward_np, dtype=torch.float32, device=device
            ).view(-1)
            
            
            next_obs = torch.tensor(
                next_obs_np, dtype=torch.float32, device=device
            )
            next_termination = torch.tensor(
                termination_np, dtype=torch.float32, device=device
            )
            next_truncation = torch.tensor(
                truncation_np, dtype=torch.float32, device=device
            )


        #  GAE & returns 계산
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0.0

            next_done = torch.maximum(next_termination, next_truncation)
            dones = torch.maximum(terminations, truncations)
            
            td_errors = torch.zeros((args.num_steps, args.num_envs), device=device)

            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_nonterminal = 1.0 - next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - dones[t + 1]
                    next_values = values[t + 1]

                delta = (
                    rewards[t]
                    + args.gamma * next_values * next_nonterminal
                    - values[t]
                )
                td_errors[t] = delta
                advantages[t] = lastgaelam = (
                    delta
                    + args.gamma
                    * args.gae_lambda
                    * next_nonterminal
                    * lastgaelam
                )

            returns = advantages + values

        #  Flatten batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        #  PPO 업데이트
        b_inds = np.arange(args.batch_size)
        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef)
                        .float()
                        .mean()
                        .item()
                    )

                mb_adv = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = (
                    pg_loss
                    - args.ent_coef * entropy_loss
                    + args.vf_coef * v_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # explained variance
        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        #  TensorBoard 로깅
        writer.add_scalar(
            "charts/learning_rate",
            optimizer.param_groups[0]["lr"],
            global_step,
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        
        
        #  TD Error Logging 
        td_error_history.extend(td_errors.flatten().tolist())

        if len(td_error_history) >= window_size:
            window = td_error_history[:window_size]
            td_window_mean = np.mean(window)
            
            # TD error값 모니터링
            writer.add_scalar("td/window_mean", td_window_mean, global_step)

            # 그 다음
            td_error_history = td_error_history[window_size:]

        sps = int(global_step / (time.time() - start_time))
        print(f"update={update}/{num_updates}, global_step={global_step}, SPS={sps}")
        writer.add_scalar("charts/SPS", sps, global_step)

    envs.close()
    writer.close()
