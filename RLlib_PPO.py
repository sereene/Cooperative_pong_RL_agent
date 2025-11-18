

import os
import ray
import supersuit as ss
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn

from pettingzoo.butterfly import cooperative_pong_v5


# Custom CNN models
class CNNModelLeft(TorchModelV2, nn.Module):
    """Left paddle policy network (for 'paddle_0')."""
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)

        # 84x84x3 입력 기준
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, [8, 8], stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, [3, 3], stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        # obs: [B, H, W, C] -> [B, C, H, W]
        x = input_dict["obs"].permute(0, 3, 1, 2)
        model_out = self.model(x)
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()


class CNNModelRight(TorchModelV2, nn.Module):

    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)

        # 약간 다른 CNN 아키텍처
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, [8, 8], stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 128, [3, 3], stride=(1, 1)),  # 채널 수 128로 변경
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6272, 512), 
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].permute(0, 3, 1, 2)
        model_out = self.model(x)
        self._value_out = self.value_fn(model_out)
        return self.policy_fn(model_out), state

    def value_function(self):
        return self._value_out.flatten()


# Environment creator
def env_creator(config):
    env = cooperative_pong_v5.parallel_env(
        ball_speed=9,
        left_paddle_speed=12,
        right_paddle_speed=12,
        cake_paddle=True,
        max_cycles=900,
        bounce_randomness=False,
        max_reward=100,
        off_screen_penalty=-10,
    )

    # Pistonball 튜토리얼과 비슷하게 wrapper 구성함
    env = ss.color_reduction_v0(env, mode="B")        # grayscale
    env = ss.dtype_v0(env, "float32")                 # float32
    env = ss.resize_v1(env, x_size=84, y_size=84)     # 84x84
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    env = ss.frame_stack_v1(env, 3)                   # 3 frame stack -> (84,84,3)

    return env


if __name__ == "__main__":
    ray.init()

    env_name = "cooperative_pong_v5_multi_ppo"

    # RLlib에 env 등록
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

    # 한 번 env를 만들어서 observation_space / action_space 확인
    temp_env = ParallelPettingZooEnv(env_creator({}))
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space
    agent_ids = temp_env.agents  # ['paddle_0', 'paddle_1']
    print("Agents:", agent_ids)
    temp_env.close()

    # 커스텀 모델 등록
    ModelCatalog.register_custom_model("CNNModelLeft", CNNModelLeft)
    ModelCatalog.register_custom_model("CNNModelRight", CNNModelRight)

    # Multi-agent 설정
    # paddle_0  : left_policy (CNNModelLeft)
    # paddle_1  : right_policy (CNNModelRight)
    policies = {
        "left_policy": (
            None,           
            obs_space,
            act_space,
            {
                "model": {
                    "custom_model": "CNNModelLeft",
                }
            },
        ),
        "right_policy": (
            None,
            obs_space,
            act_space,
            {
                "model": {
                    "custom_model": "CNNModelRight",
                }
            },
        ),
    }

    def policy_mapping_fn(agent_id, *args, **kwargs):
        # PettingZoo cooperative_pong_v5 agents: ['paddle_0', 'paddle_1']
        if agent_id == "paddle_0":
            return "left_policy"
        elif agent_id == "paddle_1":
            return "right_policy"
        else:
            return "left_policy"

    # PPO 설정
    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["left_policy", "right_policy"],
        )
        .env_runners(
        num_env_runners=4,              
        rollout_fragment_length=128     
        )
        .training(
            train_batch_size=512,
            lr=2e-5,
            gamma=0.99,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            sgd_minibatch_size=64,
            num_sgd_iter=10,
        )
        .debugging(log_level="ERROR")
        .framework("torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    # 학습 실행 
    tune.run(
        "PPO",
        name="PPO_cooperative_pong_two_policies",
        stop={
            "timesteps_total": 5_000_000 if not os.environ.get("CI") else 50_000
        },
        checkpoint_freq=10,
        local_dir="~/ray_results/" + env_name,
        config=config.to_dict(),
    )
