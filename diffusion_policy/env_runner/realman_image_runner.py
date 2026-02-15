import numpy as np
import torch
import tqdm
import collections

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
class RealmanImageRunner(BaseImageRunner):

    def __init__(
        self,
        output_dir,
        env,
        shape_meta: dict,
        env_seeds=None,
        env_prefixs=None,
        max_steps=400,
        n_obs_steps=2,
        n_action_steps=8,
        past_action=False,
        tqdm_interval_sec=5.0,
    ):
        super().__init__(output_dir)

        self.env = MultiStepWrapper(
            env,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps
        )

        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.tqdm_interval_sec = tqdm_interval_sec

        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs

    def run(self, policy: BaseImagePolicy):

        device = policy.device
        env = self.env

        n_envs = env.env.n_envs

        obs = env.reset()
        policy.reset()

        past_action = None
        done = np.zeros(n_envs, dtype=bool)

        # 用 numpy array 更高效
        max_rewards = np.zeros(n_envs)

        pbar = tqdm.tqdm(
            total=self.max_steps,
            desc="Realman Eval",
            leave=False,
            mininterval=self.tqdm_interval_sec
        )

        step_count = 0

        while (not np.all(done)) and step_count < self.max_steps:

            np_obs_dict = dict(obs)

            if self.past_action and (past_action is not None):
                np_obs_dict['past_action'] = past_action[
                    :, -(self.n_obs_steps-1):
                ].astype(np.float32)

            obs_dict = dict_apply(
                np_obs_dict,
                lambda x: torch.from_numpy(x).to(device=device)
            )

            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            np_action_dict = dict_apply(
                action_dict,
                lambda x: x.detach().cpu().numpy()
            )

            action = np_action_dict['action']

            if not np.all(np.isfinite(action)):
                raise RuntimeError("Nan or Inf action")

            # ---------- 对已完成环境屏蔽 action ----------
            action[done] = 0.0

            obs, reward, done_step, info = env.step(action)

            # ---------- 累积 done ----------
            done = np.logical_or(done, done_step)

            # ---------- 记录最大 reward ----------
            max_rewards = np.maximum(max_rewards, reward)

            past_action = action

            step_count += self.n_action_steps
            pbar.update(self.n_action_steps)

        pbar.close()

        # ---------- 统计 ----------
        log_data = {}

        for i in range(n_envs):
            prefix = self.env_prefixs[i] if self.env_prefixs else ""
            log_data[prefix + f"sim_max_reward_{i}"] = max_rewards[i]

        # mean score
        mean_score = np.mean(max_rewards)
        log_data["mean_score"] = mean_score

        return log_data




# import numpy as np
# import torch
# import tqdm
# import collections

# from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
# from diffusion_policy.common.pytorch_util import dict_apply
# from diffusion_policy.policy.base_image_policy import BaseImagePolicy
# from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper


# class RealmanImageRunner(BaseImageRunner):

#     def __init__(
#         self,
#         output_dir,
#         env,                      # 你的 vector env (带 n_envs)
#         shape_meta: dict,
#         env_seeds=None,           # 用于区分 train/test
#         env_prefixs=None,
#         max_steps=400,
#         n_obs_steps=2,
#         n_action_steps=8,
#         past_action=False,
#         tqdm_interval_sec=5.0,
#     ):
#         super().__init__(output_dir)

#         self.env = MultiStepWrapper(
#             env,
#             n_obs_steps=n_obs_steps,
#             n_action_steps=n_action_steps,
#             max_episode_steps=max_steps
#         )

#         self.max_steps = max_steps
#         self.n_obs_steps = n_obs_steps
#         self.n_action_steps = n_action_steps
#         self.past_action = past_action
#         self.tqdm_interval_sec = tqdm_interval_sec

#         # 和原代码一致
#         self.env_seeds = env_seeds
#         self.env_prefixs = env_prefixs

#     def run(self, policy: BaseImagePolicy):

#         device = policy.device
#         env = self.env

#         n_envs = env.env.n_envs   # 你的 RealmanEnv 内部并行数

#         obs = env.reset()
#         policy.reset()

#         past_action = None
#         done = np.zeros(n_envs, dtype=bool)

#         # 保存每个环境的 reward 序列
#         all_rewards = [[] for _ in range(n_envs)]

#         pbar = tqdm.tqdm(
#             total=self.max_steps,
#             desc="Realman Eval",
#             leave=False,
#             mininterval=self.tqdm_interval_sec
#         )

#         step_count = 0

#         while not np.all(done) and step_count < self.max_steps:

#             # ---------- 构造 obs ----------
#             np_obs_dict = dict(obs)

#             if self.past_action and (past_action is not None):
#                 np_obs_dict['past_action'] = past_action[
#                     :, -(self.n_obs_steps-1):
#                 ].astype(np.float32)

#             # ---------- 转 torch ----------
#             obs_dict = dict_apply(
#                 np_obs_dict,
#                 lambda x: torch.from_numpy(x).to(device=device)
#             )

#             # ---------- policy ----------
#             with torch.no_grad():
#                 action_dict = policy.predict_action(obs_dict)

#             np_action_dict = dict_apply(
#                 action_dict,
#                 lambda x: x.detach().cpu().numpy()
#             )

#             action = np_action_dict['action']

#             if not np.all(np.isfinite(action)):
#                 raise RuntimeError("Nan or Inf action")

#             # ---------- 环境 step ----------
#             obs, reward, done_step, info = env.step(action)

#             # 累积 reward
#             for i in range(n_envs):
#                 all_rewards[i].append(reward[i])

#             done = done_step
#             past_action = action

#             step_count += self.n_action_steps
#             pbar.update(self.n_action_steps)

#         pbar.close()

#         # ---------- 统计结果（完全对齐原逻辑） ----------
#         max_rewards = collections.defaultdict(list)
#         log_data = {}

#         for i in range(n_envs):

#             prefix = self.env_prefixs[i] if self.env_prefixs else ""

#             max_reward = np.max(all_rewards[i])

#             max_rewards[prefix].append(max_reward)
#             log_data[prefix + f"sim_max_reward_{i}"] = max_reward

#         # aggregate
#         for prefix, value in max_rewards.items():
#             name = prefix + "mean_score"
#             log_data[name] = np.mean(value)

#         return log_data
