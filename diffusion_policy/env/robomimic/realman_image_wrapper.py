from typing import Optional
import numpy as np
import gym
from gym import spaces

from configs.LinkerHandGrasp_config import LinkGraspCfg
from env.TaskRobotEnv import RealmanGraspSingleGym

import torch


class RealManImageWrapper(gym.Env):

    def __init__(self, env, shape_meta, render_obs_key='camera_image'):

        self.env = env
        self.n_envs = env.num_envs 
        self.render_obs_key = render_obs_key
        self.shape_meta = shape_meta
        self.render_cache = None

        # -------- Action Space --------
        action_shape = shape_meta['action']['shape']

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.n_envs, *action_shape),
            dtype=np.float32
        )

        # -------- Observation Space --------
        observation_space = spaces.Dict()

        for key, value in shape_meta['obs'].items():
            shape = (self.n_envs, *value['shape'])

            if key.endswith('image'):
                low, high = 0, 1
            else:
                low, high = -np.inf, np.inf

            observation_space[key] = spaces.Box(
                low=low,
                high=high,
                shape=shape,
                dtype=np.float32
            )

        self.observation_space = observation_space

    def _to_numpy(self, x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return x

    def reset(self):
        raw_obs = self.env.reset()
        return self.get_raw_observation(raw_obs)

    def step(self, action):

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self.env.device)

        raw_obs, privileged_obs, reward, done, info = self.env.step(action)

        obs = self.get_raw_observation(raw_obs)

        reward = self._to_numpy(reward).astype(np.float32)
        done = self._to_numpy(done).astype(bool)

        return obs, privileged_obs, reward, done, info

    def get_raw_observation(self, raw_obs=None, keys_to_return=None):

        if raw_obs is None:
            raw_obs = self.env.reset()

        if isinstance(raw_obs, tuple):
            raw_obs = raw_obs[0]

        n_envs = raw_obs.shape[0]

        img_dim = 3 * 84 * 84

        wrist_flat = raw_obs[:, :img_dim]
        head_flat = raw_obs[:, img_dim:2*img_dim]
        state_obs = raw_obs[:, 2*img_dim:]

        wrist_img = wrist_flat.view(n_envs, 3, 84, 84)
        head_img = head_flat.view(n_envs, 3, 84, 84)

        robot0_qpos = state_obs[:, :7]
        robot0_gripper_qpos = state_obs[:, 7:8]
        robot_ee_pos = state_obs[:, 8:11]
        robot_ee_orn = state_obs[:, 11:15]

        obs_dict = {
            "agentview_image": wrist_img,
            "agentview_head_image": head_img,
            "robot0_qpos": robot0_qpos,
            "robot0_gripper_qpos": robot0_gripper_qpos,
            "robot_ee_pos": robot_ee_pos,
            "robot_ee_orn": robot_ee_orn,
        }

        # 筛选 key
        if keys_to_return is not None:
            obs_dict = {k: obs_dict[k] for k in keys_to_return if k in obs_dict}

        # 转 numpy（给 diffusion policy）
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu().numpy().astype("float32")

        return obs_dict


    def render(self, mode='rgb_array'):

        if self.render_cache is None:
            raise RuntimeError("Call reset or step first")

        img = (self.render_cache[0] * 255).astype(np.uint8)
        return img



def test_dp_wrapper():

    cfg = LinkGraspCfg()
    base_env = RealmanGraspSingleGym(cfg)

    shape_meta = {
        "action": {
            "shape": (8,)
        },
        "obs": {
            "agentview_image": {"shape": (3, 84, 84)},
            "agentview_head_image": {"shape": (3, 84, 84)},
            "robot0_qpos": {"shape": (7,)},
            "robot0_gripper_qpos": {"shape": (1,)},
        }
    }

    env = RealManImageWrapper(base_env, shape_meta)

    obs = env.reset()

    print("\n--- Observation after reset ---")
    for k, v in obs.items():
        print(f"{k}: shape={v.shape}, dtype={v.dtype}")
        assert isinstance(v, np.ndarray)
        assert np.all(np.isfinite(v)), f"{k} contains NaN"

    print("\nRunning one random step...")

    action = np.random.uniform(
        low=-1,
        high=1,
        size=(env.n_envs, 8)
    ).astype(np.float32)

    obs, privileged_obs, reward, done, info = env.step(action)

    print("\n--- After step ---")

    for k, v in obs.items():
        print(f"{k}: shape={v.shape}, dtype={v.dtype}")
        assert isinstance(v, np.ndarray)

    print(f"reward shape: {reward.shape}")
    print(f"reward dtype: {reward.dtype}")

    print(f"done shape: {done.shape}")
    print(f"done dtype: {done.dtype}")

    print("\n========== TEST SUCCESS ==========")


if __name__ == "__main__":
    test_dp_wrapper()



























# from typing import Optional
# import numpy as np
# import gym
# from gym import spaces
# class RealManImageWrapper(gym.Env):

#     def __init__(self, env, shape_meta, render_obs_key='camera_image'):

#         self.env = env
#         self.n_envs = env.n_envs
#         self.render_obs_key = render_obs_key
#         self.shape_meta = shape_meta
#         self.render_cache = None

#         # -------- Action Space --------
#         action_shape = shape_meta['action']['shape']

#         self.action_space = spaces.Box(
#             low=-1,
#             high=1,
#             shape=(self.n_envs, *action_shape),  # ✅ 多环境
#             dtype=np.float32
#         )

#         # -------- Observation Space --------
#         observation_space = spaces.Dict()

#         for key, value in shape_meta['obs'].items():
#             shape = value['shape']

#             # ✅ 前面加 n_envs 维度
#             shape = (self.n_envs, *shape)

#             if key.endswith('image'):
#                 low, high = 0, 1
#             else:
#                 low, high = -1, 1

#             observation_space[key] = spaces.Box(
#                 low=low,
#                 high=high,
#                 shape=shape,
#                 dtype=np.float32
#             )

#         self.observation_space = observation_space

#     def reset(self):
#         raw_obs = self.env.reset()
#         return self.get_observation(raw_obs)

#     def step(self, action):
#         raw_obs, reward, done, info = self.env.step(action)
#         obs = self.get_observation(raw_obs)
#         return obs, reward, done, info

#     def get_observation(self, raw_obs=None):
#         if raw_obs is None:
#             raw_obs = self.env.get_raw_observation()

#         self.render_cache = raw_obs[self.render_obs_key]

#         obs = {}
#         for key in self.observation_space.keys():
#             obs[key] = raw_obs[key]

#         return obs

#     def render(self, mode='rgb_array'):
#         if self.render_cache is None:
#             raise RuntimeError("Call reset or step first")

#         img = (self.render_cache[0] * 255).astype(np.uint8)
#         return img










# from typing import Optional
# import numpy as np
# import gym
# from gym import spaces

# class RealManImageWrapper(gym.Env):

#     def __init__(self, 
#                  env: gym.Env,
#                  shape_meta: dict,
#                  render_obs_key='camera_image'):

#         self.env = env
#         self.render_obs_key = render_obs_key
#         self.shape_meta = shape_meta
#         self.render_cache = None

#         # -------- Action Space --------
#         action_shape = shape_meta['action']['shape']
#         self.action_space = spaces.Box(
#             low=-1,
#             high=1,
#             shape=action_shape,
#             dtype=np.float32
#         )

#         # -------- Observation Space --------
#         observation_space = spaces.Dict()
#         for key, value in shape_meta['obs'].items():
#             shape = value['shape']
#             if key.endswith('image'):
#                 low, high = 0, 1
#             else:
#                 low, high = -1, 1

#             observation_space[key] = spaces.Box(
#                 low=low,
#                 high=high,
#                 shape=shape,
#                 dtype=np.float32
#             )
#         self.observation_space = observation_space

#     # Reset（只做一件事：重置环境）
#     def reset(self):
#         raw_obs = self.env.reset()
#         return self.get_observation(raw_obs)

#     def step(self, action):
#         raw_obs, reward, done, info = self.env.step(action)
#         obs = self.get_observation(raw_obs)
#         return obs, reward, done, info

#     # 只负责筛选 obs + cache 图像
#     def get_observation(self, raw_obs=None):
#         if raw_obs is None:
#             raw_obs = self.env.get_raw_observation()

#         self.render_cache = raw_obs[self.render_obs_key]
#         obs = {}
#         for key in self.observation_space.keys():
#             obs[key] = raw_obs[key]
#         return obs

#     # Render 用于 rollout 可视化

#     def render(self, mode='rgb_array'):
#         if self.render_cache is None:
#             raise RuntimeError("Call reset or step first")

#         img = (self.render_cache * 255).astype(np.uint8)
#         return img
