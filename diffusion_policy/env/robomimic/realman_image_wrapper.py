from typing import Optional
import numpy as np
import gym
from gym import spaces
class RealManImageWrapper(gym.Env):

    def __init__(self, env, shape_meta, render_obs_key='camera_image'):

        self.env = env
        self.n_envs = env.n_envs
        self.render_obs_key = render_obs_key
        self.shape_meta = shape_meta
        self.render_cache = None

        # -------- Action Space --------
        action_shape = shape_meta['action']['shape']

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.n_envs, *action_shape),  # ✅ 多环境
            dtype=np.float32
        )

        # -------- Observation Space --------
        observation_space = spaces.Dict()

        for key, value in shape_meta['obs'].items():
            shape = value['shape']

            # ✅ 前面加 n_envs 维度
            shape = (self.n_envs, *shape)

            if key.endswith('image'):
                low, high = 0, 1
            else:
                low, high = -1, 1

            observation_space[key] = spaces.Box(
                low=low,
                high=high,
                shape=shape,
                dtype=np.float32
            )

        self.observation_space = observation_space

    def reset(self):
        raw_obs = self.env.reset()
        return self.get_observation(raw_obs)

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)
        return obs, reward, done, info

    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_raw_observation()

        self.render_cache = raw_obs[self.render_obs_key]

        obs = {}
        for key in self.observation_space.keys():
            obs[key] = raw_obs[key]

        return obs

    def render(self, mode='rgb_array'):
        if self.render_cache is None:
            raise RuntimeError("Call reset or step first")

        img = (self.render_cache[0] * 255).astype(np.uint8)
        return img


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
