import numpy as np
from NexusMinds_RL.env.Robot.gym_env.sim.pygym_DexGrasp import Gym
import torch


class RealmanEnv:
    """
    Isaac Gym batched environment
    支持 n_envs 并行
    """

    def __init__(self, config, args):
        self.env = Gym(args)

        self.config = config
        self.n_envs = config.get("n_envs", 8)

        # ===== 基本参数 =====
        self.max_steps = config.get("max_steps", 400)
        self.table_height = config.get("table_height", 0.0)
        self.lift_threshold = config.get("lift_threshold", 0.05)
        self.success_hold_steps = config.get("success_hold_steps", 5)


        self.asset_root = config.get("asset_root", "./assets")
        self.asset_file = config.get("asset_filt", "realman_*.urdf")
        self.base_pos = [0, 0.25, 0]
        self.base_orn = [0, 0, 0, 1]
        self.control_type = config.get("position")
        self.obs_type = None
        self.robot_type = config.get("robot_type", "realman")
        self.d_max = 0.1409
        self.q_max = -0.91

        self.env.pre_simulate(self.n_envs, self.asset_root, self.asset_file, self.base_pos, self.base_orn, self.control_type, self.obs_type, self.robot_type)
 
        # ===== batched 状态变量 =====
        self.step_count = np.zeros(self.n_envs, dtype=np.int32)
        self.success_counter = np.zeros(self.n_envs, dtype=np.int32)

        # 状态必须是 batch
        self._state = None

    # ======================================================
    # Reset
    # ======================================================
    def reset(self):

        self.env.reset_joint_states(env_ids)
        self.env.reset_object_states(env_ids)

        self.step_count[:] = 0
        self.success_counter[:] = 0

        self._state = self._init_state()  # 必须返回 batch state

        self._apply_state(self._state)

        return self.get_raw_observation()

    # ======================================================
    # Step
    # ======================================================
    def step(self, action: np.ndarray):
        """
        action shape = (n_envs, action_dim)
        """
        #action扩展成24维，前16维就初始关节角就行
        u1 = action[:, :16]
        u2 = action[:, 16:23]
        d = action[:, 23] 
        q1 = (d/self.d_max * self.q_max).unsqueeze(-1)
        q2 = -q1.unsqueeze(-1)
        q3 = q1.unsqueeze(-1)
        q4 = -q1.unsqueeze(-1)
        q5 = -q1.unsqueeze(-1)
        q6 = -q1.unsqueeze(-1)
        gripper_q = torch.cat([q1, q2, q3, q4, q5, q6], dim=1)
        u = torch.cat([u1, u2, gripper_q], dim=1)

       
        # 1️⃣ 应用 batch 动作
        self.env.step(u, self.control_type, self.obs_type)  

        # 2️⃣ 更新 batch 状态
        self._state = self._update_state()

        # 3️⃣ 步数增加
        self.step_count += 1

        # 4️⃣ 生成观测
        raw_obs = self.get_raw_observation()

        # 5️⃣ 计算 reward & success
        reward, success = self._compute_reward()

        # 6️⃣ 计算 done
        done = self._check_done(success)

        info = {
            "success": success,
            "step_count": self.step_count.copy()
        }

        return raw_obs, reward, done, info

    # ======================================================
    # Reward
    # ======================================================
    def _compute_reward(self):

        # obj_height shape = (n_envs,)
        middle_point_to_object_distance = self.env.get_right_gripper_to_object_distance()
        threshold = 0.01

        success = middle_point_to_object_distance < threshold

        # 更新 success counter（batch 版本）
        self.success_counter = np.where(
            success,
            self.success_counter + 1,
            0
        )

        stable_success = self.success_counter >= self.success_hold_steps

        reward = stable_success.astype(np.float32)

        return reward, stable_success

    # ======================================================
    # Done
    # ======================================================
    def _check_done(self, success):

        timeout = self.step_count >= self.max_steps

        done = np.logical_or(success, timeout)

        return done

    # ======================================================
    # Observation
    # ======================================================
    def get_raw_observation(self):

        return {
            # 形状必须是 (n_envs, C, H, W)
            "camera_image": self.env.get_right_wrist_image(),

            "camera_head_image": self.env.get_head_image(),

            # (n_envs, 7)
            "robot_qpos": self.env.get_joint_pos()[:, 16:23].squeeze(-1),

            # (n_envs, 1)
            "robot_end_qpos": self.env.get_gripper_width(),

            # (n_envs, 3)
            "ee_pos": self.env.get_right_ee_position(),

            # (n_envs, 3)
            "ee_quat": self.env.get_right_ee_orientation(),#这里是四元数需要转欧拉角
        }





# import numpy as np


# class RealmanEnv:
#     """
#     Isaac Gym batched environment
#     支持 n_envs 并行
#     """

#     def __init__(self, config):

#         self.config = config
#         self.n_envs = config.get("n_envs", 8)

#         # ===== 基本参数 =====
#         self.max_steps = config.get("max_steps", 400)
#         self.table_height = config.get("table_height", 0.0)
#         self.lift_threshold = config.get("lift_threshold", 0.05)
#         self.success_hold_steps = config.get("success_hold_steps", 5)

#         # ===== batched 状态变量 =====
#         self.step_count = np.zeros(self.n_envs, dtype=np.int32)
#         self.success_counter = np.zeros(self.n_envs, dtype=np.int32)

#         # 状态必须是 batch
#         self._state = None

#     # ======================================================
#     # Reset
#     # ======================================================
#     def reset(self):

#         self.step_count[:] = 0
#         self.success_counter[:] = 0

#         self._state = self._init_state()  # 必须返回 batch state

#         self._apply_state(self._state)

#         return self.get_raw_observation()

#     # ======================================================
#     # Step
#     # ======================================================
#     def step(self, action: np.ndarray):
#         """
#         action shape = (n_envs, action_dim)
#         """

#         # 1️⃣ 应用 batch 动作
#         self._apply_action(action)

#         # 2️⃣ 更新 batch 状态
#         self._state = self._update_state()

#         # 3️⃣ 步数增加
#         self.step_count += 1

#         # 4️⃣ 生成观测
#         raw_obs = self.get_raw_observation()

#         # 5️⃣ 计算 reward & success
#         reward, success = self._compute_reward()

#         # 6️⃣ 计算 done
#         done = self._check_done(success)

#         info = {
#             "success": success,
#             "step_count": self.step_count.copy()
#         }

#         return raw_obs, reward, done, info

#     def _compute_reward(self):

#         # -------------------------
#         # 1️⃣ 获取状态
#         # -------------------------
#         ee_pos = self._get_ee_pos()              # (n_envs, 3)
#         obj_pos = self._get_object_pos()         # (n_envs, 3)
#         gripper_width = self._get_gripper_width()  # (n_envs,)

#         # -------------------------
#         # 2️⃣ 计算末端与物体距离
#         # -------------------------
#         dist = np.linalg.norm(ee_pos - obj_pos, axis=-1)

#         grasp_dist_thresh = 0.03   # 3cm，可根据实际调整
#         near_object = dist < grasp_dist_thresh

#         # -------------------------
#         # 3️⃣ 夹爪闭合判定
#         # -------------------------
#         # 1000=张开，0=闭合
#         gripper_close_thresh = 600

#         gripper_closed = gripper_width < gripper_close_thresh

#         # -------------------------
#         # 4️⃣ 成功判定
#         # -------------------------
#         grasp_success = np.logical_and(near_object, gripper_closed)

#         # 稳定判定（连续多步成立才算成功）
#         self.success_counter = np.where(
#             grasp_success,
#             self.success_counter + 1,
#             0
#         )

#         stable_success = self.success_counter >= self.success_hold_steps

#         # -------------------------
#         # 5️⃣ 奖励
#         # -------------------------
#         reward = stable_success.astype(np.float32)

#         return reward, stable_success
#     # Done
#     def _check_done(self, success):

#         timeout = self.step_count >= self.max_steps

#         done = np.logical_or(success, timeout)

#         return done

#     # Observation
#     def get_raw_observation(self):

#         return {
#             # 形状必须是 (n_envs, C, H, W)
#             "camera_image": self._render_camera(),

#             "camera_head_image": self._render_camera_head(),

#             # (n_envs, 7)
#             "robot_qpos": self._get_qpos(),

#             # (n_envs, 1)
#             "robot_end_qpos": self._get_robot_end_qpos(),

#             # (n_envs, 3)
#             "ee_pos": self._get_ee_pos(),

#             # (n_envs, 3)
#             "ee_quat": self._get_ee_quat(),
#         }









# import numpy as np

# class RealmanEnv:
#     """
#     最小可用底层环境
#     目标：能被 RealManImageWrapper 正确包住
#     """

#     def __init__(self, config):
#         """
#         config: 任意环境参数
#         """
#         self.config = config

#         # ===== 基本参数 =====
#         self.max_steps = config.get("max_steps", 400)
#         self.table_height = config.get("table_height", 0.0)
#         self.lift_threshold = config.get("lift_threshold", 0.05)

#         # 稳定成功判定帧数
#         self.success_hold_steps = config.get("success_hold_steps", 5)

#         # ===== 内部变量 =====
#         self._state = None
#         self.step_count = 0
#         self.success_counter = 0

#     # Reset
#     def reset(self):
#         """
#         初始化环境状态
#         return:
#             raw_obs (dict)
#         """
#         # 清空变量
#         self.step_count = 0
#         self.success_counter = 0

#         # 1. 初始化内部状态（位置、物体等）
#         self._state = self._init_state()

#         # 2. 将状态写入仿真 / 真实环境
#         self._apply_state(self._state)

#         # 3. 返回观测
#         return self.get_raw_observation()

#     def step(self, action: np.ndarray):

#         # 1. 执行动作
#         self._apply_action(action)  #8维动作

#         # 2. 更新环境状态
#         self._state = self._update_state()

#         self.step_count += 1

#         # 3. 生成观测
#         raw_obs = self.get_raw_observation()

#         # 4. 计算 reward / done（任务相关）
#         reward, success = self._compute_reward()

#         done = self._check_done(success)

#         info = {
#             "success": success,
#             "step_count": self.step_count
#         }

#         return raw_obs, reward, done, info
    
#     def _compute_reward(self):

#         obj_height = self._get_object_height()

#         success = obj_height > (self.table_height + self.lift_threshold)

#         # 稳定成功判定
#         if success:
#             self.success_counter += 1
#         else:
#             self.success_counter = 0

#         stable_success = self.success_counter >= self.success_hold_steps

#         reward = 1.0 if stable_success else 0.0

#         return reward, stable_success

    
#     def _check_done(self, success):

#         timeout = self.step_count >= self.max_steps

#         done = success or timeout

#         return done

#     def get_raw_observation(self):
#         """
#         必须返回 dict
#         key / shape 与 shape_meta['obs'] 完全一致
#         """
#         return {
#             "camera_image": self._render_camera(),  # (C,H,W)
#             "camera_head_image": self._render_camera_head(),  # (C,H,W)
#             "robot_qpos":   self._get_qpos(),        # (N,7)
#             "robot_end_qpos": self._get_robot_end_qpos(),  # (N,1)
#             "ee_pos":       self._get_ee_pos(),      # (3)
#             "ee_quat":      self._get_ee_quat(),     # (3)rxryrz
#         }

