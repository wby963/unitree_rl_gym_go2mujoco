import os
import time
import argparse

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml

LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ================================================================
# GLFW 键码常量（MuJoCo viewer key_callback 使用）
# ================================================================
_K_SPACE = 32
_K_0 = 48
_K_2 = 50
_K_4 = 52
_K_6 = 54
_K_7 = 55
_K_8 = 56
_K_9 = 57
_K_A = 65
_K_D = 68
_K_E = 69
_K_I = 73
_K_J = 74
_K_K = 75
_K_L = 76
_K_O = 79
_K_Q = 81
_K_S = 83
_K_U = 85
_K_W = 87
_K_GRAVE = 96      # 反引号 `
_K_BACKSPACE = 259
_K_RIGHT = 262
_K_LEFT = 263
_K_DOWN = 264
_K_UP = 265

# ================================================================
# 命令档位（锁定模式，按一下就保持，不用一直按）
#   vx / vy 每档 0.5 m/s，yaw 每档 0.5 rad/s
# ================================================================
_VX_STEP = 0.5     # 前进/后退档位 (m/s)
_VY_STEP = 0.5
_WZ_STEP = 0.5

# 共享状态
_cmd_vx = 0.0
_cmd_vy = 0.0
_cmd_wz = 0.0
_reset_flag = False
_debug_flag = False

def _set_cmd(vx=None, vy=None, wz=None, zero_all=False):
    """设置命令档位（无上限，由用户自己控制）"""
    global _cmd_vx, _cmd_vy, _cmd_wz
    if zero_all:
        _cmd_vx, _cmd_vy, _cmd_wz = 0.0, 0.0, 0.0
        return
    if vx is not None:
        _cmd_vx = float(vx)
    if vy is not None:
        _cmd_vy = float(vy)
    if wz is not None:
        _cmd_wz = float(wz)


def _on_key(key):
    """
    MuJoCo viewer 键盘回调（边沿触发，按一次切换一档）。
    按键直接从 viewer 窗口获取，无需切换焦点，不依赖 pynput。
    """
    global _reset_flag, _debug_flag

    # 调试开关
    if key == _K_GRAVE:
        _debug_flag = not _debug_flag
        print(f"[DEBUG] 打印模式: {'ON' if _debug_flag else 'OFF'}")
        return

    # 停止（0 / 空格）
    if key == _K_0 or key == _K_SPACE:
        _set_cmd(zero_all=True)
        print(f"[CMD] 已停止  cmd = [{_cmd_vx:+.2f}, {_cmd_vy:+.2f}, {_cmd_wz:+.2f}]")
        return

    # 重置（Backspace）
    if key == _K_BACKSPACE:
        _reset_flag = True
        _set_cmd(zero_all=True)
        print("[CMD] 重置仿真 + 速度归零")
        return

    # 前进档（↑ / W / I / 8）：vx + 0.5
    if key in (_K_UP, _K_W, _K_I, _K_8):
        _set_cmd(vx=_cmd_vx + _VX_STEP)
    # 后退档（↓ / S / K / 2）：vx - 0.5
    elif key in (_K_DOWN, _K_S, _K_K, _K_2):
        _set_cmd(vx=_cmd_vx - _VX_STEP)
    # 左移档（← / A / J / 4）：vy + 0.5
    elif key in (_K_LEFT, _K_A, _K_J, _K_4):
        _set_cmd(vy=_cmd_vy + _VY_STEP)
    # 右移档（→ / D / L / 6）：vy - 0.5
    elif key in (_K_RIGHT, _K_D, _K_L, _K_6):
        _set_cmd(vy=_cmd_vy - _VY_STEP)
    # 左转档（Q / U / 7）：wz + 0.5
    elif key in (_K_Q, _K_U, _K_7):
        _set_cmd(wz=_cmd_wz + _WZ_STEP)
    # 右转档（E / O / 9）：wz - 0.5
    elif key in (_K_E, _K_O, _K_9):
        _set_cmd(wz=_cmd_wz - _WZ_STEP)
    else:
        return

    print(f"[CMD] cmd = [{_cmd_vx:+.2f}, {_cmd_vy:+.2f}, {_cmd_wz:+.2f}]")


def get_cmd():
    """返回已锁定的命令档位（物理单位 m/s, rad/s）"""
    return np.array([_cmd_vx, _cmd_vy, _cmd_wz], dtype=np.float32)


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def quat_to_mat(quat):
    w, x, y, z = quat
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
        [    2*(x*y + z*w),   1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),       2*(y*z + x*w),   1 - 2*(x**2 + y**2)]
    ])
    return R

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


KEYBOARD_HELP = """
================================================================
                 键盘控制说明 (Go2 MuJoCo)
           【锁定档位】按一下就保持，不用一直按
================================================================
 【按键直接从 MuJoCo 窗口获取】
   请确保鼠标焦点在 MuJoCo 仿真窗口上后按键。

 【方向键（推荐）- 每按一次 ±0.5】
   ↑          前进 +0.5 (vx += 0.5 m/s)
   ↓          后退 -0.5 (vx -= 0.5)
   ←          左移 +0.5 (vy += 0.5)
   →          右移 -0.5 (vy -= 0.5)
   Q          左转 +0.5 (yaw += 0.5 rad/s)
   E          右转 -0.5 (yaw -= 0.5)
   0 / 空格   停止：所有速度归零
   Backspace  重置仿真环境 + 速度归零
   ` (反引号) 切换调试打印

 【数字键备选 - 不触发 viewer 快捷键】
   8 前进+0.5   2 后退-0.5   4 左移+0.5   6 右移-0.5   7 左转+0.5   9 右转-0.5

 【示例】
   按一次 ↑ → [CMD] cmd = [+0.50, +0.00, +0.00]，速度 0.5 m/s
   再按一次 ↑ → cmd = [+1.00, +0.00, +0.00]，速度 1.0 m/s
   按一次 ↓ → cmd = [+0.50, +0.00, +0.00]，速度降回 0.5 m/s
   再按 ↓ 两次 → cmd = [-0.50, +0.00, +0.00]，后退 0.5 m/s
   按 0       → 全部归零，停止
================================================================
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file

    config_path = os.path.join(LEGGED_GYM_ROOT_DIR, "deploy/deploy_mujoco/configs", config_file)
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    _default_xml = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/scene1.xml"
    xml_path = config.get("xml_path", _default_xml).replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    if "xml_path" not in config:
        print(f"警告：config 中未找到 xml_path，使用默认值: {xml_path}")
        print(f"当前加载的配置文件: {config_path}")
        print(f"config 中包含的键: {list(config.keys())}")

    simulation_duration = config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)

    lin_vel_scale = config["lin_vel_scale"]
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

    num_actions = config["num_actions"]
    num_obs = config["num_obs"]

    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    print(KEYBOARD_HELP)

    # ---------- 加载 MuJoCo ----------
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    policy = torch.jit.load(policy_path)

    # ---------- 启动可视化（传入 key_callback，按键直接从 viewer 获取） ----------
    _debug_frame_counter = 0
    try:
        with mujoco.viewer.launch_passive(m, d, key_callback=_on_key) as viewer:
            start = time.time()
            counter = 0

            while viewer.is_running() and time.time() - start < simulation_duration:
                step_start = time.time()

                # ---------- 重置（Backspace） ----------
                if _reset_flag:
                    _reset_flag = False
                    mujoco.mj_resetData(m, d)
                    mujoco.mj_forward(m, d)
                    action = np.zeros(num_actions, dtype=np.float32)
                    target_dof_pos = default_angles.copy()
                    print("Environment reset!")

                # ---------- 获取当前命令档位 ----------
                cmd = get_cmd()

                # PD 控制
                tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
                tau = np.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)

                # ---------------------------------------------------------------
                # 腿序重映射
                #   MuJoCo actuator/ctrl 顺序（go2.xml）：FR → FL → RR → RL
                #   策略输出 tau 顺序（训练/默认角度）：FL → FR → RL → RR
                # ---------------------------------------------------------------
                # --- 右前腿 FR  ctrl[0..2] ---
                d.ctrl[0] = tau[3]    # FR_hip
                d.ctrl[1] = tau[4]    # FR_thigh
                d.ctrl[2] = tau[5]    # FR_calf
                # --- 左前腿 FL  ctrl[3..5] ---
                d.ctrl[3] = tau[0]    # FL_hip
                d.ctrl[4] = tau[1]    # FL_thigh
                d.ctrl[5] = tau[2]    # FL_calf
                # --- 右后腿 RR  ctrl[6..8] ---
                d.ctrl[6] = tau[9]    # RR_hip
                d.ctrl[7] = tau[10]   # RR_thigh
                d.ctrl[8] = tau[11]   # RR_calf
                # --- 左后腿 RL  ctrl[9..11] ---
                d.ctrl[9]  = tau[6]   # RL_hip
                d.ctrl[10] = tau[7]   # RL_thigh
                d.ctrl[11] = tau[8]   # RL_calf

                mujoco.mj_step(m, d)
                counter += 1

                if counter % control_decimation == 0:
                    qj = d.qpos[7:]
                    dqj = d.qvel[6:]
                    quat = d.qpos[3:7]
                    omega = d.qvel[3:6]
                    base_vel_world = d.qvel[:3]

                    R = quat_to_mat(quat)
                    base_vel_body = R.T @ base_vel_world
                    base_vel = base_vel_body * lin_vel_scale

                    qj = (qj - default_angles) * dof_pos_scale
                    dqj = dqj * dof_vel_scale
                    gravity_orientation = get_gravity_orientation(quat)
                    omega = omega * ang_vel_scale

                    obs[:3] = base_vel
                    obs[3:6] = omega
                    obs[6:9] = gravity_orientation
                    # cmd 是物理单位(m/s, rad/s)，乘 cmd_scale 归一化送入观测（与训练一致）
                    obs[9:12] = cmd * cmd_scale
                    obs[12:12+num_actions] = qj
                    obs[12+num_actions:12+2*num_actions] = dqj
                    obs[12+2*num_actions:12+3*num_actions] = action

                    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                    action = policy(obs_tensor).detach().numpy().squeeze()
                    target_dof_pos = action * action_scale + default_angles

                    # ---------- DEBUG 打印（反引号 ` 开关） ----------
                    if _debug_flag:
                        _debug_frame_counter += 1
                        if _debug_frame_counter % 5 == 0:  # ~10Hz
                            print(
                                f"[DBG] cmd_target=[{cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}]  "
                                f"base_vel=[{base_vel_body[0]:+.2f},{base_vel_body[1]:+.2f},{base_vel_body[2]:+.2f}]  "
                                f"pos=[{d.qpos[0]:+.2f},{d.qpos[1]:+.2f},{d.qpos[2]:+.2f}]  "
                                f"act_max={np.max(np.abs(action)):.3f}"
                            )

                viewer.sync()

                elapsed = time.time() - step_start
                sleep_time = m.opt.timestep - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
    except TypeError:
        # 旧版 mujoco 不支持 key_callback 参数，回退到无回调模式
        print("错误：当前 mujoco 版本不支持 key_callback，请升级 mujoco: pip install -U mujoco")
        raise

    print("Simulation finished.")
