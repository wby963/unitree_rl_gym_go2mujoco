import os


import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml

from deploy_joystick import DeployJoystick

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_

TOPIC_WIRELESS_CONTROLLER = "rt/wirelesscontroller"
LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#LEGGED_GYM_ROOT_DIR = "Home/unitree_rl_gym"

# 获取四元数
def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    # 重力方向
    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def quat_to_mat(quat):
    """将四元数 (w, x, y, z) 转换为旋转矩阵"""
    w, x, y, z = quat
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),       2*(x*z + y*w)],
        [    2*(x*y + z*w),   1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
        [    2*(x*z - y*w),       2*(y*z + x*w),   1 - 2*(x**2 + y**2)]
    ])
    return R

# PD 控制器  目标位置  当前位置  Kp 目标速度  当前速度  Kd
def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    """PD 控制器 计算输出扭矩 """
    return (target_q - q) * kp + (target_dq - dq) * kd


def WirelessControllerHandler(msg:WirelessController_):
    global joymsg
    joymsg=msg

if __name__ == "__main__":
    # get config file name from command line
    import argparse
    # 获取传入参数
    parser = argparse.ArgumentParser()
    # 设定参数名称和类型 config_file  str  提示
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()     # 获取参数
    config_file = args.config_file # 传入参数
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        # 加载yaml参数
        config = yaml.load(f, Loader=yaml.FullLoader)
        # 策略路径      replace 用具体参数替代yaml中的位置 
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        # 加载mujoco模型 
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        # 仿真时间
        simulation_duration = config["simulation_duration"]
        # 仿真时间步长
        simulation_dt = config["simulation_dt"]
        # 控制周期倍率  control freq = comtrol_decimation * dt
        control_decimation = config["control_decimation"]

        # 加载PD参数 并设置为np.array格式
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        # 加载初始化默认关节角度
        default_angles = np.array(config["default_angles"], dtype=np.float32)

        lin_vel_scale = config["lin_vel_scale"]
        # yaw角速度缩放系数?
        ang_vel_scale = config["ang_vel_scale"]
        # 关节角度缩放系数
        dof_pos_scale = config["dof_pos_scale"]
        # 关节速度缩放系数
        dof_vel_scale = config["dof_vel_scale"]
        # 动作缩放系数？
        action_scale = config["action_scale"]
        # 控制指令缩放系数
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        # 动作空间
        num_actions = config["num_actions"]
        # 观测器空间
        num_obs = config["num_obs"]
        # 初始化指令
        cmd = np.array(config["cmd_init"], dtype=np.float32)
        # joy
        use_joystick=config["joystick"]

    # define context variables
    # 初始化动作 目标位置 观测器 
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    # Joystick初始化
    if use_joystick:
        ChannelFactoryInitialize(1)
        DeployJoy=DeployJoystick()
        DeployJoy.SetupJoystick()

        wirelesscontroller = ChannelSubscriber(TOPIC_WIRELESS_CONTROLLER, WirelessController_)
        wirelesscontroller.Init(WirelessControllerHandler, 10)


    counter = 0

    # Load robot model
    # 加载机器人Mujoco模型
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt


    # 初始化输出机器人关节信息 
    print("<<------------- Link ------------->> ")
    for i in range(m.nbody):
        name = mujoco.mj_id2name(m, mujoco._enums.mjtObj.mjOBJ_BODY, i)
        if name:
            print("link_index:", i, ", name:", name)
    print(" ")

    print("<<------------- Joint ------------->> ")
    for i in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco._enums.mjtObj.mjOBJ_JOINT, i)
        if name:
            print("joint_index:", i, ", name:", name)
    print(" ")

    print("<<------------- Actuator ------------->>")
    for i in range(m.nu):
        name = mujoco.mj_id2name(
            m, mujoco._enums.mjtObj.mjOBJ_ACTUATOR, i
        )
        if name:
            print("actuator_index:", i, ", name:", name)
    print(" ")

    print("<<------------- Sensor ------------->>")
    index = 0
    for i in range(m.nsensor):
        name = mujoco.mj_id2name(
            m, mujoco._enums.mjtObj.mjOBJ_SENSOR, i
        )
        if name:
            print(
                "sensor_index:",
                index,
                ", name:",
                name,
                ", dim:",
                m.sensor_dim[i],
            )
        index = index + m.sensor_dim[i]
    print(" ")



    # load policy
    # 加载控制器策略！！！ policy.pt
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        # 仿真总时间 duration
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time() # 单步时间记录
            # PD控制器计算输出力矩 初始化力矩 目标位置 | 位置反馈| Kp | 目标速度为0 阻尼  |关节速度反馈| Kd 
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            
            # 手柄控制
            if use_joystick:
                cmd[0]=-joymsg.rx
                cmd[1]=joymsg.ry
                cmd[2]=-joymsg.lx

                joykeys_event=DeployJoy.JoykeysPress()

                if 'select' in joykeys_event["press"]:
                    mujoco.mj_resetData(m, d)
                    mujoco.mj_forward(m, d)   # 更新一次，保证观测量正确
                    print("Environment reset!")
            

            # print(target_dof_pos[3],d.qpos[7:][3],tau[3])
            # print(d.qvel[6:])
            # 发送控制指令
            # d.ctrl[:] = tau
            # print(tau)
            tau=np.nan_to_num(tau,nan=0.0,posinf=0.0,neginf=0.0)

            # --- 左前腿 (FL) ---
            d.ctrl[0] = tau[3]   # FL_hip
            d.ctrl[1] = tau[4]   # FL_thigh
            d.ctrl[2] = tau[5]   # FL_calf

            # --- 左后腿 (RL) ---
            d.ctrl[3] = tau[0]   # RL_hip
            d.ctrl[4] = tau[1]   # RL_thigh
            d.ctrl[5] = tau[2]   # RL_calf

            # --- 右前腿 (FR) ---
            d.ctrl[6] = tau[9]   # FR_hip
            d.ctrl[7] = tau[10]   # FR_thigh
            d.ctrl[8] = tau[11]   # FR_calf

            # --- 右后腿 (RR) ---
            d.ctrl[9]  = tau[6]   # RR_hip
            d.ctrl[10] = tau[7]  # RR_thigh
            d.ctrl[11] = tau[8]  # RR_calf

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d) 

            counter += 1
            #                     10
            if counter % control_decimation == 0:
                # Apply control signal here.

                # create observation
                # 观测器
                # d.qpos存储所有自由度的位置，前7个元素是根节点的位置（3d）+四元数姿态（4d），
                #       [7:]是后续关节的位置
                qj = d.qpos[7:]     # 关节位置
                # d.qvel中存储所有自由度的速度，前六个元素是根节点的线速度（3d）+角速度（3d）,
                #       [6:]是后续关节的速度
                dqj = d.qvel[6:]    # 关节速度
                quat = d.qpos[3:7]  # 四元数
                omega = d.qvel[3:6] # 角速度
                # go2新增 根节点线速度
                base_vel_world = d.qvel[:3]

                # 现实世界很难获得，需要通过卡尔曼观测器估计机体三维线速度信息
                # 应该是机体坐标系下的速度而不是世界坐标系
                base_vel_body=np.zeros(3)
                R=quat_to_mat(quat)
                base_vel_body=R.T @ base_vel_world

                base_vel=base_vel_body*lin_vel_scale
                # print(base_vel_world,base_vel_body)

                # 关节位置 = ( 关节位置 - 初始化角度 )*关节位置缩放系数（1.0）
                qj = (qj - default_angles) * dof_pos_scale
                # 关节速度 = 关节速度 * 速度缩放系数（0.05）
                dqj = dqj * dof_vel_scale
                # 重力分量
                gravity_orientation = get_gravity_orientation(quat)
   
                # yaw角速度*缩放系数（0.25）
                omega = omega * ang_vel_scale



                # print(cmd)

                # # 周期
                # period = 0.8
                # count = counter * simulation_dt
                # # 相位
                # phase = count % period / period
                # sin_phase = np.sin(2 * np.pi * phase)
                # cos_phase = np.cos(2 * np.pi * phase)

                # print(base_vel,cmd* cmd_scale)

                # 打包观测器 打包成一维
                obs[:3] = base_vel
                obs[3:6] = omega
                obs[6:9] = gravity_orientation
                obs[9:12] = cmd * cmd_scale
                #obs[12,24]
                obs[12 : 12 + num_actions] = qj
                # obs[24:36]
                obs[12 + num_actions : 12 + 2 * num_actions] = dqj
                # obs[36,48] 上一时刻的动作
                obs[12 + 2 * num_actions : 12 + 3 * num_actions] = action
                # obs[45:47] sin cos 信号 相位信息
                # obs[9 + 3 * num_actions : 9 + 3 * num_actions + 2] = np.array([sin_phase, cos_phase])
                # 将numpy数组转化为Pytorch张量（共享内存）添加批次维度 [obs_dim]->[1,obs_dim]
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                # 策略推理      前向传播      |切断梯度计算|切回numpy|去除批次纬度[1,action_dim]->[action_dim] 
                action = policy(obs_tensor).detach().numpy().squeeze()

                
                # print(action)
                # transform action to target_dof_pos
                # 返回最终的关节位置=动作缩放后+初始位置
                target_dof_pos = action * action_scale + default_angles
                # target_dof_pos = default_angles

                # print(obs)
                #print(target_dof_pos)


            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            # 更新GUI 
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
