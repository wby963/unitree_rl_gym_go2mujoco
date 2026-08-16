import sys
import os
import pygame
import struct
import numpy as np
import time

# unitree dds
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.utils.thread import RecurrentThread

os.environ["SDL_VIDEODRIVER"] = "dummy"
TOPIC_WIRELESS_CONTROLLER = "rt/wirelesscontroller"


class DeployJoystick:
    def __init__(self):
        self.joystick = None

        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.wireless_controller_puber = ChannelPublisher(
            TOPIC_WIRELESS_CONTROLLER, WirelessController_
        )
        self.wireless_controller_puber.Init()

        self.WirelessControllerThread = RecurrentThread(
            interval=0.01,
            target=self.PublishWirelessController,
            name="sim_wireless_controller",
        )
        self.WirelessControllerThread.Start()


        # joystick
        self.key_map = {
            "R1": 0,
            "L1": 1,
            "start": 2,
            "select": 3,
            "R2": 4,
            "L2": 5,
            "F1": 6,
            "F2": 7,
            "A": 8,
            "B": 9,
            "X": 10,
            "Y": 11,
            "up": 12,
            "right": 13,
            "down": 14,
            "left": 15,
        }
        # 反向字典
        self.reverse_key_map = {value: key for key, value in self.key_map.items()}
        # press
        self.last_keys=set()

    def SetupJoystick(self):
        pygame.init()
        # pygame.display.quit()  # 确保显示系统关闭
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()

        if joystick_count > 0:
            self.joystick=pygame.joystick.Joystick(0)
            self.joystick.init()
        else:
            print("No Gamepad!")
            sys.exit()

        # xbox
        self.axis_id = {
                "LX": 0,  # Left stick axis x
                "LY": 1,  # Left stick axis y
                "RX": 3,  # Right stick axis x
                "RY": 2,  # Right stick axis y

                "LT": 4,  # Left trigger
                "RT": 5,  # Right trigger
                "DX": 6,  # Directional pad x
                "DY": 7,  # Directional pad y
            }

        self.button_id = {
                "X": 2,
                "Y": 3,
                "B": 1,
                "A": 0,
                "LB": 4,
                "RB": 5,
                "SELECT": 6,
                "START": 7,
            }


    # 解析遥控器按键 
    def GetJoystickKeys(self,keys,re_keys):
            keys_bin=bin(keys)[2:].zfill(16)
            keys_bin_int=int(keys_bin,2)
            key_pos=[]
            key_press=[]

            if keys:
                for i in range(16):
                    if keys_bin_int & (1<<i):
                        key_pos.append(i)
                        key_press.append(re_keys[i])
            return key_press 


    def PublishWirelessController(self):
        if self.joystick != None:
            pygame.event.get()
            key_state = [0] * 16
            key_state[self.key_map["R1"]] = self.joystick.get_button(
                self.button_id["RB"]
            )
            key_state[self.key_map["L1"]] = self.joystick.get_button(
                self.button_id["LB"]
            )
            key_state[self.key_map["start"]] = self.joystick.get_button(
                self.button_id["START"]
            )
            key_state[self.key_map["select"]] = self.joystick.get_button(
                self.button_id["SELECT"]
            )
            key_state[self.key_map["R2"]] = (
                self.joystick.get_axis(self.axis_id["RT"]) > 0
            )
            key_state[self.key_map["L2"]] = (
                self.joystick.get_axis(self.axis_id["LT"]) > 0
            )
            key_state[self.key_map["F1"]] = 0
            key_state[self.key_map["F2"]] = 0
            key_state[self.key_map["A"]] = self.joystick.get_button(self.button_id["A"])
            key_state[self.key_map["B"]] = self.joystick.get_button(self.button_id["B"])
            key_state[self.key_map["X"]] = self.joystick.get_button(self.button_id["X"])
            key_state[self.key_map["Y"]] = self.joystick.get_button(self.button_id["Y"])
            key_state[self.key_map["up"]] = self.joystick.get_hat(0)[1] > 0
            key_state[self.key_map["right"]] = self.joystick.get_hat(0)[0] > 0
            key_state[self.key_map["down"]] = self.joystick.get_hat(0)[1] < 0
            key_state[self.key_map["left"]] = self.joystick.get_hat(0)[0] < 0

            key_value = 0
            for i in range(16):
                key_value += key_state[i] << i

            self.wireless_controller.keys = key_value
            self.wireless_controller.lx = self.joystick.get_axis(self.axis_id["LX"])
            self.wireless_controller.ly = -self.joystick.get_axis(self.axis_id["LY"])
            self.wireless_controller.rx = self.joystick.get_axis(self.axis_id["RX"])
            self.wireless_controller.ry = -self.joystick.get_axis(self.axis_id["RY"])

            self.wireless_controller_puber.Write(self.wireless_controller)
            self.press_keys=self.GetJoystickKeys(self.wireless_controller.keys,self.reverse_key_map)
            
            # if self.press_keys:
            #     print(self.press_keys)


    def JoykeysPress(self):
        """
        更新按键状态
        :param current_keys: list 或 set,例如 ['select', 'start']
        :return: 一个 dict,包含哪些键刚刚按下/刚刚释放
        """
        current_keys = self.press_keys

        current_keys = set(current_keys)
        press = current_keys - self.last_keys      # 新按下
        releas = self.last_keys - current_keys     # 刚释放

        # 保存这次的状态
        self.last_keys = current_keys

        return {
            "press": press,
            "releas": releas
        }

# if __name__ == "__main__":
#     ChannelFactoryInitialize(1)
#     DeployPad=DeployJoystick()
#     DeployPad.SetupJoystick()
#     while True:
#         DeployPad.PublishWirelessController()
#         time.sleep(0.02)
