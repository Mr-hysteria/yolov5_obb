"""
本程序完成了三个步骤的内容：
1.目标识别
2.坐标转换
3.运动控制(第二个线程）
"""
import time

from main_detect_lib import *
import threading

# 先把机器人运行到初始位置,且关闭夹爪
erobot.set_angles(initA, 1080)
erobot.set_digital_out(1,1)
erobot.set_digital_out(0,0)

# yolo开始检测
print("------等待yolov5启动------\n")
t1 = threading.Thread(target=realsense_detect)  # Press esc or 'q' to close the Thread
t1.start()
time.sleep(18)


# 开始校准
print("------机器人开始运动------\n")
move_1()
t1.join()

# 旋转夹爪
print("------旋转夹爪-----")
move_3()


# 打开夹爪,先把两者电位复位，pin0高点平打开，pin1高电平关闭
erobot.set_digital_out(1, 0)
erobot.set_digital_out(0, 1)
move_2()

erobot.set_digital_out(0, 0)
erobot.set_digital_out(1, 1)
time.sleep(2)

# # 路点1
# pose_now = erobot.get_coords()
# pose_now[2] = pose_now[2] + 50
# erobot.set_coords(pose_now, 2000)
# erobot.wait_command_done()
#
#
# # 路点2
# erobot.set_angles([0, 0, -90, -90, -90, 60], 720)
# erobot.wait_command_done()
# # 路点3
# erobot.set_angles(initA,720)
# erobot.wait_command_done()

# 下降路点
# a = [51.001237, 6.070218, 38.357388, -223.945312, 39.199219, 59.941406]
# erobot.set_angles(a,720)
# erobot.wait_command_done()
# time.sleep(2)
# erobot.set_digital_out(1,0)
# erobot.set_digital_out(0,1)
