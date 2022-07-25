"""
控制机械手去该目标点处
1.机械臂末端的旋转矩阵是确定的，平行于桌面即可（Rx，Ry，Rz）即可确定下来
2.Z轴先不纳入控制范围，先把末端的Z轴对齐杯子中心（中心点在base中的表示已经得到，又因为base的YZ轴是平行杯子的平面，因此只需要设置setcoords中的yz即可）
3.设置延时，当走到一定点的时候，再次对比，如果相距比较低了，则停止

 [172.030366, 422.283023, 173.689831, -77.640983, 63.910554, 105.44709]可以作为初始位置
"""

# 在程序的开头就得设置,运行到初始位置
import time

from elephant import elephant_command
from global_v import *

erobot = elephant_command()

# chushiweizhi = initA_g  # 最近都是在这个点进行的抓取

# robot_pos = erobot.get_coords()
# robot_pos[4] -=60
# erobot.set_coords(robot_pos, 100)
# erobot.wait_command_done()


# 夹爪打开
# time.sleep(4)
# erobot.set_digital_out(1,0)
# erobot.set_digital_out(0,1)

# guanbi
# erobot.set_digital_out(1,1)
# erobot.set_digital_out(0,0)

# 重要初始位置
chushiweizhi =  [-335,183,707,90,33,-180]
erobot.set_coords(chushiweizhi,2000)
# erobot.set_angles(chushiweizhi,720)
#
# print(erobot.get_angles())
# print(erobot.get_coords())


