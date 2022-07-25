from elephant import elephant_command

# 初始化机器人
erobot = elephant_command()


# erobot.set_digital_out(0,1)  # 打开
erobot.set_digital_out(1,0)
erobot.set_digital_out(0,1)

# guanbi
# erobot.set_digital_out(1,1)
# erobot.set_digital_out(0,0)