"""
用yolov5识别目标位置，并得到相机坐标系下的位置
注意：reaslsense采集的是bgr通道的图像，目的是为了符合opencv的要求。预测的时候需要转化成RGB
修改的：
1.模型权重文件路径
2.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)，修改分辨率与帧率。最好训练是多少就多少
"""


import pyrealsense2 as rs
import numpy as np
import cv2
import random
import torch


model = torch.hub.load('/home/desktop-tjj/yolov5_obb', 'custom', path='/home/desktop-tjj/yolov5_obb/weights/best.pt',
                           source='local')  # 加载模型
# model = torch.hub.load('ultralytics/yolov5', 'yolov5l6')
model.eval()


def get_mid_pos(frame, box, depth_data, randnum):
    distance_list = []
    mid_pos = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]  # 确定中心点
    min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1]))  # 以中心点为中心，确定深度搜索范围（方框宽度最小值）
    # randnum是为了多取一些值来取平均
    for i in range(randnum):
        bias = random.randint(-min_val // 4, min_val // 4)  # 随机偏差,控制被除数大小即可控制范围
        dist = depth_data.get_distance(int(mid_pos[0] + bias), int(mid_pos[1] + bias))  # 单位为m
        # dist = depth_data[int(mid_pos[0] + bias), int(mid_pos[1] + bias)]

        if dist:
            distance_list.append(dist)
    distance_list = np.array(distance_list)
    distance_list = np.sort(distance_list)[randnum // 2 - randnum // 4:randnum // 2 + randnum // 4]  # 冒泡排序+中值滤波
    return np.mean(distance_list)


# 这个函数主要是在原图上画框,标出深度信息,此外用毫米看比较方便
def dectshow(org_img, boxs, depth_data, intrin):
    img = org_img.copy()
    for box in boxs:
        dist = get_mid_pos(org_img, box, depth_data, 24)  # 单位m
        camera_coordinate = rs.rs2_deproject_pixel_to_point(intrin=intrin,
                                                            pixel=[(box[0] + box[2]) // 2, (box[1] + box[3]) // 2],
                                                            depth=dist)
        print('像素坐标：', [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2, dist*1000])
        print('相机坐标：', [camera_coordinate[0]*1000, camera_coordinate[1]*1000, camera_coordinate[2]*1000])
        print('\n')
        text_pixel = str((int(box[0]) + int(box[2])) // 2) + ', ' + str((int(box[1]) + int(box[3])) // 2)+ '(pixel)'
        text_camera = str(camera_coordinate[0]*1000)[:4] + ', '+str(camera_coordinate[1]*1000)[:4]+'(mm)'
        center = [(int(box[0]) + int(box[2])) // 2, (int(box[1]) + int(box[3])) // 2]

        # rectangle画框，参数表示依次为：(图片，长方形框左上角坐标, 长方形框右下角坐标， 字体颜色，字体粗细)
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.circle(img, center, 8, (0, 255, 0), -1)
        # putText各参数依次是：图片，添加的文字(标签+深度-单位m)，左上角坐标，字体，字体大小，颜色，字体粗细
        cv2.putText(img, 'distance:' + str(dist*1000)[:5] + 'mm',
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, 'box_center_pixel:' + text_pixel,
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, 'box_center_camera:' + text_camera,
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('dec_img', img)


if __name__ == "__main__":
    # 配置摄像机参数，用opencv的时候，颜色通道是bgr
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # 深度通道的分辨率最大为1280x720
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    # 开始采集
    pipeline.start(config)
    # 深度与彩色图像对齐
    alignIt = rs.align(rs.stream.color)
    num = 1

    try:
        while True:
            # 获取深度图以及彩色图像
            frames = pipeline.wait_for_frames()

            # 获取相机内参
            if num == 1:
                color_frame = frames.get_color_frame()
                intr = color_frame.profile.as_video_stream_profile().intrinsics
                num += 1

            aligned_frame = alignIt.process(frames)
            depth_frame = aligned_frame.get_depth_frame()
            color_frame = aligned_frame.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 转化成numpy格式
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 开始预测，转化通道(BGR TO RGB)
            convert_img = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = model(convert_img)
            boxs = results.pandas().xyxy[0].values
            dectshow(color_image, boxs, depth_frame, intr)  # 用的是depth_frame(没有转化成np格式的)，因为要调用get_distance

            # # 可选，展示彩色图像和深度图
            # # 在深度图像上应用colormap(图像必须先转换为每像素8位)
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # # 水平堆叠深度图和彩色图
            # images = np.hstack((color_image, depth_colormap))
            # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('RealSense', images)

            key = cv2.waitKey(1)

            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()
