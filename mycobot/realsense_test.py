import pyrealsense2 as rs
import random
import time
from scipy.spatial.transform import Rotation as R
from elephant import elephant_command
from global_v import *

import cv2
import numpy as np
import torch
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, non_max_suppression_obb, print_args, scale_coords,
                           scale_polys, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.rboxs_utils import poly2rbox, rbox2poly


# 初始化yolodetectLoad model
weights = '/home/desktop-tjj/yolov5_obb/weights/best.pt'
device = select_device()
model = DetectMultiBackend(weights, device=device)
stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
imgsz = check_img_size((640, 640), s=stride)  # check image size
# Half
half = False
half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
if pt or jit:
    model.model.half() if half else model.model.float()
# Run inference
model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup

def realsense_detect():  # 进行目标识别，显示目标识别效果，返回相机坐标系下的值
    # 配置yolov5
    # model = torch.hub.load('/home/desktop-tjj/yolov5', 'custom', path='/home/desktop-tjj/yolov5_obb/weights/best.pt',
    #                        source='local')  # 加载模型
    # model.eval()
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

            aligned_frame = alignIt.process(frames)  # 获取对齐数据
            depth_frame = aligned_frame.get_depth_frame()
            color_frame = aligned_frame.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 转化成numpy格式

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # 需要输入BGR格式
            boxs, label = yolo_run(color_image)
            dectshow(color_image, boxs, depth_frame, intr, label)  # 用的是depth_frame(没有转化成np格式的)，因为要调用get_distance
            # 当获取为nan的时候，虽然跳过了，但是还是会报错


            # # 可选，展示彩色图像和深度图
            # # 在深度图像上应用colormap(图像必须先转换为每像素8位)
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # # 水平堆叠深度图和彩色图
            # images = np.hstack((color_image, depth_colormap))
            # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('RealSense', images)
            global key
            key = cv2.waitKey(1)

            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()


@torch.no_grad()
def yolo_run(imy,  # 不进行初始化
             # weights=ROOT / 'weights/best.pt',  # model.pt path(s)
             weights='/home/desktop-tjj/yolov5_obb/weights/best.pt',  # model.pt path(s)
             # imgsz=(640, 640),  # inference size (height, width)
             conf_thres=0.3,  # confidence threshold
             iou_thres=0.4,  # NMS IOU threshold
             max_det=1000,  # maximum detections per image
             # device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
             view_img=False,  # show results
             save_txt=False,  # save results to *.txt
             save_conf=False,  # save confidences in --save-txt labels
             save_crop=False,  # save cropped prediction boxes
             nosave=False,  # do not save images/videos
             classes=None,  # filter by class: --class 0, or --class 0 2 3
             agnostic_nms=False,  # class-agnostic NMS
             augment=False,  # augmented inference
             visualize=False,  # visualize features
             update=False,  # update all models
             name='exp',  # save results to project/name
             exist_ok=False,  # existing project/name ok, do not increment
             line_thickness=3,  # bounding box thickness (pixels)
             hide_labels=False,  # hide labels
             hide_conf=False,  # hide confidences
             # half=False,  # use FP16 half-precision inference
             dnn=False,  # use OpenCV DNN for ONNX inference
             ):
    # Load model
    # device = select_device(device)
    # model = DetectMultiBackend(weights, device=device, dnn=dnn)
    # stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    # imgsz = check_img_size(imgsz, s=stride)  # check image size

    # # Half
    # half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    # if pt or jit:
    #     model.model.half() if half else model.model.float()

    # # Run inference
    # model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    # for遍历图片
    imy_copy = imy
    im = letterbox(imy_copy)[0]
    im = im.transpose((2, 0, 1))[::-1]  # BGR to RGB
    im = np.ascontiguousarray(im)
    t1 = time_sync()
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1

    # Inference
    pred = model(im, augment=augment, visualize=visualize)
    t3 = time_sync()
    dt[1] += t3 - t2

    # NMS
    # pred: list*(n, [cxcylsθ, conf, cls]) θ ∈ [-pi/2, pi/2)
    pred = non_max_suppression_obb(pred, conf_thres, iou_thres, classes, agnostic_nms, multi_label=True,
                                   max_det=max_det)
    dt[2] += time_sync() - t3

    # Process predictions
    for i, det in enumerate(pred):  # per image
        pred_poly = rbox2poly(det[:, :5])  # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
        seen += 1
        annotator = Annotator(im, line_width=line_thickness, example=str(names))

        if len(det):
            # Rescale polys from img_size to im0 size
            # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
            pred_poly = scale_polys(im.shape[2:], pred_poly, imy_copy.shape)
            det = torch.cat((pred_poly, det[:, -2:]), dim=1)  # (n, [poly conf cls])

            # Write results
            polygon_list_all = []
            for *poly, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                poly = [x.cpu().numpy() for x in poly]
                polygon_list = np.array([(poly[0], poly[1]), (poly[2], poly[3]), \
                                         (poly[4], poly[5]), (poly[6], poly[7])], np.int32)
                # polygon_list = [(poly[0], poly[1]), (poly[2], poly[3]), \
                #                          (poly[4], poly[5]), (poly[6], poly[7])]
                polygon_list_all.append(polygon_list)
                # annotator.box_label(xyxy, label, color=colors(c, True))
                # result = annotator.poly_label_return(poly, label, color=colors(c, True))
            return polygon_list_all, label
            break
        else:
            aa = np.array([(0, 0), (0, 0), (0, 0), (0, 0)], np.int32)
            polygon_list_all = []
            polygon_list_all.append(aa)
            return polygon_list_all, 'no object'


# 这个函数主要是在原图上画框,标出深度信息,此外用毫米看比较方便
def dectshow(org_img, boxs, depth_data, intrin, label):
    img = org_img
    if label == 'no object':
        cv2.putText(img, 'no_object',
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        for box in boxs:

            # rectangle画框，参数表示依次为：(图片，长方形框左上角坐标, 长方形框右下角坐标， 字体颜色，字体粗细)
            # cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            # circle画圆心，参数表示依次为：(img, center, radius, color[, thickness]),thickness为负表示绘制实心圆
            center = [(box[0][0] + box[2][0]) // 2, (box[0][1] + box[2][1]) // 2]
            cv2.circle(img, center, 8, (0, 255, 0), -1)
            # first_point = [int(box[0][0]), int(box[0][1])]
            # second_point = [int(box[1][0]), int(box[1][1])]
            # cv2.circle(img, first_point, 8, (0, 255, 0), -1)
            # cv2.circle(img, second_point, 8, (0, 0, 255), -1)
            cv2.drawContours(img, contours=[box], contourIdx=-1, color=(0, 255, 0), thickness=2)

            # cv2.putText(img, 'cup_position_in_base:' + text_base, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('dec_img', img)

realsense_detect()