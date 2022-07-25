import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, non_max_suppression_obb, print_args, scale_coords, scale_polys, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.rboxs_utils import poly2rbox, rbox2poly


@torch.no_grad()
def yolo_run(imy,  # 不进行初始化
        # weights=ROOT / 'weights/best.pt',  # model.pt path(s)
        weights='/home/desktop-tjj/yolov5_obb/weights/best.pt',  # model.pt path(s)
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.3,  # confidence threshold
        iou_thres=0.4,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
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
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
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





