import os
import math
import xml.etree.ElementTree as ET
import numpy as np
import cv2


def rotate(cx, cy, w, h, angle):
    # 用于常规坐标时，angle是顺时针旋转角度
    # 用于图像坐标时，angle是逆时针旋转角度
    # math.cos(angle)这里的angle单位是rad
    # angle = angle/180*math.pi  单位转换
    angle = -angle  # 这里是图像坐标
    points = [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2],
              [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]]
    newpoints = []
    if angle < 0:  # 逆时针
        angle = -angle
        for point in points:
            x, y = point
            newx = round((x - cx) * math.cos(angle) - (y - cy) * math.sin(angle) + cx, 1)
            newy = round((x - cx) * math.sin(angle) + (y - cy) * math.cos(angle) + cy, 1)
            newpoints.append([newx, newy])
    else:
        for point in points:
            x, y = point
            newx = round((x - cx) * math.cos(angle) + (y - cy) * math.sin(angle) + cx, 1)
            newy = round((y - cy) * math.cos(angle) - (x - cx) * math.sin(angle) + cy, 1)
            newpoints.append([newx, newy])
    return newpoints


def roxml2txt(dir):
    # dir是xml文件目录
    files = os.listdir(dir)
    parentdir, dirname = os.path.split(dir)
    txtdir = os.path.join(parentdir, 'txt')
    if not os.path.exists(txtdir):
        os.mkdir(txtdir)
    for f in files:
        xml = ET.parse(os.path.join(dir, f))
        root = xml.getroot()
        boxes = root.iter('robndbox') # 多个目标可能存在
        object =root.iter('object')
        name=[]
        i = 0
        for obj in object:
            name.append(obj.find('name').text)
        with open(os.path.join(txtdir, f.split('.')[0] + '.txt'), 'w+') as t:
            for box in boxes:
                cx = float(box.find('cx').text)
                cy = float(box.find('cy').text)
                w = float(box.find('w').text)
                h = float(box.find('h').text)
                angle = float(box.find('angle').text)
                newpoints = rotate(cx, cy, w, h, angle)  # 计算旋转后的4个点坐标
                # 用于查看坐标转换是否正确，在原图上画矩形框，自行修改图片路径
                newpoints = np.array(newpoints)
                newpoints = newpoints.astype(int)

                # img = cv2.imread(os.path.join('test', 'images', f.split('.')[0] + '.png'))
                # img = cv2.polylines(img, [newpoints], isClosed=True, color=(0, 0, 255))
                # cv2.imshow('pic', img)
                # cv2.waitKey()
                ##########################################
                line = ''
                for point in newpoints:
                    line += str(point[0]) + ' ' + str(point[1]) + ' '
                line += name[i]+' 0\n'
                t.write(line)
                i+=1
                print(line)
            t.close()


roxml2txt('/home/desktop-tjj/Desktop/mydata_2/label/val')  # 路径最后不能加/