# -*- coding: utf-8 -*-
import os
import cv2
import math
import numpy as np
import random
from xml.dom.minidom import Document


def save_to_xml(save_path, im_width, im_height, objects_axis, label_name, name, hbb=True):  # 根据需求来创建xml
    im_depth = 0
    object_num = len(objects_axis)
    doc = Document()

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    folder = doc.createElement('folder')
    folder_name = doc.createTextNode('VOC2007')
    folder.appendChild(folder_name)
    annotation.appendChild(folder)

    filename = doc.createElement('filename')
    filename_name = doc.createTextNode(name)
    filename.appendChild(filename_name)
    annotation.appendChild(filename)

    source = doc.createElement('source')
    annotation.appendChild(source)

    database = doc.createElement('database')
    database.appendChild(doc.createTextNode('The VOC2007 Database'))
    source.appendChild(database)

    annotation_s = doc.createElement('annotation')
    annotation_s.appendChild(doc.createTextNode('PASCAL VOC2007'))
    source.appendChild(annotation_s)

    image = doc.createElement('image')
    image.appendChild(doc.createTextNode('flickr'))
    source.appendChild(image)

    flickrid = doc.createElement('flickrid')
    flickrid.appendChild(doc.createTextNode('322409915'))
    source.appendChild(flickrid)

    owner = doc.createElement('owner')
    annotation.appendChild(owner)

    flickrid_o = doc.createElement('flickrid')
    flickrid_o.appendChild(doc.createTextNode('knautia'))
    owner.appendChild(flickrid_o)

    name_o = doc.createElement('name')
    name_o.appendChild(doc.createTextNode('yang'))
    owner.appendChild(name_o)

    size = doc.createElement('size')
    annotation.appendChild(size)
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(im_width)))
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(im_height)))
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(im_depth)))
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)
    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)
    for i in range(object_num):
        objects = doc.createElement('object')
        annotation.appendChild(objects)
        object_name = doc.createElement('name')
        object_name.appendChild(doc.createTextNode(label_name[0]))  # 这里对应的是分类list的名字，如果是多分类，需要更改
        objects.appendChild(object_name)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        objects.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('1'))
        objects.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        objects.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)
        if hbb:
            x0 = doc.createElement('xmin')
            x0.appendChild(doc.createTextNode(str(int(objects_axis[i][0]))))
            bndbox.appendChild(x0)
            y0 = doc.createElement('ymin')
            y0.appendChild(doc.createTextNode(str(int(objects_axis[i][1]))))
            bndbox.appendChild(y0)

            x1 = doc.createElement('xmax')
            x1.appendChild(doc.createTextNode(str(int(objects_axis[i][2]))))
            bndbox.appendChild(x1)
            y1 = doc.createElement('ymax')
            y1.appendChild(doc.createTextNode(str(int(objects_axis[i][3]))))
            bndbox.appendChild(y1)
        else:

            x0 = doc.createElement('x0')
            x0.appendChild(doc.createTextNode(str((objects_axis[i][0]))))
            bndbox.appendChild(x0)
            y0 = doc.createElement('y0')
            y0.appendChild(doc.createTextNode(str((objects_axis[i][1]))))
            bndbox.appendChild(y0)

            x1 = doc.createElement('x1')
            x1.appendChild(doc.createTextNode(str((objects_axis[i][2]))))
            bndbox.appendChild(x1)
            y1 = doc.createElement('y1')
            y1.appendChild(doc.createTextNode(str((objects_axis[i][3]))))
            bndbox.appendChild(y1)

            x2 = doc.createElement('x2')
            x2.appendChild(doc.createTextNode(str((objects_axis[i][4]))))
            bndbox.appendChild(x2)
            y2 = doc.createElement('y2')
            y2.appendChild(doc.createTextNode(str((objects_axis[i][5]))))
            bndbox.appendChild(y2)

            x3 = doc.createElement('x3')
            x3.appendChild(doc.createTextNode(str((objects_axis[i][6]))))
            bndbox.appendChild(x3)
            y3 = doc.createElement('y3')
            y3.appendChild(doc.createTextNode(str((objects_axis[i][7]))))
            bndbox.appendChild(y3)

    f = open(save_path, 'w')
    f.write(doc.toprettyxml(indent=''))
    f.close()


class_list = ['ship']  # 分类表


def format_label(txt_list):
    format_data = []
    for i in range(len(txt_list)):
        box = txt_list[i].split(' ')
        format_data.append(
            [int(float(box[0])), int(float(box[1])), int(float(box[2])), int(float(box[3]))]
        )
        # if i.split(' ')[8] not in class_list:
        #     print ('warning found a new label :', i.split(' ')[8])
        #     exit()
    return np.array(format_data)


def rotate_xy((x, y), angle, cx, cy):
    """
    点(x,y) 绕(cx,cy)点旋转
    """
    # print(cx,cy)
    angle = float(angle) / 180 * math.pi  # 在math里面的cos和sin输入的是弧度
    # add = math.cos(angle)
    # 仿射坐标公式 https://www.cnblogs.com/shijibao001/articles/1225962.html ,这个里面图片是逆时针旋转的，所以对应的为-angle
    x_new = (x - cx) * math.cos(angle) + (y - cy) * math.sin(angle) + cx
    y_new = -(x - cx) * math.sin(angle) + (y - cy) * math.cos(angle) + cy
    return x_new, y_new


def rotate_boxes(all_boxes, angle, height, weight):
    new_box = []
    for i in range(all_boxes.shape[0]):
        x1 = (all_boxes[i, 0], all_boxes[i, 1])
        x2 = (all_boxes[i, 2], all_boxes[i, 1])
        x3 = (all_boxes[i, 0], all_boxes[i, 3])
        x4 = (all_boxes[i, 2], all_boxes[i, 3])
        p1 = rotate_xy(x1, angle, weight / 2, height / 2)
        p2 = rotate_xy(x2, angle, weight / 2, height / 2)
        p3 = rotate_xy(x3, angle, weight / 2, height / 2)
        p4 = rotate_xy(x4, angle, weight / 2, height / 2)
        # 获取四个点的旋转后的坐标，根据四个坐标求最小外接矩形
        new_x0 = min([p1[0], p2[0], p3[0], p4[0]])
        new_y0 = min([p1[1], p2[1], p3[1], p4[1]])
        new_x1 = max([p1[0], p2[0], p3[0], p4[0]])
        new_y1 = max([p1[1], p2[1], p3[1], p4[1]])
        if (new_x0 + new_x1) / 2 > weight or (new_x0 + new_x1) / 2 < 0 or \
                (new_y0 + new_y1) / 2 > height or (new_y0 + new_y1) / 2 < 0:  # 超过边界的box直接舍去
            continue
        new_box.append([new_x0, new_y0, new_x1, new_y1])
    return new_box


def rotation(ran_angle=None):
    crop_data = '/home/gongen/anaconda3/5000SAR/5000_voc/'
    crop_images_dir = os.path.join(crop_data, 'crop_images')
    crop_label_dir = os.path.join(crop_data, 'crop_txt')

    save_dir = '/home/gongen/anaconda3/5000SAR/5000_voc/'

    images = [i for i in os.listdir(crop_images_dir) if 'jpg' in i]
    labels = [i for i in os.listdir(crop_label_dir) if 'txt' in i]

    print ('find image', len(images))
    print ('find label', len(labels))

    for idx, img in enumerate(images):
        # img = 'gamma_1000_k_0_xzb_1046_abcf_3_06_2_10_6_7_1_2_0_0_4_00_20_5.jpg'
        print (idx, 'read image', img)
        img_data = cv2.imread(os.path.join(crop_images_dir, img))  # 将图片读取出来为array numpy

        txt_data = open(os.path.join(crop_label_dir, img.replace('jpg', 'txt')), 'r').readlines()

        box = format_label(txt_data)
        rows, cols = img_data.shape[:2]  # row是高y col是宽x
        if ran_angle is None:
            angle = random.randint(0, 360)
        else:
            angle = ran_angle

        new_boxes = rotate_boxes(box, angle=angle, height=rows, weight=cols)

        # 第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)  # 这里的angle输入是角度不是弧度
        # 第三个参数：变换后的图像大小
        res = cv2.warpAffine(img_data, M, (rows, cols))  # 计算旋转后的图片，旋转中心点为图片中心

        name = "%s_%03d.jpg" % (img.strip('.jpg'), angle)
        print(name)
        save_img_name = os.path.join(save_dir, 'rotate_images', "%s_%03d.jpg" % (img.strip('.jpg'), angle))
        cv2.imwrite(save_img_name, res)

        save_txt_name = os.path.join(save_dir, 'rotate_txt', "%s_%03d.txt" % (img.strip('.jpg'), angle))
        # save_xml_name = os.path.join(save_dir, 'rotate_xml', "%s_%03d.xml" % (img.strip('.jpg'), angle))
        #
        # save_to_xml(save_xml_name, res.shape[0], res.shape[1], new_boxes, class_list, str(name))
        # print ('save xml : ', save_to_xml)

        with open(save_txt_name, 'w') as f:
            for i in range(len(new_boxes)):
                f.write('%d %d %d %d\n' % (int(new_boxes[i][0]), int(new_boxes[i][1]),
                                           int(new_boxes[i][2]), int(new_boxes[i][3])))
        f.close()


if __name__ == '__main__':
    print ('rotate begin')
    rotation()  # 默认是随机旋转角度，如需自定义旋转角度rotation（angle）即可，旋转方向为逆时针
    print ('rotate success')
