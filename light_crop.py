# -*- coding: utf-8 -*-
import copy
import cv2
import os
import scipy.misc as misc
from xml.dom.minidom import Document
from PIL import Image
import numpy as np

Image.MAX_IMAGE_PIXELS = None


def save_to_xml(save_path, im_width, im_height, objects_axis, label_name, name, hbb=True):
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
        object_name.appendChild(doc.createTextNode(label_name[int(objects_axis[i][-1])]))
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
            x0.appendChild(doc.createTextNode(str((objects_axis[i][0]))))
            bndbox.appendChild(x0)
            y0 = doc.createElement('ymin')
            y0.appendChild(doc.createTextNode(str((objects_axis[i][1]))))
            bndbox.appendChild(y0)

            x1 = doc.createElement('xmax')
            x1.appendChild(doc.createTextNode(str((objects_axis[i][2]))))
            bndbox.appendChild(x1)
            y1 = doc.createElement('ymax')
            y1.appendChild(doc.createTextNode(str((objects_axis[i][5]))))
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


class_list = ['ship']


def format_label(txt_list):
    format_data = []
    for i in range(len(txt_list)):
        box = txt_list[i].split(' ')
        format_data.append(
            [int(float(box[0])), int(float(box[1])), int(float(box[2])), int(float(box[3]))]
            # {
            # 'x0': float(txt_list[i][0:8].split),
            # 'x1': float(txt_list[i][18:26].split),
            # # 'x2': int(i.split(' ')[4]),
            # # 'x3': int(i.split(' ')[6]),
            # 'y0': float(txt_list[i][9:17].split),
            # 'y1': float(txt_list[i][27:35].split)
            # }
            # 'y3': int(i.split(' ')[5]),
            # 'y4': int(i.split(' ')[7])}
            # 'class': class_list.index(i.split(' ')[8]) if i.split(' ')[8] in class_list else 0,
            # 'difficulty': int(i.split(' ')[9])}
        )
        # if i.split(' ')[8] not in class_list:
        #     print ('warning found a new label :', i.split(' ')[8])
        #     exit()
    return np.array(format_data)


def resize_box(boxes, preheight, newheight):
    for num in range(boxes.shape[0]):
        boxes[num, 1] = float(boxes[num, 1]) / preheight * newheight  # 计算时，如果不转换为float，会默认为int，计算结果为0
        boxes[num, 3] = float(boxes[num, 3]) / preheight * newheight
    return boxes


def clip_image(file_idx, image, boxes_all, width, height):
    # print ('image shape', image.shape)
    if len(boxes_all) > 0:
        shape = image.size  # 注意image.size和img.shape中关于wh顺序的区别，size是wh，shape是hw
        if shape[1] < 5000:
            image = image.resize((shape[0], 5000))  # 对于5600*4700的图片,将其resize到5600*5000
            boxes_all = resize_box(boxes_all, shape[1], 5000)
            shape = image.size
        for start_h in range(0, shape[1], 1024):
            for start_w in range(0, shape[0], 1024):
                boxes = copy.deepcopy(boxes_all)
                box = np.zeros_like(boxes_all)
                start_h_new = start_h
                start_w_new = start_w
                if start_h + height > shape[1]:
                    start_h_new = shape[1] - height
                if start_w + width > shape[0]:
                    start_w_new = shape[0] - width
                top_left_row = max(start_h_new, 0)  # y
                top_left_col = max(start_w_new, 0)  # x
                bottom_right_row = min(start_h + height, shape[1])
                bottom_right_col = min(start_w + width, shape[0])

                subImage = image.crop((top_left_col, top_left_row, bottom_right_col, bottom_right_row))

                box[:, 0] = boxes[:, 0] - top_left_col
                box[:, 2] = boxes[:, 2] - top_left_col
                # box[:, 4] = boxes[:, 4] - top_left_col
                # box[:, 6] = boxes[:, 6] - top_left_col

                box[:, 1] = boxes[:, 1] - top_left_row
                box[:, 3] = boxes[:, 3] - top_left_row
                # box[:, 5] = boxes[:, 5] - top_left_row
                # box[:, 7] = boxes[:, 7] - top_left_row
                # box[:, 8] = boxes[:, 8]
                center_y = 0.5 * (box[:, 1] + box[:, 3])
                center_x = 0.5 * (box[:, 0] + box[:, 2])
                # print('center_y', center_y)
                # print('center_x', center_x)
                # print ('boxes', boxes)
                # print ('boxes_all', boxes_all)
                # print ('top_left_col', top_left_col, 'top_left_row', top_left_row)

                cond1 = np.intersect1d(np.where(center_y[:] >= 0)[0], np.where(center_x[:] >= 0)[0])
                cond2 = np.intersect1d(np.where(center_y[:] <= (bottom_right_row - top_left_row))[0],
                                       np.where(center_x[:] <= (bottom_right_col - top_left_col))[0])
                idx = np.intersect1d(cond1, cond2)  # 获取满足需求的box
                # idx = np.where(center_y[:]>=0 and center_x[:]>=0 and center_y[:] <= (bottom_right_row - top_left_row)
                # and center_x[:] <= (bottom_right_col - top_left_col))[0]
                # save_path, im_width, im_height, objects_axis, label_name
                if len(idx) > 0:
                    name = "%s_%05d_%05d.jpg" % (file_idx, top_left_row, top_left_col)
                    print(name)
                    txt = os.path.join(save_dir, 'crop_txt',
                                       "%s_%05d_%05d.txt" % (file_idx, top_left_row, top_left_col))
                    # save_to_xml(txt, subImage.shape[1], subImage.shape[0], box[idx, :], class_list, str(name))
                    # print ('save xml : ', xml)
                    with open(txt, 'w') as f:
                        for i in range(len(idx)):
                            f.write('%d %d %d %d\n' % (int(box[idx[i], 0]), int(box[idx[i], 1]),
                                                     int(box[idx[i], 2]), int(box[idx[i], 3])))
                    f.close()
                    if subImage.size[0] > 5 and subImage.size[1] > 5:
                        img = os.path.join(save_dir, 'crop_images',
                                           "%s_%05d_%05d.jpg" % (file_idx, top_left_row, top_left_col))
                        subImage.save(img)
                #         add = 1


print ('class_list', len(class_list))
raw_data = '/home/gongen/anaconda3/5000SAR/5000_voc/'
raw_images_dir = os.path.join(raw_data, 'images')
raw_label_dir = os.path.join(raw_data, 'txt')

save_dir = '/home/gongen/anaconda3/5000SAR/5000_voc/'

images = [i for i in os.listdir(raw_images_dir) if 'jpg' in i]
labels = [i for i in os.listdir(raw_label_dir) if 'xml' in i]

print ('find image', len(images))
print ('find label', len(labels))

min_length = 1e10
max_length = 1

for idx, img in enumerate(images):
    # img = 'gamma_1000_k_0_xzb_1046_abcf_3_06_2_10_6_7_1_2_0_0_4_00_20_5.jpg'
    print (idx, 'read image', img)
    # img_data = misc.imread(os.path.join(raw_images_dir, img))  # 将图片读取出来为array numpy
    img_data = Image.open(os.path.join(raw_images_dir, img))
    # if len(img_data.shape) == 2:
    # img_data = img_data[:, :, np.newaxis]
    # print ('find gray image')
    txt_data = open(os.path.join(raw_label_dir, img.replace('jpg', 'txt')), 'r').readlines()
    # print (idx, len(format_label(txt_data)), img_data.shape)
    # if max(img_data.shape[:2]) > max_length:
    # max_length = max(img_data.shape[:2])
    # if min(img_data.shape[:2]) < min_length:
    # min_length = min(img_data.shape[:2])
    # if idx % 50 ==0:
    # print (idx, len(format_label(txt_data)), img_data.shape)
    # print (idx, 'min_length', min_length, 'max_length', max_length)
    box = format_label(txt_data)
    clip_image(img.strip('.jpg'), img_data, box, 5000, 5000)  # strip移除字符串头尾指定的字符序列
