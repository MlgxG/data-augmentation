# -*- coding: utf-8 -*-
import os
import os.path
import xml.etree.ElementTree as ET
import glob
import numpy as np
from xml.dom.minidom import Document


def xml_to_txt(xmlpath, txtpath):

    os.chdir(xmlpath)  # 改变工作路径到xmlpath
    annotations = os.listdir('.')  # 返回path指定的文件夹包含的文件或文件夹的名字的列表
    annotations = glob.glob(str(annotations)+'*.xml')

    for i, xml_file in enumerate(annotations):

        file_txt = os.path.join(txtpath, xml_file.replace('xml', 'txt'))
        f_w = open(file_txt, 'w')

        in_file = open(xml_file, 'r')
        tree = ET.parse(in_file)
        root = tree.getroot()

        for obj in root.iter('object'):
            name = obj.find('name').text

            # class_num = class_names.index(name)
            xmlbox = obj.find('bndbox')

            x1 = xmlbox.find('xmin').text
            x2 = xmlbox.find('xmax').text
            y1 = xmlbox.find('ymin').text
            y2 = xmlbox.find('ymax').text

            f_w.write('%d %d %d %d\n' % (int(x1), int(y1), int(x2), int(y2)))
        f_w.close()
        # add = 1


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
        object_name.appendChild(doc.createTextNode(label_name[0]))
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


class_names = ['ship']
xmlpath = '/home/gongen/anaconda3/5000SAR/5000_voc/crop_xml/'
txtpath = '/home/gongen/anaconda3/5000SAR/5000_voc/crop_txt/'

labels = [i for i in os.listdir(txtpath) if 'txt' in i]
for txt in labels:
    txt_data = open(os.path.join(txtpath, txt), 'r').readlines()
    name = txt.replace('txt', 'jpg')
    box = format_label(txt_data)
    save_xml_name = os.path.join(xmlpath, name.replace('jpg', 'xml'))
    save_to_xml(save_xml_name, 5000, 5000, box, class_names, str(name))
# xml_to_txt(xmlpath, txtpath)
# save_to_xml(txt, 5000, 5000, box[idx, :], class_list, str(name))
    print ('save xml : ', save_xml_name)