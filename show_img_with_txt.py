import os
from numpy import *
import cv2

show_img_path = './rotate_images/'
show_img_list = os.listdir(show_img_path)

show_lab_path = './rotate_txt/'



def showimgs():
    show_size = 1096
    for img_name in show_img_list:
        print (img_name)
        start_len = img_name.find('2021')
        end_len = img_name.find('.jpg')
        if start_len == -1 or end_len == -1:
            continue
        txt_name = img_name[start_len:end_len]
        fp = open(show_lab_path + txt_name + '.txt', 'r')

        Preimage = cv2.imread(show_img_path + img_name)
        image = cv2.resize(Preimage, (show_size, show_size))
        # cv2.imshow('test1', image)
        for line in fp.readlines():
            bbox = line.split()
            x0 = float(bbox[0]) / Preimage.shape[1] * show_size
            y0 = float(bbox[1]) / Preimage.shape[0] * show_size
            x1 = float(bbox[2]) / Preimage.shape[1] * show_size
            y1 = float(bbox[3]) / Preimage.shape[0] * show_size
            x0 = max([x0, 0]) + 1
            y0 = max([y0, 0]) + 1
            x1 = min([show_size, x1]) - 1
            y1 = min([show_size, y1]) - 1
            new_img = cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 1)

        cv2.imshow('test2', new_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        fp.close()


if __name__ == '__main__':
    showimgs()
