import cv2
import numpy as np
from PIL import Image
import os
import utils
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

class_obj = ['car', 'motorbicycle', 'bicycle', 'person', 'rider', 'truck', 'bus', 'tricycle', 'road', 'siderwalk',
             'traffic_cone', 'road_pile', 'fence', 'traffic_light', 'pole', 'traffic_sign', 'wall', 'dustbin',
             'billboard', 'building', 'bridge', 'tunnel', 'overpass', 'vegatation']

standardization_dic = {
    33: 0,  # car
    34: 1,  # motorbicycle
    35: 2,  # bicycle
    36: 3,  # person
    37: 4,  # rider
    38: 5,  # truck
    39: 6,  # bus
    40: 7,  # tricycle
    49: 8,  # road
    50: 9,  # siderwalk
    65: 10,  # traffic_cone
    66: 11,  # road_pile
    67: 12,  # fence
    81: 13,  # traffic_light
    82: 14,  # pole
    83: 15,  # traffic_sign
    84: 16,  # wall
    85: 17,  # dustbin
    86: 18,  # billboard
    97: 19,  # building
    98: 20,  # bridge
    99: 21,  # tunnel
    100: 22,  # overpass
    113: 23,  # vegatation
    161: 0,  # car
    162: 1,  # motorbicycle
    163: 2,  # bicycle
    164: 3,  # person
    165: 4,  # rider
    166: 5,  # truck
    167: 6,  # bus
    168: 7,  # tricycle
}


def showImg(img):
    plt.imshow(img)
    plt.show()


def seg2ins(mask):
    label_0 = label(mask)
    props = regionprops(label_0)
    bbox_coor = []
    for prop in props:
        print("found bounding box", prop.bbox)
        prop.bbox
        bbox_coor.append(prop.bbox)
    return bbox_coor


def apollospace2yolo(imag_path, mask_path, mask_instance_path, file):
    img = Image.open(imag_path)
    mask = Image.open(mask_path)
    mask_ins = Image.open(mask_instance_path)
    new_size = (608, 608)
    img = utils.letterbox_image(img, new_size)
    mask_seg = utils.letterbox_mask(mask, new_size)
    mask_ins_new = utils.letterbox_mask_ins(mask_ins, new_size)

    mask_seg_array = np.array(mask_seg)  # 语义分割的矩阵
    mask_ins_array = np.array(mask_ins_new)  # 实例分割矩阵
    mask_ins_array_unique = (np.array(mask_ins_new) / 1000).astype(np.int8)  # 实例分割有哪些类别
    print(np.unique(mask_seg_array))
    for i in np.unique(mask_ins_array_unique):  # 在语义分割中删除掉实例
        local = np.where(mask_seg_array == i)
        mask_seg_array[local[0], local[1]] = 255
    file.write(imag_path)
    ins_shape = np.shape(mask_ins_new)
    for j in np.unique(mask_seg_array):  # 添加语义标签的识别框
        if j == 255 or j == 1 or j == 17 or j == 128:
            continue
        area = np.where(mask_seg_array == j)
        xmin = np.min(area[1])
        xmax = np.max(area[1])
        ymin = np.min(area[0])
        ymax = np.max(area[0])
        cls_id = standardization_dic[j]
        b = [xmin, ymin, xmax, ymax]

        file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

    ins_diff = np.array(new_size) - np.array(ins_shape)
    for k in np.unique(np.array(mask_ins_new)):  # 添加实例框
        if k < 10000 or k > 60000:  # 排除掉无用标签
            continue
        area = np.where(mask_ins_array == k)
        xmin = int(np.min(area[1]) + (ins_diff[1] / 2))
        xmax = int(np.max(area[1]) + (ins_diff[1] / 2))
        ymin = int(np.min(area[0]) + (ins_diff[0] / 2))
        ymax = int(np.max(area[0]) + (ins_diff[0] / 2))
        b = [xmin, ymin, xmax, ymax]
        num = int(k / 1000)
        cls_id = standardization_dic[num]
        file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
    file.write('\n')


# img_path = 'D:\学习资料\机器学习\\apllo\seg\\train\\170908_061558423_Camera_6.jpg'
# mask_path = 'D:\学习资料\机器学习\\apllo\img\\train\\170908_061558423_Camera_6.png'
# mask_ins = 'D:\学习资料\机器学习\\apllo\img\\train\\170908_061558423_Camera_6_instanceIds.png'
# list_file = open('%s_%s.txt' % ("2020", 'train'), 'w')
# apollospace2yolo(img_path, mask_path, mask_ins, list_file)
# list_file.close()

def draw_box(image_path, box_txt):
    with open(box_txt, "r") as f:  # 打开文件
        data = f.read()  # 读取文件
        dataarrag = data.split(" ", -1)
        image = Image.open(image_path)
        image = utils.letterbox_image(image, (608, 608))
        print()
        for j in range(len(dataarrag)):
            if j == 0:
                continue
            bbox = dataarrag[j].split(',', -1)
            image_2 = image
            image_2 = cv2.cvtColor(np.asarray(image_2), cv2.COLOR_RGB2BGR)  # 转成opencvde的格式
            first_point = (int(float(bbox[0])),
                           int(float(bbox[1])))
            second_point = (int(float(bbox[2])), int(float(bbox[3])))

            cv2.rectangle(image_2, first_point, second_point, (0, 255, 0), 2)  # 绘制矩形
            cv2.namedWindow(str(bbox[4]), cv2.WINDOW_NORMAL)  # 窗口大小可以改变
            cv2.imshow(str(bbox[4]), image_2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def get_one_mask(mask, id):
    mask_array = np.array(mask)
    area = np.where(mask_array == id)
    w, h = np.shape(mask_array)
    new_mask = np.zeros((w, h), dtype=np.uint8)
    new_mask[area[0], area[1]] = 1;
    return new_mask
