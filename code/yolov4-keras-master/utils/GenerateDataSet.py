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
    33: 0,  # car               1
    34: 1,  # motorbicycle      1
    35: 2,  # bicycle           1
    36: 3,  # person            1
    37: 4,  # rider             1
    38: 5,  # truck             1
    39: 6,  # bus               1
    40: 7,  # tricycle          1
    49: 8,  # road              2
    50: 9,  # siderwalk         2
    65: 10,  # traffic_cone     2
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

# bdd100K 映射到 标准ID
bdd100k2std = {
    13: 0,  # car
    17: 1,  # motorbicycle
    18: 2,  # bicycle
    11: 3,  # person
    12: 4,  # rider
    14: 5,  # truck
    15: 6,  # bus
    0: 8,  # road
    1: 9,  # siderwalk
    4: 12,  # fence
    6: 13,  # traffic_light
    5: 14,  # pole
    7: 15,  # traffic_sign
    3: 16,  # wall
    2: 19,  # building
    8: 23,  # vegatation
}

city2std = {
    26: 0,  # car
    32: 1,  # motorbicycle
    33: 2,  # bicycle
    24: 3,  # person
    25: 4,  # rider
    27: 5,  # truck
    28: 6,  # bus
    7: 8,  # road
    8: 9,  # siderwalk
    13: 12,  # fence
    14: 12,  # guard rail
    19: 13,  # traffic_light
    17: 14,  # pole
    20: 15,  # traffic_sign
    12: 16,  # wall
    11: 19,  # building
    15: 20,  # bridge
    16: 21,  # tunnel
    21: 23,  # vegatation
}


def showImg(img):
    plt.imshow(img)
    plt.show()


# 标出联通区域,area是传入box的阈值
def seg2ins(mask, area):
    label_0 = label(mask)
    props = regionprops(label_0)
    bbox_coor = []
    for prop in props:
        if prop.area < area:
            continue
        print("found bounding box", prop.bbox, " area", prop.area, " th", area)
        bbox_coor.append(prop.bbox)
    return bbox_coor


# 制作数据集
def apollospace2yolo(imag_path, mask_path, mask_instance_path, file):
    img = Image.open(imag_path)
    mask = Image.open(mask_path)
    mask_ins = Image.open(mask_instance_path)
    new_size = (608, 608)
    # img = utils.letterbox_image(img, new_size)
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
        anum = 100
        if j == 255 or j == 1 or j == 17 or j == 128 or j == 97:
            continue
        mask_new = get_one_mask(mask_seg_array, j)  # 拿到mask
        if j == 113:
            anum = 200
            mask_new = mask_erosion(mask_new)  # 如果是树木的话做腐蚀
        else:  # 其他的膨胀
            mask_new = mask_dilate(mask_new)
        # cv2.imshow("dilate_demo", mask_new)
        # cv2.waitKey(0)
        boxes = seg2ins(mask_new, anum)
        cls_id = standardization_dic[j]
        for box in boxes:
            xmin = box[1]
            ymin = box[0]
            xmax = box[3]
            ymax = box[2]
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


# 伯里克利生成yolo
def bdd100k2yolo(imag_path, mask_path, file):
    mask = Image.open(mask_path)
    new_size = (608, 608)
    mask_seg = utils.letterbox_mask(mask, new_size)
    mask_seg_array = np.array(mask_seg)  # 语义分割的矩阵
    print(np.unique(mask_seg_array))
    file.write(imag_path)
    useful_id = [0, 1, 2, 3, 4, 5, 6, 8, 11, 13, 14, 15, 17, 18]
    for j in np.unique(mask_seg_array):  # 添加语义标签的识别框
        anum = 100
        if j not in useful_id:  # id没有用的话就不要了
            continue
        stdId = bdd100k2std[j]  # 映射成标准
        mask_new = get_one_mask(mask_seg_array, j)  # 拿到mask
        if j == 8:  # 如果是树木的话做腐蚀
            anum = 200
            mask_new = mask_erosion(mask_new)
        else:  # 其他的膨胀
            mask_new = mask_dilate(mask_new)
        boxes = seg2ins(mask_new, anum)
        cls_id = stdId
        for box in boxes:
            xmin = box[1]
            ymin = box[0]
            xmax = box[3]
            ymax = box[2]
            b = [xmin, ymin, xmax, ymax]
            file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
    file.write('\n')


# Cityspace生成yolo
def city2yolo(imag_path, mask_path, file):
    mask = Image.open(mask_path)
    new_size = (608, 608)
    mask_seg = utils.letterbox_mask(mask, new_size)
    mask_seg_array = np.array(mask_seg)  # 语义分割的矩阵
    print(np.unique(mask_seg_array))
    file.write(imag_path)
    useful_id = [26, 32, 33, 24, 25, 27, 28, 7, 8, 13, 14, 19, 17, 20, 12, 11, 15, 16, 21]
    for j in np.unique(mask_seg_array):  # 添加语义标签的识别框
        anum = 100

        ins = j if j < 1000 else int(j / 1000)

        if ins not in useful_id:  # id没有用的话就不要了
            continue
        stdId = city2std[ins]  # 映射成标准
        mask_new = get_one_mask(mask_seg_array, j)  # 拿到mask
        if j == 21:  # 如果是树木的话做腐蚀
            anum = 200
            mask_new = mask_erosion(mask_new)
        else:  # 其他的膨胀
            mask_new = mask_dilate(mask_new)
        boxes = seg2ins(mask_new, anum)
        cls_id = stdId
        for box in boxes:
            xmin = box[1]
            ymin = box[0]
            xmax = box[3]
            ymax = box[2]
            b = [xmin, ymin, xmax, ymax]
            file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
    file.write('\n')


# 根据yolo文件画出box
def draw_box_txt(image_path, box_txt):
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
            print("bbox", bbox)
            first_point = (int(float(bbox[0])), int(float(bbox[1])))
            second_point = (int(float(bbox[2])), int(float(bbox[3])))

            cv2.rectangle(image_2, first_point, second_point, (0, 255, 0), 2)  # 绘制矩形
            cv2.namedWindow(str(bbox[4]), cv2.WINDOW_NORMAL)  # 窗口大小可以改变
            cv2.imshow(str(bbox[4]), image_2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# 根据regionProp的box绘制box
def draw_box(image_path, boxes):
    image = Image.open(image_path)
    image = utils.letterbox_image(image, (608, 608))
    for bbox in boxes:
        image_2 = image
        image_2 = cv2.cvtColor(np.asarray(image_2), cv2.COLOR_RGB2BGR)  # 转成opencvde的格式
        first_point = (int(float(bbox[1])),
                       int(float(bbox[0])))
        second_point = (int(float(bbox[3])), int(float(bbox[2])))

        cv2.rectangle(image_2, first_point, second_point, (0, 255, 0), 2)  # 绘制矩形
        cv2.namedWindow("text", cv2.WINDOW_NORMAL)  # 窗口大小可以改变
        cv2.imshow("text", image_2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# 获取一个种类的mask
def get_one_mask(mask, id):
    mask_array = np.array(mask)
    area = np.where(mask_array == id)
    print(id)
    # print("区域", area)
    w, h = np.shape(mask_array)
    new_mask = np.zeros((w, h), dtype=np.uint8)
    new_mask[area[0], area[1]] = 100
    # plt.figure(1)
    # plt.title("原始mask")
    # showImg(mask)
    #
    # plt.figure(2)
    # plt.title("抽取的mask" + str(id))
    # showImg(new_mask)  # matplotlib 显示图片
    return new_mask


#  对单层的mask numpy数组进行腐蚀操作
def mask_erosion(mask_new, kernel=(5, 5)):
    mask_three = np.stack((mask_new, mask_new, mask_new), axis=2)  # np数组堆叠
    gray = cv2.cvtColor(mask_three, cv2.COLOR_BGR2GRAY)  # 转cv
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
    erosion = cv2.erode(gray, kernel)
    return erosion


# 对单层的mask numpy数组进行膨胀操作
def mask_dilate(mask_new, kernel=(5, 5)):
    mask_three = np.stack((mask_new, mask_new, mask_new), axis=2)  # np数组堆叠
    gray = cv2.cvtColor(mask_three, cv2.COLOR_BGR2GRAY)  # 转cv
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel)
    dilate = cv2.dilate(gray, kernel)
    return dilate


#  D:\学习资料\机器学习\apllo\seg\train

# img_path = 'D:\学习资料\机器学习\BD\\train\img\\0004a4c0-d4dff0ad.jpg'
# mask_path = 'D:\学习资料\机器学习\BD\\train\seg\\0004a4c0-d4dff0ad_train_id.png'
# img_path = "D:\学习资料\机器学习\\apllo\seg\\train\\170908_061602315_Camera_6.jpg"
# mask_path = "D:\学习资料\机器学习\\apllo\img\\train\\170908_061602315_Camera_6.png"

# img_path = "D:\学习资料\机器学习\\apllo\seg\\train\\170908_061558423_Camera_6.jpg"
# mask_path = "D:\学习资料\机器学习\\apllo\img\\train\\170908_061558423_Camera_6.png"
#
# # img_path = "D:\学习资料\机器学习\\apllo\seg\\train\\170927_064028573_Camera_5.jpg"
# # mask_path = "D:\学习资料\机器学习\\apllo\img\\train\\170927_064028573_Camera_5_bin.png"
# mask = Image.open(mask_path)
# image = Image.open(img_path)
# mask = utils.letterbox_mask(mask, (608, 608))
# mask_new = get_one_mask(mask, 82)
#
# mask_three = np.stack((mask_new, mask_new, mask_new), axis=2)  # np数组堆叠
# print(np.shape(mask_three))
# gray = cv2.cvtColor(mask_three, cv2.COLOR_BGR2GRAY)  # 转cv
# # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# # erosion = cv2.erode(gray, kernel)  # 腐蚀操作
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
# dst = cv2.dilate(gray, kernel)  # 膨胀操作
# mask_new = dst
# cv2.imshow("dilate_demo", mask_new)
# cv2.waitKey(0)
# print(type(mask_new))
# print(np.shape(mask_new))
# print(np.unique(mask_new))
#
# boxes = seg2ins(mask_new, 50)

# 生成train.txt

# img_path = 'D:\学习资料\机器学习\BD\\train\img\\0004a4c0-d4dff0ad.jpg'
# mask_path = 'D:\学习资料\机器学习\BD\\train\seg\\0004a4c0-d4dff0ad_train_id.png'
# img_path = 'D:\学习资料\机器学习\\apllo\seg\\train\\170908_061558423_Camera_6.jpg'
# mask_path = 'D:\学习资料\机器学习\\apllo\img\\train\\170908_061558423_Camera_6.png'
# mask_ins = 'D:\学习资料\机器学习\\apllo\img\\train\\170908_061558423_Camera_6_instanceIds.png'
img_path = 'D:\学习资料\机器学习\cityscapes\leftImg8bit\\train\\bochum\\bochum_000000_000600_leftImg8bit.png'
mask_path = 'D:\学习资料\机器学习\cityscapes\gtFine\\train\\bochum\\bochum_000000_000600_gtFine_instanceIds.png'
list_file = open('%s_%s.txt' % ("2020", 'train'), 'w')
city2yolo(img_path, mask_path, list_file)
list_file.close()

draw_box_txt(img_path, "2020_train.txt")
