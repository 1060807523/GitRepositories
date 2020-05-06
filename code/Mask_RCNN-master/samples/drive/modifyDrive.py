# import PIL as Image
import os
import sys

import numpy as np
import skimage.io as Image

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from cityscapesscripts import helpers

print(helpers.labels)

# 重写load_mask
############################################################
#  Dataset
############################################################
class CityDataset(utils.Dataset):
    def load_Objection(self, dataset_dir, subset_mask, subset_image):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: mask
        """
        # 添加识别类别
        # Add classes. We have only one class to add.
        self.add_class("drive", 1, "楼房")
        self.add_class("drive", 2, "护栏")
        self.add_class("drive", 3, "路边设施")
        self.add_class("drive", 4, "行人")
        self.add_class("drive", 5, "路灯/电线杆")
        self.add_class("drive", 6, "车行道分界线")
        self.add_class("drive", 7, "马路")
        self.add_class("drive", 8, "人行道")
        self.add_class("drive", 9, "植物")
        self.add_class("drive", 10, "汽车")
        self.add_class("drive", 11, "矮墙")
        self.add_class("drive", 12, "路牌/交通信号灯")

        # Train or validation dataset?

        dataset_mask_dir = os.path.join(dataset_dir, subset_mask)
        dataset_image_dir = os.path.join(dataset_dir, subset_mask)
        print(dataset_mask_dir)
        # 装载图片信息
        # 获取mask 文件夹下所有的图片的文件名
        # 根据图片文件名，获取图片中有几种物体
        # 分别将各个物体组装成img_info
        # img_info 包含的信息：
        #   image_id ： 图片名称作为唯一标识
        #   path： 图片的路径
        #   width,heigt:图片大小
        #   mask_path：掩码地址
        #   num_obj：种类数量
        #   obj_ids：一张图上所有的种类ID
        for root, dirs, files in os.walk(dataset_mask_dir):
            print('-----------------------------', root, '--------------------------------')
            print(files)
            i = 0;
            for file in files:
                i = i + 1
                if i == 10:
                    break
                mask = Image.imread(os.path.join(root, file))  # mask图片文件
                mask = np.array(mask)  # mask图片转数组
                image_path = os.path.join(dataset_dir, subset_image, file)  # 图片路径
                mask_path = os.path.join(root, file)  # mask路径
                obj_ids = np.unique(mask)  # 所有的种类编号
                obj_id_without_bg = obj_ids[1:]
                height, width = np.shape(mask)[:2]  # mask尺寸
                num_obj = len(obj_id_without_bg)  # 每个mask有几种 物体
                self.add_image(
                    "drive",
                    image_id=file,  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height,
                    mask_path=mask_path,
                    num_obj=num_obj,
                    obj_ids=obj_ids
                )
        print(len(self.image_info))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        print("加载mask中")
        # If not a ‘drive’ dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "drive":
            return super(self.__class__, self).load_mask(image_id)

        mask_templet = np.array(Image.imread(image_info["mask_path"]))
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        obj_ids = image_info['obj_ids']
        obj_ids = obj_ids[1:]  # 刨除背景
        mask = np.zeros(([image_info['height'], image_info['width'], image_info['num_obj']]))
        print(obj_ids)
        for id_in_mask in obj_ids:
            area = np.where(mask_templet[:, :, 0] == id_in_mask)
            n = np.shape(area)[-1]
            num = 0;
            mask[area[0], area[1], num] = 1;
            num = num + 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1sss
        a = mask.astype(np.bool)
        # print('-----------')
        # print(image_info['num_obj'])
        # print(obj_ids)
        # print(np.shape(mask))
        # print(self.image_ids)
        return mask.astype(np.bool), obj_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "drive":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


