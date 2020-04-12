import numpy as np
import matplotlib.pyplot as plt
# import PIL as Image
import os
import sys
import skimage.io as Image

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize


# 重写load_mask
############################################################
#  Dataset
############################################################

class BalloonDataset(utils.Dataset):
    def load_balloon(self, dataset_dir, subset_mask, subset_image):
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
                height, width = np.shape(mask)[:2]  # mask尺寸
                num_obj = len(obj_ids)  # 每个mask有几种 物体
                self.add_image(
                    "drive",
                    image_id=files,  # use file name as a unique image id
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
        print(mask_templet)
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        obj_ids = image_info['obj_ids']
        obj_ids = obj_ids[1:]  # 刨除背景
        mask = np.zeros(([image_info['height'], image_info['width'], image_info['num_obj']])) # 创建空的mask
        #  根据mask_templet 在对应的层上画上实例
        for id_in_mask in obj_ids:
            area = np.where(mask_templet[:, :, 0] == id_in_mask)
            # print(area)
            # print(np.shape(area))
            n = np.shape(area)[-1]
            print(n, "   ----", type(n))
            num = np.ones((1, n)) * id_in_mask
            mask[area[0], area[1], id_in_mask - 1] = 1;

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        plt.imshow(mask[:, :, 4])
        plt.show()
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "drive":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "drive"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 +12  # background + 1 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 384

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50


#  封装一个训练方法
def train(model):
    """Train the model."""
    #  准备训练集
    dataset_train = BalloonDataset()
    dataset_train.load_balloon("D:\学习资料/机器学习\dataA/dataA", "CameraSeg", "CameraRGB")
    dataset_train.prepare()

    #  准备验证集
    dataset_val = BalloonDataset()
    dataset_val.load_balloon("D:\学习资料/机器学习\dataB/dataB", "CameraSeg", "CameraRGB")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

if __name__ == '__main__':


    #  项目的根目录
    ROOT_DIR = os.getcwd()
    # 模型输出路径
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    # 参数类
    config = ShapesConfig()
    config.display()

    # coco数据集权重文件路径
    COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, 'logs')

    # 模型创建，训练 模式
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    # 使用的模型参数
    init_with = "coco"  # imagenet, coco, or last
    weights_path = COCO_WEIGHTS_PATH

    if init_with == "coco":  # 使用COCO的权重
        model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"
        ])
    elif init_with == "imagenet":  # 使用ImageNet的权重
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "last":
        model.load_weights(model.find_last(), by_name=True)

    #  训练模型
    train(model)