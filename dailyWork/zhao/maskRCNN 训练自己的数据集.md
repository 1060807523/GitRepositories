# maskRCNN 训练自己的数据集

### 一、环境简单说明

搭建环境的步骤略

maskRCNN源码是基于tensorflow 1.X，如果安装的tensorflow 2.x的话是不兼容的。

数据集：https://www.kaggle.com/kumaresanmanickavelu/lyft-udacity-challenge

来源于kaggle。

### 二、模型整体

整个源码核心的部分为分为五个文件：

config.py：模型配置参数类

model.py：模型的主体

parallel_model.py：并行模型，对上面model中一些复杂的计算可以并行进行
utils.py：主要使用数据集接口*
visualize.py：

文件目录如下：

![1586747117192](image\1586747117192.png)

本次到目前为止用到的是model.py  utils.py  config.py

#### config.py: 模型的配置参数

用于配置模型的基本参数，如GPU数量，学习率，步长等参数

主要内容如下：

```python
import numpy as np

class Config(object):
    NAME = None  # Override in sub-classes
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50
    BACKBONE = "resnet101"
    COMPUTE_BACKBONE_SHAPE = None
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024
    TOP_DOWN_PYRAMID_SIZE = 256
    NUM_CLASSES = 1  # Override in sub-classes
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_ANCHOR_STRIDE = 1
    RPN_NMS_THRESHOLD = 0.7
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    PRE_NMS_LIMIT = 6000
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_SCALE = 0
    IMAGE_CHANNEL_COUNT = 3
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    TRAIN_ROIS_PER_IMAGE = 200
    ROI_POSITIVE_RATIO = 0.33
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]
    MAX_GT_INSTANCES = 100
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    DETECTION_MAX_INSTANCES = 100
    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }
    USE_RPN_ROIS = True
    GRADIENT_CLIP_NORM = 5.0

    def __init__(self):
       
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM,
                self.IMAGE_CHANNEL_COUNT])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM,
                self.IMAGE_CHANNEL_COUNT])
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES
```

其中最基本的参数有，内容来自https://blog.csdn.net/ZesenChen/article/details/79593925：

> GPU_COUNT = 1#你打算用多少块GPU
> IMAGES_PER_GPU = 2#一块12G的GPU最多能同时处理2张1024*1024的图像
> STEPS_PER_EPOCH = 1000#一个epoch的迭代步数
> NUM_CLASSES = 1#目标类别=1(背景)+object_class，如果我只打算检测细胞核，那就设置为2
> IMAGE_MIN_DIM = 800#样本图像的最小边长，设置的不对会导致训练出错
> IMAGE_MAX_DIM = 1024#样本图像的最大边长，同上
> USE_MINI_MASK = True#是否压缩目标图像
> MINI_MASK_SHAPE = (56, 56)#目标图像的压缩大小

需要关注的是`GPU_COUNT` 和`IMAGES_PER_GPU`,这两个参数会决定`BATCH_SIZE`的值，计算如下

`BATCH_SIZE` = `GPU_COUNT`×`IMAGES_PER_GPU`

另有`NAME`: 对应于`utils.py`中`dataset`类的成员变量`image_info`中的`source`字段的名字，相当于是过滤器，训练的时候只会挑选指定的`source`图片进行。实例如下：

我们想要训练的图片的`source`是drive(名字是自己起的)

![1586748968643](image\1586748968643.png)

那么装载数据的时候就要在`source`字段中填`drive`

![1586749321251](image\1586749321251.png)

其中的`add_class()` 和` add_image()` 方法定义在`util.py`中

```python
def add_image(self, source, image_id, path, **kwargs):
    # 在调用该方法时，前三个参数必要的，也是model.py运行时会读取的
    # 也可在**kwargs中传入多个自定义的参数，就如上图所写的那样，多出来的参数主要是给load_mask用的
    image_info = {
        "id": image_id,
        "source": source,
        "path": path,
    }
    image_info.update(kwargs)
    self.image_info.append(image_info)
```

```python
def add_class(self, source, class_id, class_name):
    assert "." not in source, "Source name cannot contain a dot"
    # Does the class exist already?
    for info in self.class_info:
        if info['source'] == source and info["id"] == class_id:
            # source.class_id combination already available, skip
            return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })
```



另一个重要参数`NUM_CLASSES` ,是该模型要进行检测和分割的物体种类数，一般是1+n种，前面的1是bg，n是其他的种类。



此外https://blog.csdn.net/ZesenChen/article/details/79593925 博客中还提到：

> 还有IMAGE_MIN_DIM 和IMAGE_MAX_DIM 这两个参数是不能默认的，一旦你传入的图像不在这个范围里面训练出的模型是有问题的，这也是我碰到的最坑的地方，训练不会报错但得到的模型是不能用的；还有USE_MINI_MASK 这个参数，当你的mask图像中目标很小时要把这个参数设置为False，假设目标就4，5个像素点，然后被你一压缩就没了，那训练不就出问题了么。
> ————————————————
> 版权声明：本文为CSDN博主「ZesenChen」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
> 原文链接：https://blog.csdn.net/ZesenChen/article/details/79593925

#### utils.py: 工具类以及数据集接口类

在这个文件，主要关注`Dataset`，想要训练自己的数据集，需要继承该类，然后重写`load_mask()`和`image_reference()` 两个方法，还要添加一个`load_object()`方法。该类有如下几个成员：

```python
def __init__(self, class_map=None):
    self._image_ids = []
    self.image_info = []
    # Background is always the first class
    self.class_info = [{"source": "", "id": 0, "name": "BG"}]
    self.source_class_ids = {}
```

+ 首先应该是`load_object()` 方法，该方法不会被model调用，但是初始化模型，装载数据的重要方法，主要作用就是调用`add_class` 填写`class_info`成员属性 ,告知模型，要识别的物体有哪些，物体的Id是什么。如下：

  ```python
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
  ```

  还有作用就是根据标注文件如labme生成json,或者数据集中带有的mask.png文件，填写`image_info`成员属性

  ```python
  for file in files:
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
  ```
  
  本次训练使用的是mask图片的形式标注。关于mask图片在后文细讲。
  

填写时，前三个参数是必须的。`image_id` 用来标识一张图片，可以直接使用图片的文件名，`path`是图片的路径。

后面为了方便生成训练用mask，加入了`mask_path` 图片对应的mask标注的路径。`with`，`height`  图片的大小，也是为了方便生成mask用的。`obj_ids`这一张图片里有哪些标注的数字。

+ 其次是`load_mask()` 方法：

  该方法用于生成每张图片的mask，以及id。返回的mask是一个with和height都与原图相同，有`num_obj`那么多层。

  其中每一层mask是单独的一个实例。比如在原图上标注了两个汽车，三棵树，一栋房子。

  如果是语义分割，那么mask应该是有三层，第一层汽车，第二层是树，第三层是房子。具体画法是找出原图中两辆汽车所在的所有像素点的位置，然后利用得到的位置，在生成的空mask的第一层将对应的像素点标为True，其他的为false。第二层同样是这样处理。

  如果是实例分割的话，那个每个汽车，每棵树都要占据一层。也就是2+3+1=6层。

  注意：如果数据集中所有要监测的目标比如有4种，而某一张图片上只有三种或两种，那么mask只画这两三种就行。另外背景不需要画上去。

  最后返回，每一层的实例对应的ID号，也就是一个数组。

  ![1586752829950](image\1586752829950.png)

+ 最后是`image_reference()`方法：

  ```python
  def image_reference(self, image_id):
      """Return the path of the image."""
      info = self.image_info[image_id]
      if info["source"] == "drive":
          return info["path"]
      else:
          super(self.__class__, self).image_reference(image_id)
  ```

  用于返回`image_id` 对应的图片路径

  注意这个image_id 不是填写在成员属性里面的图片名称，而是一个数字。其实就是用来从`self.image_info`里取出某一项来用的，就是1,2,3,4,... 根据下标取的。

### 三、数据集处理

maskRCNN的接收的数据有两部分

1. RGB 图片（好像是作者使用imgaug库，这个库只能用RGB）
2. RGB图片对应的mask，这是个rgb图片中要识别物体的几何，它有多层，每一层长宽大小等于RGB图片长宽的尺寸，每一层每一个数值代表RGB图片的每一个像素，每一层只有一个物体实例的区域的像素被标成true，其他部分全是false，这叫做图片的二值化。为了表示每一层是什么物体，还需要有一个class_ids数组，数组的第一个值，代表第一层实例的Id，一次类推，第二个值是第二层...

上面是maskRCNN可以处理的，我们需要将我们的数据集处理成如上所得那样才能被maskRCNN训练。我们得到数据集大值可以分为两类。

1. 原图+json
2. 原图+黑色的图片(或者底色是黑色，各种彩色的物体)

第一种数据集是用标注工具标注的，json里面存了原图中每一个实例，所在区域的像素坐标。

第二种数据集中的黑色图片是mask标注，每个图片对应一个黑色的png图片，我们在这称为mask_templet(自己起的名字)，mask_templet 是大小都与原图一致的图片，原图中每一个实例所占用的像素，都在mask_templet对应的位置，标上了该实例的ID。

在代码中 我们可以用如下代码查看，这个图片中有多少种实例，每个实例的ID是什么。

```python
np.unique(mask_temlpt)
```

本次使用是mask_templet的方式训练。

数据集目录：



想要使用数据集，需要自己继承utils.py文件中的dateset类，重写load_mask

```python
class DriveDataset(utils.Dataset):
    def load_Object(self, dataset_dir, subset_mask, subset_image):
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
            for file in files:
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
        num = 0;
        for id_in_mask in obj_ids:
            area = np.where(mask_templet[:, :, 0] == id_in_mask)
            n = np.shape(area)[-1]
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

```

`load_Object()`方法：遍历一遍所有的`mask_templet`图片，提取里面的标注信息，每个图片的信息生成一个字典然后append到`image_info`数组汇总。`model.py`里会用到`image_info`。

### 四、参数类继承

继承config.py中的参数类，用于修改一些必要的参数。

```python
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
    NUM_CLASSES = 1 + 12  # background + 1 shapes

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

```

### 五、模型创建

1. 初始化一些路径

   ```python
   #  项目的根目录
   ROOT_DIR = os.getcwd()
   # 模型输出路径
   MODEL_DIR = os.path.join(ROOT_DIR, "logs")	
   # coco数据集权重文件路径
   COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, 'logs')
   ```

   

2. 数据集和验证集的准备

   ```python
   #  准备训练集
   dataset_train = DriveDataset()
   dataset_train.load_Object("D:\学习资料/机器学习\dataA/dataA", "CameraSeg", "CameraRGB")
   dataset_train.prepare()
   
   #  准备验证集
   dataset_val = DriveDataset()
   dataset_val.load_Object("D:\学习资料/机器学习\dataB/dataB", "CameraSeg", "CameraRGB")
   dataset_val.prepare()
   ```

3. 创建参数类对象

   ```python
   # 参数类
   config = ShapesConfig()
   config.display()
   ```

4. 模型创建

   ```python
   # 模型创建，训练 模式
   model = modellib.MaskRCNN(mode="training", config=config,
                             model_dir=MODEL_DIR)
   ```

   其中`mode`参数有`training`和`inference`两种，字面意思就是训练和推理。关于`modellib.MaskRCNNMaskRCNN`的描述原文：

   >```python
   >def __init__(self, mode, config, model_dir):
   >    """
   >        mode: Either "training" or "inference"
   >        config: A Sub-class of the Config class
   >        model_dir: Directory to save training logs and trained weights
   >        """
   >    assert mode in ['training', 'inference']
   >    self.mode = mode
   >    self.config = config
   >    self.model_dir = model_dir
   >    self.set_log_dir()
   >    self.keras_model = self.build(mode=mode, config=config)
   >```

5. 初始权重加载

   ```python
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
   ```

6. 训练/预测

   ```python
   print("Training network heads")
   model.train(dataset_train, dataset_val,
               learning_rate=config.LEARNING_RATE,
               epochs=30,
               layers='heads')
   ```

   训练时，每个epoch完成之后，会使用验证集的调整。之前将验证集的路径写错导致一直报出。

   > UnboundLocalError： local variable image_id' referenced before assignment

   将验证集写对就可以了

### 六、完整代码

```python
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
        dataset_image_dir = os.path.join(dataset_dir, subset_mask)
        print('查看', dataset_mask_dir)
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
                mask = Image.imread(os.path.join(root, file))  # mask图片文件
                mask = np.array(mask)  # mask图片转数组
                image_path = os.path.join(dataset_dir, subset_image, file)  # 图片路径
                mask_path = os.path.join(root, file)  # mask路径
                obj_ids = np.unique(mask)  # 所有的种类编号
                obj_id_without_bg = obj_ids[1:]
                height, width = np.shape(mask)[:2]  # mask尺寸
                num_obj = len(obj_id_without_bg)  # 每个mask有几种 物体
                if num_obj > 13:
                    continue
                self.add_image(
                    "drive",
                    image_id=file,  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height,
                    mask_path=mask_path,
                    num_obj=num_obj,
                    obj_ids=obj_ids
                )
        print('info：', len(self.image_info))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
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
        num = 0;
        for id_in_mask in obj_ids:
            area = np.where(mask_templet[:, :, 0] == id_in_mask)
            n = np.shape(area)[-1]
            mask[area[0], area[1], num] = 1;
            num = num + 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1sss
        a = mask.astype(np.bool)
        # print('-----------')

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
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 12  # background + 12

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 576
    IMAGE_MAX_DIM = 832

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
    dataset_train.load_balloon("/media/deep/新加卷/data-soft/dataA/dataA", "CameraSeg", "CameraRGB")
    dataset_train.prepare()

    #  准备验证集
    dataset_val = BalloonDataset()
    dataset_val.load_balloon("/media/deep/新加卷/data-soft/dataA/valA", "CameraSeg", "CameraRGB")
    dataset_val.prepare()

    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')  #


if __name__ == '__main__':

    #  项目的根目录
    ROOT_DIR = os.path.abspath("../../")
    # 模型输出路径
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    # 参数类
    config = ShapesConfig()
    config.display()

    # coco数据集权重文件路径
    COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, 'logs/mask_rcnn_coco.h5')

    # 模型创建，训练 模式
    model = modellib.MaskRCNN(mode="training",
                              config=config,
                              model_dir=MODEL_DIR)

    # 使用的模型参数
    init_with = "coco"  # imagenet, coco, or last
    weights_path = COCO_WEIGHTS_PATH
    print("加载权重")
    if init_with == "coco":  # 使用COCO的权重
        model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"
        ])
    elif init_with == "imagenet":  # 使用ImageNet的权重
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "last":  # 使用最后一次训练的权重
        model.load_weights(model.find_last(), by_name=True)

    #  训练模型
    train(model)

    print("训练完成")


```

