# Res系列

### resNet

基本Resedual block 块，该块的模型由两个层组成，输入层和输出要相加。示意图如下：

![img](D:\GitRepositories\dailyWork\zhao\image\webp)

​																				图1.残差基本单元结构

图中输入为x层，经过第一个weightLayer（卷积3*3卷积） ,这个weightLayer 经过relu 激活函数再通过第二个weightLayer（卷积层）得到输出。

这里x代表输入的层，F() 函数代表`卷积`、`relu激活函数`、`卷积`的一通操作(Resedual)，F(x)代表x经过操作之后的输出。

该块最终的输出为F(x)+x，记为y = F(x)+x， 而F(x)又称为残差。更为具体的图再下方。

![img](D:\GitRepositories\dailyWork\zhao\image\v2-bd76d0f10f84d74f90505eababd3d4a1_720w.jpg)

​																		图2. 详细结构

weight 代表卷积层，BN是归一化层。

上面的图片只是两层的残差单元，对应于下图中左边那个。

![残差单元](D:\GitRepositories\dailyWork\zhao\image\watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNzYwNzY3,size_16,color_FFFFFF,t_70)

​													图3. 两种残差结构（又叫残差单元）

而做相加的时候，有可能会存在通道不一致的情况。也为了简化计算，作者提出了右边的结构，专门用于resnet50_101_152等残差网络。我们只针对于resnet101来看，怎么样保证通道相同呢。



对于上图3，右边的残差单元，

	1. 直线分支：输入的图像首先经过1X1的卷积进行降维（通道降为64），然后激活，然后3*3卷积（通道还是64），激活，然后再1X1卷积升维度（通道数提升到256）。 
 	2. 曲线分支，输入的图像，如果通道数与直线分不一致则经过1X1X256 的卷积将图像的通道数变成与直线分支输出的通道数一致。
 	3. 直线分支与曲线分支相加然后经过relu激活。
 	4. 完成。

resnet101中的结构：



![img](D:\GitRepositories\dailyWork\zhao\image\watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3lhb2NodW5jaHU=,size_16,color_FFFFFF,t_70)



101代表又101层，其中不包括池化层，激活层。

1. stem(conv1 or stage 1):101中首先经过7x7步长为2的卷积，然后又经过池化。这是第一层，1层
2. conv2_x (stage 2): 3次残差单元，单元内又一层1x1卷积降维，一层3x3卷积提取特征，一层1x1卷积升维，其中第一次的残差单元的曲线分支需要经过1x1的卷积进行通道的匹配。后面的两次可以不用了，因为第一次对通道进行了统一了，后面的通道都是统一的。3\*3 = 9 层
3. conv3_x (stage 3): 4次残差单元，与上面相同，也是第一次，要对曲线进行卷积统一通道数。4\*3 = 12层
4. conv4_x (stage 4): 23次残差单元，与上面相同，第一次统一。有23\*3=69层
5. conv5_x (stage 5): 3次残差单元，第一次统一。有3\*3=9 层
6. 全连接FC。1层

总共：1+9+12+69+9+1 = 101层。

在mask RCNN中的Resnet的代码实现（基于keras）

曲线分支带有1x1卷积的残差单元

```python
def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                                           '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)  # 从往上是直线分支

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,# 从这往下是曲线分支
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x
```

画成图：

![1589478735055](D:\GitRepositories\dailyWork\zhao\image\1589478735055.png)

曲线分支不带1*1卷积的代码部分：

```python
def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    x = KL.Add()([x, input_tensor])  # 直接相加
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x
```

化成图：

![1589478825719](D:\GitRepositories\dailyWork\zhao\image\1589478825719.png)

构建resnet101的代码：

```python
def resnet_graph(input_image, architecture, stage5=False, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101  字符串
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet50", "resnet101"]

    '''
          input_tensor: input tensor
          kernel_size: default 3, the kernel size of middle conv layer at main path
          filters: list of integers, the nb_filters of 3 conv layer at main path
          stage: integer, current stage label, used for generating layer names
          block: 'a','b'..., current block label, used for generating layer names
          use_bias: Boolean. To use or not use a bias in conv layers.
          train_bn: Boolean. Train or freeze Batch Norm layers
    '''

    # Stage 1  stem
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=True)(x)
    x = BatchNorm(name='bn_conv1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)


    # Stage 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), train_bn=train_bn)  # 统一通道数
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', train_bn=train_bn)
    

    # Stage 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', train_bn=train_bn)  # 统一
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', train_bn=train_bn)


    # Stage 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', train_bn=train_bn)  # 统一
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i), train_bn=train_bn)
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', train_bn=train_bn)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]
```

总体结构：

![2019062022131876](D:\GitRepositories\dailyWork\zhao\image\2019062022131876.jpg)

### resNeXt

resNext 在resnet的基础上增加了宽度，