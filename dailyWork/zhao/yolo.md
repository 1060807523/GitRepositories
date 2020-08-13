# yolo v3

### LeakyReLU:

类似于RelU,不过负半轴不在像Relu那样一刀切成0，而是给了一个很小的a，作为斜率，这个a是一个可学习参数。公式就变成下面这样。

![img](D:\GitRepositories\dailyWork\zhao\image\1244340-20190704145445485-869978198.png)



另外补充非线性定义：导数不为一个常数的即非线性函数。

单独来看RelU系列函数，正负均是线性，但组合在一起就不是线性了。以全连接为例子，每个神经元的输出，都是上一层所有神经元的乘加和经过激活后的值，而一层神经元有多个，有些神经元的输入是非正的，那么这个神经元就被抑制，这样这一层的输出结果就不再是上一层的线性变换了。



### Yolo v3 的网络结构：

![img](D:\GitRepositories\dailyWork\zhao\image\yolo.jpg)

`DBL`：	就是卷积+BN+Leaky relu。对于v3来说，BN和leaky relu已经是和卷积层不可分离的部分了(最后一层卷积除外)，共同构成了最小组件

`res*n`: 	n代表数字，有res1，res2 ，res8，表示这个res_block里含有多少个res_unit，res_unit就是残差单元。

`concat`:	张量拼接，会扩充两个张量的维度，例如**26×26×256**和**26×26×512**两个张量拼接，结果是**26×26×768**。Concat和cfg文件中的route功能一样

`add`:	张量相加，张量直接相加，不会扩充维度，例如**104×104×128**和**104×104×128**相加，结果还是**104×104×128**。add和cfg文件中的shortcut功能一样

#### 主干网络 

主干网路是DarkNet53，从上图就可以看出darkNet的结构。这里的darkNet53没有全连接层，其实是52层。

其生成的特征金字塔有三层。分别是1/8倍的特征图，1/16倍特征图，1/32倍特征图（因此输入图片必须为32的整数倍）。这里进行尺寸的改变的时候都用的卷积+步长来调整的，没有使用池化。



#### anchor机制

通过对ground True 的尺寸进行k-mean聚类，k为人为设定的值（k种尺寸，提高多尺度性），v2是5，v3是9。由此来确定anchor的尺寸，这里不同于maskrcnn的固定尺寸。特征图的每一个像素点，都生成三个框，三个特征图平分9种尺寸的anchor(9种尺寸分为大中小，三种)。

体验一下anchor框：

![å¨è¿éæå¥å¾çæè¿°](D:\GitRepositories\dailyWork\zhao\image\20190331091342340.png)

每个anchor框会使用**logistic regression** 得到目标性评分，用于去掉低分框，减少不必要的anchor。

这里每个框要包含class_num +5个元素，分别是各种分类的概率，x(中心坐标)，y(中心坐标)，w(尺寸)，h(尺寸)，con（是否包含识别目标的概率）。

训练的学习目标就是x,y,w,h

# yolov4

### mish 激活函数

公式：
$$
Mish(x) = x*tanh(ln(1+e^x))
$$
![img](D:\GitRepositories\dailyWork\zhao\image\mish.jpg)

mish函数优于relu函数。

### 网络结构图

![2020052108583188](D:\GitRepositories\dailyWork\zhao\image\2020052108583188.png)

CBM：对标对标v3的DBL，即Conv+Bn+Mish三者组成，是v4的最小的组件

CBL：就是v3的DBL，Conv+Bn+leaky_relu

Res unit: 残差单元，

CSPX：CSPnet的网络结构，CBL+x个残差块+CBL 在与输入层经过一个CBL（近道shotcut）

spp:平行的三个池化，然后堆叠，

### 创新部分

1. 输入端数据增强，采用Mosaic数据增强
2. 主干网络使用CSPDarknet53，Dropblock正则化防止过拟合，Mish激活函数
3. Neck使用FPN+PAN结构和SPP模块
4. 损失函数使用CIOU_loss，综合考虑边界框的宽高，中心点距离，重叠率
5. 非极大值抑制采用DIOU_nms

### Mosaic数据增强：







## 学习率调整

ReduceLROnPlateau

当指标停止提升时，降低学习速率。

一旦学习停止，模型通常可使学习率降低2-10倍。该重新监测数量，如果没有看到epoch的'patience'数量的改善，那么学习率就会降低。

```python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
```

- monitor：要监测的数量。
- factor：学习速率降低的因素。new_lr= lr * factor
- 耐心：没有提升的时代数，之后学习率将降低。
- verbose：int。0：安静，1：更新消息。
- 模式：{auto，min，max}之一。在最小模式下，当监测量停止下降时，lr将减少；在最大模式下，当监测数量停止增加时，减少减少；在自动模式下，从监测数量的名称自动预测方向。
- min_delta：对于测量新的最优化的阀值，仅关注重大变化。
- cooldown：在学习速率被降低之后，重新恢复正常操作之前等待的epoch数量。
- min_lr：学习率的下限。

