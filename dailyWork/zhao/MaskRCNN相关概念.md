# MaskRCNN

## 基础知识：

### `梯度`：

高数知识，对于u = f(x,y,z)

梯度：grad u = {φu/φx, φu/φy, φu/φz}

实际就是对每个参数求偏导得到的一个向量。梯度的方向是这个函数上升最快的方向，其方向导数最大。

方向导数的求法：

对于固定点M<sub>0</sub>(x<sub>0</sub>,y<sub>0</sub>,z<sub>0</sub>) ，u 向任意一点M(x,y,z) 方向变化的速度。也可看做，M<sub>0</sub>向着M的切线 L 的方向。方向导数定义如下
$$
\frac{φu}{φl} = \frac{φu}{φx}cosα+\frac{φu}{φy}cosβ+\frac{φu}{φz}cosγ
$$
方向导数写成向量的形式就是：
$$
\frac{φu}{φl} = (\frac{φu}{φx},\frac{φu}{φy},\frac{φu}{φz})⋅(cosα,cosβ,cosγ)
$$


上式前半截那一堆偏导组成的向量是个常向量，固定了点M<sub>0</sub> 前半截就确定了。 后半截是切线 L 的单位方向向量。两个向量相乘等于
$$
\sqrt{(\frac{φu}{φx}^2 ,\frac{φu}{φy}^2,\frac{φu}{φz}^2)}·cosΘ
$$
Θ是前半截向量和后半截向量的夹角。由于后半截是单位向量，所以模为1，就没了。由此可见，夹角Θ为0的时候，也就是没有夹角，方向向量与前半截同向的时候，变化速度最大。



### `正向传播`：

对一个神经元，他接受与他相连的所有的神经元的输出与连接权重的乘加和，放入到自己的激励函数中得到输出值传递给下一层的神经元，重复如此操作。

![1587156241554](image/1587156241554.png)

正向传播公式对于图中的神经元，wij是第i层，第j个神经元与图中神经元的连接权重，Oi是上一层每个神经元的输出。设该神经元激励函数为sigmoid函数

该神经元向前传播输出的值Oj:
$$
O_j =sigmoid(\sum{O_iw_{ij}}) 
$$

$$

$$

### `反向传播`：

back propagate

改编自https://blog.csdn.net/weixin_38347387/article/details/82936585，原文写的更好。

对于如下网络

![img](image/format,png)

就是通过对误差的贡献，从输出层结果，不停计算每个节点对最终误差的贡献，得到该结点的误差值，在上一层与该结点相连的连接。

从局部开始看：

![img](image/partNetwork,png)

https://blog.csdn.net/henreash/article/details/102925892

1. 计算误差

   ​	所有输出节点的总误差target是每个输出节点的期望值，outpu<sub>i</sub>是该输出节点实际的输出
   $$
   E_{total} = E_{out1}+E_{out2} =∑\frac{1}{2}(target-output_i)^2
   $$
   公式中每个节点的误差前面的1/2是为了求偏导时消除指数2变成的系数。

2. 更新隐藏层到输出层的参数（w5,w6,w7,w8）

   以权重w5为例

   E<sub>total</sub> 是一个关于未知量output<sub>1</sub>，output<sub>2</sub>的函数，而与w5有关的output<sub>1</sub>计算如下
   $$
   output_1 = sigmoid(y_{h1}*w5+y_{h2}*w6+b_2)
   $$
   ​	我设net<sub>o1</sub>为sigmoid函数的输入：
   $$
   net_{o1}=y_{h1}*w5+y_{h2}*w6+b_2
   $$
   ​	采用梯度下降法，E<sub>total</sub> 实际是一个关于所有权重的参数，但是我们要修改隐藏层到输出层权重的，那么从输入层输出的数值保持向前传播时的值，即为一个常量。所以E<sub>total</sub> 就是关于w5 , w6, w7, w8的函数。 根据梯度的概念，E<sub>total</sub> 沿着梯度向量变化最快，怎么然他沿着呢，当然就是在这个向量上平移即 w5 w6 w7 w8 按照方向向量的比例加减。方向向量如下：
   $$
   \{\frac{φE_{total}}{φw_5},\frac{φE_{total}}{φw_6} ,\frac{φE_{total}}{φw_7},\frac{φE_{total}}{φw_8}\}
   $$
   也就是说w5等其他参数只要减去等比例的梯度，就能最快的收敛。
   $$
   \{w5,w6,w7,w \}-α \{\frac{φE_{total}}{φw_5},\frac{φE_{total}}{φw_6} ,\frac{φE_{total}}{φw_7},\frac{φE_{total}}{φw_8}\}
   $$
   这个比例α 就是学习率（learning rate）,相减的过程就叫做惩罚。梯度向量中的每一个元素叫做局部梯度（自己起的名字，也可能真叫这个）

   w5的局部梯度求解(链式偏导)：
   $$
   \frac{φE_{total}}{φw_5} =\frac{φE_{total}}{φoutput_1}*\frac{φoutput_1}{φnet_{o1}}*\frac{φnet_{o1}}{φw5}
   $$
   步步求偏导之后：
   $$
   \frac{φE_{total}}{φw_5} =-(target_{o1}-output_{o1})output_1(1-output_1)*y_{h1}
   $$
   

   令：
   $$
   δ_{o1} =\frac{φE_{total}}{φoutput_1}*\frac{φoutput_1}{φnet_{o1}} =-(target_{o1}-output_{o1})output_1(1-output_1)
   $$
   

   因此：
   $$
   \frac{φE_{total}}{φw_5} =δ_{o1}*y_{h1}
   $$
   

   

   3. 更新输入层到隐藏层的权重w1,w2,w3,w4。

      ![img](image/inputWeight,png)

      以w1 为例，h1节点的误差受到E<sub>o1</sub>和E<sub>o2</sub>

      因此公式组如下：其他步骤如上即可。
      $$
      \frac{φE_{total}}{φw1} = \frac{φE_{total}}{φoutput_{h1}} *\frac{φoutput_{h1}}{φnet_{h1}}*\frac{φnet_{h1}}{φw1}
      $$

      $$
      \frac{φE_{total}}{φoutput_{h1}} = \frac{φE_{out1}}{φoutput_{h1}}+\frac{φE_{out2}}{φoutput_{h1}}
      $$

      $$
      φoutput_{h1} = sigmoid[Σ(y_{input_1}*w1 + y_{input_2}*w2+b1)]
      $$

   $$
   net_{h1} =y_{input_1}*w1 + y_{input_2}*w2+b1
   $$

   

maskRCNN 是基于fasterRCNN的和FPN的模型。

FPN主要包含两部分：

1. bottom-up pathway:正常网络的向前传播过程，基于resnet网络（残差网络）

   

2. top-down pathway：



## 报错信息:

### 测试图片路径是否正确

```shell
xdg-open ***.png
```

打开图片指令



### TypeError

```python
The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/deep/maskRcnn/Mask-Rcnn-master/samples/city/citytarinning.py", line 234, in <module>
    train(model)
  File "/home/deep/maskRcnn/Mask-Rcnn-master/samples/city/citytarinning.py", line 199, in train
    layers='heads')  #
  File "/home/deep/maskRcnn/Mask-Rcnn-master/mrcnn/model.py", line 2374, in train
    use_multiprocessing=True,
  File "/home/deep/.local/lib/python3.6/site-packages/keras/legacy/interfaces.py", line 91, in wrapper
    return func(*args, **kwargs)
  File "/home/deep/.local/lib/python3.6/site-packages/keras/engine/training.py", line 1418, in fit_generator
    initial_epoch=initial_epoch)
  File "/home/deep/.local/lib/python3.6/site-packages/keras/engine/training_generator.py", line 181, in fit_generator
    generator_output = next(output_generator)
  File "/home/deep/.local/lib/python3.6/site-packages/keras/utils/data_utils.py", line 709, in get
    six.reraise(*sys.exc_info())
  File "/home/deep/.local/lib/python3.6/site-packages/six.py", line 693, in reraise
    raise value
  File "/home/deep/.local/lib/python3.6/site-packages/keras/utils/data_utils.py", line 685, in get
    inputs = self.queue.get(block=True).get()
  File "/home/deep/anaconda3/envs/tensorflow-gpu/lib/python3.6/multiprocessing/pool.py", line 644, in get
    raise self._value
TypeError: only integer scalar arrays can be converted to a scalar index
ERROR:root:Error processing image {'id': 'bremen_000194_000019_gtFine_instanceIds.png', 'source': 'city', 'path': '/media/deep/新加卷/data-soft/cityscapes/leftImg8bit/train/bremen/bremen_000194_000019_leftImg8bit.png', 'width': 2048, 'height': 1024, 'mask_path': '/media/deep/新加卷/data-soft/cityscapes/gtFine/train/bremen/bremen_000194_000019_gtFine_instanceIds.png', 'num_objs': 20, 'obj_ids': array([    6,     7,     8,    11,    17,    19,    20,    21,    23,
          24,    26, 24000, 24001, 26000, 26001, 26002, 26003, 26004,
       26005, 26006, 26007], dtype=uint16)}
Traceback (most recent call last):
  File "/home/deep/maskRcnn/Mask-Rcnn-master/mrcnn/model.py", line 1709, in data_generator
    use_mini_mask=config.USE_MINI_MASK)
  File "/home/deep/maskRcnn/Mask-Rcnn-master/mrcnn/model.py", line 1265, in load_image_gt
    class_ids = class_ids[_idx]
TypeError: only integer scalar arrays can be converted to a scalar index

Process finished with exit code 1
```



maskRCNN通篇的数据结构均是np数组，爆出此错误时我在load_mask 中返回的class_id使用了一般的array类型。

nparray 和 array 在打印的时候表现也不一样:

```
[1, 2, 3, 4] <class 'list'>
[1 2 3 4] <class 'numpy.ndarray'>
```



### ResourceExhaustedError

>ResourceExhaustedError: OOM when allocating tensor with shape[1,128,256,256] and type float ……

gpu不足，使用 

```shell
nvidia-smi
```

查看GPU的使用

![1587219123281](image/1587219123281.png)

`Memory-Usage`一栏里的数值是：已用显存大小/可用显存大小

`volatile Uncorr. ECC GPU-Util compute M.` 一栏显示的GPU利用率

GPU不足一般要杀死一些进程。如果`nvidia-smi`中没有显示占用的话使用`fuser -v /dev/nvidia*`查看所有进程

![1587226371538](image/1587226371538.png)

杀死`/dev/nvidia0` 中的某些进程，在上图我们要杀死6144号进程。



### MemoryError

```python
Unable to allocate array with shape (XX, XX, XX) and data type float64
```

这个错误报在load_mask 中，说的是创建的mask数组占用的内存过大无法创建(每个数字占用64位)，将创建时使用的类型改成int8(每个数字占用8位).

爆出该错误的原因是（节选自解决方案https://blog.csdn.net/Ryxiong728/article/details/105016093/）：

据传 [1] 是因为触发了系统的 overcommit handing 模式。

事情是这样的，我打算生成一个形状为[430949, 430949]的稀疏矩阵，结果就报了上述错误。大致是因为这个矩阵似乎要占用的空间太大，所以系统就提前禁止创建了。



### ValueError

```
Traceback (most recent call last):
 
  File "<ipython-input-1-fdee81fb82fb>", line 1, in <module>
    runfile('G:/labelme/test.py', wdir='G:/labelme')
 
  File "C:\Users\34905\Anaconda3\envs\cv2\lib\site-packages\spyder_kernels\customize\spydercustomize.py", line 704, in runfile
    execfile(filename, namespace)
 
  File "C:\Users\34905\Anaconda3\envs\cv2\lib\site-packages\spyder_kernels\customize\spydercustomize.py", line 108, in execfile
    exec(compile(f.read(), filename, 'exec'), namespace)
 
  File "G:/labelme/test.py", line 48, in <module>
    model.load_weights(COCO_MODEL_PATH, by_name=True)
 
  File "G:\Mask_RCNN\mrcnn\model.py", line 2131, in load_weights
    saving.load_weights_from_hdf5_group_by_name(f, layers)
 
  File "C:\Users\34905\Anaconda3\envs\cv2\lib\site-packages\keras\engine\saving.py", line 1149, in load_weights_from_hdf5_group_by_name
    str(weight_values[i].shape) + '.')
 
ValueError: Layer #389 (named "mrcnn_bbox_fc"), weight <tf.Variable 'mrcnn_bbox_fc/kernel:0' shape=(1024, 8) dtype=float32_ref> has shape (1024, 8), but the saved weight has shape (1024, 324).

```

如果你不想测试coco里默认的81类，只想测试2类，那一定记住要把model.load_weights(COCO_MODEL_PATH, by_name=True)改为model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])
————————————————
版权声明：本文为CSDN博主「唤醒沉睡的你」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_16065939/article/details/84769397



```python


Traceback (most recent call last):

  File "<ipython-input-10-4a900268c3f6>", line 1, in <module>
    runfile('D:/Mask R-CNN/train.py', wdir='D:/Mask R-CNN')

  File "D:\Anaconda3\Anaconda3-5.3.0\envs\cv2\lib\site-packages\spyder_kernels\customize\spydercustomize.py", line 704, in runfile
    execfile(filename, namespace)

  File "D:\Anaconda3\Anaconda3-5.3.0\envs\cv2\lib\site-packages\spyder_kernels\customize\spydercustomize.py", line 108, in execfile
    exec(compile(f.read(), filename, 'exec'), namespace)

  File "D:/Mask R-CNN/train.py", line 158, in <module>
    model.load_weights(COCO_MODEL_PATH, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc","mrcnn_bbox", "mrcnn_mask"])

  File "D:\Mask R-CNN\mrcnn\model.py", line 2131, in load_weights
    saving.load_weights_from_hdf5_group_by_name(f, layers)

  File "D:\Anaconda3\Anaconda3-5.3.0\envs\cv2\lib\site-packages\keras\engine\saving.py", line 1104, in load_weights_from_hdf5_group_by_name
    g = f[name]

  File "h5py\_objects.pyx", line 54, in h5py._objects.with_phil.wrapper

  File "h5py\_objects.pyx", line 55, in h5py._objects.with_phil.wrapper

  File "D:\Anaconda3\Anaconda3-5.3.0\envs\cv2\lib\site-packages\h5py\_hl\group.py", line 177, in __getitem__
    oid = h5o.open(self.id, self._e(name), lapl=self._lapl)

  File "h5py\_objects.pyx", line 54, in h5py._objects.with_phil.wrapper

  File "h5py\_objects.pyx", line 55, in h5py._objects.with_phil.wrapper

  File "h5py\h5o.pyx", line 190, in h5py.h5o.open

KeyError: 'Unable to open object (wrong B-tree signature)'
```



这个bug是我目前遇到最难察觉问题的bug，而且网上关于这个bug的解决方法很少，经过我仔细的排查，我发现是我的mask_rcnn_coco.h5文件有问题。我重新下载后，就解决了这个bug，建议大家在重新搭建平台的时候，不要用上一个平台已经用过的预权重 mask_rcnn_coco.h5，会发生很多不可知的错误。最好还是重新去网上下载一个。
————————————————
版权声明：本文为CSDN博主「唤醒沉睡的你」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_16065939/article/details/84769397



### AttributeError: 

```
AttributeError:'NoneType' object has no attribute 'terminate'
```

线程的原因，在这个错误上面肯定有一个关于线程创建的错误，如果最后输出的程序状态没有异常（状态码：0）可以不用理会。



### OSError

```python
OSError: [Errno 12] Cannot allocate memory
```

无法分配内存，内存被沾满了，可以尝试减小batch_size。或者监控内存使用情况分析原因

```shell
watch -n 2 free -m
```



## 多线程模型改成单线程模型：

遇到这个问题大家不要慌，这是因为大家设置了多线程，但是线程不同步造成的。大家耐心等一会，就会出现loss了。同时如果大家不希望出现多线程，大家可以改为单线程。更改方法如下：在Mask RCNN\mrcnn\model.py中

```python
self.keras_model.fit_generator(
    train_generator,
    initial_epoch=self.epoch,
    epochs=epochs,
    steps_per_epoch=self.config.STEPS_PER_EPOCH,
    callbacks=callbacks,
    validation_data=val_generator,
    validation_steps=self.config.VALIDATION_STEPS,
    max_queue_size=100,
    workers=workers,
    use_multiprocessing=True,
    # use_multiprocessing=False,
)
```



大家如果只是想要把多线程改为单线程，就要把use_multiprocessing=False,同时要让workers=1。因为单线程的情况下让workers大于1会报错！
————————————————
版权声明：本文为CSDN博主「唤醒沉睡的你」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_16065939/article/details/84769397



## 不明白的地方：

```python

#model.train(dataset_train, dataset_val,learning_rate=config.LEARNING_RATE,epochs=1,layers='heads')
model.train(dataset_train, dataset_val,learning_rate=config.LEARNING_RATE,epochs=20,layers='heads')
 
model.train(dataset_train, dataset_val,learning_rate=config.LEARNING_RATE / 10,epochs=20,layers="all")
#model.train(dataset_train, dataset_val,learning_rate=config.LEARNING_RATE / 10,epochs=1,layers="all")
```

训练了两次？有什么深意吗？

————————————————
版权声明：本文为CSDN博主「唤醒沉睡的你」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_16065939/article/details/84769397