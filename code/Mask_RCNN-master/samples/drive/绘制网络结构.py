import numpy as np
from keras.datasets import mnist
# Convolution2D:2为的卷积
# MaxPooling2D:2为的池化
# Flatten:扁平化,例如把二位数据扁平化为一维的数据
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.models import Sequential
# 导入 adam 优化器
from keras.optimizers import SGD, Adam
# install pydot and graphviz

from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# (60000,28,28)=>(60000,28,28,1) 1是图片的深度，-1自动匹配
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0  # -1 是自适应
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
# 换成one hot格式
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)


# 定义顺序模型
model = Sequential()

# 第一个卷积层
model.add(Convolution2D(
    input_shape=(28, 28, 1),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',
    activation='relu'
))

# 第一个池化层，不需要设定输入形状了
model.add(MaxPooling2D(
    pool_size=2,  # 池化窗口2*2
    strides=2,  # 步长
    padding='same'
))
# 第二个卷积层，64个卷积核，大小5*5
model.add(Convolution2D(64, 5, strides=1, padding='same', activation='relu'))

# 第二个池化层
model.add(MaxPooling2D(2, 2, 'same'))

# 第二个池化层的输出扁平化为1维
model.add(Flatten())

# 第一个全连接层
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))

# 第二个全连接层
model.add(Dense(10, activation='softmax'))

adam = Adam(lr=1e-4)

model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, batch_size=32, epochs=10)

#  评估模型
loss, accuracy = model.evaluate(x_test, y_test)

print('test loss', loss)
print('test accuracy', accuracy)

# 绘制模型
plot_model(model,to_file="model.png",show_shapes=True,show_layer_names=True,rankdir='TB')
plt.figure(figsize=(10,10))
img = plt.imread("model.png")
plt.imshow(img)
plt.axis('off')
plt.show()