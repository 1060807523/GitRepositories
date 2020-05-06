import numpy as np
from keras.datasets import mnist
# Convolution2D:2为的卷积
# MaxPooling2D:2为的池化
# Flatten:扁平化,例如把二位数据扁平化为一维的数据
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten

# 常用SimoleRNN，LSTM，GRU三种循环神经网络
from keras.layers.recurrent import SimpleRNN
from keras.utils import np_utils
from keras.models import Sequential
# 导入 adam 优化器
from keras.optimizers import SGD, Adam

# 隐藏层cell个数
cell_size = 50

# 序列长度一共28行，每行是一个序列
time_steps = 28

# 数据长度一行有28个像素
input_size = 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# (60000,28,28)=>(60000,28,28,1) 1是图片的深度，-1自动匹配
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
# 换成one hot格式
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 创建模型
model = Sequential()

# 循环神经网络
model.add(SimpleRNN(
    units=cell_size,
    input_shape=(time_steps, input_size)
))

# 输出层
model.add(Dense(10, activation='softmax'))

# 定义优化器
adam = Adam(lr=1e-4)

# 模型编译
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, batch_size=64, epochs=2)

# 评价模型
loss, accuracy = model.evaluate(x_test, y_test)
print('test loss', loss)
print('test accuracy', accuracy)
