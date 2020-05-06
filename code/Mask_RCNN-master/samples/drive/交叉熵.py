import numpy as np
from keras.datasets import mnist
from keras.layers import Dense
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import SGD

# 载入数据
(x_train, y_trian), (x_test, y_test) = mnist.load_data()
print("x_shape", x_train.shape)
print("y_shape", y_trian.shape)

# (60000,28,28)->(60000,784)
# -1 代表不一定有多少行，自适应行数
# 255 归一化处理。
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# 换成one hot格式
y_train = np_utils.to_categorical(y_trian, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)  #

# 创建模型
# 输入784个神经元，输出10个神经元,
model = Sequential([
    # 偏执的初始值是1，激活函数定义为softmax函数，用来将输出变成概率
    Dense(units=10, input_dim=784, bias_initializer="one", activation="softmax")
])

# 定义优化器
sgd = SGD(lr=0.2)

model.compile(
    optimizer=sgd,
    loss="categorical_crossentropy",  # 交叉熵损失，分类问题用交叉熵速度块，效果好
    metrics=['accuracy']  # 同时计算准确率
)

# 60000/32次为一个周期。
# 训练模型
model.fit(x_train, y_train, batch_size=32, epochs=10)

# 评估模型
loss, accuracy = model.evaluate(x_test, y_test)

print('\ntest loss', loss)
print('accuracy', accuracy)
