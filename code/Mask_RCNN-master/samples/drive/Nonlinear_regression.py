import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
# Dense 全连接层，Activation 激活函数
from keras.layers import Dense, Activation
# 优化器
from keras.optimizers import SGD


def replace_char(string, char, index):
    string = list(string)
    string[index] = char
    return ''.join(string)


# 随机生成200个随机点
x_data = np.linspace(-0.5, 0.5, 200)
noise = np.random.normal(0, 0.02, x_data.shape)
# square 是平方
y_data = np.square(x_data) + noise

plt.scatter(x_data, y_data)
plt.show()

# 构建一个顺序模型
model = Sequential()

# 在模型中添加一个全连接层
# 1-10-1
# 构建一个隐藏层输入出是10个神经元，输入是一个神经元
model.add(Dense(units=10, input_dim=1))

# 激活函数默认下没有，需要指定激活函数，导入激活函数包,tanh大多数时候比sigmod函数好用
model.add(Activation('tanh'))
# 下面这样也行
# model.add(Dense(units=10, input_dim=1,activate='relu'))


# 构建输出层，上一层是10个神经元，这里不用指定，只指定输出即可
model.add(Dense(units=1))

# 添加激活函数
model.add(Activation('tanh'))

# sgd是随机梯度下降默认学习率很小，大概是0.01，loss均平方误差
# model.compile(optimizer='sgd', loss='mse')

# 修改学习率需要导入kera.optimizers impprt SGD
# 定义优化算法
sgd = SGD(lr=0.3)

# 将优化器装入神经网络中(上面的compile注释掉)
model.compile(optimizer=sgd, loss='mse')

progress = '[..............................]'
for step in range(3000):
    cost = model.train_on_batch(x_data, y_data)
    if step % 100 == 0:
        progress = replace_char(progress, "=", progress.index("."))
        print(progress)
        print('cost', cost)
W, b = model.layers[0].get_weights()
print('W', W, 'b', b)

# x_data 输入网络中，得到预测值
y_pred = model.predict(x_data)

# 显示随机点
# plt.scatter(x_data, y_data, c='r')
plt.plot(x_data, y_pred, "r-")
plt.show()
