import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
# Dense 全连接层
from keras.layers import Dense


def replace_char(string, char, index):
    string = list(string)
    string[index] = char
    return ''.join(string)


x_data = np.random.rand(100)
noise = np.random.normal(0, 0.01, x_data.shape)
y_data = x_data * 0.1 + 0.2 + noise

print("noise:", noise)
# 显示随机点
plt.scatter(x_data, y_data)
plt.show()

# 构建一个顺序模型
model = Sequential()

# 在模型中添加一个全连接层
model.add(Dense(units=1, input_dim=1))

model.compile(optimizer='sgd', loss='mse')
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
