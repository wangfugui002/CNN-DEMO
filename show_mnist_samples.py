import numpy as np
import torch
import matplotlib.pyplot as plt

# 加载mnist.npz
mnist = np.load('./data/mnist.npz')
x_train = mnist['x_train']
y_train = mnist['y_train']

# 转为Tensor并加通道维度
x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1) / 255.0

plt.figure(figsize=(10, 1.5))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(x_train[i][0], cmap='gray')
    plt.title(str(y_train[i]))
    plt.axis('off')
plt.suptitle('MNIST训练集部分图片')
plt.show()
