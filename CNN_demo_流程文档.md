# CNN Demo 流程文档

## 概述
本文档详细说明了 `cnn_demo.py` 文件中卷积神经网络（CNN）的完整实现流程，用于MNIST手写数字分类任务。

## 1. 环境准备和导入库

```python
import torch  # 导入PyTorch主库
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
import torch.nn.functional as F  # 导入常用函数（如激活函数）
import numpy as np  # 导入Numpy用于数据处理
from torch.utils.data import DataLoader, TensorDataset  # 导入数据加载和数据集工具
import matplotlib.pyplot as plt  # 导入matplotlib用于可视化
import random  # 用于随机选取测试样本
```

## 2. 设备配置

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
- **作用**：自动检测是否有GPU可用，如果有则使用GPU，否则使用CPU
- **目的**：利用GPU加速神经网络计算，大幅提升训练速度

## 3. 超参数设置

```python
batch_size = 64  # 每批次训练样本数
epochs = 3       # 训练轮数
learning_rate = 0.01  # 学习率
```

## 4. 数据加载和预处理

### 4.1 加载MNIST数据集
```python
data = np.load('./data/mnist.npz')
x_train = data['x_train']  # 训练图片，[60000, 28, 28]
y_train = data['y_train']  # 训练标签，[60000]
x_test = data['x_test']    # 测试图片，[10000, 28, 28]
y_test = data['y_test']    # 测试标签，[10000]
```

### 4.2 数据预处理
```python
# 归一化并转为Tensor，增加通道维度
x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1) / 255.0  # [60000, 1, 28, 28]
x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1) / 255.0    # [10000, 1, 28, 28]
y_train = torch.tensor(y_train, dtype=torch.long)  # [60000]
y_test = torch.tensor(y_test, dtype=torch.long)    # [10000]
```

**处理步骤说明：**
1. **转为Tensor**：把Numpy数组转为PyTorch的Tensor
2. **数据类型**：图片用float32，标签用long
3. **增加通道维度**：`.unsqueeze(1)` 把形状从 `(N, 28, 28)` 变成 `(N, 1, 28, 28)`
4. **归一化**：`/ 255.0` 把像素值从0~255缩放到0~1

### 4.3 构建数据加载器
```python
train_dataset = TensorDataset(x_train, y_train)  # 训练集
test_dataset = TensorDataset(x_test, y_test)     # 测试集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 训练数据加载器
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)   # 测试数据加载器
```

## 5. 定义卷积神经网络模型

### 5.1 网络结构
```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # 第一层卷积
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 第二层卷积
        self.pool = nn.MaxPool2d(2, 2)  # 2x2最大池化
        self.fc = nn.Linear(32 * 7 * 7, 10)  # 全连接层

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 第一层卷积+ReLU+池化
        x = self.pool(F.relu(self.conv2(x)))  # 第二层卷积+ReLU+池化
        x = x.view(-1, 32 * 7 * 7)            # 展平
        x = self.fc(x)                        # 全连接层输出
        return x
```

### 5.2 网络结构说明
- **输入**：`[batch_size, 1, 28, 28]` - 灰度图片
- **第一层卷积**：`Conv2d(1, 16, 3, padding=1)` - 输出16个特征图
- **第一次池化**：`MaxPool2d(2, 2)` - 尺寸减半，变成14×14
- **第二层卷积**：`Conv2d(16, 32, 3, padding=1)` - 输出32个特征图
- **第二次池化**：`MaxPool2d(2, 2)` - 尺寸再减半，变成7×7
- **展平**：`32 × 7 × 7 = 1568` 个特征
- **全连接层**：`Linear(1568, 10)` - 输出10个类别的分数

### 5.3 创建模型实例
```python
model = SimpleCNN().to(device)
```

## 6. 定义损失函数和优化器

```python
criterion = nn.CrossEntropyLoss()  # 交叉熵损失，适用于多分类
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # 随机梯度下降优化器
```

## 7. 训练函数

```python
def train():
    model.train()  # 设置为训练模式
    loss_list = []  # 记录每个batch的损失
    for epoch in range(epochs):  # 遍历每一轮
        total_loss = 0  # 累计损失
        for batch_idx, (data, target) in enumerate(train_loader):  # 遍历每个batch
            data, target = data.to(device), target.to(device)  # 数据转到设备
            optimizer.zero_grad()  # 梯度清零
            output = model(data)   # 前向传播
            loss = criterion(output, target)  # 计算损失
            loss.backward()       # 反向传播
            optimizer.step()      # 更新参数
            total_loss += loss.item()  # 累加损失
            loss_list.append(loss.item())  # 记录损失
            if (batch_idx + 1) % 100 == 0:  # 每100步打印一次损失
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        print(f'Epoch [{epoch+1}] 平均损失: {total_loss/len(train_loader):.4f}')
    return loss_list
```

### 7.1 训练流程详解
1. **`model.train()`**：设置为训练模式，启用dropout等
2. **遍历epoch**：每个epoch遍历整个训练集
3. **遍历batch**：每个batch包含64张图片
4. **数据转移**：把数据移到GPU/CPU
5. **梯度清零**：防止梯度累积
6. **前向传播**：计算模型输出
7. **计算损失**：比较预测和真实标签
8. **反向传播**：计算梯度
9. **参数更新**：用梯度更新模型参数

## 8. 测试函数

```python
def test():
    model.eval()  # 设置为评估模式
    correct = 0  # 正确预测数
    total = 0    # 总样本数
    with torch.no_grad():  # 测试时不计算梯度，节省内存
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)  # 前向传播
            _, predicted = torch.max(outputs.data, 1)  # 取概率最大的类别
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print(f'测试集准确率: {100 * correct / total:.2f}%')
```

### 8.1 测试流程详解
1. **`model.eval()`**：设置为评估模式，关闭dropout等
2. **`torch.no_grad()`**：不计算梯度，节省内存
3. **前向传播**：只做推理，不做训练
4. **统计准确率**：计算预测正确的样本比例

## 9. 可视化函数

### 9.1 训练损失曲线
```python
def plot_loss(loss_list):
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('训练损失曲线')
    plt.show()
```

### 9.2 预测结果可视化
```python
def visualize_predictions():
    model.eval()
    idxs = random.sample(range(len(x_test)), 10)  # 随机选10张
    images = x_test[idxs].to(device)
    labels = y_test[idxs]
    with torch.no_grad():
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    images = images.cpu().numpy()
    plt.figure(figsize=(12, 2))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(images[i][0], cmap='gray')
        plt.title(f'T:{labels[i].item()}\nP:{preds[i].item()}')
        plt.axis('off')
    plt.suptitle('测试集样本及预测结果')
    plt.show()
```

## 10. 主程序执行

```python
if __name__ == '__main__':
    loss_list = train()  # 训练并获取损失
    test()   # 测试模型
    plot_loss(loss_list)  # 可视化训练损失曲线
    visualize_predictions()  # 可视化部分测试集预测结果
```

## 11. 完整流程总结

1. **数据准备**：加载MNIST数据，预处理，构建DataLoader
2. **模型定义**：创建SimpleCNN网络结构
3. **训练配置**：设置损失函数和优化器
4. **模型训练**：多轮训练，逐步优化参数
5. **模型测试**：在测试集上评估性能
6. **结果可视化**：展示训练过程和预测结果

## 12. 关键概念解释

- **Batch**：小批量数据，用于一次前向和反向传播
- **Epoch**：完整遍历一次训练集
- **前向传播**：数据从输入到输出的计算过程
- **反向传播**：根据损失计算梯度，用于参数更新
- **梯度下降**：用梯度更新参数，使损失变小
- **交叉熵损失**：衡量预测概率与真实标签的差距

## 13. 预期结果

- **训练损失**：随着训练进行，损失应该逐渐下降
- **测试准确率**：通常在90%以上
- **可视化结果**：训练损失曲线和预测结果图片 