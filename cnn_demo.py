import torch  # 导入PyTorch主库
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
import torch.nn.functional as F  # 导入常用函数（如激活函数）
import numpy as np  # 导入Numpy用于数据处理
from torch.utils.data import DataLoader, TensorDataset  # 导入数据加载和数据集工具
import matplotlib.pyplot as plt  # 导入matplotlib用于可视化
import random  # 用于随机选取测试样本

# 设置训练设备，如果有GPU则用GPU，否则用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择设备

# 定义超参数
batch_size = 64  # 每批次训练样本数
epochs = 3       # 训练轮数
learning_rate = 0.01  # 学习率

# 加载本地MNIST数据（假设已准备好.npz文件，包含x_train, y_train, x_test, y_test）
data = np.load('./data/mnist.npz')  # 读取npz文件
x_train = data['x_train']  # 训练图片，[60000, 28, 28]
y_train = data['y_train']  # 训练标签，[60000]
x_test = data['x_test']    # 测试图片，[10000, 28, 28]
y_test = data['y_test']    # 测试标签，[10000]

# 归一化并转为Tensor，增加通道维度
x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1) / 255.0  # [60000, 1, 28, 28]
x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1) / 255.0    # [10000, 1, 28, 28]
y_train = torch.tensor(y_train, dtype=torch.long)  # [60000]
y_test = torch.tensor(y_test, dtype=torch.long)    # [10000]

# 构建DataLoader，便于批量训练和测试
train_dataset = TensorDataset(x_train, y_train)  # 训练集
test_dataset = TensorDataset(x_test, y_test)     # 测试集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 训练数据加载器
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)   # 测试数据加载器

# 定义卷积神经网络结构
class SimpleCNN(nn.Module):  # 继承nn.Module
    def __init__(self):
        super(SimpleCNN, self).__init__()  # 继承父类初始化
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)   # 第一层卷积，输入1通道，输出16通道
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 第二层卷积，输入16通道，输出32通道
        self.pool = nn.MaxPool2d(2, 2)  # 2x2最大池化
        self.fc = nn.Linear(32 * 7 * 7, 10)  # 全连接层，输出10类

    def forward(self, x):  # 前向传播
        x = self.pool(F.relu(self.conv1(x)))  # 第一层卷积+ReLU+池化
        x = self.pool(F.relu(self.conv2(x)))  # 第二层卷积+ReLU+池化
        x = x.view(-1, 32 * 7 * 7)            # 展平成一维向量
        x = self.fc(x)                        # 全连接层输出
        return x

# 实例化模型并移动到设备上
model = SimpleCNN().to(device)  # 创建模型并放到设备上

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失，适用于多分类
optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # 随机梯度下降优化器

# 训练模型，返回每个batch的损失用于可视化
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
        print(f'Epoch [{epoch+1}] 平均损失: {total_loss/len(train_loader):.4f}')  # 每轮结束打印平均损失
    return loss_list  # 返回损失列表

# 测试模型
def test():
    model.eval()  # 设置为评估模式
    correct = 0  # 正确预测数
    total = 0    # 总样本数
    with torch.no_grad():  # 测试时不计算梯度，节省内存
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # 数据转到设备
            outputs = model(data)  # 前向传播
            _, predicted = torch.max(outputs.data, 1)  # 取概率最大的类别
            total += target.size(0)  # 累加总数
            correct += (predicted == target).sum().item()  # 累加正确数
    print(f'测试集准确率: {100 * correct / total:.2f}%')  # 打印准确率

# 可视化训练损失曲线
def plot_loss(loss_list):
    plt.figure()
    plt.plot(loss_list)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('训练损失曲线')
    plt.show()

# 可视化部分测试集预测结果
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

if __name__ == '__main__':  # 主程序入口
    loss_list = train()  # 训练并获取损失
    test()   # 测试模型
    
    # 保存训练好的模型
    torch.save(model.state_dict(), 'trained_model.pth')
    print("模型已保存为 trained_model.pth")
    
    plot_loss(loss_list)  # 可视化训练损失曲线
    visualize_predictions()  # 可视化部分测试集预测结果 