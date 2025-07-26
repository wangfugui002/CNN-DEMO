import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# 加载MNIST数据
data = np.load('./data/mnist.npz')
x_train = torch.tensor(data['x_train'], dtype=torch.float32).unsqueeze(1) / 255.0
y_train = torch.tensor(data['y_train'], dtype=torch.long)

# 创建简化的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 14 * 14, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = self.fc(x)
        return x

# 可视化卷积核的函数
def visualize_kernels(model, title="卷积核可视化"):
    conv1_kernels = model.conv1.weight.data.cpu().numpy()
    
    # 创建4x4的子图布局
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle(title, fontsize=16)
    
    for i in range(16):
        row = i // 4
        col = i % 4
        
        # 获取第i个卷积核
        kernel = conv1_kernels[i, 0]  # 取第一个通道，因为输入是灰度图
        
        # 显示卷积核
        im = axes[row, col].imshow(kernel, cmap='viridis', interpolation='nearest')
        axes[row, col].set_title(f'Kernel {i+1}')
        axes[row, col].axis('off')
        
        # 添加颜色条
        plt.colorbar(im, ax=axes[row, col], shrink=0.8)
    
    plt.tight_layout()
    plt.show()

# 可视化卷积核响应的函数
def visualize_kernel_responses(model, sample_image, title="卷积核响应"):
    model.eval()
    with torch.no_grad():
        # 获取第一层卷积的输出
        conv_output = F.relu(model.conv1(sample_image.unsqueeze(0)))
        conv_output = conv_output.squeeze(0).cpu().numpy()
    
    # 创建4x4的子图布局
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(title, fontsize=16)
    
    for i in range(16):
        row = i // 4
        col = i % 4
        
        # 显示第i个卷积核的响应
        response = conv_output[i]
        im = axes[row, col].imshow(response, cmap='viridis')
        axes[row, col].set_title(f'Response {i+1}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# 主程序
if __name__ == '__main__':
    # 创建模型
    model = SimpleCNN()
    
    # 1. 显示训练前的卷积核（随机初始化）
    print("显示训练前的卷积核（随机初始化）...")
    visualize_kernels(model, "训练前的卷积核（随机初始化）")
    
    # 2. 训练模型
    print("开始训练模型...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # 创建数据加载器
    train_dataset = TensorDataset(x_train[:1000], y_train[:1000])  # 只用1000个样本快速训练
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 训练几个epoch
    for epoch in range(5):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    # 3. 显示训练后的卷积核
    print("显示训练后的卷积核...")
    visualize_kernels(model, "训练后的卷积核（学习到的特征）")
    
    # 4. 显示卷积核对样本图片的响应
    print("显示卷积核对样本图片的响应...")
    sample_image = x_train[0]  # 取第一张图片作为样本
    visualize_kernel_responses(model, sample_image, "卷积核对样本图片的响应")
    
    # 5. 显示原始样本图片
    plt.figure(figsize=(6, 6))
    plt.imshow(sample_image.squeeze().numpy(), cmap='gray')
    plt.title(f'样本图片 (标签: {y_train[0].item()})')
    plt.axis('off')
    plt.show()
    
    print("可视化完成！") 