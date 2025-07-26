import torch
import matplotlib.pyplot as plt
import numpy as np

# 定义与训练时相同的CNN模型结构
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    # 加载模型
    model = SimpleCNN()
    model.load_state_dict(torch.load('trained_model.pth', map_location='cpu'))
    model.eval()

    # 获取全连接层权重
    fc_weights = model.fc.weight.data.cpu().numpy()  # 形状: [10, 1568]
    fc_bias = model.fc.bias.data.cpu().numpy()       # 形状: [10]

    print(f"全连接层权重矩阵形状: {fc_weights.shape}")
    print(f"全连接层偏置形状: {fc_bias.shape}")

    # 可视化：每一类的权重向量reshape成32x7x7
    for i in range(10):
        weight_vec = fc_weights[i]  # 取出第i类的权重
        weight_maps = weight_vec.reshape(32, 7, 7)  # 还原为32个特征图
        # 画出前8个特征图
        fig, axes = plt.subplots(2, 4, figsize=(10, 5))
        fig.suptitle(f'数字类别 {i} 的全连接层权重（前8个特征图）')
        for j in range(8):
            ax = axes[j // 4, j % 4]
            im = ax.imshow(weight_maps[j], cmap='bwr', vmin=-np.max(np.abs(weight_maps)), vmax=np.max(np.abs(weight_maps)))
            ax.set_title(f'特征图 {j+1}')
            ax.axis('off')
        plt.colorbar(im, ax=axes, orientation='vertical', fraction=0.02)
        plt.tight_layout()
        plt.show()

    # 可视化：所有类别的权重热力图
    plt.figure(figsize=(12, 6))
    plt.imshow(fc_weights, aspect='auto', cmap='bwr', vmin=-np.max(np.abs(fc_weights)), vmax=np.max(np.abs(fc_weights)))
    plt.colorbar()
    plt.xlabel('展平特征编号 (32*7*7)')
    plt.ylabel('数字类别 (0-9)')
    plt.title('全连接层权重热力图（每行对应一个类别）')
    plt.tight_layout()
    plt.show() 