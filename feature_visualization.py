import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

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

# 可视化特征提取过程
def visualize_feature_extraction(model, sample_image):
    model.eval()
    with torch.no_grad():
        # 获取第一层卷积的输出
        conv_output = F.relu(model.conv1(sample_image.unsqueeze(0)))
        conv_output = conv_output.squeeze(0).cpu().numpy()
        
        # 获取最终分类结果
        final_output = model(sample_image.unsqueeze(0))
        probabilities = F.softmax(final_output, dim=1).squeeze().cpu().numpy()
        predicted_class = torch.argmax(final_output).item()
    
    # 创建可视化
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 原始图片
    plt.subplot(3, 6, 1)
    plt.imshow(sample_image.squeeze().numpy(), cmap='gray')
    plt.title(f'原始图片\n(标签: {y_train[0].item()})')
    plt.axis('off')
    
    # 2. 显示前5个特征图
    for i in range(5):
        plt.subplot(3, 6, i + 2)
        plt.imshow(conv_output[i], cmap='viridis')
        plt.title(f'特征图 {i+1}')
        plt.axis('off')
    
    # 3. 显示分类结果
    plt.subplot(3, 6, 7)
    classes = list(range(10))
    bars = plt.bar(classes, probabilities)
    plt.xlabel('数字类别')
    plt.ylabel('概率')
    plt.title(f'分类结果\n(预测: {predicted_class})')
    
    # 高亮预测的类别
    bars[predicted_class].set_color('red')
    
    # 4. 显示所有特征图的统计信息
    plt.subplot(3, 6, 8)
    feature_means = np.mean(conv_output, axis=(1, 2))
    plt.bar(range(16), feature_means)
    plt.xlabel('卷积核编号')
    plt.ylabel('平均激活值')
    plt.title('各卷积核的平均激活')
    
    # 5. 显示特征图的热力图
    plt.subplot(3, 6, 9)
    all_features_combined = np.mean(conv_output, axis=0)
    plt.imshow(all_features_combined, cmap='viridis')
    plt.title('所有特征图的平均')
    plt.axis('off')
    
    # 6. 显示最大激活的特征图
    plt.subplot(3, 6, 10)
    max_activation_idx = np.argmax(feature_means)
    plt.imshow(conv_output[max_activation_idx], cmap='viridis')
    plt.title(f'最大激活特征图\n(卷积核 {max_activation_idx+1})')
    plt.axis('off')
    
    # 7. 显示最小激活的特征图
    plt.subplot(3, 6, 11)
    min_activation_idx = np.argmin(feature_means)
    plt.imshow(conv_output[min_activation_idx], cmap='viridis')
    plt.title(f'最小激活特征图\n(卷积核 {min_activation_idx+1})')
    plt.axis('off')
    
    # 8. 显示特征图的方差
    plt.subplot(3, 6, 12)
    feature_vars = np.var(conv_output, axis=(1, 2))
    plt.bar(range(16), feature_vars)
    plt.xlabel('卷积核编号')
    plt.ylabel('方差')
    plt.title('各卷积核的方差')
    
    plt.tight_layout()
    plt.show()
    
    # 打印详细信息
    print(f"原始标签: {y_train[0].item()}")
    print(f"预测标签: {predicted_class}")
    print(f"预测概率: {probabilities[predicted_class]:.4f}")
    print(f"最大激活的卷积核: {max_activation_idx+1}")
    print(f"最小激活的卷积核: {min_activation_idx+1}")

# 主程序
if __name__ == '__main__':
    # 创建并训练模型
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # 快速训练
    train_dataset = torch.utils.data.TensorDataset(x_train[:500], y_train[:500])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    for epoch in range(3):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # 可视化特征提取过程
    sample_image = x_train[0]  # 取第一张图片
    visualize_feature_extraction(model, sample_image)
    
    print("特征提取可视化完成！") 