import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 定义与训练时相同的CNN模型结构
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

def load_model(model_path='trained_model.pth'):
    """加载训练好的模型"""
    model = SimpleCNN()
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"模型已从 {model_path} 加载")
    except FileNotFoundError:
        print(f"未找到模型文件 {model_path}，将使用随机初始化的模型")
        print("注意：随机模型预测结果不准确，仅用于演示")
    return model

def preprocess_image(image_path):
    """预处理图片，使其符合模型输入要求"""
    if image_path.endswith('.npy'):
        # 加载numpy文件
        img_array = np.load(image_path)
    else:
        # 加载图片文件
        from PIL import Image
        img = Image.open(image_path).convert('L')  # 转换为灰度图
        img = img.resize((28, 28))  # 调整尺寸
        img_array = np.array(img) / 255.0  # 归一化
    
    # 确保形状正确
    if len(img_array.shape) == 2:
        img_array = img_array.reshape(1, 1, 28, 28)  # 添加batch和channel维度
    elif len(img_array.shape) == 3:
        img_array = img_array.reshape(1, 1, 28, 28)
    
    # 转换为tensor
    img_tensor = torch.tensor(img_array, dtype=torch.float32)
    
    return img_tensor

def predict_digit(model, image_tensor):
    """使用模型预测数字"""
    model.eval()
    with torch.no_grad():
        # 前向传播
        output = model(image_tensor)
        
        # 计算概率
        probabilities = F.softmax(output, dim=1)
        
        # 获取预测结果
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        return predicted_class, confidence, probabilities[0].numpy()

def visualize_prediction(image_tensor, predicted_class, confidence, probabilities):
    """可视化预测结果"""
    # 获取原始图片
    img_array = image_tensor.squeeze().numpy()
    
    # 创建可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 显示图片
    axes[0].imshow(img_array, cmap='gray')
    axes[0].set_title(f'测试图片\n预测: {predicted_class} (置信度: {confidence:.3f})')
    axes[0].axis('off')
    
    # 显示预测概率
    classes = list(range(10))
    bars = axes[1].bar(classes, probabilities)
    axes[1].set_xlabel('数字类别')
    axes[1].set_ylabel('概率')
    axes[1].set_title('预测概率分布')
    axes[1].set_ylim(0, 1)
    
    # 高亮预测的类别
    bars[predicted_class].set_color('red')
    
    # 添加概率值标签
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    print("=== CNN模型测试程序 ===")
    
    # 1. 加载模型
    print("\n1. 加载模型...")
    model = load_model()
    
    # 2. 加载测试图片
    print("\n2. 加载测试图片...")
    try:
        image_tensor = preprocess_image('test_image.npy')
        print("成功加载 test_image.npy")
    except FileNotFoundError:
        print("未找到 test_image.npy，请先运行 generate_test_image.py 生成测试图片")
        return
    
    # 3. 进行预测
    print("\n3. 进行预测...")
    predicted_class, confidence, probabilities = predict_digit(model, image_tensor)
    
    # 4. 显示结果
    print(f"\n预测结果:")
    print(f"预测数字: {predicted_class}")
    print(f"置信度: {confidence:.3f}")
    print(f"所有类别的概率:")
    for i, prob in enumerate(probabilities):
        print(f"  数字 {i}: {prob:.3f}")
    
    # 5. 可视化结果
    print("\n4. 显示可视化结果...")
    visualize_prediction(image_tensor, predicted_class, confidence, probabilities)
    
    print("\n测试完成！")

if __name__ == '__main__':
    main() 