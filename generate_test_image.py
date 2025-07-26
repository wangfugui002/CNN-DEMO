import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

if __name__ == '__main__':
    # 加载MNIST数据
    data = np.load('./data/mnist.npz')
    x_train = data['x_train']  # [60000, 28, 28]
    y_train = data['y_train']  # [60000]

    # 找到第一个标签为6的图片
    idx = np.where(y_train == 6)[0][0]
    img_array = x_train[idx] / 255.0  # 归一化到0~1

    # 保存为npy
    np.save('test_image.npy', img_array)

    # 保存为PNG
    img_display = Image.fromarray((img_array * 255).astype(np.uint8))
    img_display.save('test_image.png')

    # 可视化
    plt.figure(figsize=(4, 4))
    plt.imshow(img_array, cmap='gray')
    plt.title('MNIST原始数字6样本')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    print('已从MNIST数据集中提取数字6的图片，保存为 test_image.npy 和 test_image.png') 