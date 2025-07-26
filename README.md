# CNN-demo

基于PyTorch的卷积神经网络（CNN）手写数字识别项目

## 项目简介
本项目实现了一个用于MNIST手写数字识别的卷积神经网络（CNN），包含数据加载、模型训练、测试、可视化、单张图片预测、权重可视化等完整流程。适合深度学习初学者学习CNN原理与PyTorch实践。

GitHub仓库地址：[git@github.com:wangfugui002/CNN-DEMO.git](git@github.com:wangfugui002/CNN-DEMO.git)

## 环境依赖
- Python 3.7+
- torch
- torchvision
- numpy
- matplotlib
- pillow
- scipy

安装依赖：
```bash
pip install torch torchvision numpy matplotlib pillow scipy
```

## 主要文件说明
- `cnn_demo.py`：主程序，包含模型定义、训练、测试、可视化等
- `generate_test_image.py`：生成或提取测试图片（如数字6），保存为`test_image.png`和`test_image.npy`
- `test_model.py`：加载训练好的模型，对单张图片进行预测并可视化结果
- `visualize_kernels.py`：可视化卷积核及其对图片的响应
- `feature_visualization.py`：可视化特征提取与分类全过程
- `visualize_fc_weights.py`：可视化全连接层权重矩阵
- `卷积核与全连接层关系说明.md`：卷积核与全连接层关系的详细说明
- `CNN_demo_流程文档.md`：项目完整流程与原理说明
- `data/mnist.npz`：MNIST数据集（需自动下载或手动准备）
- `trained_model.pth`：训练好的模型权重
- `test_image.png`/`test_image.npy`：测试图片

## 运行步骤

1. **准备数据**
   - 确保`data/mnist.npz`存在（可自动下载或手动下载）

2. **训练模型**
   ```bash
   python3 cnn_demo.py
   ```
   - 训练完成后会自动保存模型为`trained_model.pth`

3. **生成/提取测试图片**
   ```bash
   python3 generate_test_image.py
   ```
   - 默认会从MNIST中提取一张数字6的图片，保存为`test_image.png`和`test_image.npy`

4. **模型预测测试图片**
   ```bash
   python3 test_model.py
   ```
   - 会输出预测结果和概率分布，并可视化

5. **可视化卷积核、特征、全连接层权重**
   ```bash
   python3 visualize_kernels.py
   python3 feature_visualization.py
   python3 visualize_fc_weights.py
   ```

## 可视化说明
- 训练损失曲线、测试集预测结果、卷积核权重、特征图、全连接层权重等均有可视化脚本
- 运行相关脚本后会弹出matplotlib窗口展示结果

## 常见问题
- 字体警告：matplotlib默认字体不支持中文，不影响功能
- 数据集下载慢：可手动下载mnist.npz放到`data/`目录
- 预测图片需为28x28灰度图，或用`generate_test_image.py`自动生成

## Git操作上传项目
1. 初始化git仓库（如未初始化）
   ```bash
   git init
   git remote add origin git@github.com:wangfugui002/CNN-DEMO.git
   ```
2. 添加并提交所有文件
   ```bash
   git add .
   git commit -m "init: upload CNN-demo project"
   ```
3. 推送到GitHub
   ```bash
   git branch -M main
   git push -u origin main
   ```

---

如有问题欢迎在GitHub仓库提issue或联系作者。 