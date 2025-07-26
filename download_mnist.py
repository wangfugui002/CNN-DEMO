import os
import urllib.request

# 多个MNIST数据集源，按优先级排列
SOURCES = [
    'http://yann.lecun.com/exdb/mnist/',  # 官方源
    'https://ossci-datasets.s3.amazonaws.com/mnist/',  # torchvision备用源
    'https://mirrors.tuna.tsinghua.edu.cn/git/MNIST-data/',  # 清华镜像
    'https://ossci-datasets.oss-cn-zhangjiakou.aliyuncs.com/mnist/',  # 阿里云镜像
]
FILES = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz'
]

data_dir = './data/MNIST/raw/'
os.makedirs(data_dir, exist_ok=True)

def download_from_sources(filename):
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        print(f'{filename} 已存在，跳过下载。')
        return True
    for base_url in SOURCES:
        url = base_url + filename
        try:
            print(f'尝试下载 {filename} 来自: {url}')
            urllib.request.urlretrieve(url, filepath)
            print(f'{filename} 下载成功！')
            return True
        except Exception as e:
            print(f'下载失败: {e}')
    print(f'所有镜像均下载失败: {filename}')
    return False

if __name__ == '__main__':
    all_ok = True
    for file in FILES:
        ok = download_from_sources(file)
        all_ok = all_ok and ok
    if all_ok:
        print('所有文件下载完成！')
    else:
        print('部分文件下载失败，请检查网络或手动下载。') 