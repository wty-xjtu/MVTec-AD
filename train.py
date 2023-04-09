import os.path
import torch
import tqdm
from my_datasets import MVTecADDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from models.model import VAE
import torch.nn as nn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

"""
迁移学习：就是模型在A数据集上训练过  得到权重    再在B数据集上进行训练/测试

参数微调（Fine-Tuning）就是一种最简单的迁移学习的方法

ImageNet数据集（几十万张图片），VGG在上面训练得到pretrained-model
自己的数据集，使用pretrained-model接着训练，很小的学习率微调模型的参数，这就是迁移学习。
可以快速收敛，且不容易出现过拟合现象
"""


# 定义设置参数
def get_args():
    parser = argparse.ArgumentParser()
    # 训练多少次
    parser.add_argument('--epochs', metavar='E', type=int, default=100)
    # 一次处理多少mfcc特征   4 - 8  20*150
    parser.add_argument('--batch-size', metavar='B', type=int, nargs='?', default=20)
    # 5e-4 1e-3   1e-5
    parser.add_argument('--lr', metavar='LR', type=float, nargs='?', default=1e-4)
    # 线程数量  针对Linux系统做的  Linux：8 6 16
    parser.add_argument('--num-works', metavar='latent', type=int, nargs='?', default=0)
    # 数据集的根目录
    parser.add_argument('--root-path', metavar='latent', type=str, default=r"F:\xianyu_datasets\LA_dataset\mfccs")
    # 保存模型的目录
    parser.add_argument('--model-save-path', metavar='latent', type=str, default=r"saved")
    # 继续训练  重新开始 None
    parser.add_argument('--pretrained', metavar='latent', type=int, default=None)
    return parser.parse_args()


# 创建目录
def make_dir(path):
    if not os.path.exists(path):
        print("创建目录：", path)
        os.makedirs(path)


# 训练类
class Model:
    def __init__(self):
        # 交叉熵损失函数
        self.criterion = None
        # 数据迭代器
        self.train_loader = None
        # 优化器
        self.optimizer = None
        # 模型
        self.model = None
        # 获取设备 cpu or gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"使用设备为：{self.device}")
        # 获取参数
        self.args = get_args()
        # 初始化训练数据
        self.init_data()
        # 初始化模型
        self.init_model()
        # 创建目录
        make_dir(self.args.model_save_path)

        self.best_acc = 0.0

    def init_data(self):
        # 数据处理的方法   [0, 1] -> [-1, 1]   （x - 0.5) / 0.5
        # 数据归一化目的    可以让模型快速收敛
        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 生成数据集 迭代集合   验证机防止过拟合
        train_dataset = MVTecADDataset(os.path.join(self.args.root_path, "train.pckl"), transformer)

        # 生成迭代器  把mfcc进行分组打包  drop_last 不足以满足分组的全部丢掉不要  shuffle打乱数据
        # 训练不打乱顺序，有可能让模型训练效果不好
        self.train_loader = DataLoader(train_dataset, self.args.batch_size, shuffle=True,
                                       num_workers=self.args.num_works, drop_last=True)

    def init_model(self):
        # 定义模型 加载了预训练的参数
        self.model = VAE(128).to(self.device)
        # 加载保存的参数
        if self.args.pretrained is not None:
            print("加载预训练权重......")
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.model_save_path, f"best_model.pth"),
                           map_location=self.device))

        # 定义优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        # 定义损失函数  1 ： 9  权重限制
        weights = torch.tensor([0.1, 0.9])
        # self.criterion =


    def train(self):
        # 循环训练
        for epoch in range(self.args.epochs):
            # 开启训练模式
            self.model.train()
            # 生成进度条
            pbar = tqdm.tqdm(self.train_loader)
            # 迭代所有数据
            for imgs in pbar:
                # 按照设定的batchsize  取出batchsize个数据
                # 拷贝到device
                imgs = imgs.to(self.device)
                # 前向传播
                outputs = self.model(imgs)
                # 清空梯度
                self.optimizer.zero_grad()
                # 计算损失
                loss = self.criterion(outputs, imgs)
                # 反向传播  链式求导
                loss.backward()
                # 调整参数  根据公式
                self.optimizer.step()

                # 更新进度条
                pbar.set_description(f"Epoch : {epoch + 1}, loss : {loss.item():.3f}")

            # 验证效果
            if (epoch + 1) % self.args.dev_step == 0:
                print("\nDeving......")
                self.eval()


if __name__ == '__main__':
    model = Model()
    model.train()
