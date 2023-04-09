from torch.utils.data import Dataset
import glob
import os
from PIL import Image
import torch

# 获取数据
class MVTecADDataset(Dataset):
    def __init__(self, data_path, suffix="png", transformer=None):
        super(MVTecADDataset, self).__init__()
        self.transformer = transformer
        self.names = []
        # 获取data_path下所有图片
        self.names = glob.glob(os.path.join(data_path, "train", "good", "rgb", f"*.{suffix}"))
        print("数据加载完毕!")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # 返回一张图片
        name = self.names[index]
        # 打开图片
        x = Image.open(name).convert("L")
        # 数据处理
        if self.transformer is not None:
            x = self.transformer(x)
            # 返回
        return x