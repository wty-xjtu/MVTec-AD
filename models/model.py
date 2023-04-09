import torch.nn as nn
import torch


# 编码器
class Encoder(nn.Module):
    def __init__(self, lat_dim):
        super(Encoder, self).__init__()
        self.lat_dim = lat_dim

    def forward(self, x):
        pass



# 解码器
class Decoder(nn.Module):
    def __init__(self, lat_dim):
        super(Decoder, self).__init__()
        self.lat_dim = lat_dim

    def forward(self, x):
        pass




# VAE模型代码
class VAE(nn.Module):
    def __init__(self, lat_dim):
        super(VAE, self).__init__()
        self.lat_dim = lat_dim
        self.encoder = Encoder(lat_dim)
        self.decoder = Decoder(lat_dim)


    def forward(self, x):
        pass