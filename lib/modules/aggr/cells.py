import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def activateFunc(x):
    x = torch.tanh(x)
    return F.relu(x)

class Router(nn.Module):
    def __init__(self, num_out_path, embed_size, hid):
        super(Router, self).__init__()
        self.num_out_path = num_out_path
        self.mlp = nn.Sequential(nn.Linear(embed_size, hid), 
                                    nn.ReLU(True), 
                                    nn.Linear(hid, num_out_path))
        self.init_weights()

    def init_weights(self):
        self.mlp[2].bias.data.fill_(1.5)

    def forward(self, x):
        # print("Shape of x:", x.size())
        # x = x.mean(-2)
        # print("Shape of x after mean:", x.size())
        x = self.mlp(x)
        soft_g = activateFunc(x) 
        # print("Shape of soft_g:", soft_g.size())
        return soft_g

class MaxPoolingCell(nn.Module):
    def __init__(self, embed_size, hid_router, num_out_path):
        super(MaxPoolingCell, self).__init__()
        self.keep_mapping = nn.ReLU()
        self.embed_size = embed_size
        self.hid_router = hid_router
        self.router = Router(num_out_path, embed_size, hid_router)
        self.MaxPool = nn.AdaptiveMaxPool1d(1024)
        # self.MaxPool = torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

    def forward(self, x):
        path_prob = self.router(x)
        # emb = x[:, :1, :]
        # emb, _ = torch.max(x, dim=1, keepdim=True)
        emb = self.MaxPool(x)


        return emb, path_prob
    
class AveragePoolingCell(nn.Module):
    def __init__(self, embed_size, hid_router, num_out_path):
        super(AveragePoolingCell, self).__init__()
        self.keep_mapping = nn.ReLU()
        self.embed_size = embed_size
        self.hid_router = hid_router
        self.router = Router(num_out_path, embed_size, hid_router)
        self.AvgPool = nn.AdaptiveAvgPool1d(1024)
        # self.AvgPool = torch.nn.AvgPool1d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
    def forward(self, x):
        path_prob = self.router(x)
        # emb = x.mean(dim=1, keepdim=True)
        emb = self.AvgPool(x)

        return emb, path_prob
    
class K_MaxPoolingCell(nn.Module):
    def __init__(self, embed_size, hid_router, num_out_path, k=3, output_size=(128, 1024)):
        super(K_MaxPoolingCell, self).__init__()
        self.keep_mapping = nn.ReLU()
        self.embed_size = embed_size
        self.hid_router = hid_router
        self.router = Router(num_out_path, embed_size, hid_router)
        self.k = k
        self.output_size = output_size
        self.fc = nn.Linear(k, output_size[1])

    def k_max_pooling(self, x, k):
        # 在最后一个维度上选择前 k 个最大值
        index = x.topk(k, dim=-1)[1].sort(dim=-1)[0]
        return x.gather(-1, index)

    def forward(self, x):
        path_prob = self.router(x)
        k_max_emb = self.k_max_pooling(x, self.k)
        emb = self.fc(k_max_emb)
        return emb, path_prob
        
    
class RectifiedIdentityCell(nn.Module):
    def __init__(self, embed_size, hid_router, num_out_path):
        super(RectifiedIdentityCell, self).__init__()
        self.keep_mapping = nn.ReLU()
        self.router = Router(num_out_path, embed_size, hid_router)

    def forward(self, x):
        path_prob = self.router(x)
        emb = x

        return emb, path_prob
    
class ZeroLikeCell(nn.Module):
    def __init__(self, embed_size, hid_router, num_out_path):
        super(ZeroLikeCell, self).__init__()
        self.keep_mapping = nn.ReLU()
        self.router = Router(num_out_path, embed_size, hid_router)

    def forward(self, x):
        path_prob = self.router(x)
        emb = torch.zeros_like(x[:, :1, :])

        return emb, path_prob
    


