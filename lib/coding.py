# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.module import  get_fgsims, get_fgmask,get_mask

EPS = 1e-8 # epsilon 
MASK = -1 # padding value

# Visual Hard Assignment Coding


class VHACoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, imgs, caps, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        sims = get_fgsims(imgs, caps)[:,:,:max_r,:max_w]
        mask = get_fgmask(img_lens,cap_lens)
        sims = sims.masked_fill(mask == 0, MASK)
        sims = sims.max(dim=-2)[0]
        return sims

class VSACoding(nn.Module):
    def __init__(self,temperature = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, imgs, caps, img_lens, cap_lens, return_attn=False):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        if len(imgs.shape) == 2:
            imgs = imgs.unsqueeze(1)  # 添加一个维度，变成3维
        if len(caps.shape) == 2:
            caps = caps.unsqueeze(1)  # 添加一个维度，变成3维
        imgs = imgs[:,:max_r,:]
        caps = caps[:,:max_w,:]
        sims = get_fgsims(imgs, caps)[:,:,:max_r,:max_w]
        mask = get_fgmask(img_lens,cap_lens)

        # calculate attention
        sims = sims / self.temperature

        sims = torch.softmax(sims.masked_fill(mask==0, -torch.inf),dim=-1) # Bi x Bt x K x L
        sims = sims.masked_fill(mask == 0, 0)
        sims = torch.matmul(sims,caps) # Bi x Bt x K x D
        sims = torch.mul(sims.permute(1,0,2,3),imgs).permute(1,0,2,3).sum(dim=-1) \
                    /(torch.norm(sims,p=2,dim=-1,keepdim=False)+EPS) # Bi x Bt x K

        mask = get_mask(img_lens).permute(0,2,1).repeat(1,cap_lens.size(0),1)
        sims = sims.masked_fill(mask==0, -1)
        return sims

# Texual Hard Assignment Coding
class THACoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, imgs, caps, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        sims = get_fgsims(imgs, caps)[:,:,:max_r,:max_w]
        mask = get_fgmask(img_lens,cap_lens)
        sims = sims.masked_fill(mask == 0, MASK)
        sims = sims.max(dim=-1)[0]
        return sims


# log-sum-exp pooling
class LSEPooling(nn.Module):
    def __init__(self,temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, sims):
        assert len(sims.shape)==3
        sims[sims==MASK] = -torch.inf
        sims = torch.logsumexp(sims/self.temperature,dim=-1)
        return sims


# mean pooling
class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims):
        assert len(sims.shape)==3
        lens = (sims!=MASK).sum(dim=-1)
        sims[sims==MASK] = 0
        sims = sims.sum(dim=-1)/lens
        return sims
    

# softmax pooling
class SoftmaxPooling(nn.Module):
    def __init__(self,temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, sims):
        # assert len(sims.shape)==3
        sims[sims==MASK] = -torch.inf
        weight = torch.softmax(sims/self.temperature,dim=-1)
        sims = (weight*sims).sum(dim=-1)
        return sims

def get_coding():
    temperature = 0.1
    return VHACoding()
    

def get_pooling():
    temperature=0.060000000000000005
    belta = 0.1
    return LSEPooling(temperature)



