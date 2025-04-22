import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import init
from lib.modules.aggr.dynamicPoolingInteraction import DynamicInteraction_Layer0, DynamicInteraction_Layer

class InteractionModule(nn.Module):
    def __init__(self, embed_size, hid_router, num_layer_routing=3, num_cells=4, path_hid=128):
        super(InteractionModule, self).__init__()
        self.embed_size = embed_size
        self.hid_router = hid_router
        self.num_cells = num_cells = 4
        self.dynamic_itr_l0 = DynamicInteraction_Layer0(embed_size, hid_router, num_cells, num_cells)
        self.dynamic_itr_l1 = DynamicInteraction_Layer(embed_size, hid_router, num_cells, num_cells)
        self.dynamic_itr_l2 = DynamicInteraction_Layer(embed_size, hid_router, num_cells, 1)
        total_paths = num_cells ** 2 * (num_layer_routing - 1) + num_cells
        self.path_mapping = nn.Linear(total_paths, path_hid)
        self.bn = nn.BatchNorm1d(embed_size)

    def forward(self, x,a):
        pairs_emb_lst, paths_l0 = self.dynamic_itr_l0(x) # batch_size,  1024
        # print("I===Shape of pairs_emb_lst l0:", pairs_emb_lst[0].size()) # torch.Size([32, 1024])       
        # print("I===Shape of paths_l0:", paths_l0.size()) # torch.Size([32, 4, 4])
        
        pairs_emb_lst, paths_l1 = self.dynamic_itr_l1(pairs_emb_lst)   # 这行pairs_emb_lst, x修改为pairs_emb_lst
        # print("II===Shape of pairs_emb_lst l1:", pairs_emb_lst[0].size()) # torch.Size([32, 1024])        
        # print("II===Shape of paths_l1:", paths_l1.size()) # torch.Size([32, 4, 4])                        
        
        pairs_emb_lst, paths_l2 = self.dynamic_itr_l2(pairs_emb_lst)   # 这行pairs_emb_lst, x修改为pairs_emb_lst
        # print("III===Shape of pairs_emb_lst l2:", pairs_emb_lst[0].size()) #     torch.Size([32, 1024])    
        # print("III===Shape of paths_l2:", paths_l2.size()) #  torch.Size([32, 1, 4])


        # print("III============================Length of pairs_emb_lst :", len(pairs_emb_lst))
        # for i, result in enumerate(pairs_emb_lst):
        #     print(f"Size of element {i}: {result.size()}")
        
        n_img, n_stc = paths_l2.size()[:2] 
        
        paths_l0 = paths_l0.contiguous().view(n_img, -1).unsqueeze(1).expand(-1, n_stc, -1)
        paths_l1 = paths_l1.view(n_img, n_stc, -1)
        paths_l2 = paths_l2.view(n_img, n_stc, -1)
        paths = torch.cat([paths_l0, paths_l1, paths_l2], dim=-1) # (n_img, n_stc, total_paths)
        paths = paths.mean(dim=1)
            
        paths = self.path_mapping(paths)
        paths = F.normalize(paths, dim=-1)
        sim_paths = paths.matmul(paths.t())
        
        if a == True:
            return pairs_emb_lst[0]
        else:
            return sim_paths