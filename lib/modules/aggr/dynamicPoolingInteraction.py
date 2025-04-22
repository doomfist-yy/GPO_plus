import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pickle

from lib.modules.aggr.cells import MaxPoolingCell, AveragePoolingCell, K_MaxPoolingCell, RectifiedIdentityCell, ZeroLikeCell

def unsqueeze1d(x):
    return x.unsqueeze(-1)

def unsqueeze2d(x):
    return x.unsqueeze(-1).unsqueeze(-1)

def unsqueeze3d(x):
    return x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

def clones(module, N):
    '''Produce N identical layers.'''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class DynamicInteraction_Layer0(nn.Module):
    def __init__(self, embed_size, hid_router, num_cell, num_out_path):
        super(DynamicInteraction_Layer0, self).__init__()
        self.embed_size = embed_size
        self.hid_router = hid_router
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = num_cell
        self.num_out_path = num_out_path
        self.ric = RectifiedIdentityCell(embed_size, hid_router, num_out_path)
        self.mpc = MaxPoolingCell(embed_size, hid_router, num_out_path)
        self.apc = AveragePoolingCell(embed_size, hid_router, num_out_path)
        self.kmpc = K_MaxPoolingCell(embed_size, hid_router, num_out_path,k=6, output_size=(128, 1024))
        self.zlc = ZeroLikeCell(embed_size, hid_router, num_out_path)


    
         
    def forward(self, x):
        path_prob = [None] * self.num_cell # 4
        emb_lst = [None] * self.num_cell # 4
        emb_lst[0], path_prob[0] = self.ric(x) # 保持不变 1024 - 1024 
        # print("Shape of emb_lst[0]:", emb_lst[0].size()) #[128, 1024]
        # print("Shape of path_prob[0]:", path_prob[0].size()) #[128, 4]
        emb_lst[1], path_prob[1] = self.mpc(x) # 保持不变 1024 - 1024
        # print("Shape of emb_lst[1]:", emb_lst[1].size()) #[128, 1024]
        # print("Shape of path_prob[1]:", path_prob[1].size()) #[128, 4]
        emb_lst[2], path_prob[2] = self.apc(x) # 保持不变 1024 - 1024
        # print("Shape of emb_lst[2]:", emb_lst[2].size()) #[128, 1024]
        # print("Shape of path_prob[2]:", path_prob[2].size()) #[128, 4]
        emb_lst[3], path_prob[3] = self.kmpc(x) #  保持不变 1024 - 1024 
        # print("Shape of emb_lst[3]:", emb_lst[3].size()) #[128, 1024]
        # print("Shape of path_prob[3]:", path_prob[3].size()) #[128, 4]

        
        gate_mask = (sum(path_prob) < self.threshold).float()
        # print("Shape of gate_mask:", gate_mask.size()) # torch.Size([128, 4])
        all_path_prob = torch.stack(path_prob, dim=2)   # 改dim=1 为dim=2，思考下为什么？ tags

        all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps) #归一化，确保每一行的元素之和等于1。每个元素除以其所在行的总和
        # print(all_path_prob.size())
        # print("Shape of all_path_prob:", all_path_prob.size()) # [128, 4, 4])

        path_prob = [all_path_prob[:, :,i] for i in range(all_path_prob.size(2))]   #将 all_path_prob 张量在路径维度上的每一列分别提取出来，形成一个列表，使得每个元素都是一个表示不同路径上概率分布的列向量
        # print("Shape of path_prob----:", path_prob[0].size()) # ([4, 128, 4])  Shape of path_prob[0]: torch.Size([128, 4])
        aggr_res_lst = []
        for i in range(self.num_out_path):
            # print(gate_mask.size())

            # skip_emb = (gate_mask[:,i]).unsqueezed() * emb_lst[0]  
            skip_emb = torch.unsqueeze(gate_mask[:,i], -1) * emb_lst[0]
            # print("Shape of gate_mask[:, i]:", gate_mask[:, i].size()) # torch.Size([128])
            # print("Shape of  torch.unsqueeze(gate_mask[:,i], -1):", torch.unsqueeze(gate_mask[:,i], -1).size()) # torch.Size([128, 1])
            # print("Shape of emb_lst[0]:", emb_lst[0].size()) # torch.Size([128, 1024])
            # print("Shape of skip_emb:", skip_emb.size()) # torch.Size([128, 1024])
            res = 0
            for j in range(self.num_cell):
                cur_path = torch.unsqueeze(path_prob[j][:, i], -1)      #[:, i])
                # print("I===Shape of path_prob[j]:", path_prob[j][:, i].size())  #  torch.Size([128])
                # print("I===Shape of unsqueeze1d(path_prob[j][:, i]):", torch.unsqueeze(path_prob[j][:, i], -1).size())  # [128, 1]
                # print("I===Shape of emb_lst[j]:",emb_lst[j].shape)
                
                # if emb_lst[j].dim() == 2:
                #     cur_emb = emb_lst[j].unsqueeze(1)   #加了unsqueeze(1)
                # else:   # 3
                cur_emb = emb_lst[j]
                res = res + cur_path * cur_emb
                # print("T===Shape of res:", res.size())
                # print("T===Shape of cur_path:", cur_path.size())
                # print("T===Shape of cur_emb:", cur_emb.size())
            # res = res + skip_emb.unsqueeze(0)
            res = res + skip_emb
            aggr_res_lst.append(res)





        return aggr_res_lst, all_path_prob

    

class DynamicInteraction_Layer(nn.Module):
    def __init__(self, embed_size, hid_router, num_cell, num_out_path):
        super(DynamicInteraction_Layer, self).__init__()
        self.embed_size = embed_size
        self.hid_router = hid_router
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = num_cell
        self.num_out_path = num_out_path

        self.ric = RectifiedIdentityCell(embed_size, hid_router, num_out_path)
        self.mpc = MaxPoolingCell(embed_size, hid_router, num_out_path)
        self.apc = AveragePoolingCell(embed_size, hid_router, num_out_path)
        self.kmpc = K_MaxPoolingCell(embed_size, hid_router, num_out_path,k=6, output_size=(128, 1024))
        self.zlc = ZeroLikeCell(embed_size, hid_router, num_out_path)
        

    def forward(self, ref_rgn):
        # print("Length of ref_rgn:", len(ref_rgn)) #4
        # for i, result in enumerate(ref_rgn):
        #     print(f"Size of element {i}: {result.size()}") #Size of element 0,1,2,3: torch.Size([128, 1024])

        # print("Dimension of the first element in ref_rgn:", ref_rgn[0].dim()) 
        # print("Shape of the first element in ref_rgn:", ref_rgn[0].size())

        # assert len(ref_rgn) == self.num_cell and ref_rgn[0].dim() == 4
        assert len(ref_rgn) == self.num_cell and ref_rgn[0].dim() == 2     
        
        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell
        emb_lst[0], path_prob[0] = self.ric(ref_rgn[0])
        # print("Shape of emb_lst[0]:", emb_lst[0].size())
        # print("Shape of path_prob[0]:", path_prob[0].size())
        emb_lst[1], path_prob[1] = self.mpc(ref_rgn[1])
        emb_lst[2], path_prob[2] = self.apc(ref_rgn[2])
        emb_lst[3], path_prob[3] = self.kmpc(ref_rgn[3])

        if self.num_out_path == 1:
            aggr_res_lst = []
            gate_mask_lst = []
            res = 0
            for j in range(self.num_cell):
                gate_mask = (path_prob[j] < self.threshold / self.num_cell).float() 
                # print("III===Shape of path_prob[j]:", path_prob[j].size()) # torch.Size([32, 1])
                # print("III===Shape of gate_mask:", gate_mask.size()) # torch.Size([32, 1])
                gate_mask_lst.append(gate_mask)
                # print("III===Shape of ref_rgn[j]:", ref_rgn[j].size()) # torch.Size([32, 1024])
                # print("III===Shape of emb_lst[j]:", emb_lst[j].size()) # torch.Size([32, 1024])
                skip_emb = gate_mask * ref_rgn[j]
                res += path_prob[j] * emb_lst[j]
                res += skip_emb
                # print("III===Shape of res:", res.size())


            res = res / (sum(gate_mask_lst) + sum(path_prob))
            # print("III===Shape of res2:", res.size())
            # all_path_prob = torch.stack(path_prob, dim=3) 
            all_path_prob = torch.stack(path_prob, dim=2)          # all_path_prob = torch.stack(path_prob, dim=2) 
            # print("III===Shape of all_path_prob:", all_path_prob.size()) # 
            aggr_res_lst.append(res)
        else:
            gate_mask = (sum(path_prob) < self.threshold).float()   
            # print("Shape of gate_mask:", gate_mask.size()) 
            # for i, p in enumerate(path_prob):
            #     print(f"Dimension of path_prob[{i}]:", p.dim()) #  2
            #     print(f"Shape of path_prob[{i}]:", p.size()) # torch.Size([32, 4])

            # all_path_prob = torch.stack(path_prob, dim=3)  
            all_path_prob = torch.stack(path_prob, dim=2)          # all_path_prob = torch.stack(path_prob, dim=2) 
            
            all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
            # print("Shape of all_path_prob:", all_path_prob.size())  # torch.Size([32, 4, 4])
            path_prob = [all_path_prob[ :, :, i] for i in range(all_path_prob.size(2))]
            # print("Length of path_prob————:", len(path_prob)) 
            # print("Shape of path_prob----:", path_prob[0].size())  # 4  *  torch.Size([32, 4])
            aggr_res_lst = []
            for i in range(self.num_out_path): 
                skip_emb = unsqueeze1d(gate_mask[:, i])* emb_lst[0]    # 
                # print("DII===Shape of gate_mask[:, i]:", gate_mask[:, i].size()) #  torch.Size([32])
                # print("DII===Shape of  unsqueeze1d(gate_mask[:, i])", unsqueeze1d(gate_mask[:, i]).size()) # torch.Size([32, 1])
                # print("DII===Shape of emb_lst[0]:", emb_lst[0].size()) # torch.Size([32, 1024])
                # print("DII===Shape of skip_emb:", skip_emb.size()) # torch.Size([32, 1024])
                res = 0
                for j in range(self.num_cell):
                    cur_path = unsqueeze1d(path_prob[j][:, i])
                    # print("II===Shape of path_prob[j]:", path_prob[j][:, i].size())  # torch.Size([32])
                    # print("II===Shape of unsqueeze2d(path_prob[j][:, i]):", unsqueeze1d(path_prob[j][:, i]).size())  # torch.Size([32, 1])
                    # print("II===Shape of cur_path:", cur_path.size())  # torch.Size([32, 1])
                    # print("II===Shape of emb_lst[j]:", emb_lst[j].size()) #  torch.Size([32, 1024])
                    res = res + cur_path * emb_lst[j]
                res = res + skip_emb
                aggr_res_lst.append(res)

        return aggr_res_lst, all_path_prob



    


