import torch
import torch.nn.functional as F  

def similarity(h1, h2):
    h1 = F.normalize(h1, p=2, dim=-1)
    h2 = F.normalize(h2, p=2, dim=-1)
    return torch.matmul(h1, h2.T)

xc = torch.tensor([[1, 1], [2, 2], [3, 3]]).float()  # batch_size = 3, embedding_dim = 2
xo = torch.tensor([[10, 10], [20, 20], [30, 30]]).float()
h_graph = torch.tensor([[11, 11], [22, 22], [33, 33]]).float()

batch_size = xc.size(0)
print('batch_size',batch_size)  # 3
# 创建混合图索引矩阵，避免 i == j
idx = torch.arange(batch_size, device=xc.device)
print('idx',idx)  # tensor([0, 1, 2])
# causal_idx = idx.repeat(batch_size - 1)
# print('causal_idx',causal_idx)  # tensor([0, 0, 1, 1, 2, 2])

causal_idx = idx.unsqueeze(1).repeat(1, batch_size - 1).flatten()
print('causal_idx',causal_idx)
bias_idx = idx.unsqueeze(1).repeat(1, batch_size).T 
print('bias_idx',bias_idx)  # tensor([1, 2, 0, 2, 0, 1])
# 移除对角线
mask = ~torch.eye(batch_size, dtype=torch.bool, device=xc.device)  # 对角线为 False 的布尔掩码
print('mask',mask)  # tensor([[False,  True,  True],
bias_idx = bias_idx[mask]  # 移除对角线后展平
# mask = causal_idx != bias_idx
# print('mask',mask)  # tensor([False,  True,  True,  True,  True, False])
# causal_idx = causal_idx[mask]
# print('causal_idx',causal_idx)  # tensor([0, 1, 1, 2])
# bias_idx = bias_idx[mask]
print('bias_idx',bias_idx)  # tensor([1, 0, 2, 1])

h_mix_list = xc[causal_idx] + xo[bias_idx]  # 批量计算混合图嵌入
print('h_mix_list',h_mix_list)
        # 调整 h_mix_list 的形状为 (batch_size - 1, batch_size, embedding_dim)
h_mix_list = h_mix_list.view(batch_size, batch_size-1, -1)#.permute(0, 1, 2)
print('h_mix_list',h_mix_list)  # tensor([[ 3,  3],
                                #         [21, 21],
                                #         [32, 32],
                                #         [23, 23]])
'''
loss=0
for i in range(batch_size):
    for j in range(batch_size-1):
        print('i',i)
        print('j',j)
        # Positive sample: h_mix_list[j][i] and h_graph[i]
        pos_sim = similarity(h_mix_list[i][j], h_graph[i])
        print('h_mix_list[i][j]',h_mix_list[i][j])
        print('h_graph[i]',h_graph[i])
        print('pos_sim',pos_sim)
        neg_sims = similarity(h_mix_list[i][j], h_graph)
        print('neg_sims',neg_sims)
        neg_sims[i] = -float('inf')  # Mask self similarity
        print('neg_sims_aftter',neg_sims)
        numerator = torch.exp(pos_sim)
        denominator = torch.exp(pos_sim) + torch.exp(neg_sims).sum()
        print('numerator',numerator)
        print('denominator',denominator)
        print('(neg_sims).sum()',torch.exp(neg_sims).sum())
        loss += -torch.log(numerator / denominator)
'''


h_graph_normalized = F.normalize(h_graph, p=2, dim=-1)  # Precompute normalization
print('h_graph_normalized',h_graph_normalized)


# Compute similarities for all environments in one batch
loss = 0
K = batch_size - 1

for j in range(K):
    print('j',j)
    print('h_mix_list[j]',h_mix_list[j])
    h_mix_normalized = F.normalize(h_mix_list[j], p=2, dim=-1)
    print('h_mix_normalized',h_mix_normalized)
    print('h_mix_normalized.shape',h_mix_normalized.shape)
    print('h_graph_normalized.shape',h_graph_normalized.shape)
    # Compute batch similarity (B, B)
    sim_matrix = torch.matmul(h_mix_normalized, h_graph_normalized.T)
    
    # Extract positive similarities (diagonal elements)
    pos_sim = torch.diag(sim_matrix)

    # Mask self-similarities for negatives
    sim_matrix.fill_diagonal_(-float('inf'))

    # Compute negative similarities
    neg_sims = sim_matrix.exp().sum(dim=-1)

    # Compute InfoNCE loss for this environment
    loss += (-pos_sim + torch.log(neg_sims)).mean()

