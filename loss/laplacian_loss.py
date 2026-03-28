import torch
def laplacian_loss(V, V_DEF, L_torch):
    """
    控制Laplacian 坐标
    L_torch: torch sparse tensor (N, N) 拉普拉斯矩阵
    """
    lap_ref = torch.sparse.mm(L_torch, V)     # (N, 3)
    lap_def = torch.sparse.mm(L_torch, V_DEF)  # (N, 3)
    return ((lap_def - lap_ref) ** 2).sum()