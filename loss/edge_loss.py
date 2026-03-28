import torch
def edge_loss(V, V_DEF, edges):
    """
    V : 原始顶点位置 (n_vertices, 3)
    V_DEF : 变形后的顶点位置 (n_vertices, 3)
    """
    vi = V[edges[:, 0]]
    vj = V[edges[:, 1]]
    e = torch.norm(vi - vj, dim=1)

    vi0 = V_DEF[edges[:, 0]]
    vj0 = V_DEF[edges[:, 1]]
    e_def = torch.norm(vi0 - vj0, dim=1)

    return ((e_def - e) ** 2).sum()