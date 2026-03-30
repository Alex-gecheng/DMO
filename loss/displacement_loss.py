# 惩罚项  使每个顶点形变尽可能的少 ，增加平滑性

def displacement_loss(delta_v):
    return (delta_v ** 2).sum()