import torch
def anchor_loss(V_def, anchor_ids, anchor_targets):
    """
    Args:
        V_def: (N, 3) deformed vertex positions
        anchor_ids: (K,) vertex indices for anchors
        anchor_targets: (K, 3) target positions for anchors

    Returns:
        loss: scalar tensor
    """
    V_anchor = V_def[anchor_ids]  # (K, 3)
    loss = torch.mean(torch.sum((V_anchor - anchor_targets) ** 2, dim=1))
    return loss