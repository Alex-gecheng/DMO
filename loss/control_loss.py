def control_loss(V,control_ids, target):
    """
    V :(n_vertices, 3) 
    target: (n_control_points, 3)
    """
    V_control = V[control_ids]
    return ((V_control - target) ** 2).sum()
