import torch
from loss.control_loss import control_loss
from loss.laplacian_loss import laplacian_loss
from loss.edge_loss import edge_loss
from loss.anchor_loss import anchor_loss



def optimize_mesh(
    V,
    F,
    edges,
    L_torch,
    control_ids,
    control_targets,
    anchor_ids,
    anchor_targets,
    num_iters=1000,
    lr=1e-2,
    w_ctrl=1000.0,
    w_lap=1.0,
    w_edge=1.0,
    w_anchor=1000,
    device="cuda",
):
    """
    Args:
        V: (N, 3) torch.float32
        F: (M, 3) torch.long 
        edges: (E, 2) torch.long
        L_torch: torch sparse (N, N)
        control_ids: (K,) torch.long
        control_targets: (K, 3) torch.float32
        anchor_ids: (L,) torch.long
        anchor_targets: (L, 3) torch.float32


    Returns:
        V_def: (N, 3)
        delta_v: (N, 3)
    """
    V = V.to(device)
    F = F.to(device)
    edges = edges.to(device)
    control_ids = control_ids.to(device)
    control_targets = control_targets.to(device)
    anchor_ids = anchor_ids.to(device)
    anchor_targets = anchor_targets.to(device)
    L_torch = L_torch.to(device)

    delta_v = torch.nn.Parameter(torch.zeros_like(V))

    optimizer = torch.optim.Adam([delta_v], lr=lr)

    for it in range(num_iters):
        V_def = V + delta_v

        l_ctrl = control_loss(V_def, control_ids, control_targets)
        l_anchor = anchor_loss(V_def, anchor_ids, anchor_targets)
        l_lap = laplacian_loss(V, V_def, L_torch)
        l_edge = edge_loss(V, V_def, edges)

        loss = (
            w_ctrl * l_ctrl
            + w_lap * l_lap
            + w_edge * l_edge
            + w_anchor * l_anchor
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if  (it % 100 == 0 or it == num_iters - 1):
            print(
                f"[{it:04d}/{num_iters}] "
                f"total={loss.item():.6f} | "
                f"ctrl_raw={l_ctrl.item():.6f}, "
                f"lap_raw={l_lap.item():.6f}, "
                f"edge_raw={l_edge.item():.6f}, "
                f"anchor_raw={l_anchor.item():.6f}, "
                f"ctrl_w={(w_ctrl*l_ctrl).item():.6f}, "
                f"lap_w={(w_lap*l_lap).item():.6f}, "
                f"edge_w={(w_edge*l_edge).item():.6f}"
                f"anchor_w={(w_anchor*l_anchor).item():.6f}"
            )

    V_def = V + delta_v
    return V_def, delta_v