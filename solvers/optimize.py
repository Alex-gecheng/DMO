import torch
from loss.control_loss import control_loss
from loss.laplacian_loss import laplacian_loss
from loss.edge_loss import edge_loss
from loss.anchor_loss import anchor_loss
from loss.displacement_loss import displacement_loss
from loss.arap_loss import arap_loss


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
    optimizer_type="lbfgs",
    w_ctrl=1000.0,
    w_lap=1.0,
    w_edge=1.0,
    w_anchor=1000,
    w_disp=0.01,
    w_arap=50.0,
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
        optimizer_type: "adam" or "lbfgs"


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

    optimizer_name = optimizer_type.lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam([delta_v], lr=lr)
    elif optimizer_name == "lbfgs":
        optimizer = torch.optim.LBFGS([delta_v], lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer_type: {optimizer_type}. Use 'adam' or 'lbfgs'.")

    for it in range(num_iters):
        if optimizer_name == "lbfgs":
            def closure():
                optimizer.zero_grad()
                V_def_local = V + delta_v
                l_ctrl_local = control_loss(V_def_local, control_ids, control_targets)
                l_anchor_local = anchor_loss(V_def_local, anchor_ids, anchor_targets)
                l_lap_local = laplacian_loss(V, V_def_local, L_torch)
                l_edge_local = edge_loss(V, V_def_local, edges)
                l_disp_local = displacement_loss(delta_v)
                l_arap_local = arap_loss(V, V_def_local, edges)

                loss_local = (
                    w_ctrl * l_ctrl_local
                    + w_lap * l_lap_local
                    + w_edge * l_edge_local
                    + w_anchor * l_anchor_local
                    + w_disp * l_disp_local
                    + w_arap * l_arap_local
                )
                loss_local.backward()
                return loss_local

            optimizer.step(closure)
        else:
            optimizer.zero_grad()
            V_def_step = V + delta_v
            l_ctrl_step = control_loss(V_def_step, control_ids, control_targets)
            l_anchor_step = anchor_loss(V_def_step, anchor_ids, anchor_targets)
            l_lap_step = laplacian_loss(V, V_def_step, L_torch)
            l_edge_step = edge_loss(V, V_def_step, edges)
            l_disp_step = displacement_loss(delta_v)
            l_arap_step = arap_loss(V, V_def_step, edges)
            loss_step = (
                w_ctrl * l_ctrl_step
                + w_lap * l_lap_step
                + w_edge * l_edge_step
                + w_anchor * l_anchor_step
                + w_disp * l_disp_step
                + w_arap * l_arap_step
            )
            loss_step.backward()
            optimizer.step()

        

        if  (it % 100 == 0 or it == num_iters - 1):
            with torch.no_grad():
                V_def = V + delta_v
                l_ctrl = control_loss(V_def, control_ids, control_targets)
                l_anchor = anchor_loss(V_def, anchor_ids, anchor_targets)
                l_lap = laplacian_loss(V, V_def, L_torch)
                l_edge = edge_loss(V, V_def, edges)
                l_disp = displacement_loss(delta_v)
                l_arap = arap_loss(V, V_def, edges)
                loss = (
                    w_ctrl * l_ctrl
                    + w_lap * l_lap
                    + w_edge * l_edge
                    + w_anchor * l_anchor
                    + w_disp * l_disp
                    + w_arap * l_arap
                )
            print(
                f"[{it:04d}/{num_iters}] "
                f"total={loss.item():.6f} | "
                f"ctrl_raw={l_ctrl.item():.6f}, "
                f"lap_raw={l_lap.item():.6f}, "
                f"edge_raw={l_edge.item():.6f}, "
                f"anchor_raw={l_anchor.item():.6f}, "
                f"ctrl_w={(w_ctrl*l_ctrl).item():.6f}, "
                f"lap_w={(w_lap*l_lap).item():.6f}, "
                f"edge_w={(w_edge*l_edge).item():.6f}, "
                f"anchor_w={(w_anchor*l_anchor).item():.6f}, "
                f"disp_w={(w_disp*l_disp).item():.6f}, "
                f"arap_w={(w_arap*l_arap).item():.6f}"
            )

    V_def = V + delta_v
    return V_def, delta_v