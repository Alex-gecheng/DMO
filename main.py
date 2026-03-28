import os
import numpy as np
import torch

from core.mesh_io import load_mesh, save_mesh
from solvers.laplacian import build_edges, build_uniform_laplacian, scipy_sparse_to_torch_sparse,build_cotangent_laplacian
from solvers.optimize import optimize_mesh


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # =========================
    # 1. Load source mesh
    # =========================
    mesh_path = "data/body.obj"
    V_np, F_np = load_mesh(mesh_path)

    print("Loaded mesh:")
    print("  vertices:", V_np.shape)
    print("  faces   :", F_np.shape)

    # =========================
    # 2. Build geometry
    # =========================
    edges_np = build_edges(F_np)
    L_scipy = build_cotangent_laplacian(V_np, F_np)

    print("Built geometry:")
    print("  edges:", edges_np.shape)
    print("  Laplacian shape:", L_scipy.shape)

    # =========================
    # 3. Define control points
    # =========================
    # 从 CSV 读取控制点: 每行格式为 index,x,y,z
    control_csv_path = "data/p.csv"
    ctrl_data = np.loadtxt(control_csv_path, delimiter=",", dtype=np.float32)
    if ctrl_data.ndim == 1:
        ctrl_data = ctrl_data[None, :]
    if ctrl_data.shape[1] != 4:
        raise ValueError(f"Invalid control csv format in {control_csv_path}, expected 4 columns: index,x,y,z")

    control_ids_np = ctrl_data[0, 0].astype(np.int64)
    control_targets_np = ctrl_data[0, 1:4].astype(np.float32)

    if np.any(control_ids_np < 0) or np.any(control_ids_np >= V_np.shape[0]):
        raise ValueError("control vertex index out of range")
    

    print("Control points:")
    print("  ids:", control_ids_np)
    print("  targets shape:", control_targets_np.shape)

    anchor_ids_np = ctrl_data[1:, 0].astype(np.int64)
    anchor_targets_np = ctrl_data[1:, 1:4].astype(np.float32)

    if  np.any(anchor_ids_np >= V_np.shape[0]):
        raise ValueError("anchor vertex index out of range")
    print("Anchor points:")
    print("  ids:", anchor_ids_np)
    print("  targets shape:", anchor_targets_np.shape)

    # =========================
    # 4. Convert to torch
    # =========================
    V0 = torch.tensor(V_np, dtype=torch.float32, device=device)
    F = torch.tensor(F_np, dtype=torch.long, device=device)
    edges = torch.tensor(edges_np, dtype=torch.long, device=device)

    control_ids = torch.tensor(control_ids_np, dtype=torch.long, device=device)
    control_targets = torch.tensor(control_targets_np, dtype=torch.float32, device=device)
    anchor_ids = torch.tensor(anchor_ids_np, dtype=torch.long, device=device)
    anchor_targets = torch.tensor(anchor_targets_np, dtype=torch.float32, device=device)

    L_torch = scipy_sparse_to_torch_sparse(L_scipy, device=device, dtype=torch.float32)

    # =========================
    # 5. Optimize
    # =========================
    V_def, delta_v = optimize_mesh(
        V=V0,
        F=F,
        edges=edges,
        L_torch=L_torch,
        control_ids=control_ids,
        control_targets=control_targets,
        anchor_ids=anchor_ids,
        anchor_targets=anchor_targets,
        num_iters=1500,
        lr=1e-2,
        w_ctrl=1000.0,
        w_lap=5.0,
        w_edge=10.0,
        w_anchor=1000.0,
        device=device,
    )

    # =========================
    # 6. Save result
    # =========================
    os.makedirs("output", exist_ok=True)
    out_path = "output/deformed.obj"
    save_mesh(out_path, V_def, F)

    print(f"Saved deformed mesh to: {out_path}")


if __name__ == "__main__":
    main()