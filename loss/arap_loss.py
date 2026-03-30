import torch


def build_vertex_neighbors(num_vertices, edges, device=None):
    """
    Build adjacency list from undirected edge list.

    Args:
        num_vertices: int
        edges: (E, 2) torch.long
        device: torch device

    Returns:
        neighbors: list of length N, neighbors[i] is 1D torch.long tensor
    """
    if device is None:
        device = edges.device

    neighbor_sets = [set() for _ in range(num_vertices)]

    edges_cpu = edges.detach().cpu()
    for e in edges_cpu:
        i, j = int(e[0]), int(e[1])
        if i == j:
            continue
        neighbor_sets[i].add(j)
        neighbor_sets[j].add(i)

    neighbors = []
    for i in range(num_vertices):
        if len(neighbor_sets[i]) == 0:
            neighbors.append(torch.empty(0, dtype=torch.long, device=device))
        else:
            neighbors.append(torch.tensor(
                sorted(list(neighbor_sets[i])),
                dtype=torch.long,
                device=device
            ))
    return neighbors


def cotangent_like_weights_from_edges(num_vertices, edges, device=None):
    """
    Simple uniform weights per neighbor (NOT true cotangent weights).
    This is a placeholder helper if you want explicit per-neighbor weights.

    Returns:
        weights: list of length N, weights[i] is 1D float tensor aligned with neighbors[i]
    """
    neighbors = build_vertex_neighbors(num_vertices, edges, device=device)
    weights = []
    for nbr in neighbors:
        if len(nbr) == 0:
            weights.append(torch.empty(0, dtype=torch.float32, device=device))
        else:
            w = torch.ones(len(nbr), dtype=torch.float32, device=device)
            weights.append(w)
    return neighbors, weights


def arap_loss(V, V_def, edges, neighbors=None, weights=None, eps=1e-8):
    """
    Differentiable ARAP loss (per-vertex local rotation via SVD).

    Energy:
        sum_i sum_{j in N(i)} w_ij || (v'_i - v'_j) - R_i (v_i - v_j) ||^2

    Args:
        V:      (N, 3) original vertices
        V_def:  (N, 3) deformed vertices
        edges:  (E, 2) undirected edges, torch.long
        neighbors: optional precomputed adjacency list, list[N] of 1D torch.long
        weights:   optional precomputed weight list, aligned with neighbors
                   if None -> uniform weights
        eps: small number for numerical stability

    Returns:
        loss: scalar tensor
    """
    device = V.device
    N = V.shape[0]

    if neighbors is None:
        neighbors = build_vertex_neighbors(N, edges, device=device)

    if weights is None:
        # uniform weights
        weights = []
        for nbr in neighbors:
            if len(nbr) == 0:
                weights.append(torch.empty(0, dtype=V.dtype, device=device))
            else:
                weights.append(torch.ones(len(nbr), dtype=V.dtype, device=device))

    total_loss = V.new_tensor(0.0)
    valid_vertices = 0

    for i in range(N):
        nbr = neighbors[i]
        if nbr.numel() == 0:
            continue

        w = weights[i].to(device=device, dtype=V.dtype)  # (k,)
        if w.numel() != nbr.numel():
            raise ValueError(f"weights[{i}] size mismatch with neighbors[{i}]")

        # original local edges: p_ij = v_i - v_j
        p = V[i].unsqueeze(0) - V[nbr]         # (k, 3)

        # deformed local edges: q_ij = v'_i - v'_j
        q = V_def[i].unsqueeze(0) - V_def[nbr] # (k, 3)

        # weighted covariance S_i = sum_j w_ij * q_ij * p_ij^T
        # shape: (3, 3)
        # equivalent to: S = (w[:,None] * q).T @ p
        S = (w.unsqueeze(1) * q).transpose(0, 1) @ p

        # SVD for best-fit rotation
        # S = U Sigma V^T
        try:
            U, _, Vh = torch.linalg.svd(S, full_matrices=False)
        except RuntimeError:
            # fallback if SVD occasionally fails
            # add tiny diagonal jitter
            S_jitter = S + eps * torch.eye(3, device=device, dtype=V.dtype)
            U, _, Vh = torch.linalg.svd(S_jitter, full_matrices=False)

        R = U @ Vh

        # reflection correction: enforce det(R)=+1
        detR = torch.det(R)
        if detR < 0:
            # flip last column of U
            U_fix = U.clone()
            U_fix[:, -1] *= -1.0
            R = U_fix @ Vh

        # ARAP residual
        Rp = (R @ p.transpose(0, 1)).transpose(0, 1)   # (k, 3)
        residual = q - Rp                              # (k, 3)

        # weighted squared norm
        vertex_loss = torch.sum(w * torch.sum(residual * residual, dim=1))
        total_loss = total_loss + vertex_loss
        valid_vertices += 1

    if valid_vertices == 0:
        return V.new_tensor(0.0)

    # normalize by number of valid vertices
    return total_loss / valid_vertices