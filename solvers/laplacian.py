import numpy as np
import scipy.sparse as sp
import torch


def build_edges(F):
    """
    F: (M, 3) np.ndarray
    Returns:
        edges: (E, 2) np.int64, unique undirected edges
    """
    edge_set = set()

    for tri in F:
        i, j, k = tri
        pairs = [(i, j), (j, k), (k, i)]
        for a, b in pairs:
            if a > b:
                a, b = b, a
            edge_set.add((a, b))

    edges = np.array(list(edge_set), dtype=np.int64)
    return edges


def build_uniform_laplacian(V, F):
    """
    Build uniform graph Laplacian:
        L[i, i] = 1
        L[i, j] = -1 / deg(i)  if j in N(i)

    Returns:
        scipy sparse csr_matrix, shape (N, N)
    """
    N = V.shape[0]
    edges = build_edges(F)

    neighbors = [[] for _ in range(N)]
    for i, j in edges:
        neighbors[i].append(j)
        neighbors[j].append(i)

    rows = []
    cols = []
    vals = []

    for i in range(N):
        nbrs = neighbors[i]
        deg = len(nbrs)

        # diagonal
        rows.append(i)
        cols.append(i)
        vals.append(1.0)

        if deg > 0:
            w = -1.0 / deg
            for j in nbrs:
                rows.append(i)
                cols.append(j)
                vals.append(w)

    L = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    return L


def build_cotangent_laplacian(V, F):
    """
    Build cotangent Laplacian:
        L[i, i] = sum_j w_ij
        L[i, j] = -w_ij  if j in N(i)
    where w_ij = (cot alpha + cot beta) / 2, and alpha,beta are angles opposite to edge (i,j)

    Returns:
        scipy sparse csr_matrix, shape (N, N)
    """
    N = V.shape[0]
    edges = build_edges(F)

    # Build adjacency list for faces
    vertex_faces = [[] for _ in range(N)]
    for idx, tri in enumerate(F):
        for v in tri:
            vertex_faces[v].append(idx)

    rows = []
    cols = []
    vals = []

    for i in range(N):
        nbrs = set()
        for f_idx in vertex_faces[i]:
            tri = F[f_idx]
            for v in tri:
                if v != i:
                    nbrs.add(v)

        w_sum = 0.0
        for j in nbrs:
            # Find common faces of i and j
            common_faces = set(vertex_faces[i]) & set(vertex_faces[j])
            w_ij = 0.0

            for f_idx in common_faces:
                tri = F[f_idx]
                # Find the third vertex k opposite to edge (i,j)
                k = [v for v in tri if v != i and v != j][0]

                vi = V[i]
                vj = V[j]
                vk = V[k]

                # Compute angles at k opposite to edge (i,j)
                u = vi - vk
                v = vj - vk
                cos_angle = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)
                cot_angle = 1.0 / np.tan(angle)
                w_ij += cot_angle

            w_ij *= 0.5
            w_sum += w_ij

            rows.append(i)
            cols.append(j)
            vals.append(-w_ij)

        # diagonal
        rows.append(i)
        cols.append(i)
        vals.append(w_sum)

    L = sp.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    return L

def scipy_sparse_to_torch_sparse(sparse_mtx, device="cpu", dtype=torch.float32):
    """
    scipy csr/csc/coo -> torch sparse COO tensor
    """
    coo = sparse_mtx.tocoo()
    indices = np.vstack((coo.row, coo.col))
    indices = torch.tensor(indices, dtype=torch.long, device=device)
    values = torch.tensor(coo.data, dtype=dtype, device=device)
    shape = coo.shape

    return torch.sparse_coo_tensor(indices, values, size=shape, device=device).coalesce()