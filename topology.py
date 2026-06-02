import numpy as np
from scipy.sparse.csgraph import shortest_path


def compute_lambda2(A):
    """Compute algebraic connectivity (λ₂) of a graph given its adjacency matrix."""
    degree = A.sum(axis=1)
    D = np.diag(degree)
    L = D - A
    eigenvalues = np.linalg.eigvalsh(L)
    return sorted(eigenvalues)[1]

def build_grid_adjacency(n, m):
    """Natural grid neighbors — reproduces CoLight's current behavior."""
    N = n * m
    A = np.zeros((N, N))
    for i in range(N):
        r, c = i // m, i % m
        if r > 0:
            j = (r - 1) * m + c
            A[i][j] = A[j][i] = 1
        if r < n - 1:
            j = (r + 1) * m + c
            A[i][j] = A[j][i] = 1
        if c > 0:
            j = r * m + (c - 1)
            A[i][j] = A[j][i] = 1
        if c < m - 1:
            j = r * m + (c + 1)
            A[i][j] = A[j][i] = 1
    return A

def build_torus_adjacency(n, m):
    """Grid + wrap-around edges — every agent has exactly 4 neighbors."""
    N = n * m
    A = np.zeros((N, N))
    for i in range(N):
        r, c = i // m, i % m
        neighbors = [
            ((r - 1) % n) * m + c,  # Up
            ((r + 1) % n) * m + c,  # Down
            r * m + ((c - 1) % m),  # Left
            r * m + ((c + 1) % m),  # Right
        ]
        for j in neighbors:
            A[i][j] = A[j][i] = 1
    return A

def build_optimal_adjacency(n, m, iterations=500):
    """Hill-climb on λ₂ starting from torus to find near-optimal topology."""
    A = build_torus_adjacency(n, m)
    best_lambda2 = compute_lambda2(A)

    edges = [(i, j) for i in range(n * m) for j in range(i + 1, n * m) if A[i][j] == 1]

    for _ in range(iterations):
        idx1, idx2 = np.random.choice(len(edges), 2, replace=False)
        u, v = edges[idx1]
        x, y = edges[idx2]

        if len({u, v, x, y}) < 4:
            continue
        if A[u][x] == 1 or A[v][y] == 1:
            continue

        # Try swap
        A[u][v] = A[v][u] = 0
        A[x][y] = A[y][x] = 0
        A[u][x] = A[x][u] = 1
        A[v][y] = A[y][v] = 1

        new_lambda2 = compute_lambda2(A)

        if new_lambda2 > best_lambda2:
            best_lambda2 = new_lambda2
            edges[idx1] = (u, x)
            edges[idx2] = (v, y)
        else:
            # Undo swap
            A[u][x] = A[x][u] = 0
            A[v][y] = A[y][v] = 0
            A[u][v] = A[v][u] = 1
            A[x][y] = A[y][x] = 1

    print(f"Optimal topology λ₂: {best_lambda2:.4f}")
    return A

def build_none_adjacency(n, m):
    """No inter-agent communication — independent baseline."""
    return np.eye(n * m)

if __name__ == "__main__":
    n = 28
    m = 7
    A_grid  = build_grid_adjacency(n, m)
    A_torus = build_torus_adjacency(n, m)
    A_opt   = build_optimal_adjacency(n, m)
    A_none  = build_none_adjacency(n, m)

    D_grid = shortest_path(A_grid, directed=False)
    D_torus = shortest_path(A_torus, directed=False)
    D_opt = shortest_path(A_opt, directed=False)

    print("Grid diameter:", int(D_grid[D_grid != np.inf].max()))
    print("Torus diameter:", int(D_torus[D_torus != np.inf].max()))
    print("Optimal diameter:", int(D_opt[D_opt != np.inf].max()))
    print("Grid    λ₂:", compute_lambda2(A_grid))
    print("Torus   λ₂:", compute_lambda2(A_torus))
    print("None    λ₂:", compute_lambda2(A_none))
    print("Torus degree (all should be 4):", A_torus.sum(axis=1))

