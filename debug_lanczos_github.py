import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from feiguin_lanczos import mapping, block_lanczos, tridiagonal_lanczos

def check_block_lanczos(seeds, N):
    column_vector, coord_touples, alphas_list, betas_list, w, coord_to_index, v_init = mapping(
        [(0, 0)], 2
    )
    print("Column Vector: ", column_vector)
    print("Coordinate Tuples: ", coord_touples)
    print("Alphas List: ", alphas_list)
    print("Betas List: ", betas_list)
    print("W: ", w)
    print("Coordinate to Index Mapping: ", coord_to_index)
    print("Initial Vector v_init: ", v_init)

    coords_all, coord_to_index, Vs, all_maps, Ws, last_Q, As, Bs, Qblocks = block_lanczos(
        1, [(0, 0)], 2
    )

    Qblocks = Qblocks[: len(As)]

    def build_M_from_coords(coords_all, coord_to_index):
        n = len(coords_all)
        M = np.zeros((n, n), dtype=float)

        for i, (x, y) in enumerate(coords_all):
            # neighbors in the same pattern as apply_H
            for nx, ny in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)):
                j = coord_to_index.get((nx, ny))
                if j is not None:
                    M[i, j] = 1.0  # or your coupling value

        # (Optionally enforce symmetry explicitly)
        M = 0.5 * (M + M.T)
        return M

    M = build_M_from_coords(coords_all, coord_to_index)

    Qfull = np.hstack(Qblocks)
    print("Qfull shape:", Qfull.shape)
    print("Qfull:\n", Qfull)

    # Check symmetry
    is_symmetric = np.allclose(M, M.T)
    print("Mapping matrix M:\n", M)
    print("Is M symmetric?: ", is_symmetric)

    # Eigenvalues (first with numpy)
    M_eigenvals = np.linalg.eigvalsh(M)
    tol = 1e-10
    M_eigenvals[np.abs(M_eigenvals) < tol] = 0.0
    print("Eigenvalues of M:", M_eigenvals)

    print("Coords all:", coord_touples)
    print("Coord to index:", coord_to_index)
    if is_symmetric:
        print("Mapping matrix M is symmetric.")
    else:
        print("Mapping matrix M is not symmetric.")

    # Eigenvalues with scipy (overwrites previous)
    M_eigenvals = sc.linalg.eigvalsh(
        M, b=None, lower=True, overwrite_a=False, overwrite_b=False, check_finite=True
    )
    tol = 1e-10
    M_eigenvals = [ev.real for ev in M_eigenvals if abs(ev.imag) < tol]
    for i in range(len(M_eigenvals)):
        if abs(M_eigenvals[i]) < tol:
            print(f"Eigenvalue {M_eigenvals[i]} at index {i} is approximately zero.")
            M_eigenvals[i] = 0.0
    print("Eigenvalues of M: ", M_eigenvals)

    print("Coords all", coords_all)
    print("Coord to index", coord_to_index)

    print("Ws: ", np.hstack(Ws))
    print("Ws:", Ws)
    print("As: ", As)
    print("Bs: ", Bs)

    def build_T_variable_blocks(As, Bs):
        """
        Build symmetric block tridiagonal from variable-sized As, Bs.

        As[i] shape: (r_i, r_i)
        Bs[i] couples block i and i+1, with shape (r_i, r_{i+1}).
        """
        # Scalars from your Lanczos run
        alphas = np.array(As)  # length 2
        betas = np.array(Bs)   # length 2

        # 1) Include a block for the seed (alpha_0 = 0.0)
        As = [np.array([[0.0]])]               # seed block (v0)
        As += [np.array([[a]]) for a in alphas]  # v1, v2

        # 2) Off-diagonal blocks between successive vectors
        Bs = [np.array([[b]]) for b in betas]  # (v0↔v1), (v1↔v2)

        block_sizes = [A.shape[0] for A in As]  # [1, 1, 1]
        offsets = np.concatenate(([0], np.cumsum(block_sizes)))
        total = offsets[-1]
        T = np.zeros((total, total), dtype=float)

        for i, A in enumerate(As):
            i0, i1 = offsets[i], offsets[i + 1]

            # Diagonal
            T[i0:i1, i0:i1] = 0.5 * (A + A.T)

            if i > 0:
                # Bs[i-1] couples block (i-1) ↔ i
                B = Bs[i - 1]
                j0, j1 = offsets[i - 1], offsets[i]

                # upper (i-1, i)
                T[j0:j1, i0:i1] = B

                # lower (i, i-1)
                T[i0:i1, j0:j1] = B.T

        return T

    print("0: Ws[0]", Ws[0])
    print("1: Ws[1]", Ws[1])
    print("Qblocks: ", Qblocks)
    print(
        "Dot of Q columns (should be 0): ",
        np.dot(np.hstack(Qblocks[0]), np.hstack(Qblocks[1])),
    )
    print([Q.shape for Q in Qblocks])

    T = build_T_variable_blocks(As, Bs)
    print("T built from As and Bs:\n", T)

    T_Q = Qfull.T @ M @ Qfull
    print("T shape:", T.shape)
    print("T_Q shape:", T_Q.shape)
    print("Lanczos Tridiagonal Matrix T:\n", T)

    T_eigenvals = sc.linalg.eigvalsh(
        T, b=None, lower=True, overwrite_a=False, overwrite_b=False, check_finite=True
    )

    print("M largest eigenvalue:", max(M_eigenvals))
    print("T largest eigenvalue:", max(T_eigenvals))
    print("M smallest eigenvalue:", min(M_eigenvals))
    print("T smallest eigenvalue:", min(T_eigenvals))
    print("Print T_Q:\n", T_Q)

    # 3. Check convergence of top-k eigenvalues:
    k = 5
    M_top = sorted(M_eigenvals, reverse=True)[:k]
    T_top = sorted(T_eigenvals, reverse=True)[:k]
    print(f"Top {k} eigenvalues comparison:")
    print("M:", M_top)
    print("T:", T_top)

    # For each T eigenvalue, find closest M eigenvalue
    for t_eig in sorted(T_eigenvals):
        closest_m_eig = M_eigenvals[np.argmin(np.abs(M_eigenvals - t_eig))]
        error = abs(t_eig - closest_m_eig)
        print(f"T: {t_eig:8.4f} ≈ M: {closest_m_eig:8.4f}  (error: {error:.6f})")

    plt.plot(sorted(M_eigenvals), label="M Eigenvalues", marker="o")
    plt.plot(sorted(T_eigenvals), label="T Eigenvalues", marker="x")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.title("Eigenvalue Comparison between M and T")
    plt.legend()
    plt.show()

    # Now plotting heatmap of T block matrix with values in each block
    plt.imshow(T, aspect="auto", cmap="viridis")
    plt.colorbar(label="Value")
    plt.xlabel("Block Index")
    plt.ylabel("Block Index")
    plt.title("Lanczos Tridiagonal Matrix T")
    plt.show()

    # plotting M matrix as heatmap
    plt.imshow(M, aspect="auto", cmap="viridis")
    plt.colorbar(label="Value")
    plt.xlabel("Index")
    plt.ylabel("Index")
    plt.title("Mapping Matrix M")
    plt.show()

    for i, A in enumerate(As):
        print(f"A[{i}] shape:", A.shape)
    for i, B in enumerate(Bs):
        print(f"B[{i}] shape:", B.shape)


def check_lanczos(seeds, N):
    column_vector, coord_touples, alphas_list, betas_list, w, coord_to_index, v_init = mapping(
        [(0, 0)], 2
    )
    print("Column Vector: ", column_vector)
    print("Coordinate Tuples: ", coord_touples)
    print("Alphas List: ", alphas_list)
    print("Betas List: ", betas_list)
    print("W: ", w)
    print("Coordinate to Index Mapping: ", coord_to_index)
    print("Initial Vector v_init: ", v_init)

    tridiagonal_lanczos_matrix = tridiagonal_lanczos(len(alphas_list), alphas_list, betas_list)
    print("Tridiagonal Lanczos Matrix:\n", tridiagonal_lanczos_matrix)


# def main():
#     seeds = [(2,2)]
#     N = 1
#     coords_all, coord_to_index, Vs, all_maps, Ws, last_Q, As, Bs = block_lanczos(len(seeds), seeds, N)
#     print("Final Vs: ", Vs)
#     print("Final Ws: ", Ws)
#     print("Final last_Q: ", last_Q)


M2 = [
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
]


if __name__ == "__main__":
    # Keep your exact test behavior, just behind the main guard
    check_block_lanczos([(0, 0), (2, 0)], 4)
    eigenvals_M2 = np.linalg.eigvalsh(M2)
    print("Eigenvalues of M2:", eigenvals_M2)
