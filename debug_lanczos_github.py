import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

"""
Block Lanczos Implementation with Dictionary-based Initial Vectors
==================================================================

Supports arbitrary initial vectors specified as dictionaries mapping
coordinates to amplitudes. This allows for superposition states and
negative amplitudes.

Example usage:
    # Two initial vectors with custom amplitudes
    initial_vectors = [
        {(0, 0): 1.0, (1, 0): -1.0},   # Antisymmetric superposition
        {(3, 3): 1.0, (3, 4): 0.5},    # Another superposition
    ]
    results = block_lanczos(initial_vectors, N=10)
"""


# ============ Mapping Functions ============

def unified_mapping(initial_vectors: List[Dict[Tuple[int, int], float]], N: int, forbidden = None):
    """
    Creates unified coordinate mapping from initial vectors.
    Grows outward from ALL seed coordinates using BFS.
    
    Parameters:
    -----------
    initial_vectors : list of dicts
        Each dict maps (x, y) -> amplitude for one initial vector
    N : int
        Lattice size parameter: max_pts = 1 + 2*N*(N+1)
    
    Returns:
    --------
    coords_all : list of tuples
        Sorted list of all coordinates in the mapping
    coord_to_index : dict
        Maps (x, y) -> index in the coordinate list
    """
    # Collect all coordinates from initial vectors
    if forbidden is None:
        forbidden = set()
    else:
        forbidden = set(tuple(map(int, c)) for c in forbidden)
    
    seed_coords = set()
    for vec_dict in initial_vectors:
        for coord in vec_dict.keys():
            seed_coords.add(tuple(map(int, coord)))
    
    coord_set = set(seed_coords)
    coord_queue = list(seed_coords)
    
    max_pts = 1 + 2 * N * (N + 1)
    visited = set()
    i = 0
    
    # BFS expansion from all seeds
    while len(coord_set) < max_pts and i < len(coord_queue):
        x, y = coord_queue[i]
        
        if (x, y) in visited:
            i += 1
            continue
        
        neighbors = [(x+1, y), (x, y+1), (x-1, y), (x, y-1)]
        for nx, ny in neighbors:
            new_coord = (nx, ny)
            if new_coord in forbidden:
                continue
            if new_coord not in coord_set and len(coord_set) < max_pts:
                coord_set.add(new_coord)
                coord_queue.append(new_coord)
        
        visited.add((x, y))
        i += 1
    
    # Consistent ordering
    coords_all = sorted(coord_set, key=lambda t: (t[0], t[1]))
    coord_to_index = {c: i for i, c in enumerate(coords_all)}
    
    return coords_all, coord_to_index


def parse_initial_vectors(initial_vectors: List[Dict[Tuple[int, int], float]], 
                          coord_to_index: Dict, n: int) -> np.ndarray:
    """
    Convert list of amplitude dictionaries to Q0 matrix.
    
    Parameters:
    -----------
    initial_vectors : list of dicts
        Each dict maps (x, y) -> amplitude
    coord_to_index : dict
        Global coordinate to index mapping
    n : int
        Total number of lattice sites
    
    Returns:
    --------
    Q0 : ndarray of shape (n, num_vectors)
        Initial vectors as columns
    """
    num_vectors = len(initial_vectors)
    Q0 = np.zeros((n, num_vectors), dtype=float)
    
    for col, vec_dict in enumerate(initial_vectors):
        for coord, amplitude in vec_dict.items():
            coord = tuple(map(int, coord))
            if coord in coord_to_index:
                idx = coord_to_index[coord]
                Q0[idx, col] = amplitude
            else:
                print(f"Warning: coordinate {coord} not in mapping, skipping")
    
    return Q0


# ============ Hamiltonian Functions ============

def apply_H(v: np.ndarray, coord_to_index: Dict, coords_all: List) -> np.ndarray:
    """
    Apply tight-binding Hamiltonian (nearest-neighbor hopping) to vector v.
    H|i> = sum over neighbors j of |j>
    
    Parameters:
    -----------
    v : ndarray
        Input vector
    coord_to_index : dict
        Coordinate to index mapping
    coords_all : list
        List of all coordinates
    
    Returns:
    --------
    Av : ndarray
        H applied to v
    """
    v = np.asarray(v, dtype=float)
    Av = np.zeros_like(v)
    
    for idx, (x, y) in enumerate(coords_all):
        amp = v[idx]
        if amp == 0:
            continue
        for nx, ny in ((x+1, y), (x, y+1), (x-1, y), (x, y-1)):
            j = coord_to_index.get((nx, ny))
            if j is not None:
                Av[j] += amp
    return Av


def build_full_hamiltonian(coords_all: List, coord_to_index: Dict) -> np.ndarray:
    """
    Build the full Hamiltonian matrix for verification.
    
    Parameters:
    -----------
    coords_all : list
        List of all coordinates
    coord_to_index : dict
        Coordinate to index mapping
    
    Returns:
    --------
    H : ndarray
        Full Hamiltonian matrix
    """
    n = len(coords_all)
    H = np.zeros((n, n), dtype=float)
    
    for idx, (x, y) in enumerate(coords_all):
        for nx, ny in ((x+1, y), (x, y+1), (x-1, y), (x, y-1)):
            j = coord_to_index.get((nx, ny))
            if j is not None:
                H[idx, j] = 1.0
    return H



#Block Lanczos Algorithm 

def block_lanczos(initial_vectors: List[Dict[Tuple[int, int], float]], 
                  N: int, 
                  m_steps: Optional[int] = None, 
                  tol: float = 1e-10, 
                  normalize: bool = True,
                  verbose: bool = False,
                  forbidden=None) -> Dict:
    """
    Block Lanczos algorithm with dictionary-based initial vectors.
    
    Each initial vector is specified as a dictionary mapping coordinates
    to amplitudes, allowing for superposition states and negative amplitudes.
    
    Parameters:
    -----------
    initial_vectors : list of dicts
        Each dict maps (x, y) coordinates to amplitudes.
        Example: [{(0,0): 1.0, (1,0): -1.0}, {(3,3): 1.0}]
    N : int
        Lattice size parameter: max_pts = 1 + 2*N*(N+1)
    m_steps : int, optional
        Number of block Lanczos iterations (default: auto)
    tol : float
        Tolerance for rank detection
    normalize : bool
        Whether to normalize initial vectors before QR
    verbose : bool
        Print debug information
    
    Returns:
    --------
    dict with keys:
        'coords_all': list of all coordinates
        'coord_to_index': coordinate to index mapping
        'Q0': initial vectors matrix (before QR)
        'Qblocks': list of orthonormal Q blocks
        'As': list of diagonal blocks (A_k = Q_k^T H Q_k)
        'Bs': list of off-diagonal blocks
        'Ws': list of residual matrices
        'block_sizes': list of block sizes at each step
    """
    num_vectors = len(initial_vectors)
    
    # Step 1: Create unified mapping
    coords_all, coord_to_index = unified_mapping(initial_vectors, N, forbidden = forbidden)
    n = len(coords_all)
    
    if verbose:
        print(f"Lattice size: {n} sites")
        print(f"Number of initial vectors: {num_vectors}")
        for i, vec_dict in enumerate(initial_vectors):
            print(f"  Vector {i+1}: {vec_dict}")
    
    # Step 2: Parse initial vectors into Q0 matrix
    Q0 = parse_initial_vectors(initial_vectors, coord_to_index, n)
    
    # Step 3: Optionally normalize each vector
    if normalize:
        for j in range(Q0.shape[1]):
            norm = np.linalg.norm(Q0[:, j])
            if norm > tol:
                Q0[:, j] /= norm
                if verbose:
                    print(f"  Vector {j+1} normalized (original norm: {norm:.6f})")
    
    # Step 4: Orthonormalize via QR
    Q, R = np.linalg.qr(Q0, mode='reduced')
    
    # Check rank and remove linearly dependent vectors
    diag_R = np.abs(np.diag(R))
    initial_rank = np.sum(diag_R > tol)
    
    if verbose:
        print(f"QR decomposition: rank = {initial_rank}/{num_vectors}")
        print(f"R diagonal: {diag_R}")
    
    if initial_rank < num_vectors:
        print(f"Warning: Initial vectors are linearly dependent!")
        print(f"  Effective rank: {initial_rank}/{num_vectors}")
        keep_cols = diag_R > tol
        Q = Q[:, keep_cols]
    
    # Step 5: Set up iteration
    if m_steps is None:
        m_steps = min(N, n // max(1, Q.shape[1]))
    
    Qblocks = [Q.copy()]
    As = []
    Bs = []
    Ws = []
    block_sizes = [Q.shape[1]]
    
    Q_prev = None
    B_prev = None
    
    if verbose:
        print(f"\nRunning {m_steps} Block Lanczos iterations...")
        print("-" * 50)
    
    # Step 6: Block Lanczos iteration
    for step in range(m_steps):
        Q_curr = Qblocks[-1]
        b_curr = Q_curr.shape[1]
        
        if b_curr == 0:
            if verbose:
                print(f"Step {step}: Block size = 0, stopping")
            break
        
        # Apply H to each column of current Q block
        HQ = np.zeros((n, b_curr), dtype=float)
        for j in range(b_curr):
            HQ[:, j] = apply_H(Q_curr[:, j], coord_to_index, coords_all)
        
        # W = H*Q_k - Q_{k-1} * B_{k-1}^T
        if Q_prev is None:
            W = HQ.copy()
        else:
            W = HQ - Q_prev @ B_prev.T
        
        # A_k = Q_k^T * W (projection onto current block)
        A = Q_curr.T @ W
        As.append(A)
        
        # W = W - Q_k * A_k (remove component in current block)
        W = W - Q_curr @ A
        
        # Full reorthogonalization against ALL previous blocks
        for Qj in Qblocks:
            overlap = Qj.T @ W
            W = W - Qj @ overlap
        
        Ws.append(W.copy())
        
        # QR decomposition to get next orthonormal block
        Q_next, R_next = np.linalg.qr(W, mode='reduced')
        
        # Check for rank deficiency
        diag = np.abs(np.diag(R_next))
        rank = np.sum(diag > tol)
        
        if verbose:
            print(f"Step {step+1}:")
            print(f"  A shape: {A.shape}")
            print(f"  A =\n{A}")
            print(f"  Next rank: {rank}")
        
        if rank == 0:
            if verbose:
                print(f"  Invariant subspace reached!")
            break
        
        # Truncate if rank-deficient
        if rank < Q_next.shape[1]:
            keep = diag > tol
            Q_next = Q_next[:, keep]
            R_next = R_next[keep, :]
            if verbose:
                print(f"  Rank deficiency: keeping {rank} vectors")
        
        B_prev = R_next
        Bs.append(B_prev)
        block_sizes.append(Q_next.shape[1])
        
        if verbose:
            print(f"  B shape: {B_prev.shape}")
            print(f"  B =\n{B_prev}")
        
        Q_prev = Q_curr
        Qblocks.append(Q_next)
    
    if verbose:
        print("-" * 50)
        print(f"Block Lanczos complete:")
        print(f"  Total A blocks: {len(As)}")
        print(f"  Total B blocks: {len(Bs)}")
        print(f"  Block sizes: {block_sizes[:len(As)]}")
    
    return {
        'coords_all': coords_all,
        'coord_to_index': coord_to_index,
        'Q0': Q0,
        'Qblocks': Qblocks,
        'As': As,
        'Bs': Bs,
        'Ws': Ws,
        'block_sizes': block_sizes
    }


def single_seed_lanczos(seed_coord: Tuple[int, int], 
                        N: int, 
                        m_steps: Optional[int] = None,
                        tol: float = 1e-10) -> Dict:
    """
    Standard (non-block) Lanczos for a single seed point.
    Convenience wrapper around block_lanczos.
    
    Parameters:
    -----------
    seed_coord : tuple
        (x, y) coordinate of the seed
    N : int
        Lattice size parameter
    m_steps : int, optional
        Number of Lanczos iterations
    tol : float
        Tolerance for convergence
    
    Returns:
    --------
    dict with additional keys 'alphas' and 'betas' (scalar values)
    """
    initial_vectors = [{seed_coord: 1.0}]
    results = block_lanczos(initial_vectors, N, m_steps, tol)
    
    # Extract scalar alphas and betas from 1x1 blocks
    results['alphas'] = np.array([A[0, 0] for A in results['As']])
    results['betas'] = np.array([B[0, 0] for B in results['Bs']]) if results['Bs'] else np.array([])
    
    return results


# ============ Block Tridiagonal Matrix ============

def build_block_tridiagonal(As: List[np.ndarray], 
                            Bs: List[np.ndarray], 
                            block_sizes: Optional[List[int]] = None) -> np.ndarray:
    """
    Build the full block-tridiagonal matrix T from A and B blocks.
    
    Structure:
        T = | A_0   B_0^T   0      0    ... |
            | B_0   A_1     B_1^T  0    ... |
            | 0     B_1     A_2    B_2^T... |
            | ...                           |
    
    Parameters:
    -----------
    As : list of ndarray
        Diagonal blocks
    Bs : list of ndarray
        Off-diagonal blocks
    block_sizes : list of int, optional
        Block sizes (inferred from As if not provided)
    
    Returns:
    --------
    T : ndarray
        Block tridiagonal matrix
    """
    if len(As) == 0:
        return np.array([[]])
    
    # Infer block sizes from A blocks if not provided
    if block_sizes is None:
        block_sizes = [A.shape[0] for A in As]
    else:
        block_sizes = block_sizes[:len(As)]
    
    # Compute offsets
    offsets = [0]
    for bs in block_sizes:
        offsets.append(offsets[-1] + bs)
    
    total_dim = offsets[-1]
    T = np.zeros((total_dim, total_dim), dtype=float)
    
    # Fill diagonal A blocks
    for i, A in enumerate(As):
        r0, r1 = offsets[i], offsets[i+1]
        T[r0:r1, r0:r1] = A
    
    # Fill off-diagonal B blocks
    for i, B in enumerate(Bs):
        if i + 1 >= len(offsets):
            continue
        
        r0_curr = offsets[i]
        r1_curr = offsets[i+1]
        r0_next = offsets[i+1]
        r1_next = offsets[i+2] if i+2 < len(offsets) else offsets[-1]
        
        bs_curr = r1_curr - r0_curr
        bs_next = r1_next - r0_next
        
        # Handle potential shape mismatches
        rows_to_use = min(B.shape[0], bs_next)
        cols_to_use = min(B.shape[1], bs_curr)
        B_use = B[:rows_to_use, :cols_to_use]
        
        # B below diagonal
        T[r0_next:r0_next+rows_to_use, r0_curr:r0_curr+cols_to_use] = B_use
        # B^T above diagonal
        T[r0_curr:r0_curr+cols_to_use, r0_next:r0_next+rows_to_use] = B_use.T
    
    return T


# ============ Verification Functions ============

def verify_lanczos_relation(results: Dict) -> Dict:
    """
    Verify that T = Q^T H Q (the fundamental Lanczos relation).
    
    Parameters:
    -----------
    results : dict
        Output from block_lanczos
    
    Returns:
    --------
    dict with verification metrics
    """
    coords_all = results['coords_all']
    coord_to_index = results['coord_to_index']
    Qblocks = results['Qblocks']
    As = results['As']
    Bs = results['Bs']
    block_sizes = results['block_sizes']
    
    # Build full matrices
    H = build_full_hamiltonian(coords_all, coord_to_index)
    Q_full = np.hstack(Qblocks[:len(As)])
    T = build_block_tridiagonal(As, Bs, block_sizes)
    
    if T.size == 0:
        return {'error': 'T is empty'}
    
    # Compute Q^T H Q
    H_proj = Q_full.T @ H @ Q_full
    
    # Metrics
    metrics = {}
    metrics['T_shape'] = T.shape
    metrics['H_proj_shape'] = H_proj.shape
    metrics['Q_full_shape'] = Q_full.shape
    
    # Symmetry check
    metrics['T_symmetry_error'] = np.max(np.abs(T - T.T))
    
    # Orthonormality check
    QtQ = Q_full.T @ Q_full
    metrics['orthonormality_error'] = np.max(np.abs(QtQ - np.eye(QtQ.shape[0])))
    
    # Lanczos relation check
    if T.shape == H_proj.shape:
        metrics['lanczos_relation_error'] = np.max(np.abs(T - H_proj))
        
        # Eigenvalue comparison
        T_eigs = np.sort(np.linalg.eigvalsh(T))
        H_proj_eigs = np.sort(np.linalg.eigvalsh(H_proj))
        metrics['eigenvalue_error'] = np.max(np.abs(T_eigs - H_proj_eigs))
        metrics['T_eigenvalues'] = T_eigs
        metrics['H_proj_eigenvalues'] = H_proj_eigs
    else:
        metrics['lanczos_relation_error'] = float('inf')
        metrics['shape_mismatch'] = True
    
    # Full H eigenvalues for comparison
    H_eigs = np.sort(np.linalg.eigvalsh(H))
    metrics['H_eigenvalues'] = H_eigs
    metrics['H_eig_range'] = (H_eigs[0], H_eigs[-1])
    
    return metrics


# ============ Visualization Functions ============

def plot_vector_on_lattice(coords: List, vec: np.ndarray, 
                           title: str = "Vector on Lattice", 
                           cmap: str = 'RdBu_r',
                           show_sign: bool = True):
    """Plot a vector as a heatmap on the lattice."""
    coords = np.asarray(coords, dtype=int)
    coords = coords - coords.min(axis=0)
    W = coords[:, 0].max() + 1
    H = coords[:, 1].max() + 1
    
    grid = np.full((H, W), np.nan)
    
    if show_sign:
        for (x, y), a in zip(coords, vec):
            grid[y, x] = a
        vmax = np.nanmax(np.abs(grid))
        vmin = -vmax
    else:
        for (x, y), a in zip(coords, np.abs(vec)):
            grid[y, x] = a
        vmax = np.nanmax(grid)
        vmin = 0
    
    plt.figure(figsize=(8, 6))
    plt.imshow(grid, origin='lower', interpolation='nearest', 
               cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label='Amplitude')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()


def plot_initial_vectors(results: Dict, initial_vectors: List[Dict]):
    """Plot initial vectors on the lattice."""
    coords_all = results['coords_all']
    Q0 = results['Q0']
    
    coords = np.asarray(coords_all, dtype=int)
    coords = coords - coords.min(axis=0)
    W = coords[:, 0].max() + 1
    H = coords[:, 1].max() + 1
    
    num_vecs = Q0.shape[1]
    fig, axes = plt.subplots(1, num_vecs, figsize=(5*num_vecs, 4))
    if num_vecs == 1:
        axes = [axes]
    
    for j, ax in enumerate(axes):
        grid = np.full((H, W), np.nan)
        vec = Q0[:, j]
        
        for (x, y), amp in zip(coords, vec):
            grid[y, x] = amp
        
        vmax = np.nanmax(np.abs(grid))
        if vmax == 0:
            vmax = 1
        im = ax.imshow(grid, origin='lower', cmap='RdBu_r', 
                       vmin=-vmax, vmax=vmax, interpolation='nearest')
        plt.colorbar(im, ax=ax, label='Amplitude')
        ax.set_title(f'Vector {j+1}\n{initial_vectors[j]}')
    
    plt.tight_layout()
    plt.show()


def plot_Qblock_heatmaps(coords: List, Qblock: np.ndarray, 
                         step: Optional[int] = None, 
                         cmap: str = 'RdBu_r'):
    """Plot each column of a Q block as a separate heatmap."""
    coords = np.asarray(coords, dtype=int)
    coords = coords - coords.min(axis=0)
    W = coords[:, 0].max() + 1
    H = coords[:, 1].max() + 1
    
    num_cols = Qblock.shape[1]
    fig, axes = plt.subplots(1, num_cols, figsize=(4*num_cols, 4))
    if num_cols == 1:
        axes = [axes]
    
    for j, ax in enumerate(axes):
        grid = np.full((H, W), np.nan)
        vec = Qblock[:, j]
        
        for (x, y), a in zip(coords, vec):
            grid[y, x] = a
        
        vmax = np.nanmax(np.abs(grid))
        if vmax == 0:
            vmax = 1
        im = ax.imshow(grid, origin='lower', interpolation='nearest', 
                       cmap=cmap, vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Amplitude')
        title = f"Q column {j+1}"
        if step is not None:
            title = f"Step {step}, " + title
        ax.set_title(title)
    
    plt.tight_layout()
    plt.show()


def plot_T_structure(T: np.ndarray, block_sizes: List[int]):
    """Visualize block tridiagonal matrix."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Full matrix
    ax = axes[0]
    vmax = np.max(np.abs(T)) if T.size > 0 else 1
    im = ax.imshow(T, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_title('Block Tridiagonal Matrix T')
    
    # Draw block boundaries
    offset = 0
    for bs in block_sizes[:-1]:
        offset += bs
        ax.axhline(offset - 0.5, color='k', linewidth=0.5)
        ax.axvline(offset - 0.5, color='k', linewidth=0.5)
    
    # Sparsity pattern
    ax = axes[1]
    ax.spy(np.abs(T) > 1e-10, markersize=3)
    ax.set_title('Sparsity Pattern')
    
    plt.tight_layout()
    plt.show()




# ============ Example Usage ============

if __name__ == "__main__":
    print("=" * 60)
    print("EXAMPLE 1: Two single-site seeds")
    print("=" * 60)
    
    initial_vectors_1 = [
        {(0, 0): 1.0},
        {(3, 3): 1.0},
    ]
    
    results1 = block_lanczos(initial_vectors_1, N=8, m_steps=10, verbose=True)
    
    print("\nVerification:")
    metrics1 = verify_lanczos_relation(results1)
    print(f"  T = Q^T H Q error: {metrics1['lanczos_relation_error']:.2e}")
    print(f"  Orthonormality error: {metrics1['orthonormality_error']:.2e}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Antisymmetric superposition")
    print("=" * 60)
    
    initial_vectors_2 = [
        {(0, 0): 1.0, (1, 0): -1.0},   # Antisymmetric
        {(0, 0): 1.0, (1, 0): 1.0},    # Symmetric
    ]
    
    results2 = block_lanczos(initial_vectors_2, N=8, m_steps=10, verbose=True)
    
    print("\nVerification:")
    metrics2 = verify_lanczos_relation(results2)
    print(f"  T = Q^T H Q error: {metrics2['lanczos_relation_error']:.2e}")
    print(f"  Orthonormality error: {metrics2['orthonormality_error']:.2e}")
    
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Complex multi-site states")
    print("=" * 60)
    
    initial_vectors_3 = [
        {(0, 0): 1.0, (1, 0): -1.0, (0, 1): 1.0},
        {(2, 2): 1.0, (3, 2): -1.0, (2, 3): 1.0},
        {(5, 5): 1.0},
    ]
    
    results3 = block_lanczos(initial_vectors_3, N=10, m_steps=12, verbose=True)
    
    print("\nVerification:")
    metrics3 = verify_lanczos_relation(results3)
    print(f"  T = Q^T H Q error: {metrics3['lanczos_relation_error']:.2e}")
    print(f"  Orthonormality error: {metrics3['orthonormality_error']:.2e}")
    
    # Visualizations
    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)
    
    plot_initial_vectors(results2, initial_vectors_2)
    
    T = build_block_tridiagonal(results2['As'], results2['Bs'], results2['block_sizes'])
    if T.size > 0:
        plot_T_structure(T, results2['block_sizes'][:len(results2['As'])])
    
    plot_T_structure(T, results2['block_sizes'][:len(results2['As'])])
    plot_Qblock_heatmaps(results2['coords_all'], results2['Qblocks'][0], step=0)
