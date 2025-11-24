from py_compile import main
from matplotlib.pyplot import title
import numpy as np 
import math
import sympy as sp 
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from collections import defaultdict
import scipy.sparse as sps

"""
Feiguin-Lanczos mapping and block lanczos implementation for multiple seed points on a
2D lattice. Starts from user-defined seed points and grows the mapping outwards, generating a 
Hamiltonian application function. Then performs block Lanczos/Lanczos iterations to obtain orthogonal vectors
spanning the subspace defined by the initial seed points. Block Lanczos is used to handle multiple seed points (impurities) simultaneously.
Lanczos function is used for single seed point cases with 1 impurity on the lattice. 

There is the option to visualize the final Q matrix and W matrices as heatmaps on the lattice. W matrices represent the residuals at each step of the Lanczos process.
Q matrices are the orthogonal basis vectors generated during the Lanczos iterations.

"""

#mapping function:
def mapping(coord_touples, N, tol = 1e-10):

    #initializing all of the variables
    #create the mapping from coordinates to indices
    column_vector = [1]
    coord_to_index = {coord_touples[0]: 0} #{(0,0):0, (1,0):1, (0,1):2, (-1,0):3, (0,-1):4} etc.
    seen = set()
    beta = 0
    w = 0 
    n = 1 + 2 * N * (N + 1)  # max number of points in the mapping
    betas_list = []
    alphas_list = []    
    print(n)

    #max points is based on the formula for number of sites in a square lattice up to N hops (even iterations)
    #i.e n= 1 -> 4 neighbors, then n = 2 -> all neighbors of those 4 neighbors, etc. 

    max_pts = 1 + 2 * N * (N + 1)
    i = 0

    # mapping loop
    while len(column_vector) < max_pts:

        x, y = coord_touples[i]

        if i in seen: 
            i += 1
            continue  # skip already processed points

        # generate neighbors
        neighbors = [(x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)]

        for nx, ny in neighbors:
            new_coord = (nx, ny)
            if new_coord in coord_to_index:
                # already seen: increment the corresponding index
                idx = coord_to_index[new_coord]
                column_vector[idx] += 1
                print("Index: ", idx)
            else:
                # new coordinate: append to everything
                coord_touples.append(new_coord)
                coord_to_index[new_coord] = len(column_vector)
                column_vector.append(1) 

        seen.add(i)
        i += 1

    v = np.asarray(column_vector, dtype=float)
    v_prev = np.zeros_like(v)

    # After the mapping loop finishes and coord_touples is complete:
    num_sites = len(coord_touples)
    column_vector = np.zeros(num_sites, dtype=float)
    column_vector[0] = 1.0         # seed at index 0 only
    v = column_vector.copy()
    v_prev = np.zeros_like(v)
    v /= np.linalg.norm(v)
    v_init = v.copy()


    def apply_H(colvec, coord_to_idx): 

        Av = np.zeros_like(colvec)
        for idx, (x, y) in enumerate(coord_touples):

            if idx > len(colvec) - 1:
                break
            amplitude = colvec[idx]

            if amplitude == 0:
                continue  # skip zero amplitudes

            for nx, ny in ((x+1,y), (x, y + 1), (x-1,y), (x,y-1)):
            # only add if that neighbor exists in the map
                if (nx, ny) in coord_to_idx:
                    j = coord_to_idx[(nx, ny)]
                    Av[j] += amplitude

        return Av
    
    m_steps = min(N, len(v))  # sensible cap
    Vs = []

    for step in range(m_steps):

        Av = apply_H(v, coord_to_index)
        
        vv = np.dot(v, v)
        if vv == 0:
            raise ValueError("v vector is zero during iteration")

        alpha = float(np.dot(v, Av) / vv)
        alphas_list.append(alpha)

        if step == 0: 
            w = Av - alpha * v
            beta = np.linalg.norm(w)
        else:
            w = Av - (alpha * v) - (beta * v_prev)
            beta = np.linalg.norm(w)

        for u in Vs:
            proj = np.dot(u, w)
            w -= proj * u

        if beta < tol or beta == 0.0:
                # cannot compute b_n^2; stop
                break
        betas_list.append(beta)
        # rotate: v_prev <- v, v <- v_next
        v_prev, v = v, w / beta
        Vs.append(v_prev)

    return np.array(column_vector), coord_touples, np.array(alphas_list), np.array(betas_list), w, coord_to_index, np.array(v_init)

def tridiagonal_lanczos(num_steps, alphas, betas):
    T = np.zeros((num_steps, num_steps), dtype=float)
    for i in range(num_steps):
        T[i, i] = alphas[i]
        if i < num_steps - 1:
            T[i, i + 1] = betas[i]
            T[i + 1, i] = betas[i]
    return T

#block lanczos function that uses the mapping function above (only its mapping capability) to generate the hamiltonian application function
def block_lanczos(num_seeds, init_coord_touples, N):

    #recursive mapping for all seed points
    coords_set = set()
    coord_to_index = {}
    #going through each seed point to create mapping
    for i in range(num_seeds):
       colvec, coords, alphas, betas, w, coord_to_index, v_init = mapping(([init_coord_touples[i]]), N)
       coords_set.update(coords)
       print(f"Mapping for seed point {i+1}:")
    
    #global list of coordinates and index map
    coords_all = sorted(coords_set, key=lambda t: (t[0], t[1]))   # global ordered list
    coord_to_index = {c: i for i, c in enumerate(coords_all)}     # the ONLY index map
    
    Vs = []
    for i in range(num_seeds):
        v = np.zeros(len(coords_all), dtype=float)
        sx, sy = init_coord_touples[i]
        idx = coord_to_index[(int(sx), int(sy))]
        key = (int(sx), int(sy))
        v[coord_to_index[key]] = 1.0  # initial vector for this seed
        # optionally normalize if you want:
        # v /= np.linalg.norm(v)
        Vs.append(v)
        print("Vs list updated: ", Vs)
        print("Vs list updated: ", Vs)
        print(f"DEBUG: num_seeds = {num_seeds}")
        print(f"DEBUG: len(Vs) = {len(Vs)}")
        print(f"DEBUG: len(coords_all) = {len(coords_all)}")

    #generating the hamiltonian application function
    def apply_H(v, coord_to_index, coords_all):

        v = np.asarray(v)
        Av = np.zeros_like(v)
        print("Applying H to vector: ", v)
        for idx, (x, y) in enumerate(coords_all):
            
            amplitude = v[idx]
            print(amplitude)
            if amplitude == 0:
                continue  # skip zero amplitudes
            
            #for loop that iterates through neighbors to check if they exist in the mapping
            for nx, ny in ((x+1,y), (x, y + 1), (x-1,y), (x,y-1)):
            # only add if that neighbor exists in the map
                j = coord_to_index.get((nx, ny))
                if j is not None:
                    Av[j] += amplitude


        return Av
    
    m_steps = N  # sensible cap

    n = len(coords_all)
      # initial Q blocks are just the seed vectors
    Q = np.column_stack(Vs).astype(float)
    print(f"DEBUG: Q.shape before QR = {Q.shape}")
    Q, _ = np.linalg.qr(Q, mode='reduced')
    print(f"DEBUG: Q.shape after QR = {Q.shape}")
    print(f"DEBUG: First A will be shape ({Q.shape[1]}, {Q.shape[1]})")
    last_Q = Q.copy() 
    Qblocks = [Q.copy()]
    Ws = [] 
    As = []
    Bs = []
    W = None

    Q_prev = None
    B_prev = None

    dim = len(coords_all)

    # choose some factor c >= 1; c = 2 is a decent heuristic
    c = 1
    m_steps = min(c * N, dim)  

    for i in range(m_steps):

        """
        Computing block lanczos steps using the apply H function to each vector in Vs, which is the initial state
        for each seed. Then we compute the b x b Hamiltonian block for this step. The matrix for alphas is built 
        in the following manner M = < v_i | H | v_j > for i,j in [1,b] where b is the number of seeds. X = < v_i | v_j >
        A_n is then given by MX^{-1}

        Each seed vector is orthogonalized against all previous seed vectors to ensure orthogonality.
        The formula for each state update is:
        W_{n+1} = H v_n - A_n v_n - B_n v_{n-1}.... and it gets longer if you have more seeds.
        """


        #generating the Q array, array of all Vectors at current step for each seed
        Q = Qblocks[-1]
        # Transpose Q for later use
        print(f"DEBUG: Q.shape at step {i+1} = {Q.shape}")
        

        #HQ is the hamiltonian applied to each vector in Q
        HQ = np.zeros_like(Q)  # to store H applied to each vector
        for j in range(Q.shape[1]):
            HQ[:, j] = apply_H(Q[:, j], coord_to_index, coords_all)
            print(f"H at point {j + 1}: ", HQ[:, j])

        B_prevt = B_prev.T if B_prev is not None else None
        if Q_prev is None: 
            W = HQ.copy()
        else:
            W = HQ - Q_prev @ B_prevt
        

        QT = Q.T
        #Compute the block matrices A and B
        A = QT @ W
        As.append(A)
        
        W = W - Q @ A

        if len(Qblocks) > 2:
            for qs in Qblocks[:-2]:
                W = W - qs @ (qs.T @ W)
                print("Qs:", qs)
                print("QBLOCKS:", Qblocks[-1])
        Ws.append(W.copy())
        
        
        Q_next, R_next = np.linalg.qr(W, mode='reduced')


        if Q_next.shape[1] == 0:
            break

        B_prev = R_next
        Bs.append(B_prev)
        last_Q = Q_next.copy()
        Q_prev = Q
        Qblocks.append(Q_next)
    # for i in range(Q.shape[1]):
    #     Q[:, i] = Q[:, i]  # flip sign for consistency

    final_block = np.array(last_Q)
    final_vectors = [final_block[:, c] for c in range(final_block.shape[1])]

    # map amplitudes to coords for convenience
    all_maps = [{coords_all[i]: final_block[i, j] for i in range(n)}
            for j in range(final_block.shape[1])]
    
    for j, v in enumerate(final_vectors, 1):
        print(f"  column {j}: ||v||={np.linalg.norm(v):.6f}")

    
    return coords_all, coord_to_index, np.column_stack(Vs), all_maps, Ws, last_Q, As, Bs, Qblocks



#plots the last Q matrix as a heatmap on the lattice
def plot_last_Q_heatmap(coords, last_Q, title="Last Q on lattice"):
    coords = np.asarray(coords, dtype=int)
    # shift to positive quadrant
    coords = coords - coords.min(axis=0)
    W = coords[:,0].max() + 1
    H = coords[:,1].max() + 1
    grid = np.zeros((H, W), dtype=float)
    amp = np.abs(last_Q[:,0])  # first vector
    for (x,y), a in zip(coords, amp):
        grid[y, x] = a
    plt.imshow(grid, origin='lower', interpolation='nearest', cmap = 'inferno')
    plt.title(title); plt.xticks([]); plt.yticks([])
    plt.colorbar(label='|Amplitude|')
    plt.tight_layout(); plt.show()

#plots W matrices (residuals) as heatmaps for each lanczos step
def plot_Ws_heatmap(coords, Ws, title="|W| on lattice"):
    coords = np.asarray(coords, dtype=int)
    # shift to positive quadrant
    coords = coords - coords.min(axis=0)
    W = coords[:,0].max() + 1
    H = coords[:,1].max() + 1
    
    for step, W_mat in enumerate(Ws):
        grid = np.zeros((H, W), dtype=float)

        # Sum the state vectors FIRST (coherent superposition)
        superposition = W_mat.sum(axis=1)  # shape: (num_sites,)
        amp = np.abs(superposition)  # THEN take absolute value
        
        for (x, y), a in zip(coords, amp):
            grid[y, x] = a

        im = plt.imshow(grid, origin='lower', interpolation='nearest', cmap='inferno')
        plt.colorbar(im, label='|Amplitude|')
        plt.title(f"{title} at step {step}")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()

def plot_last_W_heatmap(coords, W, title="Last W on lattice"):
    coords = np.asarray(coords, dtype=int)
    # shift to positive quadrant
    coords = coords - coords.min(axis=0)
    W_dim = coords[:,0].max() + 1
    H_dim = coords[:,1].max() + 1
    grid = np.zeros((H_dim, W_dim), dtype=float)
    amp = np.abs(W)  # last W matrix
    for (x,y), a in zip(coords, amp):
        grid[y, x] = a
    plt.imshow(grid, origin='lower', interpolation='nearest', cmap='inferno')
    plt.title(title); plt.xticks([]); plt.yticks([])
    plt.colorbar(label='|Amplitude|')
    plt.tight_layout(); plt.show()

#plots lanczos iteration for one seed
def lanczos_heatmap(coords, w, title="Lanczos Vector Heatmap"):
    coords = np.asarray(coords, dtype=int)
    # shift to positive quadrant
    coords = coords - coords.min(axis=0)
    W = coords[:,0].max() + 1
    H = coords[:,1].max() + 1
    grid = np.zeros((H, W), dtype=float)
    amp = np.abs(w)  # lanczos vector
    for (x,y), a in zip(coords, amp):
        grid[y, x] = a
    plt.imshow(grid, origin='lower', interpolation='nearest', cmap='inferno')
    plt.title(title); plt.xticks([]); plt.yticks([])
    plt.colorbar(label='|Amplitude|')
    plt.tight_layout(); plt.show()









    



































	
	
	





	
	

	



























































