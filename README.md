# Lattice - Lanczos 

A Python implementation of **Lanczos** and **Block Lanczos** algorithms to describe the states of electrons within a substrate in different bases following the method of Allerdt & Feiguin *A Numerically Exact Approach to Quantum Impurity Problems in Realistic Lattice Geometries*.
This repository generates the following:
- Adjacency-based Hamiltonian mapped to 2D coordinate system off of one (lanczos) or multiple (block lanczos) seeds
- Applies block lanczos for multi seed (impurity) systems
- Built-in visualization of lanczos residuals W, and mapped lattice amplitudes

The lanczos transformation recursively generates new orthogonal bases that expand radially outward from one or more "seeds" within a substrate. 
For one "seed" the basis forms a 1D chain with coefficients: 

$$
a_n = \frac{\langle n | H | n \rangle}{\langle n | n \rangle}, \qquad b_n = \frac{\langle n|n\rangle}{\langle n-1 | n-1 \rangle}
$$

exacty as in Feiguins eq. (7) - (8). 

For **multiple impurities**, the block Lanczos method generates a **ladder geometry** with **2×2 (or k×k) blocks**, per Feiguin’s Eq. (23)–(30): 

$$
|\alpha_{n+1}\rangle = H |\alpha_n\rangle - a^{\alpha\alpha}_n |\alpha_n\rangle - a^{\alpha\beta}_n |\beta_n\rangle - b^{\alpha\alpha}_n |\alpha_{n-1}\rangle - b^{\alpha\beta}_n |\beta_{n-1}\rangle
$$

$$
|\beta_{n+1}\rangle = H |\beta_n\rangle - a^{\beta\beta}_n |\beta_n\rangle - a^{\beta\alpha}_n |\alpha_n\rangle - b^{\beta\beta}_n |\beta_{n-1}\rangle - b^{\beta\alpha}_n |\alpha_{n-1}\rangle
$$

Where $\alpha$ and $\beta$ are different seeds at different points within the lattice. To solve for the a's and b's, one must use the following matrix formulas : 

```math
\begin{pmatrix}
\langle \alpha_n | H | \alpha_n \rangle & \langle \beta_n | H | \alpha_n \rangle \\
\langle \alpha_n | H | \beta_n \rangle & \langle \beta_n | H | \beta_n \rangle
\end{pmatrix}
=
\begin{pmatrix}
a_n^{\alpha\alpha} & a_n^{\alpha\beta} \\
a_n^{\beta\alpha} & a_n^{\beta\beta}
\end{pmatrix}
\begin{pmatrix}
\langle \alpha_n | \alpha_n \rangle & \langle \beta_n | \alpha_n \rangle \\
\langle \alpha_n | \beta_n \rangle & \langle \beta_n | \beta_n \rangle
\end{pmatrix}
```

and 

```math
\begin{pmatrix}
\langle \alpha_{n-1} | H | \alpha_n \rangle & \langle \beta_{n-1} | H | \alpha_n \rangle \\
\langle \alpha_{n-1} | H | \beta_n \rangle & \langle \beta_{n-1} | H | \beta_n \rangle
\end{pmatrix}
=
\begin{pmatrix}
b_n^{\alpha\alpha} & b_n^{\alpha\beta} \\
b_n^{\beta\alpha} & b_n^{\beta\beta}
\end{pmatrix}
\begin{pmatrix}
\langle \alpha_{n-1} | \alpha_{n-1} \rangle & \langle \beta_{n-1} | \alpha_{n-1} \rangle \\
\langle \alpha_{n-1} | \beta_{n-1} \rangle & \langle \beta_{n-1} | \beta_{n-1} \rangle
\end{pmatrix}
```
The code alters the formula such that the a's and b's are: 

```math
\begin{pmatrix}
a_n^{\alpha\alpha} & a_n^{\alpha\beta} \\
a_n^{\beta\alpha} & a_n^{\beta\beta}
\end{pmatrix}
= 
\begin{pmatrix}
\langle \alpha_n | H | \alpha_n \rangle & \langle \beta_n | H | \alpha_n \rangle \\
\langle \alpha_n | H | \beta_n \rangle & \langle \beta_n | H | \beta_n \rangle
\end{pmatrix}
\begin{pmatrix}
\langle \alpha_n | \alpha_n \rangle & \langle \beta_n | \alpha_n \rangle \\
\langle \alpha_n | \beta_n \rangle & \langle \beta_n | \beta_n \rangle
\end{pmatrix}^{-1}
```

**Compute the b-coefficients** (coupling to previous iteration):
```math
\begin{pmatrix}
b_n^{\alpha\alpha} & b_n^{\alpha\beta} \\
b_n^{\beta\alpha} & b_n^{\beta\beta}
\end{pmatrix}
= 
\begin{pmatrix}
\langle \alpha_{n-1} | H | \alpha_n \rangle & \langle \beta_{n-1} | H | \alpha_n \rangle \\
\langle \alpha_{n-1} | H | \beta_n \rangle & \langle \beta_{n-1} | H | \beta_n \rangle
\end{pmatrix}
\begin{pmatrix}
\langle \alpha_{n-1} | \alpha_{n-1} \rangle & \langle \beta_{n-1} | \alpha_{n-1} \rangle \\
\langle \alpha_{n-1} | \beta_{n-1} \rangle & \langle \beta_{n-1} | \beta_{n-1} \rangle
\end{pmatrix}^{-1}
```

## Code to Math Translation 

### Hamiltonian generation
The hamiltonian is generated and mapped to a 2D coordinate system such that one is able to retrieve the coordinates and input coordinates for a seed to get the states for N iterations.
For example, N = 1 means that the Hamiltonian has ones (amplitudes) corresponding to the sites that are direct "neighbors" to the seed. N = 2 finds the neighbors of the seeds neighbors, 
and so on. The amplitudes of the coordinates are then assigned within a dictionary. The neighbors are: 

```python
neighbors = [(x+1, y), (x, y+1), (x-1, y), (x, y-1)]
```

This generates a coordinate-to-index mapping:
```python
coord_to_index = {(0,0): 0, (1,0): 1, (0,1): 2, (-1,0): 3, ...}
```

The maximum number of sites is:
```math
n = 1 + 2N(N+1)
```
The `apply_H()` function implements the hamiltonian for the "neighbor" hopping:
```math
H|\psi\rangle = \sum_{\langle i,j \rangle} (|i\rangle\langle j| + |j\rangle\langle i|)
```

For each site, the amplitude spreads to its four nearest neighbors:
```python
Av[neighbor_idx] += amplitude[site_idx]
```

The standard Lanczos algorithm computes the tridiagonal matrix representation for one seed (block lanczos for multiple seeds which is described further below):

```math
\alpha_n = \frac{\langle v_n | H | v_n \rangle}{\langle v_n | v_n \rangle}
```
```math
\beta_n = ||H|v_n\rangle - \alpha_n |v_n\rangle - \beta_{n-1} |v_{n-1}\rangle||
```


### Variables for block laczos: 
- `Q` = Current block of vectors: $[|\alpha_n\rangle, |\beta_n\rangle]$ (as columns)
- `Q_prev` = Previous block: $[|\alpha_{n-1}\rangle, |\beta_{n-1}\rangle]$
- `HQ` = Hamiltonian applied to current block: $[H|\alpha_n\rangle, H|\beta_n\rangle]$
- `A` = Current coefficient matrix (the **a-coefficients**)
- `B_prev` = Previous coefficient matrix (the **b-coefficients** from last iteration)
- `W` = Working vector during orthogonalization

### Step-by-Step Correspondence

**1. Apply Hamiltonian to current block:**
```python
HQ[:, j] = apply_H(Q[:, j], ...)
```
This computes:
```math
HQ = [H|\alpha_n\rangle, H|\beta_n\rangle]
```

**2. Subtract previous block contribution:**
```python
W = HQ - Q_prev @ B_prevt
```
This performs:
```math
W = [H|\alpha_n\rangle, H|\beta_n\rangle] - [|\alpha_{n-1}\rangle, |\beta_{n-1}\rangle] 
\begin{pmatrix}
b_n^{\alpha\alpha} & b_n^{\beta\alpha} \\
b_n^{\alpha\beta} & b_n^{\beta\beta}
\end{pmatrix}
```

**3. Compute A coefficients:**
```python
A = QT @ W
```
This calculates:
```math
A_n = Q_n^T W = 
\begin{pmatrix}
a_n^{\alpha\alpha} & a_n^{\alpha\beta} \\
a_n^{\beta\alpha} & a_n^{\beta\beta}
\end{pmatrix}
```

Since `Q` is orthonormal, this is equivalent to:
```math
A_n = Q_n^T H Q_n
```

**4. Subtract current block contribution:**
```python
W = W - Q @ A
```
This performs:
```math
W = W - Q_n A_n
```

**5. Full reorthogonalization (for numerical stability):**
```python
for qs in Qblocks[:-2]:
    W = W - qs @ (qs.T @ W)
```
Ensures orthogonality against all earlier blocks:
```math
W = W - \sum_{k < n-1} Q_k (Q_k^T W)
```

**6. QR factorization to get next block:**
```python
Q_next, R_next = np.linalg.qr(W, mode='reduced')
B_prev = R_next
```
This decomposes:
```math
W = Q_{n+1} B_{n+1}
```

where:
- `Q_next` = $Q_{n+1}$ is the orthonormal next block
- `R_next` = $B_{n+1}$ is the upper triangular B matrix for the next iteration

**7. Convergence check:**
```python
if Q_next.shape[1] == 0:
    break
```
Stops when the Krylov subspace is exhausted (invariant subspace found).

### Full Recursion

The code implements the block Lanczos three-term recurrence:
```math
H Q_n = Q_{n-1} B_n^T + Q_n A_n + Q_{n+1} B_{n+1}
```

Rearranging:
```math
Q_{n+1} B_{n+1} = H Q_n - Q_{n-1} B_n^T - Q_n A_n
```

And finally 
```math
T = \begin{pmatrix}
A_1 & B_2^T & 0 & 0 & \cdots \\
B_2 & A_2 & B_3^T & 0 & \cdots \\
0 & B_3 & A_3 & B_4^T & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{pmatrix}
```

The eigenvalues of $T$ approximate the eigenvalues of $H$.

