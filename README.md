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
