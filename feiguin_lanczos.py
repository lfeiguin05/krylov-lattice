"""
Block Lanczos Debug 
==============================
Save as: debug_test_suite.py
Requires: block_lanczos_dict.py in same directory

Usage: python debug_test_suite.py
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json
from datetime import datetime
import pickle


# Import from main implementation
from feiguin_lanczos2 import (
    unified_mapping,
    parse_initial_vectors,
    apply_H,
    build_full_hamiltonian,
    block_lanczos,
    single_seed_lanczos,
    build_block_tridiagonal,
    verify_lanczos_relation,
    plot_vector_on_lattice,
    plot_initial_vectors,
    plot_Qblock_heatmaps,
    plot_T_structure,

)


# ============ Test Utilities ============

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.messages = []

    def log(self, msg: str):
        self.messages.append(msg)
        print(f"  {msg}")

    def fail(self, msg: str):
        self.passed = False
        print(f"  ✗ FAIL: {msg}")

    def ok(self, msg: str):
        print(f"  ✓ OK: {msg}")


def header(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


# ============ Individual Tests ============

def test_mapping_correctness(initial_vectors, N):
    result = TestResult("Mapping Correctness")
    print(f"\n--- {result.name} ---")
    
    coords_all, coord_to_index = unified_mapping(initial_vectors, N)
    
    all_seeds = set()
    for vec_dict in initial_vectors:
        for coord in vec_dict.keys():
            all_seeds.add(tuple(map(int, coord)))
    
    missing = all_seeds - set(coords_all)
    if missing:
        result.fail(f"Missing seeds: {missing}")
    else:
        result.ok(f"All {len(all_seeds)} seed coordinates present")
    
    result.log(f"Mapping size: {len(coords_all)}")
    return result


def test_hamiltonian_symmetry(initial_vectors, N):
    result = TestResult("Hamiltonian Symmetry")
    print(f"\n--- {result.name} ---")
    
    coords_all, coord_to_index = unified_mapping(initial_vectors, N)
    H = build_full_hamiltonian(coords_all, coord_to_index)
    
    sym_error = np.max(np.abs(H - H.T))
    result.log(f"max|H - H^T| = {sym_error:.2e}")
    
    if sym_error < 1e-14:
        result.ok("Hamiltonian is symmetric")
    else:
        result.fail("Hamiltonian not symmetric")
    
    return result


def test_apply_H_consistency(initial_vectors, N):
    result = TestResult("apply_H Consistency")
    print(f"\n--- {result.name} ---")
    
    coords_all, coord_to_index = unified_mapping(initial_vectors, N)
    n = len(coords_all)
    H = build_full_hamiltonian(coords_all, coord_to_index)
    
    max_error = 0.0
    for _ in range(5):
        v = np.random.randn(n)
        Hv_matrix = H @ v
        Hv_func = apply_H(v, coord_to_index, coords_all)
        error = np.max(np.abs(Hv_matrix - Hv_func))
        max_error = max(max_error, error)
    
    result.log(f"max|H@v - apply_H(v)| = {max_error:.2e}")
    
    if max_error < 1e-12:
        result.ok("apply_H matches matrix multiplication")
    else:
        result.fail("apply_H mismatch")
    
    return result


def test_initial_vector_parsing(initial_vectors, N):
    result = TestResult("Initial Vector Parsing")
    print(f"\n--- {result.name} ---")
    
    coords_all, coord_to_index = unified_mapping(initial_vectors, N)
    n = len(coords_all)
    Q0 = parse_initial_vectors(initial_vectors, coord_to_index, n)
    
    result.log(f"Q0 shape: {Q0.shape}")
    
    all_correct = True
    for col, vec_dict in enumerate(initial_vectors):
        for coord, expected_amp in vec_dict.items():
            coord = tuple(map(int, coord))
            idx = coord_to_index[coord]
            actual_amp = Q0[idx, col]
            if not np.isclose(actual_amp, expected_amp):
                result.fail(f"Vector {col}, coord {coord}: expected {expected_amp}, got {actual_amp}")
                all_correct = False
    
    if all_correct:
        result.ok("All amplitudes correctly parsed")
    
    return result


def test_negative_amplitudes(N):
    result = TestResult("Negative Amplitude Handling")
    print(f"\n--- {result.name} ---")
    
    initial_vectors = [{(0, 0): 1.0, (1, 0): -1.0}]
    
    coords_all, coord_to_index = unified_mapping(initial_vectors, N)
    n = len(coords_all)
    Q0 = parse_initial_vectors(initial_vectors, coord_to_index, n)
    
    idx_00 = coord_to_index[(0, 0)]
    idx_10 = coord_to_index[(1, 0)]
    
    result.log(f"Amplitude at (0,0): {Q0[idx_00, 0]}")
    result.log(f"Amplitude at (1,0): {Q0[idx_10, 0]}")
    
    if Q0[idx_00, 0] == 1.0 and Q0[idx_10, 0] == -1.0:
        result.ok("Negative amplitudes preserved")
    else:
        result.fail("Negative amplitudes not preserved")
    
    return result


def test_orthonormality(initial_vectors, N, m_steps):
    result = TestResult("Orthonormality of Q Blocks")
    print(f"\n--- {result.name} ---")
    
    results = block_lanczos(initial_vectors, N, m_steps=m_steps, verbose=False)
    Qblocks = results['Qblocks']
       
    Qblocks = results['Qblocks']
    As = results['As']
    Bs = results['Bs']
    
    Q_all = np.hstack(Qblocks[:len(As)])
    result.log(f"Total Q blocks: {len(Qblocks)}, Q_all shape: {Q_all.shape}")
    
    QtQ = Q_all.T @ Q_all
    ortho_error = np.max(np.abs(QtQ - np.eye(QtQ.shape[0])))
    result.log(f"max|Q^T Q - I| = {ortho_error:.2e}")
    
    if ortho_error < 1e-10:
        result.ok("All Q blocks are orthonormal")
    else:
        result.fail("Orthonormality violated")
    
    return result


def test_block_tridiagonal_structure(initial_vectors, N, m_steps):
    result = TestResult("Block Tridiagonal Structure")
    print(f"\n--- {result.name} ---")
    
    results = block_lanczos(initial_vectors, N, m_steps=m_steps, verbose=False)
    As = results['As']
    Bs = results['Bs']
    Qblocks = results['Qblocks']
    
    result.log(f"A blocks: {len(As)}, B blocks: {len(Bs)}, Q blocks: {len(Qblocks)}")
    result.log(f"For T: {len(As)} A blocks, {min(len(Bs), len(As)-1)} B blocks used")
    
    # Check A symmetry
    for i, A in enumerate(As):
        sym_err = np.max(np.abs(A - A.T))
        if sym_err > 1e-10:
            result.fail(f"A[{i}] not symmetric")
            return result
    result.ok("All A blocks symmetric")
    
    # Build and check T
    T = build_block_tridiagonal(As, Bs, results['block_sizes'])
    result.log(f"T shape: {T.shape}")
    
    T_sym_error = np.max(np.abs(T - T.T))
    result.log(f"max|T - T^T| = {T_sym_error:.2e}")
    
    if T_sym_error < 1e-10:
        result.ok("T is symmetric")
    else:
        result.fail("T not symmetric")
    
    return result


def test_lanczos_relation(initial_vectors, N, m_steps):
    result = TestResult("Lanczos Relation: T = Q^T H Q")
    print(f"\n--- {result.name} ---")
    
    results = block_lanczos(initial_vectors, N, m_steps=m_steps, verbose=False)
    metrics = verify_lanczos_relation(results)
    
    result.log(f"T shape: {metrics['T_shape']}")
    result.log(f"Q^T H Q shape: {metrics['H_proj_shape']}")
    
    if 'shape_mismatch' in metrics:
        result.fail("Shape mismatch!")
        return result
    
    error = metrics['lanczos_relation_error']
    result.log(f"max|T - Q^T H Q| = {error:.2e}")
    
    if error < 1e-10:
        result.ok("Lanczos relation satisfied")
    else:
        result.fail("Lanczos relation violated")
    
    return result


def test_eigenvalue_approximation(initial_vectors, N, m_steps):
    result = TestResult("Eigenvalue Approximation")
    print(f"\n--- {result.name} ---")
    
    results = block_lanczos(initial_vectors, N, m_steps=m_steps, verbose=False)
    metrics = verify_lanczos_relation(results)
    
    H_eigs = metrics['H_eigenvalues']
    T_eigs = metrics.get('T_eigenvalues', np.array([]))
    
    if len(T_eigs) == 0:
        result.fail("No T eigenvalues")
        return result
    
    result.log(f"H eigenvalues: {len(H_eigs)}, T eigenvalues: {len(T_eigs)}")
    result.log(f"H range: [{H_eigs[0]:.4f}, {H_eigs[-1]:.4f}]")
    result.log(f"T range: [{T_eigs[0]:.4f}, {T_eigs[-1]:.4f}]")
    
    result.ok("Ritz values computed")
    return result


def test_single_seed_lanczos(N):
    result = TestResult("Single Seed Lanczos")
    print(f"\n--- {result.name} ---")
    
    results = single_seed_lanczos((0, 0), N, m_steps=10)
    
    alphas = results['alphas']
    betas = results['betas']
    
    result.log(f"Alphas: {len(alphas)}, Betas: {len(betas)}")
    
    # Build scalar T using len(alphas)-1 betas
    betas_for_T = betas[:len(alphas)-1]
    T_scalar = np.diag(alphas)
    if len(betas_for_T) > 0:
        T_scalar += np.diag(betas_for_T, k=1) + np.diag(betas_for_T, k=-1)
    
    result.log(f"T_scalar shape: {T_scalar.shape}")
    
    # Verify via the standard method
    metrics = verify_lanczos_relation(results)
    error = metrics['lanczos_relation_error']
    result.log(f"Lanczos relation error: {error:.2e}")
    
    if error < 1e-10:
        result.ok("Single seed works correctly")
    else:
        result.fail("Lanczos relation violated")
    
    return result


def test_linear_dependence_handling(N):
    result = TestResult("Linear Dependence Handling")
    print(f"\n--- {result.name} ---")
    
    # Two vectors at same location
    initial_vectors = [
        {(0, 0): 1.0},
        {(0, 0): 2.0},
        {(1, 1): 1.0},
    ]
    
    result.log(f"Input: 3 vectors, 2 at same location")
    
    results = block_lanczos(initial_vectors, N, m_steps=5, verbose=False)
    
    initial_block_size = results['Qblocks'][0].shape[1]
    result.log(f"Initial block size after QR: {initial_block_size}")
    
    if initial_block_size == 2:
        result.ok("Reduced to 2 independent vectors")
    else:
        result.fail(f"Expected 2, got {initial_block_size}")
    
    return result


def test_invariant_subspace(N=3):
    result = TestResult("Invariant Subspace Detection")
    print(f"\n--- {result.name} ---")
    
    initial_vectors = [{(0, 0): 1.0}]
    results = block_lanczos(initial_vectors, N, m_steps=100, verbose=False)
    
    num_A = len(results['As'])
    Q_for_T = np.hstack(results['Qblocks'][:num_A])
    krylov_dim = Q_for_T.shape[1]
    lattice_size = len(results['coords_all'])
    
    result.log(f"Lattice size: {lattice_size}")
    result.log(f"Krylov dimension: {krylov_dim}")
    
    if krylov_dim <= lattice_size:
        result.ok(f"Stopped at dim {krylov_dim}")
    else:
        result.fail("Exceeded lattice size")
    
    return result


def test_superposition_states(N):
    result = TestResult("Superposition States")
    print(f"\n--- {result.name} ---")
    
    initial_vectors = [
        {(0, 0): 1.0, (1, 0): 1.0},
        {(0, 0): 1.0, (1, 0): -1.0},
    ]
    
    results = block_lanczos(initial_vectors, N, m_steps=10, verbose=False)
    metrics = verify_lanczos_relation(results)
    
    error = metrics['lanczos_relation_error']
    result.log(f"Lanczos relation error: {error:.2e}")
    
    if error < 1e-10:
        result.ok("Superposition states work correctly")
    else:
        result.fail("Lanczos relation violated")
    
    return result


def test_multi_site_vectors(N=10):
    result = TestResult("Multi-Site Vectors")
    print(f"\n--- {result.name} ---")
    
    initial_vectors = [
        {(0, 0): 1.0, (1, 0): -0.5, (0, 1): 0.5},
        {(3, 3): 1.0, (4, 3): 0.7, (3, 4): -0.7},
    ]
    
    results = block_lanczos(initial_vectors, N, m_steps=12, verbose=False)
    metrics = verify_lanczos_relation(results)
    
    error = metrics['lanczos_relation_error']
    result.log(f"Lanczos relation error: {error:.2e}")
    
    if error < 1e-10:
        result.ok("Multi-site vectors work correctly")
    else:
        result.fail("Lanczos relation violated")
    
    return result


def test_residual_orthogonality(initial_vectors, N, m_steps):
    result = TestResult("Residual Orthogonality")
    print(f"\n--- {result.name} ---")
    
    results = block_lanczos(initial_vectors, N, m_steps=m_steps, verbose=False)
    Qblocks = results['Qblocks']
    Ws = results['Ws']
    
    result.log(f"W matrices: {len(Ws)}, Q blocks: {len(Qblocks)}")
    
    max_overlap = 0.0
    for i, W in enumerate(Ws):
        for j, Q in enumerate(Qblocks[:i+1]):
            overlap = np.max(np.abs(Q.T @ W))
            max_overlap = max(max_overlap, overlap)
    
    result.log(f"Max W-Q overlap: {max_overlap:.2e}")
    
    if max_overlap < 1e-9:
        result.ok("Residuals are orthogonal")
    else:
        result.fail("Residuals not orthogonal")
    
    return result


def test_extra_blocks(initial_vectors, N, m_steps):
    result = TestResult("Extra Block Generation")
    print(f"\n--- {result.name} ---")
    
    results = block_lanczos(initial_vectors, N, m_steps=m_steps, verbose=False)
    
    num_A = len(results['As'])
    num_B = len(results['Bs'])
    num_Q = len(results['Qblocks'])
    
    result.log(f"A blocks: {num_A}, B blocks: {num_B}, Q blocks: {num_Q}")
    result.log(f"Expected: B = A ({num_A}), Q = A+1 ({num_A + 1})")
    
    if num_B == num_A and num_Q == num_A + 1:
        result.ok("Correct block counts")
    else:
        result.fail(f"Wrong counts: B={num_B}, Q={num_Q}")
    
    return result

def test_eigenvalue_diagnostic():
    """Diagnose why eigenvalues don't match."""
    result = TestResult("Eigenvalue Diagnostic")
    print(f"\n--- {result.name} ---")
    
    N = 2
    initial_vectors = [{(0, 0): 1.0}, {(0, 1): 1.0, (1, 0): -1.0}]
    
    results = block_lanczos(initial_vectors, N, m_steps=50, verbose=False)
    
    coords_all = results['coords_all']
    coord_to_index = results['coord_to_index']
    Qblocks = results['Qblocks']
    As = results['As']
    Bs = results['Bs']
    
    result.log(f"A blocks: {len(As)}, B blocks: {len(Bs)}, Q blocks: {len(Qblocks)}")
    
    # Build matrices
    H = build_full_hamiltonian(coords_all, coord_to_index)
    
    # Use correct number of Q blocks
    num_Q_for_T = len(As)
    Q_full = np.hstack(Qblocks[:num_Q_for_T])
    
    result.log(f"Q_full shape: {Q_full.shape}")
    result.log(f"Using {num_Q_for_T} Q blocks for T")
    
    # Check 1: Q orthonormality
    QtQ = Q_full.T @ Q_full
    ortho_error = np.max(np.abs(QtQ - np.eye(QtQ.shape[0])))
    result.log(f"\n1. Orthonormality error: {ortho_error:.2e}")
    if ortho_error > 1e-10:
        result.fail("Q is NOT orthonormal!")
    
    # Check 2: Compute Q^T H Q directly
    H_proj = Q_full.T @ H @ Q_full
    result.log(f"2. Q^T H Q shape: {H_proj.shape}")
    
    # Check 3: Build T from blocks
    T = build_block_tridiagonal(As, Bs, results['block_sizes'])
    result.log(f"3. T shape: {T.shape}")
    
    # Check 4: Compare T and Q^T H Q
    if T.shape == H_proj.shape:
        diff = np.max(np.abs(T - H_proj))
        result.log(f"4. max|T - Q^T H Q|: {diff:.2e}")
        if diff > 1e-10:
            result.fail("T ≠ Q^T H Q !")
            result.log(f"\nT:\n{np.array2string(T, precision=4)}")
            result.log(f"\nQ^T H Q:\n{np.array2string(H_proj, precision=4)}")
    else:
        result.fail(f"Shape mismatch: T={T.shape}, Q^T H Q={H_proj.shape}")
    
    # Check 5: Eigenvalues
    H_eigs = np.sort(np.linalg.eigvalsh(H))
    T_eigs = np.sort(np.linalg.eigvalsh(T))
    H_proj_eigs = np.sort(np.linalg.eigvalsh(H_proj))
    
    result.log(f"\n5. Eigenvalues:")
    result.log(f"   H range: [{H_eigs[0]:.4f}, {H_eigs[-1]:.4f}]")
    result.log(f"   T range: [{T_eigs[0]:.4f}, {T_eigs[-1]:.4f}]")
    result.log(f"   Q^T H Q range: [{H_proj_eigs[0]:.4f}, {H_proj_eigs[-1]:.4f}]")
    
    # T eigenvalues MUST be within H eigenvalue range
    if T_eigs[0] < H_eigs[0] - 1e-10 or T_eigs[-1] > H_eigs[-1] + 1e-10:
        result.fail("T eigenvalues OUTSIDE H range - this is impossible if T = Q^T H Q!")
    
    # Check 6: Are Q^T H Q eigenvalues correct?
    if H_proj_eigs[0] >= H_eigs[0] - 1e-10 and H_proj_eigs[-1] <= H_eigs[-1] + 1e-10:
        result.log("   Q^T H Q eigenvalues are within H range (correct)")
    else:
        result.fail("Q^T H Q eigenvalues outside H range")
    
    result.log(f"\n6. Full comparison:")
    result.log(f"   H      : {np.array2string(H_eigs, precision=4)}")
    result.log(f"   T      : {np.array2string(T_eigs, precision=4)}")
    result.log(f"   Q^T H Q: {np.array2string(H_proj_eigs, precision=4)}")
    
    return result


# ============ Run All Tests ============

def run_all_tests():
    header("BLOCK LANCZOS TEST SUITE")
    all_results = {}
    INV = 1/np.sqrt(2)
    #first run with these initial vectors varying x coordinate of second cluster

    x_vals = list(range(5, 25))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for x in x_vals:
        c = x + 1  # center at x+1 
        print(f"\n{'='*60}")
        print(f"TESTING WITH SEPARATION x = {x}")
        print(f"{'='*60}")

        # moving cross around (c,0): left, top, right, bottom
        left   = (c - 1, 0)
        right  = (c + 1, 0)
        top    = (c,  1)
        bottom = (c, -1)

        
        #case 1 
        
        initial_vectors = [
            {
                (1, 0):  INV,
                (0, 1): -INV,
                (-1, 0): INV,
                (0, -1): -INV,

                left:   INV,
                top:   -INV,
                right: INV,
                bottom:-INV,
            },
            {
                (1, 0):  INV,
                (0, 1): -INV,
                (-1, 0): INV,
                (0, -1): -INV,

                left:   -INV,
                top:    INV,
                right:  -INV,
                bottom: INV,
            },
            {
                (1, 0):  INV,
                (0, 1):  INV,
                (-1, 0): INV,
                (0, -1): INV,

                left:   INV,
                top:    INV,
                right:  INV,
                bottom: INV,
            },
            {
                (1, 0):  INV,
                (0, 1):  INV,
                (-1, 0): INV,
                (0, -1): INV,

                left:   -INV,
                top:    -INV,
                right:  -INV,
                bottom: -INV,
            },
    ]
        """
        #case 2
        initial_vectors = [
            {
                (1, 0):  INV,
                (0, 1): -INV,
                (-1, 0): INV,
                (0, -1): -INV,
            }, 
            {
                left:   INV,
                top:   -INV,
                right: INV,
                bottom:-INV,
            },
            {
                (1, 0):  INV,
                (0, 1):  INV,
                (-1, 0): INV,
                (0, -1): INV, 
            },
            {
                left:   INV,
                top:    INV,
                right:  INV,
                bottom: INV,
            },
        ]
        """
        N = 100
        m_steps = 100

        forbidden_coords = [(0, 0)]

        print(f"\nConfiguration: N={N}, m_steps={m_steps}")
        print(f"Initial vectors: {initial_vectors}")


        # Run block Lanczos
        results = block_lanczos(initial_vectors, N, m_steps=m_steps, 
                               verbose=False, forbidden=forbidden_coords)
        
        # Verify and collect metrics
        metrics = verify_lanczos_relation(results)
        
        # Run critical tests
        test_results = []
        test_results.append(test_orthonormality(initial_vectors, N, m_steps))
        test_results.append(test_lanczos_relation(initial_vectors, N, m_steps))
        test_results.append(test_eigenvalue_approximation(initial_vectors, N, m_steps))
        
        passed = sum(1 for r in test_results if r.passed)
        print(f"  Tests passed: {passed}/{len(test_results)}")
        
        # Store results (convert numpy arrays to lists for JSON)
        all_results[str(x)] = {
            'separation': x,
            'center': c,
            'lanczos_error': float(metrics['lanczos_relation_error']),
            'H_eigenvalues': metrics['H_eigenvalues'].tolist(),
            'T_eigenvalues': metrics.get('T_eigenvalues', np.array([])).tolist(),
            'num_A_blocks': len(results['As']),
            'num_B_blocks': len(results['Bs']),
            'num_Q_blocks': len(results['Qblocks']),
            'block_sizes': results['block_sizes'],
            'lattice_size': len(results['coords_all']),
            'krylov_dimension': sum(results['block_sizes'][:len(results['As'])]),
            'tests_passed': passed,
            'separation': x,
            'T_eigenvalues': metrics['T_eigenvalues'].tolist(),
            
            #  A matrices (list of 4x4 matrices)
            'A_matrices': [A.tolist() for A in results['As']],
            
            #  B matrices (list of 4x4 matrices)
            'B_matrices': [B.tolist() for B in results['Bs']],
            
            # Also  dimensions for verification
            'num_A_blocks': len(results['As']),
            'num_B_blocks': len(results['Bs']),
            'block_sizes': results['block_sizes'],
            'tests_total': len(test_results),
            'test_details': [
                {
                    'name': r.name,
                    'passed': r.passed,
                    'messages': r.messages
                }
                for r in test_results
            ],
            'initial_vectors': [
                {str(k): v for k, v in vec.items()} 
                for vec in initial_vectors
            ],
            
        }
    
        # Save to JSON
        json_filename = f"lanczos_results_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Saved JSON results to: {json_filename}")

        json_filename = f"lanczos_matrices_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"✓ Saved matrices to: {json_filename}")
        
        # Save human-readable text summary
        txt_filename = f"lanczos_summary_{timestamp}.txt"
        write_text_summary(all_results, x_vals, txt_filename)
        print(f"✓ Saved text summary to: {txt_filename}")
        
        # Save full results with numpy arrays (for later analysis)
        pkl_filename = f"lanczos_full_{timestamp}.pkl"
        with open(pkl_filename, 'wb') as f:
            pickle.dump(all_results, f)
        print(f"✓ Saved pickle file to: {pkl_filename}")
        
        # Analysis and plotting
        header("ANALYSIS ACROSS SEPARATIONS")
        print_summary_table(all_results, x_vals)
        plot_separation_analysis(all_results, x_vals, timestamp)
        
    return all_results, timestamp


    # All tests
    """
    tests = [
        lambda: test_mapping_correctness(initial_vectors, N),
        lambda: test_hamiltonian_symmetry(initial_vectors, N),
        lambda: test_apply_H_consistency(initial_vectors, N),
        lambda: test_initial_vector_parsing(initial_vectors, N),
        lambda: test_negative_amplitudes(N),
        lambda: test_orthonormality(initial_vectors, N, m_steps),
        lambda: test_block_tridiagonal_structure(initial_vectors, N, m_steps),
        lambda: test_lanczos_relation(initial_vectors, N, m_steps),
        lambda: test_eigenvalue_approximation(initial_vectors, N, m_steps),
        lambda: test_single_seed_lanczos(N),
        lambda: test_linear_dependence_handling(N),
        lambda: test_invariant_subspace(N=3),
        lambda: test_superposition_states(N),
        lambda: test_multi_site_vectors(N=10),
        lambda: test_residual_orthogonality(initial_vectors, N, m_steps),
        lambda: test_extra_blocks(initial_vectors, N, m_steps),
        lambda: test_eigenvalue_diagnostic(),
    ]
    """

   
    
   # return results



def write_text_summary(all_results, x_vals, filename):
    """Write human-readable text summary."""
    with open(filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BLOCK LANCZOS TEST RESULTS SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Separation range: {min(x_vals)} to {max(x_vals)}\n")
        f.write(f"N (hops): 10\n")
        f.write(f"m_steps: 10\n")
        f.write(f"Initial vectors: 4 (two clusters with symmetric/antisymmetric states)\n\n")
        
        f.write("RESULTS BY SEPARATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Sep':<5} {'Lanczos Err':<15} {'Tests':<12} {'H eigs':<8} {'T eigs':<8} {'Lattice':<10} {'Krylov':<8}\n")
        f.write("-" * 80 + "\n")
        
        for x in x_vals:
            res = all_results[str(x)]
            f.write(f"{res['separation']:<5} "
                   f"{res['lanczos_error']:<15.2e} "
                   f"{res['tests_passed']}/{res['tests_total']:<10} "
                   f"{len(res['H_eigenvalues']):<8} "
                   f"{len(res['T_eigenvalues']):<8} "
                   f"{res['lattice_size']:<10} "
                   f"{res['krylov_dimension']:<8}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for x in x_vals:
            res = all_results[str(x)]
            f.write(f"\nSEPARATION x = {x}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Center coordinate: {res['center']}\n")
            f.write(f"Lanczos relation error: {res['lanczos_error']:.2e}\n")
            f.write(f"Lattice size: {res['lattice_size']}\n")
            f.write(f"Krylov dimension: {res['krylov_dimension']}\n")
            f.write(f"Block structure: A={res['num_A_blocks']}, B={res['num_B_blocks']}, Q={res['num_Q_blocks']}\n")
            f.write(f"Block sizes: {res['block_sizes']}\n\n")
            
            f.write("H eigenvalues:\n")
            H_eigs = np.array(res['H_eigenvalues'])
            f.write(f"  Range: [{H_eigs.min():.6f}, {H_eigs.max():.6f}]\n")
            f.write(f"  Values: {np.array2string(H_eigs[:10], precision=6, suppress_small=True)}...\n\n")
            
            if len(res['T_eigenvalues']) > 0:
                T_eigs = np.array(res['T_eigenvalues'])
                f.write("T eigenvalues:\n")
                f.write(f"  Range: [{T_eigs.min():.6f}, {T_eigs.max():.6f}]\n")
                f.write(f"  Values: {np.array2string(T_eigs, precision=6, suppress_small=True)}\n\n")
            
            f.write("Test Results:\n")
            for test in res['test_details']:
                status = "✓ PASS" if test['passed'] else "✗ FAIL"
                f.write(f"  {status}: {test['name']}\n")
                for msg in test['messages']:
                    f.write(f"    {msg}\n")
            f.write("\n")
        
        # Overall statistics
        f.write("\n" + "="*80 + "\n")
        f.write("OVERALL STATISTICS\n")
        f.write("="*80 + "\n")
        
        errors = [all_results[str(x)]['lanczos_error'] for x in x_vals]
        pass_rates = [all_results[str(x)]['tests_passed']/all_results[str(x)]['tests_total'] for x in x_vals]
        
        f.write(f"Lanczos error:\n")
        f.write(f"  Min: {min(errors):.2e}\n")
        f.write(f"  Max: {max(errors):.2e}\n")
        f.write(f"  Mean: {np.mean(errors):.2e}\n")
        f.write(f"  Median: {np.median(errors):.2e}\n\n")
        
        f.write(f"Test pass rate:\n")
        f.write(f"  Min: {min(pass_rates)*100:.1f}%\n")
        f.write(f"  Max: {max(pass_rates)*100:.1f}%\n")
        f.write(f"  Mean: {np.mean(pass_rates)*100:.1f}%\n")
        
        configs_all_passed = sum(1 for x in x_vals if all_results[str(x)]['tests_passed'] == all_results[str(x)]['tests_total'])
        f.write(f"  Configurations with all tests passed: {configs_all_passed}/{len(x_vals)}\n")


def print_summary_table(all_results, x_vals):
    """Print summary table to console."""
    print("\nSummary:")
    print(f"{'x':<5} {'Lanczos Error':<15} {'Tests':<12} {'H eigs':<8} {'T eigs':<8} {'Lattice':<10}")
    print("-" * 70)
    for x in x_vals:
        res = all_results[str(x)]
        print(f"{res['separation']:<5} "
              f"{res['lanczos_error']:<15.2e} "
              f"{res['tests_passed']}/{res['tests_total']:<10} "
              f"{len(res['H_eigenvalues']):<8} "
              f"{len(res['T_eigenvalues']):<8} "
              f"{res['lattice_size']:<10}")


def plot_separation_analysis(all_results, x_vals, timestamp):
    """Plot how algorithm behaves vs cluster separation."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Lanczos error vs separation
    errors = [all_results[str(x)]['lanczos_error'] for x in x_vals]
    axes[0, 0].semilogy(x_vals, errors, 'o-')
    axes[0, 0].set_xlabel('Separation (x)')
    axes[0, 0].set_ylabel('Lanczos Relation Error')
    axes[0, 0].set_title('Numerical Accuracy vs Separation')
    axes[0, 0].grid(True)
    axes[0, 0].axhline(y=1e-10, color='r', linestyle='--', alpha=0.5, label='Target: 1e-10')
    axes[0, 0].legend()
    
    # 2. Eigenvalue spectrum evolution
    for i, x in enumerate(x_vals[::3]):  # Sample every 3rd
        T_eigs = np.array(all_results[str(x)]['T_eigenvalues'])
        if len(T_eigs) > 0:
            axes[0, 1].plot(T_eigs, [x]*len(T_eigs), 'o', 
                           label=f'x={x}', alpha=0.6, markersize=4)
    axes[0, 1].set_xlabel('Eigenvalue')
    axes[0, 1].set_ylabel('Separation (x)')
    axes[0, 1].set_title('Eigenvalue Spectrum Evolution')
    axes[0, 1].legend(fontsize=8, ncol=2)
    axes[0, 1].grid(True)
    
    # 3. Krylov dimension vs lattice size
    krylov_dims = [all_results[str(x)]['krylov_dimension'] for x in x_vals]
    lattice_sizes = [all_results[str(x)]['lattice_size'] for x in x_vals]
    axes[1, 0].plot(x_vals, krylov_dims, 'o-', label='Krylov dim')
    axes[1, 0].plot(x_vals, lattice_sizes, 's-', alpha=0.5, label='Lattice size')
    axes[1, 0].set_xlabel('Separation (x)')
    axes[1, 0].set_ylabel('Dimension')
    axes[1, 0].set_title('Subspace Dimensions')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 4. Test pass rate
    pass_rates = [all_results[str(x)]['tests_passed']/all_results[str(x)]['tests_total'] 
                  for x in x_vals]
    axes[1, 1].plot(x_vals, pass_rates, 'o-')
    axes[1, 1].fill_between(x_vals, pass_rates, alpha=0.3)
    axes[1, 1].set_xlabel('Separation (x)')
    axes[1, 1].set_ylabel('Test Pass Rate')
    axes[1, 1].set_ylim([0, 1.1])
    axes[1, 1].set_title('Algorithm Reliability')
    axes[1, 1].axhline(y=1.0, color='g', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_filename = f'separation_analysis_{timestamp}.png'
    plt.savefig(plot_filename, dpi=150)
    print(f"✓ Saved analysis plot: {plot_filename}")
    plt.show()


def load_results(json_filename):
    """Load results from JSON file for later analysis."""
    with open(json_filename, 'r') as f:
        results = json.load(f)
    print(f"Loaded {len(results)} configurations from {json_filename}")
    return results


def compare_eigenvalues_from_files(json_file_1, json_file_2, tolerance=1e-6):
    """
    Compare eigenvalue spectra from two saved JSON result files.
    
    Parameters:
    -----------
    json_file_1, json_file_2 : str
        Paths to JSON files from previous runs
    tolerance : float
        Tolerance for eigenvalue matching
        
    Returns:
    --------
    dict with comparison results
    """
    print("="*60)
    print("EIGENVALUE EQUIVALENCE CHECK")
    print("="*60)
    
    
    # Load both files
    print(f"\nLoading: {json_file_1}")
    with open(json_file_1, 'r') as f:
        results_1 = json.load(f)
    
    print(f"Loading: {json_file_2}")
    with open(json_file_2, 'r') as f:
        results_2 = json.load(f)
    
    # Get x values
    x_vals_1 = sorted([int(k) for k in results_1.keys()])
    x_vals_2 = sorted([int(k) for k in results_2.keys()])
    
    print(f"\nFile 1: {len(x_vals_1)} configurations (x = {x_vals_1[0]} to {x_vals_1[-1]})")
    print(f"File 2: {len(x_vals_2)} configurations (x = {x_vals_2[0]} to {x_vals_2[-1]})")
    
    # Find common x values
    common_x = sorted(set(x_vals_1) & set(x_vals_2))
    
    if not common_x:
        print("\n✗ NO COMMON CONFIGURATIONS TO COMPARE")
        return {'match': False, 'reason': 'no_common_x'}
    
    print(f"\nComparing {len(common_x)} common configurations...")
    print("-"*60)
    
    # Compare each x value
    all_match = True
    comparison_results = {}
    
    for x in common_x:
        eigs_1 = np.array(results_1[str(x)]['T_eigenvalues'])
        eigs_2 = np.array(results_2[str(x)]['T_eigenvalues'])
        
        # Sort for comparison
        eigs_1 = np.sort(eigs_1)
        eigs_2 = np.sort(eigs_2)
        
        same_dimension = (len(eigs_1) == len(eigs_2))
        
        if same_dimension:
            diff = np.abs(eigs_1 - eigs_2)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            match = max_diff < tolerance
        else:
            max_diff = np.inf
            mean_diff = np.inf
            match = False
        
        comparison_results[x] = {
            'n_eigs_1': len(eigs_1),
            'n_eigs_2': len(eigs_2),
            'same_dimension': same_dimension,
            'max_diff': float(max_diff),
            'mean_diff': float(mean_diff),
            'match': match
        }
        
        status = "✓" if match else "✗"
        print(f"x={x:2d}  {status}  dims: {len(eigs_1)} vs {len(eigs_2)}  "
              f"max_diff: {max_diff:.2e}  mean_diff: {mean_diff:.2e}")
        
        all_match = all_match and match
    
    # Summary
    print("\n" + "="*60)
    if all_match:
        print("✓ ALL EIGENVALUES MATCH")
    else:
        print("✗ EIGENVALUES DO NOT MATCH")
    
    matched = sum(1 for r in comparison_results.values() if r['match'])
    print(f"\nMatched: {matched}/{len(common_x)} configurations")
    
    return {
        'all_match': all_match,
        'matched': matched,
        'total': len(common_x),
        'tolerance': tolerance,
        'details': comparison_results
    }


# Quick usage function
def quick_check(file1, file2, tol=1e-6):
    #One-liner to check if two result files match.
    result = compare_eigenvalues_from_files(file1, file2, tolerance=tol)
    return result['all_match']


"""
if __name__ == "__main__":
    # Compare two JSON files
    result = compare_eigenvalues_from_files(
        "lanczos_results_20260126_180041.json",
        "lanczos_results_20260126_183213.json",
        tolerance=1e-6
    )
    
    # Or just quick check:
    # match = quick_check("file1.json", "file2.json")
    # print(f"Match: {match}")

"""


# Usage example
if __name__ == "__main__":
    results, timestamp = run_all_tests()

# ============ Visual Tests ============

def run_visual_tests():
    header("VISUAL TESTS")
    
    initial_vectors = [
        {(1, 0): 0.5, (0, 1): -0.5, (-1, 0): 0.5, (0, -1): -0.5, (6, 0): 0.5, (5, 1): -0.5, (4, 0): 0.5, (5, -1): -0.5},
        {(1, 0): 0.5, (0, 1): -0.5, (-1, 0): 0.5, (0, -1): -0.5, (6, 0): -0.5, (5, 1): 0.5, (4, 0): -0.5, (5, -1): 0.5},
        {(1, 0): 0.5, (0, 1): 0.5, (-1, 0): 0.5, (0, -1): 0.5, (6, 0): 0.5, (5, 1): 0.5, (4, 0): 0.5, (5, -1): 0.5},
        {(1, 0): 0.5, (0, 1): 0.5, (-1, 0): 0.5, (0, -1): 0.5, (6, 0): -0.5, (5, 1): -0.5, (4, 0): -0.5, (5, -1): -0.5},

    ]
    
    forbidden_coords = [(0, 0)]
    results = block_lanczos(initial_vectors, N=10, m_steps=10, verbose=True, forbidden = forbidden_coords)
    
    print("\n1. Initial vectors:")
    plot_initial_vectors(results, initial_vectors)
    
    print("\n2. Q block evolution:")
    for i in range(len(results['Qblocks'])):
        plot_Qblock_heatmaps(results['coords_all'], results['Qblocks'][i], step=i)
    
    print("\n3. T matrix structure:")
    T = build_block_tridiagonal(results['As'], results['Bs'], results['block_sizes'])
    plot_T_structure(T, results['block_sizes'][:len(results['As'])])
    
    print("\n4. Eigenvalue convergence:")
    



