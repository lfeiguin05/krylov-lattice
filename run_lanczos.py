from feiguin_lanczos import mapping, block_lanczos, lanczos_heatmap, plot_last_Q_heatmap, plot_Ws_heatmap, plot_last_W_heatmap
import numpy as np
import matplotlib.pyplot as plt


def main():
    
    seeds = [(0,0), (2, 0)]
    num_seeds = len(seeds)
    N = 4

    if num_seeds == 1:
        column_vector, coord_touples, alphas_list, betas_list, w, coord_to_index, v_init = mapping(([seeds[0]]), N)
        lanczos_heatmap(coord_touples, w, title="Lanczos Vector Heatmap")
    else: 
        column_vector, coord_touples, alphas_list, betas_list, w, coord_to_index, v_init = mapping(seeds, N)
        coords_all, coord_to_index, Vs, all_maps, Ws, last_Q, As, Bs, Qblocks = block_lanczos(num_seeds, seeds, N)
        plot_last_Q_heatmap(coords_all, last_Q, title="Last Q Matrix Heatmap")
        plot_Ws_heatmap(coords_all, Ws, title="W Matrices Heatmap")
        plot_last_W_heatmap(coords_all, Ws[-1], title="Last W Matrix Heatmap")

        for i in range(len(Ws) -1): 
            W_i = Ws[i]
            print(f"W at step {i}: ", W_i)
            W_next = Ws[i + 1]
            print(f"W at step {i + 1}: ", W_next)
            overlap = np.dot(W_i.T, W_next)
            print(f"Overlap between W at step {i} and W at step {i + 1}: ", overlap)
    
    
if __name__ == "__main__":
    main()


