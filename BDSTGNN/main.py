import pandas as pd
import numpy as np
import torch
import dgl
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from data.dataParser import data_processing_japan, df_japan, loc_list, loc_col

def create_dgl_graph_from_processed(train_x_true, loc_list, threshold=0.6):
    num_nodes = train_x_true.shape[1]
    active_cases = train_x_true[:, :, :, 0].permute(1, 0, 2).reshape(num_nodes, -1).cpu().numpy()

    corr_matrix = np.corrcoef(active_cases)
    np.fill_diagonal(corr_matrix, 0)

    adj_matrix = np.where(corr_matrix >= threshold, corr_matrix, 0.0)
    src, dst = np.nonzero(adj_matrix)
    weights = adj_matrix[src, dst]

    g = dgl.graph((torch.tensor(src, dtype=torch.int64), torch.tensor(dst, dtype=torch.int64)), num_nodes=num_nodes)
    g.edata['weight'] = torch.tensor(weights, dtype=torch.float32)

    return g, corr_matrix


def plot_dgl_graph(g, corr_matrix, loc_list, threshold=0.6):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    im = axes[0].imshow(corr_matrix, cmap="viridis", vmin=0, vmax=1)

    axes[0].set_xticks(range(len(loc_list)))
    axes[0].set_yticks(range(len(loc_list)))
    axes[0].set_xticklabels(loc_list, rotation=90, fontsize=7)
    axes[0].set_yticklabels(loc_list, fontsize=7)

    cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label('Fuerza de conexion (Correlacion)', fontsize=9)

    nx_g = dgl.to_networkx(g, edge_attrs=['weight'])
    mapping = {i: loc for i, loc in enumerate(loc_list)}
    nx_g = nx.relabel_nodes(nx_g, mapping)

    pos = nx.spring_layout(nx_g, k=0.5, seed=42)

    weights = []
    for u, v, data in nx_g.edges(data=True):
        w = data.get('weight', 1.0)
        if hasattr(w, 'item'):
            w = w.item()
        weights.append(float(w) * 3)

    nx.draw_networkx_nodes(nx_g, pos, ax=axes[1], node_size=500, node_color='skyblue', edgecolors='black')
    nx.draw_networkx_labels(nx_g, pos, ax=axes[1], font_size=8, font_weight='bold')

    if weights:
        nx.draw_networkx_edges(nx_g, pos, ax=axes[1], width=weights, edge_color='crimson', alpha=0.6, arrowsize=12)
    else:
        nx.draw_networkx_edges(nx_g, pos, ax=axes[1], edge_color='crimson', alpha=0.6, arrowsize=12)

    axes[1].set_title(f"Grafo DGL Dirigido (Correlacion >= {threshold})", fontsize=13, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        print(f"Nombre de la GPU   : {torch.cuda.get_device_name(0)}")
        print(f"Memoria VRAM Total : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("OJITO: Estando en CPU...")


    history_window = 5
    pred_window = 5
    slide_step = 5
    test_window = 50
    valid_window = 50

    (train_x, train_y, train_x_true, val_x, val_y, val_x_true, test_x, test_y, test_x_true, S_min, S_max, I_min, I_max, R_min, R_max, N) = data_processing_japan(loc_list=loc_list,raw_data=df_japan,loc_col=loc_col,test_window=test_window,valid_window=valid_window,history_window=history_window,pred_window=pred_window,slide_step=slide_step,device=device)

    print("--- Procesamiento finalizado con exito ---")
    print(f"Dimension tensor train_x_true: {train_x_true.shape}")

    threshold = 0.4
    g, corr_matrix = create_dgl_graph_from_processed(train_x_true, loc_list, threshold=threshold)

    print()
    print("--- Objeto DGL Graph Creado ---")
    print(g)
    print(f"Numero de Nodos: {g.num_nodes()}")
    print(f"Numero de Aristas (Edges): {g.num_edges()}")

    plot_dgl_graph(g, corr_matrix, loc_list, threshold=threshold)

if __name__ == '__main__':
    main()