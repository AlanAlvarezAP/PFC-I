import numpy as np
import torch
import dgl
import matplotlib.pyplot as plt
import networkx as nx

from data.dataParser import process_mepognn_pipeline, load_raw_data


def create_dgl_graph_from_mepo(raw_cases: np.ndarray, threshold: float = 0.5):
    inf_cases = raw_cases.T

    corr_matrix = np.corrcoef(inf_cases)
    np.fill_diagonal(corr_matrix, 0.0)
    corr_matrix = np.nan_to_num(corr_matrix, 0.0)

    adj_matrix = np.where(corr_matrix >= threshold, corr_matrix, 0.0)

    src, dst = np.nonzero(adj_matrix)
    weights = adj_matrix[src, dst]

    num_nodes = inf_cases.shape[0]
    g = dgl.graph((torch.tensor(src, dtype=torch.int64), torch.tensor(dst, dtype=torch.int64)), num_nodes=num_nodes)
    g.edata['weight'] = torch.tensor(weights, dtype=torch.float32)

    return g, corr_matrix


def plot_dgl_graph(g: dgl.DGLGraph, corr_matrix: np.ndarray, threshold: float = 0.5, node_labels: list = None):
    num_nodes = g.num_nodes()
    if node_labels is None:
        node_labels = [f"N-{i}" for i in range(num_nodes)]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    im = axes[0].imshow(corr_matrix, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_xticks(range(num_nodes))
    axes[0].set_yticks(range(num_nodes))
    axes[0].set_xticklabels(node_labels, rotation=90, fontsize=6)
    axes[0].set_yticklabels(node_labels, fontsize=6)
    axes[0].set_title("Matriz de correlacion entre nodos", fontsize=12, fontweight='bold')

    cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label('Correlacion de Pearson', fontsize=10)

    nx_g = dgl.to_networkx(g, edge_attrs=['weight'])
    mapping = {i: label for i, label in enumerate(node_labels)}
    nx_g = nx.relabel_nodes(nx_g, mapping)

    pos = nx.spring_layout(nx_g, k=0.4, seed=42)

    weights = []
    for u, v, data in nx_g.edges(data=True):
        w = data.get('weight', 1.0)
        if hasattr(w, 'item'):
            w = w.item()
        weights.append(float(w) * 2.5)

    nx.draw_networkx_nodes(nx_g, pos, ax=axes[1], node_size=350, node_color='skyblue', edgecolors='black')
    nx.draw_networkx_labels(nx_g, pos, ax=axes[1], font_size=7, font_weight='bold')

    if weights:
        nx.draw_networkx_edges(nx_g, pos, ax=axes[1], width=weights, edge_color='crimson', alpha=0.6, arrowsize=10)
    else:
        nx.draw_networkx_edges(nx_g, pos, ax=axes[1], edge_color='crimson', alpha=0.6, arrowsize=10)

    axes[1].set_title(f"Grafo DGL (Filtrado por Correlación >= {threshold})", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("INFORMACION DEL DISPOSITIVO")
    print("=" * 60)
    if device.type == 'cuda':
        print(f"Dispositivo activo: GPU ({torch.cuda.get_device_name(0)})")
        print(f"VRAM Disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Dispositivo activo: CPU")

    obs_len = 14
    pred_len = 14
    batch_size = 32
    split_ratio = (6, 1, 1)
    corr_threshold = 0.5

    dataloaders, static_commute_graph = process_mepognn_pipeline(obs_len=obs_len,pred_len=pred_len,split_ratio=split_ratio,batch_size=batch_size,device=device,verbose=True)

    train_batches = len(dataloaders['train'])
    val_batches   = len(dataloaders['validate'])
    test_batches  = len(dataloaders['test'])

    print("ESTADISTICAS DE BATCHES:")
    print(f"Batches Train({split_ratio[0]} parts): {train_batches} (Muestras = {train_batches * batch_size})")
    print(f"Batches Val  ({split_ratio[1]} parts): {val_batches} (Muestras = {val_batches * batch_size})")
    print(f"Batches Test ({split_ratio[2]} parts): {test_batches} (Muestras = {test_batches * batch_size})")
    print(f"Escalar max_od (Normalizado): {dataloaders['max_od']:.4f}")

    x_od_b, x_node_b, x_sir_b, y_b = next(iter(dataloaders['train']))

    print()
    print("DIMENSIONES DE TENSORES POR BATCH:")
    print(f"Flujo OD Dinamico (x_od) : {x_od_b.shape} -> [Batch, Obs_Len, Nodos, Nodos]")
    print(f"Atributos Nodos (x_node) : {x_node_b.shape} -> [Batch, Obs_Len, Nodos, Feats]")
    print(f"Atributos SIR (x_sir)    : {x_sir_b.shape} -> [Batch, Obs_Len, Nodos, SIR_dim]")
    print(f"Target Futuro (y_true)   : {y_b.shape} -> [Batch, Pred_Len, Nodos, 1]")

    print("\n" + "=" * 60)
    print("-------- CONSTRUCCION Y ANALISIS DEL GRAFO DGL ------------")
    print("=" * 60)

    raw_data = load_raw_data()
    raw_cases = raw_data['y'].squeeze()
    node_labels = raw_data['locations']

    g, corr_matrix = create_dgl_graph_from_mepo(raw_cases, threshold=corr_threshold)

    num_nodes = g.num_nodes()
    print(f"Grafo DGL Generado: {g}")
    print(f"Cantidad de Nodos: {num_nodes}")
    print(f"Cantidad de Aristas (Edges): {g.num_edges()}")
    
    density = g.num_edges() / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
    print(f"Densidad del Grafo: {density:.4f}")

    print()
    print("Generando graficos del mapa de calor y la topologia del grafo...")
    plot_dgl_graph(g, corr_matrix, threshold=corr_threshold, node_labels=node_labels)


if __name__ == '__main__':
    main()