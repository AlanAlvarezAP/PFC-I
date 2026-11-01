import os, copy, argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import dgl

from data.dataParser import process_mepognn_pipeline, load_raw_data, loc_list
from model.MepoGNN import MepoGNNComplete

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

    axes[1].set_title(f"Grafo DGL (Filtrado por Correlacion >= {threshold})", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

def imprimir_valores_explicitos(preds_tensor, loc_list, batch_idx=0, title="EVOLUCION EPIDEMIOLOGICA"):
    model_preds = preds_tensor[batch_idx].detach().cpu().numpy()
    pred_len, num_nodes = model_preds.shape
    
    print("\n" + "="*90)
    print(f" {title} (Proximos {pred_len} dias)")
    print("="*90)
    
    header = f"{'Region':<20}" + "".join([f"Dia +{d+1:<9}" for d in range(pred_len)])
    print(header)
    print("-" * len(header))
    
    for i, loc_name in enumerate(loc_list):
        if i >= num_nodes:
            break
        vals = model_preds[:, i]
        row_str = f"{loc_name:<20}" + "".join([f"{v:<10.2f}" for v in vals])
        print(row_str)
    print("="*90)

def evaluate(y_pred, y_true):
    mse = F.mse_loss(y_pred, y_true)
    rmse = torch.sqrt(mse)
    mae = F.l1_loss(y_pred, y_true)
    lab = y_true + 1.0 
    mape = torch.mean(torch.abs(y_pred - lab) / lab)
    rae = torch.sum(torch.abs(y_pred - y_true) / torch.abs(y_true - y_true.mean()).sum())
    return dict(RMSE=rmse.item(), MAE=mae.item(), MAPE=100 * mape.item(), RAE=rae.item())


def run_model(model, batch, graph_type, max_od):
    x_od, x_node, x_sir, y = batch
    if graph_type == 'Adaptive':
        return model(x_node, x_sir)
    return model(x_node, x_sir, x_od, max_od)


def train_and_test(dataloaders, commute, graph_type, obs_len, pred_len, seed, epochs, patience=20, lr=1e-3, weight_decay=1e-8, clip=5.0, verbose=True):
    device = next(iter(dataloaders['train']))[0].device
    num_nodes = next(iter(dataloaders['train']))[1].shape[2]
    max_od = float(dataloaders['max_od'])
    torch.manual_seed(seed); np.random.seed(seed)
    model = MepoGNNComplete(num_nodes=num_nodes, adpinit=commute if graph_type == 'Adaptive' else None,glm_type=graph_type, in_dim=4, in_len=obs_len, out_len=pred_len, dropout=0.5).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    task_level, best_val, best_state, bad = 1, np.inf, None, patience

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in dataloaders['train']:
            y_true = batch[3]
            y_pred = run_model(model, batch, graph_type, max_od)
            opt.zero_grad()
            if epoch % 2 == 0 and task_level <= pred_len:
                task_level += 1
            loss = F.l1_loss(y_pred[:, :task_level], y_true[:, :task_level])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()

        model.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for batch in dataloaders['validate']:
                tot += F.l1_loss(run_model(model, batch, graph_type, max_od), batch[3]).item() * batch[3].shape[0]
                n += batch[3].shape[0]
        val = tot / n
        if verbose: print(f"Epoca {epoch:3d} | train(L1,t<={task_level}) {loss.item():8.3f} | val L1 {val:8.3f}", flush=True)
        if val <= best_val:
            best_val, bad = val, patience
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad -= 1
            if bad == 0:
                print(f"Early stopping en la epoca {epoch}"); break

    model.load_state_dict(best_state); model.eval()
    fc, gt = [], []
    with torch.no_grad():
        for batch in dataloaders['test']:
            fc.append(run_model(model, batch, graph_type, max_od).cpu()); gt.append(batch[3].cpu())
    fc, gt = torch.cat(fc), torch.cat(gt)
    res = {'3 dias': evaluate(fc[:, 2], gt[:, 2]), '7 dias': evaluate(fc[:, 6], gt[:, 6]), '14 dias': evaluate(fc[:, 13], gt[:, 13]), 'Overall': evaluate(fc, gt)}
    return model, fc, gt, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graph', default='Dynamic', choices=['Dynamic', 'Adaptive'])
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--no_plot', action='store_true')
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60 + "\nINFORMACION DEL DISPOSITIVO (MEPOGNN)\n" + "=" * 60)
    if device.type == 'cuda':
        print(f"Dispositivo activo: GPU ({torch.cuda.get_device_name(0)})")
    else:
        print("Dispositivo activo: CPU")

    obs_len, pred_len, batch_size, split_ratio, corr_threshold = 14, 14, 32, (6, 1, 1), 0.5
    dataloaders, static_commute_graph = process_mepognn_pipeline(obs_len=obs_len, pred_len=pred_len, split_ratio=split_ratio, batch_size=batch_size, device=device, verbose=True)

    print("\n" + "=" * 40 + f"\n INICIANDO ENTRENAMIENTO MEPOGNN ({args.graph}) \n" + "=" * 40)
    model_mepo, fc, gt, res = train_and_test(dataloaders, static_commute_graph, args.graph, obs_len, pred_len,args.seed, args.epochs)

    print("\n" + "=" * 70 + f"\n RESULTADOS TEST - MepoGNN({args.graph[:3]}) - semilla {args.seed}\n" + "=" * 70)
    print(f"{'Horizonte':<10}{'RMSE':>10}{'MAE':>10}{'MAPE(%)':>10}{'RAE':>8}")
    for k, v in res.items():
        print(f"{k:<10}{v['RMSE']:>10.1f}{v['MAE']:>10.1f}{v['MAPE']:>10.1f}{v['RAE']:>8.2f}")

    imprimir_valores_explicitos(fc.squeeze(-1), loc_list=loc_list, title="MEPOGNN - PREDICCION DE CASOS (SIR)")

    if not args.no_plot:
        raw_data = load_raw_data()
        raw_cases = raw_data['y'].squeeze()
        g, corr_matrix = create_dgl_graph_from_mepo(raw_cases, threshold=corr_threshold)
        plot_dgl_graph(g, corr_matrix, threshold=corr_threshold, node_labels=loc_list)

if __name__ == '__main__':
    main()