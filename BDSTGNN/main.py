import os
import sys
import gc
import traceback
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import copy
import dgl
import matplotlib.pyplot as plt
import networkx as nx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import MAE,RMSE,MAPE,CCC,PCC

from data.dataParser import data_processing_japan, df_japan, loc_list, loc_col
from model.BDSTGNN import BDSTGNNWithDGL, check_dgl_parity

WINDOW_SIZES = [5, 10, 15]
PLOT_GRAPH = False
SEED = 1234
OUTPUT_DIR = "results"

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

def evaluar_y_exportar_casos(preds_tensor, loc_list, pred_window, batch_idx=0, title="EVOLUCIÓN EPIDEMIOLÓGICA", verbose=True):
    model_preds = preds_tensor[batch_idx].detach().cpu().numpy()

    if model_preds.shape[0] != len(loc_list) and model_preds.shape[1] == len(loc_list):
        model_preds = model_preds.T

    num_nodes, pred_len = model_preds.shape
    real_cases_preds = np.clip(model_preds, a_min=0, a_max=None)

    day_columns = [f"Dia +{d+1}" for d in range(pred_len)]
    df_results = pd.DataFrame(real_cases_preds, index=loc_list[:num_nodes], columns=day_columns)

    if verbose:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', lambda x: '%.2f' % x)

        print("\n" + "="*85)
        print(f" {title} (Casos Reales Estimados)")
        print("="*85)
        print(df_results)
        print("="*85 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filename = f"BDSTGNN_pred_window_{pred_window}.csv"
    file_path = os.path.join(OUTPUT_DIR, filename)
    df_results.to_csv(file_path, index=True)
    print(f"¡Casos exactos exportados exitosamente a '{file_path}'!")
    return real_cases_preds


def run_experiment(L, device, plot_graph=False):
    torch.manual_seed(SEED); np.random.seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    history_window = L
    pred_window = L
    slide_step = L
    test_window = max(30, 2 * L)
    valid_window = max(30, 2 * L)

    print("\n" + "#"*85)
    print(f"#  EXPERIMENTO  L = {L}  (history={history_window}, pred={pred_window}, slide={slide_step})")
    print("#"*85)

    (train_x, train_y, train_x_true, val_x, val_y, val_x_true, test_x, test_y, test_x_true, S_min, S_max, I_min, I_max, R_min, R_max, N) = data_processing_japan(loc_list=loc_list, raw_data=df_japan, loc_col=loc_col,test_window=test_window, valid_window=valid_window,history_window=history_window, pred_window=pred_window,slide_step=slide_step, device=device)

    print("--- DIAGNOSTICO DE FORMAS (SHAPES) ---")
    print("train_x shape :", tuple(train_x.shape), " (muestras, nodos, tiempo, features)")
    print("train_y shape :", tuple(train_y.shape))
    print("val_x shape   :", tuple(val_x.shape))
    print("test_x shape  :", tuple(test_x.shape))
    print("test_y shape  :", tuple(test_y.shape))
    print("---------------------------------------")

    if train_x.shape[0] == 0 or val_x.shape[0] == 0 or test_x.shape[0] == 0:
        raise ValueError(f"Con L={L} algun split quedo sin muestras " f"(train={train_x.shape[0]}, val={val_x.shape[0]}, test={test_x.shape[0]}) " f"Aumenta valid_window/test_window o reduce slide_step")

    num_nodes, in_feats = train_x.shape[1], train_x.shape[-1]
    model_bdst = BDSTGNNWithDGL(num_nodes=num_nodes, in_feats=in_feats, hidden_dim=32,history_window=history_window, pred_window=pred_window,kernel_size=15, tcn_channels=(8,), kernel_size_graph=3, dropout=0.0).to(device)
    optimizer = torch.optim.Adam(model_bdst.parameters(), lr=1e-4)
    epochs = 200

    I_min_b, I_max_b, N_b = I_min.unsqueeze(0), I_max.unsqueeze(0), N.unsqueeze(0)
    fwd = lambda x, xt: model_bdst(x, xt, I_min_b, I_max_b, N_b)

    print("\n" + "="*40 + f"\n ENTRENAMIENTO BDSTGNN (L={L}, {epochs} EPOCAS) \n" + "="*40)

    training_history = []
    best_val, patience_count, patience, min_delta = 1e10, 0, 10, 0.01
    stopped_epoch = epochs

    for epoch in range(epochs):
        model_bdst.train()
        for b in range(train_x.shape[0]):
            y_pred_st, y_pred_phy = fwd(train_x[b:b+1], train_x_true[b:b+1])
            loss_st = F.l1_loss(y_pred_st, train_y[b:b+1])
            loss_phy = F.l1_loss(y_pred_phy, train_y[b:b+1])
            total_loss = loss_st + loss_phy
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        model_bdst.eval()
        with torch.no_grad():
            val_loss = F.l1_loss(fwd(val_x, val_x_true)[0], val_y).item()

        training_history.append({"Epoch": epoch + 1, "Total_Loss": total_loss.item(), "ST_Loss": loss_st.item(), "Phy_Loss": loss_phy.item(), "Val_ST_Loss": val_loss})
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[L={L}] Epoca [{epoch+1}/{epochs}] | Loss Total: {total_loss.item():.4f} " f"(ST: {loss_st.item():.4f}, Phy: {loss_phy.item():.4f}) | Val ST: {val_loss:.4f}")

        if val_loss - best_val > min_delta:
            patience_count += 1
            if patience_count > patience:
                print(f"Early stopping en la epoca {epoch + 1}...")
                stopped_epoch = epoch + 1
                break
        else:
            patience_count, best_val = 0, val_loss

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    loss_filepath = os.path.join(OUTPUT_DIR, f"BDSTGNN_loss_history_win_{pred_window}.csv")
    pd.DataFrame(training_history).to_csv(loss_filepath, index=False)
    print(f"\nHistorial de perdidas exportado a '{loss_filepath}'")

    rows = []
    model_bdst.eval()
    with torch.no_grad():
        pred_nn, pred_phy = fwd(test_x, test_x_true)
        rng, lo = (I_max - I_min)[None, :, None], I_min[None, :, None]
        pred_nn_real  = (pred_nn  * rng + lo).cpu()
        pred_phy_real = (pred_phy * rng + lo).cpu()
        y_true = test_y.cpu()

        for name, key, p in [("NN  (salida que reporta el paper)", "NN", pred_nn_real), ("PHY (SIR)", "PHY", pred_phy_real)]:
            mae, rmse, mape = float(MAE(p, y_true)), float(RMSE(p, y_true)), float(MAPE(p, y_true))
            pcc, ccc = float(PCC(p, y_true)), float(CCC(p, y_true))
            print(f"{name} I:(history_window:{history_window},pred_window:{pred_window},slide_step:{slide_step}," f"MAE:{mae:.2f},RMSE:{rmse:.2f},MAPE:{mape:.4f}," f"PCC:{pcc:.4f},CCC:{ccc:.4f})")
            rows.append({"L": L, "history_window": history_window, "pred_window": pred_window, "slide_step": slide_step, "salida": key, "epocas": stopped_epoch, "MAE": mae, "RMSE": rmse, "MAPE": mape, "PCC": pcc, "CCC": ccc})

    evaluar_y_exportar_casos(pred_nn_real, loc_list, pred_window, batch_idx=0, verbose=False)

    if plot_graph:
        threshold = 0.4
        g, corr_matrix = create_dgl_graph_from_processed(train_x_true, loc_list, threshold=threshold)
        plot_dgl_graph(g, corr_matrix, loc_list, threshold=threshold)

    del model_bdst, optimizer, train_x, train_y, train_x_true, val_x, val_y, val_x_true, test_x, test_y, test_x_true
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print(f"Nombre de la GPU   : {torch.cuda.get_device_name(0)}")
        print(f"Memoria VRAM Total : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("OJITO: Estando en CPU...")

    check_dgl_parity(device=device)

    all_rows = []
    failed = []
    for i, L in enumerate(WINDOW_SIZES):
        try:
            all_rows.extend(run_experiment(L, device, plot_graph=PLOT_GRAPH))
        except Exception as e:
            print(f"\n[ERROR] L={L} fallo: {e}")
            traceback.print_exc()
            failed.append(L)

    if all_rows:
        df_summary = pd.DataFrame(all_rows)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        summary_path = os.path.join(OUTPUT_DIR, "BDSTGNN_summary_all_windows.csv")
        df_summary.to_csv(summary_path, index=False)

        pd.set_option('display.width', 1000)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.float_format', lambda x: '%.4f' % x)
        print("\n" + "="*85)
        print(" RESUMEN FINAL - TODAS LAS VENTANAS")
        print("="*85)
        print(df_summary[["L", "salida", "epocas", "MAE", "RMSE", "MAPE", "PCC", "CCC"]].to_string(index=False))
        print("="*85)
        print(f"Resumen exportado a '{summary_path}'")

    if failed:
        print(f"Ventanas que fallaron: {failed}")

if __name__ == '__main__':
    main()
