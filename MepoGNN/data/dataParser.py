import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_raw_data():
    data = np.load("data/jp20200401_20210921.npy", allow_pickle=True).item()
    commute = np.load("data/commute_jp.npy")

    node_inf = np.log(data['node'][..., [0]] + 1.0)
    node_other = data['node'][..., [1, 2, 3]]
    node_features = np.concatenate((node_inf, node_other), axis=-1)

    locations = None
    for key in ['location', 'locations', 'prefectures', 'nodes']:
        if key in data:
            locations = list(data[key])
            break   

    return {'od': data['od'],'node': node_features,'sir': data['SIR'],'y': data['node'][..., [0]],'commute': commute,'locations': locations}


def create_sliding_windows(raw_data: dict, obs_len: int = 14, pred_len: int = 14):
    x_od, x_node, x_sir, y = [], [], [], []
    num_timesteps = raw_data['od'].shape[0]

    for i in range(obs_len, num_timesteps - pred_len + 1):
        x_od.append(raw_data['od'][i - obs_len : i])
        x_node.append(raw_data['node'][i - obs_len : i])
        x_sir.append(raw_data['sir'][i - obs_len : i])
        y.append(raw_data['y'][i : i + pred_len])

    return np.array(x_od), np.array(x_node), np.array(x_sir), np.array(y)


def normalize_features(x_od: np.ndarray, x_node: np.ndarray, train_len: int):
    max_od = x_od[:train_len, ..., 0].max()
    x_od_norm = x_od / max_od

    x_node_norm = np.copy(x_node)
    for f in range(x_node.shape[-1]):
        mean = x_node[:train_len, ..., f].mean()
        std = x_node[:train_len, ..., f].std()
        x_node_norm[..., f] = (x_node[..., f] - mean) / (std + 1e-8)

    return x_od_norm, x_node_norm, max_od


def build_dataloaders(x_od, x_node, x_sir, y, split_ratio=(6, 1, 1), batch_size=32, device='cpu'):
    total_samples = len(y)
    ratio_sum = sum(split_ratio)
    
    train_len = int((split_ratio[0] / ratio_sum) * total_samples)
    val_len = int((split_ratio[1] / ratio_sum) * total_samples)

    x_od_norm, x_node_norm, max_od = normalize_features(x_od, x_node, train_len)

    x_od_t = torch.from_numpy(x_od_norm).float().to(device)
    x_node_t = torch.from_numpy(x_node_norm).float().to(device)
    x_sir_t = torch.from_numpy(x_sir).float().to(device)
    y_t = torch.from_numpy(y).float().to(device)

    v_start = train_len
    t_start = train_len + val_len

    ds_train = TensorDataset(x_od_t[:v_start], x_node_t[:v_start], x_sir_t[:v_start], y_t[:v_start])
    ds_val   = TensorDataset(x_od_t[v_start:t_start], x_node_t[v_start:t_start], x_sir_t[v_start:t_start], y_t[v_start:t_start])
    ds_test  = TensorDataset(x_od_t[t_start:], x_node_t[t_start:], x_sir_t[t_start:], y_t[t_start:])

    dataloaders = {'max_od': max_od,'train': DataLoader(ds_train, batch_size=batch_size, shuffle=True),'validate': DataLoader(ds_val, batch_size=batch_size, shuffle=False),'test': DataLoader(ds_test, batch_size=batch_size, shuffle=False)}

    return dataloaders


def print_dataset_summary(raw_data: dict, start_date: str = "2020-04-01"):
    od_shape = raw_data['od'].shape
    
    if len(od_shape) == 3:
        T, N, _ = od_shape
    elif len(od_shape) == 2:
        T = od_shape[0]
        N = 47
    else:
        T = od_shape[0]
        N = raw_data['node'].shape[1]

    common_dates = pd.date_range(start=start_date, periods=T, freq='D')
    pivot_active = raw_data['y'].squeeze()
    null_values = np.isnan(raw_data['od']).sum() + np.isnan(raw_data['node']).sum()

    print("\n" + "=" * 65)
    print(" RESULTADOS DEL PROCESAMIENTO DE DATOS (dataParser.py)")
    print("=" * 65)
    print(f"Rango de fechas alineadas: {common_dates.min().strftime('%Y-%m-%d')} - {common_dates.max().strftime('%Y-%m-%d')}")
    print(f"Dias continuos totales: {len(common_dates)} dias")
    print(f"Numero de Nodos/Regiones: 47 prefecturas")
    print(f"Forma Matriz OD: {od_shape}")
    print(f"Forma Matriz Pivote (T x N): {pivot_active.shape}")
    print(f"Valores Nulos/Incompletos: {null_values}")
    print("-" * 65)
    print("-------- ESTADISTICAS DE CASOS ACTIVOS (POS FILTRADO): -------------- ")
    print(f"Min : {pivot_active.min():.0f}")
    print(f"Max : {pivot_active.max():.0f}")
    print(f"Mean: {round(pivot_active.mean())}")
    print(f"Std : {round(pivot_active.std(ddof=0))}")
    print("-" * 65)
    print("=" * 65 + "\n")


def process_mepognn_pipeline(obs_len=14, pred_len=14, split_ratio=(6, 1, 1), batch_size=32, device='cpu', verbose=True):
    raw_data = load_raw_data()

    if verbose:
        print_dataset_summary(raw_data)

    x_od, x_node, x_sir, y = create_sliding_windows(raw_data, obs_len, pred_len)
    dataloaders = build_dataloaders(x_od, x_node, x_sir, y, split_ratio, batch_size, device)
    commute_graph = torch.from_numpy(raw_data['commute']).float().to(device)
    
    return dataloaders, commute_graph