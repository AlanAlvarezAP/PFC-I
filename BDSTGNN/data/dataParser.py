import pandas as pd
import numpy as np
import torch

df_japan = pd.read_csv('data/covid19_jp.csv')
df_japan = df_japan[df_japan['administrative_area_level'] == 2].copy()

special_entries = ['Charter Flight', 'Quarantine', 'Diamond Princess', 'Costa Atlantica']
loc_col = 'administrative_area_level_2'
df_japan = df_japan[~df_japan[loc_col].isin(special_entries)].copy()

df_japan['date'] = pd.to_datetime(df_japan['date'])
mask = (df_japan['date'] >= '2022-01-15') & (df_japan['date'] <= '2022-06-14')
df_japan = df_japan.loc[mask].copy()

df_japan['recovered'] = df_japan['recovered'].fillna(0)
df_japan['deaths'] = df_japan['deaths'].fillna(0)
df_japan['active'] = df_japan['confirmed'] - df_japan['recovered'] - df_japan['deaths']
df_japan['active'] = df_japan['active'].clip(lower=0)

df_japan = df_japan.sort_values([loc_col, 'date']).reset_index(drop=True)

pivot_active = df_japan.pivot(index='date', columns=loc_col, values='active').dropna(how='any')
common_dates = pivot_active.index

df_japan = df_japan[df_japan['date'].isin(common_dates)].sort_values([loc_col, 'date']).reset_index(drop=True)
loc_list = sorted(df_japan[loc_col].astype(str).unique())

print("\n" + "="*65)
print(" RESULTADOS DEL PROCESAMIENTO DE DATOS (dataParser.py)")
print("="*65)
print(f"Rango de fechas alineadas: {common_dates.min().strftime('%Y-%m-%d')} - {common_dates.max().strftime('%Y-%m-%d')}")
print(f"Dias continuos totales (T): {len(common_dates)} dias")
print(f"Numero de Nodos/Regiones(N): {len(loc_list)} prefecturas")
print(f"Forma Matriz Pivote (T x N): {pivot_active.shape}")
print(f"Valores Nulos/Incompletos: {pivot_active.isna().sum().sum()}")
print("-" * 65)
print("-------- ESTADISTICAS DE CASOS ACTIVOS (POS FILTRADO): -------------- ")
print(f"Min : {df_japan['active'].min()}")
print(f"Max : {df_japan['active'].max()}")
print(f"Mean: {round(df_japan['active'].mean())}")
print(f"Std : {round(df_japan['active'].std(ddof=0))}")
print("-" * 65)
print("Lista de Nodos Detectados (47 Prefecturas):")
print(loc_list)
print("="*65 + "\n")


def prepare_data(data, history_window, pred_window, slide_step):
    x = []
    y = []

    for i in range(0, data.shape[1], slide_step):
        if i + history_window + pred_window > data.shape[1]:
            break
        x.append(data[:, i:i + history_window, :])
        y.append(data[:, i + history_window:i + history_window + pred_window, 0])

    x = np.array(x)
    y = np.array(y)
    return x, y


def data_processing_japan(loc_list, raw_data, loc_col, test_window, valid_window, history_window, pred_window, slide_step, device):
    active_cases = []
    confirmed_cases = []
    populations = []

    for each_loc in loc_list:
        loc_df = raw_data[raw_data[loc_col] == each_loc]
        active_cases.append(loc_df['active'].values)
        confirmed_cases.append(loc_df['confirmed'].values)
        populations.append(loc_df['population'].iloc[0])

    active_cases = np.array(active_cases)
    confirmed_cases = np.array(confirmed_cases)
    populations = np.array(populations)
    recovered_cases = confirmed_cases - active_cases
    susceptible_cases = np.expand_dims(populations, -1) - active_cases - recovered_cases

    normalizer = {'S': {}, 'I': {}, 'R': {}}
    for i, each_loc in enumerate(loc_list):
        normalizer['S'][each_loc] = (np.max(susceptible_cases[i]), np.min(susceptible_cases[i]))
        normalizer['I'][each_loc] = (np.max(active_cases[i]), np.min(active_cases[i]))
        normalizer['R'][each_loc] = (np.max(recovered_cases[i]), np.min(recovered_cases[i]) + 10)

    data_truth = np.concatenate((np.expand_dims(active_cases, axis=-1),np.expand_dims(recovered_cases, axis=-1),np.expand_dims(susceptible_cases, axis=-1)), axis=-1)
    data_feat = data_truth.copy()

    eps = 1e-8
    for i, each_loc in enumerate(loc_list):
        data_feat[i, :, 0] = (data_feat[i, :, 0] - normalizer['I'][each_loc][1]) / (normalizer['I'][each_loc][0] - normalizer['I'][each_loc][1] + eps)
        data_feat[i, :, 1] = (data_feat[i, :, 1] - normalizer['R'][each_loc][1]) / (normalizer['R'][each_loc][0] - normalizer['R'][each_loc][1] + eps)
        data_feat[i, :, 2] = (data_feat[i, :, 2] - normalizer['S'][each_loc][1]) / (normalizer['S'][each_loc][0] - normalizer['S'][each_loc][1] + eps)

    I_max = np.array([normalizer['I'][loc][0] for loc in loc_list], dtype=np.float32)
    I_min = np.array([normalizer['I'][loc][1] for loc in loc_list], dtype=np.float32)
    R_max = np.array([normalizer['R'][loc][0] for loc in loc_list], dtype=np.float32)
    R_min = np.array([normalizer['R'][loc][1] for loc in loc_list], dtype=np.float32)
    S_max = np.array([normalizer['S'][loc][0] for loc in loc_list], dtype=np.float32)
    S_min = np.array([normalizer['S'][loc][1] for loc in loc_list], dtype=np.float32)

    train_feat = data_feat[:, :-valid_window-test_window, :]
    val_feat   = data_feat[:, -valid_window-test_window:-test_window, :]
    test_feat  = data_feat[:, -test_window:, :]
    train_truth = data_truth[:, :-valid_window-test_window, :]
    val_truth   = data_truth[:, -valid_window-test_window:-test_window, :]
    test_truth  = data_truth[:, -test_window:, :]

    train_x, train_y = prepare_data(train_feat, history_window, pred_window, slide_step)
    train_x_true, _  = prepare_data(train_truth, history_window, pred_window, slide_step)
    val_x, val_y     = prepare_data(val_feat, history_window, pred_window, slide_step)
    val_x_true, _    = prepare_data(val_truth, history_window, pred_window, slide_step)
    test_x, _        = prepare_data(test_feat, history_window, pred_window, slide_step)
    _, test_y        = prepare_data(test_truth, history_window, pred_window, slide_step)
    test_x_true, _   = prepare_data(test_truth, history_window, pred_window, slide_step)

    to_tensor = lambda arr: torch.tensor(arr, dtype=torch.float32).to(device)
    return (to_tensor(train_x), to_tensor(train_y), to_tensor(train_x_true),to_tensor(val_x), to_tensor(val_y), to_tensor(val_x_true),to_tensor(test_x), to_tensor(test_y), to_tensor(test_x_true),to_tensor(S_min), to_tensor(S_max),to_tensor(I_min), to_tensor(I_max),to_tensor(R_min), to_tensor(R_max),to_tensor(populations))