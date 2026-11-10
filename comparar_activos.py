import argparse
import copy
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from BDSTGNN.model.BDSTGNN import BDSTGNNWithDGL
from MepoGNN.model.MepoGNN import MepoGNNComplete
from MepoGNN.data.dataParser import process_mepognn_pipeline, load_raw_data, create_sliding_windows
from datos_y_metricas import (check, origins, read_csv_external, prepare_mepo_csv,export_report, compare_active_outputs,)


CARPETA = Path(__file__).resolve().parent
CSV = CARPETA / 'BDSTGNN/data/covid19_jp.csv'
SALIDA = Path(os.environ.get('RESULTADOS_DIR', str(CARPETA / 'resultados_comparacion')))
PESOS = Path(os.environ.get('PESOS_DIR', str(CARPETA / 'pesos')))

INICIO_TEST = '2022-07-01'
FIN_TEST = '2022-09-05'
VENTANA = 14
PASO_TEST = 1
EPOCAS_BDST = 200
EPOCAS_MEPO = 200
PACIENCIA = 15
BATCH_MEPO = 32
PROTOCOLO = 'comun'
PERIODO_TRAIN = ('2022-01-15', '2022-06-14')
VAL_DIAS, TEST_DIAS = 30, 30
SOLO_REVISAR_DATOS = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cargar_bdst():
    carpeta_anterior = Path.cwd()
    os.environ['BDST_INICIO'], os.environ['BDST_FIN'] = PERIODO_TRAIN
    try:
        os.chdir(CARPETA / 'BDSTGNN')
        from BDSTGNN.data.dataParser import (data_processing_japan, df_japan, loc_list, loc_col,)
        resultado = data_processing_japan(loc_list=loc_list, raw_data=df_japan, loc_col=loc_col,test_window=30, valid_window=30,history_window=VENTANA, pred_window=VENTANA,slide_step=1, device=DEVICE,)
    finally:
        os.chdir(carpeta_anterior)

    (train_x, train_y, train_real, val_x, val_y, val_real, test_x_antiguo, test_y_antiguo, test_real_antiguo,S_min, S_max, I_min, I_max, R_min, R_max, poblacion) = resultado

    return {'train_x': train_x, 'train_y': train_y, 'train_real': train_real,'val_x': val_x, 'val_y': val_y, 'val_real': val_real,'S_min': S_min, 'S_max': S_max,'I_min': I_min, 'I_max': I_max,'R_min': R_min, 'R_max': R_max,'poblacion': poblacion, 'regiones': loc_list,}


def cargar_mepo_comun(objetivo='nuevos'):
    datos, _ = prepare_mepo_csv(CARPETA, CSV, PERIODO_TRAIN[0], PERIODO_TRAIN[1], history=1, flexible=True)
    nuevos, node, sir = datos['node'][..., 0], datos['node'].copy(), datos['SIR']
    n_dias = len(datos['dates'])
    
    corte_val, corte_test = n_dias - VAL_DIAS - TEST_DIAS, n_dias - TEST_DIAS
    y_base = nuevos[..., None] if objetivo == 'nuevos' else sir[..., 1:2]
    node[..., 0] = np.log1p(node[..., 0])
    train_node = node[:corte_val]
    media = np.array([train_node[..., f].mean() for f in range(3)])
    desviacion = np.array([train_node[..., f].std() for f in range(3)])
    node = (node - media) / (desviacion + 1e-8)

    def ventanas(a, b):
        idx = range(a, b - 2 * VENTANA + 1)
        x = np.stack([node[t:t + VENTANA] for t in idx]).astype(np.float32)
        s_ = np.stack([sir[t:t + VENTANA] for t in idx]).astype(np.float32)
        y = np.stack([y_base[t + VENTANA:t + 2 * VENTANA] for t in idx]).astype(np.float32)
        od = torch.zeros(len(x), 1)
        return DataLoader(TensorDataset(od, torch.tensor(x), torch.tensor(s_), torch.tensor(y)), batch_size=BATCH_MEPO, shuffle=(a == 0))
    
    loaders = {'train': ventanas(0, corte_val), 'validate': ventanas(corte_val, corte_test)}
    for e in loaders:
        loaders[e] = DataLoader(TensorDataset(*[t.to(DEVICE) for t in loaders[e].dataset.tensors]), batch_size=BATCH_MEPO, shuffle=(e == 'train'))
    
    print('MepoGNN (protocolo comun): ventanas train', len(loaders['train'].dataset), '| val', len(loaders['validate'].dataset), flush=True)
    carpeta_anterior = Path.cwd()
    try:
        os.chdir(CARPETA / 'MepoGNN')
        commute = torch.from_numpy(load_raw_data()['commute']).float().to(DEVICE)
    finally:
        os.chdir(carpeta_anterior)
    return loaders, commute, media, desviacion


def cargar_mepo(objetivo="nuevos"):
    if PROTOCOLO == 'comun':
        return cargar_mepo_comun(objetivo)
    
    carpeta_anterior = Path.cwd()
    try:
        os.chdir(CARPETA / 'MepoGNN')
        loaders, commute = process_mepognn_pipeline(obs_len=VENTANA, pred_len=VENTANA, split_ratio=(6, 1, 1),batch_size=BATCH_MEPO, device=DEVICE, verbose=True,)
        originales = load_raw_data()
        _, ventanas_node, _, _ = create_sliding_windows(originales, VENTANA, VENTANA,)
    finally:
        os.chdir(carpeta_anterior)


    desplazamiento = 0
    for etapa in ['train', 'validate']:
        od, node, sir, y = loaders[etapa].dataset.tensors
        if objetivo == 'activos':
            inicio = desplazamiento + VENTANA
            activos = originales['sir'][..., 1:2]
            etiquetas = np.stack([activos[t:t + VENTANA] for t in range(inicio, inicio + len(y))])
            y = torch.as_tensor(etiquetas, dtype=y.dtype, device=y.device)

        desplazamiento += len(y)
        dataset = TensorDataset(od, node[..., [0, 2, 3]], sir, y)
        loaders[etapa] = DataLoader(dataset, batch_size=BATCH_MEPO, shuffle=(etapa == 'train'),)

    n_train = len(loaders['train'].dataset)
    ventanas_node = ventanas_node[..., [0, 2, 3]]
    media = np.array([ventanas_node[:n_train, ..., f].mean() for f in range(3)])
    desviacion = np.array([ventanas_node[:n_train, ..., f].std() for f in range(3)])
    del loaders['test']
    return loaders, commute, media, desviacion

def entrenar_bdst(datos):
    torch.manual_seed(1234)
    np.random.seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    modelo = BDSTGNNWithDGL(num_nodes=len(datos['regiones']), in_feats=3, hidden_dim=32,history_window=VENTANA, pred_window=VENTANA,kernel_size=15, tcn_channels=(8,), kernel_size_graph=3, dropout=0.0,).to(DEVICE)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-4)
    minimo = datos['I_min'][None, :]
    maximo = datos['I_max'][None, :]
    poblacion = datos['poblacion'][None, :]

    mejor_val = np.inf
    mejor_estado = None
    paciencia = PACIENCIA
    historial = []

    for epoca in range(EPOCAS_BDST):
        modelo.train()
        for b in range(len(datos['train_x'])):
            pred_nn, pred_phy = modelo(datos['train_x'][b:b+1], datos['train_real'][b:b+1],minimo, maximo, poblacion,)
            y = datos['train_y'][b:b+1]
            loss_nn = F.l1_loss(pred_nn, y)
            loss_phy = F.l1_loss(pred_phy, y)
            loss = loss_nn + loss_phy

            optimizador.zero_grad()
            loss.backward()
            optimizador.step()

        modelo.eval()
        with torch.no_grad():
            pred_val, _ = modelo(datos['val_x'], datos['val_real'], minimo, maximo, poblacion,)
            val_loss = F.l1_loss(pred_val, datos['val_y']).item()

        historial.append({'epoca': epoca+1, 'loss_ultimo_batch': loss.item(), 'val_loss': val_loss})
        print(f'BDSTGNN | época {epoca+1} | val: {val_loss:.4f}', flush=True)

        if val_loss <= mejor_val:
            mejor_val = val_loss
            mejor_estado = copy.deepcopy(modelo.state_dict())
            paciencia = PACIENCIA
        else:
            paciencia -= 1
            if paciencia == 0:
                break

    modelo.load_state_dict(mejor_estado)
    pd.DataFrame(historial).to_csv(SALIDA / 'BDSTGNN_entrenamiento.csv', index=False)
    torch.save({
        'state_dict': modelo.state_dict(), 'ventana': VENTANA,
        'regiones': datos['regiones'],
        'normalizadores': {k: datos[k].cpu() for k in ['S_min', 'S_max', 'I_min', 'I_max', 'R_min', 'R_max', 'poblacion',]},
    }, PESOS / 'BDSTGNN.pt')
    return modelo


def salida_mepo(modelo, node, sir, objetivo):
    if objetivo == 'activos':
        _, pred_activos = modelo(node, sir, return_active=True)
        return pred_activos
    return modelo(node, sir)


def entrenar_mepo(loaders, commute, media, desviacion, regiones, objetivo='nuevos'):
    nombre = 'MepoGNN_activos' if objetivo == 'activos' else 'MepoGNN'
    torch.manual_seed(3)
    np.random.seed(3)
    modelo = MepoGNNComplete(num_nodes=47, adpinit=commute, glm_type='Adaptive',in_dim=3, in_len=VENTANA, out_len=VENTANA, dropout=0.5,).to(DEVICE)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3, weight_decay=1e-8)
    mejor_val = np.inf
    mejor_estado = None
    paciencia = PACIENCIA
    nivel = 1
    historial = []

    for epoca in range(1, EPOCAS_MEPO+1):
        modelo.train()
        for _, node, sir, y in loaders['train']:
            pred = salida_mepo(modelo, node, sir, objetivo)
            optimizador.zero_grad()

            if epoca % 2 == 0 and nivel <= VENTANA:
                nivel += 1
            loss = F.l1_loss(pred[:, :nivel], y[:, :nivel])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
            optimizador.step()

        modelo.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for _, node, sir, y in loaders['validate']:
                pred = salida_mepo(modelo, node, sir, objetivo)
                total += F.l1_loss(pred, y).item() * len(y)
                n += len(y)

        val_loss = total / n
        historial.append({'epoca': epoca, 'loss_ultimo_batch': loss.item(), 'val_loss': val_loss})
        print(f'MepoGNN | época {epoca} | val {objetivo}: {val_loss:.4f}', flush=True)

        if val_loss <= mejor_val:
            mejor_val = val_loss
            mejor_estado = copy.deepcopy(modelo.state_dict())
            paciencia = PACIENCIA
        else:
            paciencia -= 1
            if paciencia == 0:
                break

    modelo.load_state_dict(mejor_estado)
    pd.DataFrame(historial).to_csv(SALIDA / f'{nombre}_entrenamiento.csv', index=False)
    torch.save({
        'state_dict': modelo.state_dict(), 'ventana': VENTANA,
        'objetivo': objetivo, 'graph': 'Adaptive', 'in_dim': 3, 'regiones': regiones,
        'media': torch.tensor(media), 'desviacion': torch.tensor(desviacion),
        'commute': commute.cpu(),
    }, PESOS / f'{nombre}.pt')
    return modelo


def evaluar_bdst(modelo, datos):
    fechas, reales, poblacion = read_csv_external(CSV, datos['regiones'], INICIO_TEST, FIN_TEST, VENTANA,)
    minimo = np.stack([datos[k].cpu().numpy() for k in ['I_min', 'R_min', 'S_min']], axis=-1)
    maximo = np.stack([datos[k].cpu().numpy() for k in ['I_max', 'R_max', 'S_max']], axis=-1)
    entradas = (reales - minimo) / (maximo - minimo + 1e-8)
    indices = origins(fechas, INICIO_TEST, FIN_TEST, VENTANA, VENTANA, PASO_TEST)
    predicciones = []
    verdades = []
    modelo.eval()

    with torch.no_grad():
        for i in indices:
            x = torch.tensor(entradas[i-VENTANA:i].transpose(1, 0, 2)[None], dtype=torch.float32, device=DEVICE)
            x_real = torch.tensor(reales[i-VENTANA:i].transpose(1, 0, 2)[None], dtype=torch.float32, device=DEVICE)
            pred_nn, _ = modelo(x, x_real, datos['I_min'][None], datos['I_max'][None], datos['poblacion'][None])
            pred_activos = pred_nn * (datos['I_max'] - datos['I_min'])[None, :, None] + datos['I_min'][None, :, None]
            predicciones.append(pred_activos[0].T.cpu().numpy())
            verdades.append(reales[i:i+VENTANA, :, 0])

    export_report(SALIDA, 'BDSTGNN_L14_NN', 'active_cases',np.array(predicciones), np.array(verdades), fechas, indices, datos['regiones'])


def evaluar_mepo(modelo, test, media, desviacion, objetivo='nuevos'):
    node = test['node'].copy()
    node[..., 0] = np.log1p(node[..., 0])
    node = (node - media) / (desviacion + 1e-8)
    indices = origins(test['dates'], INICIO_TEST, FIN_TEST, VENTANA, VENTANA, PASO_TEST)
    nuevos = []
    activos = []
    nuevos_reales = []
    activos_reales = []
    modelo.eval()

    with torch.no_grad():
        for inicio in range(0, len(indices), BATCH_MEPO):
            lote = indices[inicio:inicio+BATCH_MEPO]
            x_node = torch.tensor(np.stack([node[i-VENTANA:i] for i in lote]), dtype=torch.float32, device=DEVICE)
            x_sir = torch.tensor(np.stack([test['SIR'][i-VENTANA:i] for i in lote]), dtype=torch.float32, device=DEVICE)

            pred_nuevos, pred_activos = modelo(x_node, x_sir, return_active=True)
            nuevos.append(pred_nuevos.squeeze(-1).cpu().numpy())
            activos.append(pred_activos.squeeze(-1).cpu().numpy())
            nuevos_reales.extend([test['node'][i:i+VENTANA, :, 0] for i in lote])
            activos_reales.extend([test['SIR'][i:i+VENTANA, :, 1] for i in lote])

    if objetivo == 'nuevos':
        export_report(SALIDA, 'MepoGNN_new', 'daily_new_cases',np.concatenate(nuevos), np.array(nuevos_reales), test['dates'], indices, test['locations'])
    nombre = 'MepoGNN_active_supervised' if objetivo == 'activos' else 'MepoGNN_active_derived'
    export_report(SALIDA, nombre, 'active_cases',np.concatenate(activos), np.array(activos_reales), test['dates'], indices, test['locations'])


def cargar_bdst_guardado():
    ruta = PESOS / 'BDSTGNN.pt'
    checkpoint = torch.load(ruta, map_location='cpu', weights_only=True)

    regiones = checkpoint['regiones']
    normalizadores = checkpoint['normalizadores']
    campos = ['S_min', 'S_max', 'I_min', 'I_max', 'R_min', 'R_max', 'poblacion']

    datos = {'regiones': regiones}
    for k in campos:
        valor = normalizadores[k]
        datos[k] = valor.to(DEVICE)

    modelo = BDSTGNNWithDGL(num_nodes=47, in_feats=3, hidden_dim=32,history_window=VENTANA, pred_window=VENTANA,kernel_size=15, tcn_channels=(8,), kernel_size_graph=3, dropout=0.0,).to(DEVICE)
    modelo.load_state_dict(checkpoint['state_dict'], strict=True)
    modelo.eval()
    print('BDSTGNN: pesos cargados desde', ruta)
    return modelo, datos


def cargar_mepo_guardado(regiones_test, objetivo='nuevos'):
    ruta = PESOS / ('MepoGNN_activos.pt' if objetivo == 'activos' else 'MepoGNN.pt')
    checkpoint = torch.load(ruta, map_location='cpu', weights_only=True)
    media = checkpoint['media'].cpu().numpy()
    desviacion = checkpoint['desviacion'].cpu().numpy()
    commute = checkpoint['commute']

    modelo = MepoGNNComplete(num_nodes=47, adpinit=commute.to(DEVICE), glm_type='Adaptive',in_dim=3, in_len=VENTANA, out_len=VENTANA, dropout=0.5,).to(DEVICE)
    modelo.load_state_dict(checkpoint['state_dict'], strict=True)
    modelo.eval()
    print('MepoGNN: pesos cargados desde', ruta)
    return modelo, media, desviacion


def comparar_tres(carpeta):
    from datos_y_metricas import metric_values
    claves = ['origin_date', 'target_date', 'horizon', 'location']
    base = pd.read_csv(carpeta / 'active_predictions_aligned.csv')
    extra = pd.read_csv(carpeta / 'MepoGNN_active_supervised_predictions.csv')
    datos = base.merge(extra, on=claves, how='outer', validate='one_to_one', indicator=True)

    datos = datos.drop(columns=['_merge', 'y_true']).rename(columns={'y_pred': 'y_pred_Mepo_supervised'})
    datos.to_csv(carpeta / 'active_predictions_tres.csv', index=False)
    resumen = pd.read_csv(carpeta / 'active_comparison_metrics.csv')
    filas = []
    for h in ['all', *sorted(datos.horizon.unique())]:
        grupo = datos if h == 'all' else datos[datos.horizon == h]
        filas.append(dict(model='MepoGNN_active_supervised', target='active_cases', horizon=h, n_values=len(grupo), **metric_values(grupo.y_pred_Mepo_supervised, grupo.active_true)))
    resumen = pd.concat([resumen, pd.DataFrame(filas)], ignore_index=True)
    resumen.to_csv(carpeta / 'active_comparison_tres.csv', index=False)
    print(resumen[resumen.horizon == 'all'].to_string(index=False))


def elegir_accion():
    print('\n1. Entrenar y evaluar ambos (guarda/reemplaza los pesos)')
    print('2. Solo evaluar ambos usando pesos guardados')
    print('3. Solo entrenar ambos y guardar pesos')
    print('4. Revisar los datos, sin entrenar')
    print('5. Entrenar solo MepoGNN para activos (guarda/reemplaza MepoGNN_activos.pt)')
    print('6. Evaluar los tres modelos con pesos guardados')
    print('0. Salir')
    opciones = {'1': 'entrenar-evaluar', '2': 'evaluar', '3': 'entrenar', '4': 'revisar', '5': 'entrenar-activos', '6': 'evaluar-tres', '0': 'salir'}
    while True:
        try:
            opcion = input('Elige una opción: ').strip()
        except EOFError:
            raise ValueError('No hay terminal interactiva. Usa --accion evaluar, entrenar-evaluar, entrenar o revisar.')
        if opcion in opciones:
            return opciones[opcion]
        print('Elige una opción del 0 al 6.')


def main():
    global DEVICE, PESOS, SALIDA, EPOCAS_BDST, EPOCAS_MEPO, PROTOCOLO, INICIO_TEST, FIN_TEST, PERIODO_TRAIN
    parser = argparse.ArgumentParser(description='Entrenar una vez o evaluar con pesos guardados.')
    parser.add_argument('--accion', choices=['entrenar-evaluar', 'evaluar', 'entrenar', 'revisar', 'entrenar-activos', 'evaluar-tres'])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default=os.environ.get('DEVICE', 'auto'))
    parser.add_argument('--pesos', type=Path, default=PESOS)
    parser.add_argument('--resultados', type=Path, default=SALIDA)
    parser.add_argument('--epocas-bdst', type=int, default=EPOCAS_BDST)
    parser.add_argument('--epocas-mepo', type=int, default=EPOCAS_MEPO)
    parser.add_argument('--protocolo', choices=['comun', 'paper'], default=PROTOCOLO)
    parser.add_argument('--inicio-test', default=INICIO_TEST)
    parser.add_argument('--inicio-train', default=PERIODO_TRAIN[0])
    parser.add_argument('--fin-train', default=PERIODO_TRAIN[1])
    parser.add_argument('--fin-test', default=FIN_TEST)
    args = parser.parse_args()
    PROTOCOLO = args.protocolo
    INICIO_TEST, FIN_TEST = args.inicio_test, args.fin_test
    PERIODO_TRAIN = (args.inicio_train, args.fin_train)
    PESOS, SALIDA = args.pesos.resolve(), args.resultados.resolve()
    EPOCAS_BDST, EPOCAS_MEPO = args.epocas_bdst, args.epocas_mepo
    accion = args.accion or ('revisar' if SOLO_REVISAR_DATOS else elegir_accion())
    if accion == 'salir':
        return

    DEVICE = torch.device('cuda' if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')

    limite = pd.Timestamp('2022-05-15') if PROTOCOLO == 'comun' else pd.Timestamp('2022-06-14')
    entrenar = accion in ['entrenar', 'entrenar-evaluar']
    evaluar = accion in ['evaluar', 'entrenar-evaluar', 'evaluar-tres']
    if entrenar:
        check(EPOCAS_BDST > 0 and EPOCAS_MEPO > 0, 'Épocas deben ser positivas.')
    if accion == 'entrenar-activos':
        check(EPOCAS_MEPO > 0, 'Las épocas deben ser positivas.')
    if accion in ['evaluar', 'evaluar-tres']:
        archivos = ['BDSTGNN.pt', 'MepoGNN.pt']
        if accion == 'evaluar-tres':
            archivos.append('MepoGNN_activos.pt')
        faltantes = [str(PESOS / nombre) for nombre in archivos if not (PESOS / nombre).is_file()]
        check(not faltantes, 'Faltan checkpoints: ' + ', '.join(faltantes) + '. Copia tus .pt o elige entrenar. NO se entrenó automáticamente.')

    print('Acción:', accion, '| Dispositivo:', DEVICE)
    print('Pesos:', PESOS)
    test_mepo, auditoria = prepare_mepo_csv(CARPETA, CSV, INICIO_TEST, FIN_TEST)
    indices = origins(test_mepo['dates'], INICIO_TEST, FIN_TEST, VENTANA, VENTANA, PASO_TEST)
    print('Ventanas por modelo:', len(indices), '| Prefecturas:', len(test_mepo['locations']))
    print('Activos negativos recortados:', auditoria['active_negative_rows_clipped'])
    if accion == 'revisar':
        print('Datos revisados; no se cargaron pesos ni se entrenó.')
        return
    SALIDA.mkdir(parents=True, exist_ok=True)
    if accion == 'entrenar-activos':
        PESOS.mkdir(parents=True, exist_ok=True)
        loaders, commute, media, desviacion = cargar_mepo('activos')
        entrenar_mepo(loaders, commute, media, desviacion, test_mepo['locations'], 'activos')
        print('Pesos guardados en', PESOS / 'MepoGNN_activos.pt')
        return
    if entrenar:
        PESOS.mkdir(parents=True, exist_ok=True)
    (SALIDA / 'auditoria_datos.json').write_text(json.dumps(auditoria, indent=2), encoding='utf-8')
    protocolo = {
        'accion': accion, 'test': [INICIO_TEST, FIN_TEST], 'ventana': VENTANA,
        'paso_test': PASO_TEST, 'device': str(DEVICE), 'pesos': str(PESOS),
        'epocas_bdst_solicitadas': EPOCAS_BDST if entrenar else None,
        'epocas_mepo_solicitadas': EPOCAS_MEPO if entrenar else None,
        'mepo': 'Adaptive de 3 entradas', 'protocolo': PROTOCOLO,
        'periodo_entrenamiento': list(PERIODO_TRAIN) if PROTOCOLO == 'comun' else 'BDSTGNN: 2022-01-15..2022-06-14 | MepoGNN: 2020-04-01..2021-09-21',
        'objetivos_mepo': ['nuevos', 'activos'] if accion == 'evaluar-tres' else ['nuevos'],
        'periodos': 'Originales de cada parser; test externo no se usa para entrenar ni ajustar escalas',
    }
    (SALIDA / 'protocolo.json').write_text(json.dumps(protocolo, indent=2), encoding='utf-8')

    if entrenar:
        datos_bdst = cargar_bdst()
        check(len(datos_bdst['train_x']) > 0 and len(datos_bdst['val_x']) > 0, 'Split BDSTGNN vacío.')
        modelo_bdst = entrenar_bdst(datos_bdst)
    else:
        modelo_bdst, datos_bdst = cargar_bdst_guardado()
    if evaluar:
        evaluar_bdst(modelo_bdst, datos_bdst)
    del modelo_bdst, datos_bdst
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if entrenar:
        loaders, commute, media, desviacion = cargar_mepo()
        modelo_mepo = entrenar_mepo(loaders, commute, media, desviacion, test_mepo['locations'])
    else:
        modelo_mepo, media, desviacion = cargar_mepo_guardado(test_mepo['locations'])
    if evaluar:
        evaluar_mepo(modelo_mepo, test_mepo, media, desviacion)
        compare_active_outputs(SALIDA)
        if accion == 'evaluar-tres':
            del modelo_mepo
            modelo, media, desviacion = cargar_mepo_guardado(test_mepo['locations'], 'activos')
            evaluar_mepo(modelo, test_mepo, media, desviacion, 'activos')
            comparar_tres(SALIDA)
    print('Terminado. Pesos:', PESOS, '| Resultados:', SALIDA)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise SystemExit(f'ERROR: {error}')
