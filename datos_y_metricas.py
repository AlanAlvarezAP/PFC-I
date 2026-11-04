from pathlib import Path
import numpy as np
import pandas as pd


def check(ok, message):
    if not ok:
        raise ValueError(message)

def metric_values(p, y):
    p, y = np.asarray(p, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()

    check(p.shape == y.shape and np.isfinite(p).all() and np.isfinite(y).all(), 'Predicciones/verdad inválidas')
    e = p-y
    vy = y.var()
    vp = p.var()
    cov = ((p-p.mean())*(y-y.mean())).mean()

    nz = y != 0; den = np.abs(y)+np.abs(p)
    cden = vy+vp+(y.mean()-p.mean())**2
    return {'MAE': float(np.abs(e).mean()), 'RMSE': float(np.sqrt((e**2).mean())),
            'MAPE_nonzero_pct': float(100*np.abs(e[nz]/y[nz]).mean()) if nz.any() else np.nan,
            'MAPE_coverage': float(nz.mean()),
            'sMAPE_pct': float(200*np.divide(np.abs(e),den,out=np.zeros_like(e),where=den!=0).mean()),
            'RAE': float(np.abs(e).sum()/np.abs(y-y.mean()).sum()) if vy else np.nan,
            'R2': float(1-(e**2).mean()/vy) if vy else np.nan,
            'PCC': float(cov/np.sqrt(vp*vy)) if vp*vy else np.nan,
            'CCC': float(2*cov/cden) if cden else np.nan}

def origins(dates, start, end, history, horizon, stride):
    dates = pd.DatetimeIndex(dates)

    check(len(dates)>0 and not dates.hasnans, 'Fechas inválidas')
    check(dates.equals(pd.date_range(dates[0],dates[-1])), 'Fechas no consecutivas/duplicadas')

    a, b = pd.Timestamp(start), pd.Timestamp(end)

    check(a<=b and a in dates and b in dates, f'Se requiere {start} a {end}; disponible {dates[0].date()} a {dates[-1].date()}.')

    ia, ib = dates.get_loc(a), dates.get_loc(b)

    check(ia>=history, f'Se necesitan {history} días previos a {start}')

    ix = np.arange(ia,ib-horizon+2,stride)

    check(len(ix)>0, 'No hay ventanas completas de prueba')
    return ix

def read_csv_external(path, names, start, end, history):
    df=pd.read_csv(path)
    cols=['date','administrative_area_level','administrative_area_level_2','confirmed','recovered','deaths','population']

    check(all(c in df for c in cols), f'CSV requiere {cols}')

    df['date']=pd.to_datetime(df['date'])
    c='administrative_area_level_2'
    df=df[(df.administrative_area_level==2)&df[c].isin(names)].copy()

    dates=pd.date_range(pd.Timestamp(start)-pd.Timedelta(days=history),end)
    df=df[df.date.isin(dates)]

    check(not df.duplicated(['date',c]).any(),'CSV con fechas/regiones duplicadas')

    full=pd.MultiIndex.from_product([dates,names],names=['date',c])
    indexed=df.set_index(['date',c])
    check(full.isin(indexed.index).all(), f'CSV incompleto para {dates[0].date()} a {dates[-1].date()}: faltan fechas/regiones este termina el 05/09/2022')
    indexed=indexed.reindex(full)

    vals=indexed[['confirmed','recovered','deaths','population']].to_numpy(float)

    check(np.isfinite(vals).all(), 'CSV externo tiene nulos; resolver en la fuente, especialmente recuperados')

    confirmed,recovered,deaths,pop=(vals[:,i].reshape(len(dates),len(names)) for i in range(4))
    active=np.maximum(confirmed-recovered-deaths,0)
    removed=confirmed-active; susceptible=pop-confirmed

    check((susceptible>=0).all() and (pop>0).all(), 'Población/SIR inválidos en CSV.')
    return dates,np.stack([active,removed,susceptible],axis=-1),pop

def export_report(out, model, target, p, y, dates, ids, names):
    check(p.shape==y.shape and p.ndim==3,'Se espera (muestras,horizonte,nodos).')
    rows=[]
    for h in [None]+list(range(p.shape[1])):
        rows.append(dict(model=model,target=target,horizon='all' if h is None else h+1, **metric_values(p if h is None else p[:,h],y if h is None else y[:,h])))
    
    pd.DataFrame(rows).to_csv(out/f'{model}_metrics.csv',index=False)
    data=[]
    for b,i in enumerate(ids):
        for h in range(p.shape[1]):
            for n,name in enumerate(names):
                data.append((str(dates[i-1].date()),str(dates[i+h].date()),h+1,name,y[b,h,n],p[b,h,n]))

    pd.DataFrame(data,columns=['origin_date','target_date','horizon','location','y_true','y_pred']).to_csv(out/f'{model}_predictions.csv',index=False)
    print(model, target, rows[0],flush=True)

def prepare_mepo_csv(root, csv_path, start, end):
    df=pd.read_csv(csv_path)
    df=df[(df.administrative_area_level==2)&df.jis_code.between(1,47)].copy()
    mapping=df.groupby('administrative_area_level_2').jis_code.agg(['min','max'])

    check((mapping['min']==mapping['max']).all(), 'Codigos regionales cambiantes')

    mapping=mapping.sort_values('min')
    check(mapping['min'].tolist()==list(range(1,48)), 'Se requieren los 47 codigos unicos')
    names=mapping.index.tolist()
    raw=np.load(root/'MepoGNN/data/jp20200401_20210921.npy',allow_pickle=True).item()
    hist_dates=pd.date_range('2020-04-01',periods=len(raw['node']))

    check(str(hist_dates[-1].date())=='2021-09-21', 'NPY historico distinto al esperado')

    node=raw['node'];sir=raw['SIR']

    check(np.all(node[...,2]==hist_dates.dayofweek.to_numpy()[:,None]), 'Dia de semana histórico no coincide')
    check(np.allclose(node[...,3],node[...,0]/(sir[...,1]+1)), 'Ratio histórico distinto de incidencia/(I+1)')

    cumulative=df.pivot(index='date',columns='administrative_area_level_2',values='confirmed').reindex(columns=names).sort_index()
    historic_inc=cumulative.diff().reindex(hist_dates.strftime('%Y-%m-%d')).to_numpy()

    check(np.isfinite(historic_inc).all(),'Faltan datos históricos para validar orden regional')
    corr=np.corrcoef(historic_inc.T,node[...,0].T)[:47,47:]
    check(np.array_equal(corr.argmax(0),np.arange(47)) and np.all(np.diag(corr)>0.99), 'No se pudo alinear el orden regional del NPY con JIS del CSV')

    dates,irs,pop=read_csv_external(csv_path,names,start,end,15)
    confirmed=cumulative.reindex(dates.strftime('%Y-%m-%d')).to_numpy()
    daily=np.diff(confirmed,axis=0);dates=dates[1:];irs=irs[1:];pop=pop[1:]

    check(np.isfinite(daily).all() and (daily>=0).all(),'Incidencias negativas/incompletas, revisar correcciones en la fuente')

    sir_test=irs[..., [2,0,1]].astype(np.float32)
    node_test=np.stack([daily,np.broadcast_to(dates.dayofweek.to_numpy()[:,None],daily.shape),daily/(irs[...,0]+1)],axis=-1).astype(np.float32)
    relevant=df[df.date.between(str(dates[0].date()),str(dates[-1].date()))]
    negatives=int((relevant.confirmed-relevant.recovered-relevant.deaths<0).sum())
    audit=dict(order='JIS, validado por correlación historica uno a uno',locations=names,min_historical_correlation=float(np.diag(corr).min()),active_negative_rows_clipped=negatives,features=['daily_new_cases','day_of_week','daily_new_cases/(active+1)'],historic_population=sir[0].sum(-1).tolist(),test_population=pop[0].tolist(),source_note='Train/validación NPY original; test CSV. Poblaciones y fuentes pueden diferir')
    return dict(dates=dates,locations=names,node=node_test,SIR=sir_test),audit

def compare_active_outputs(out):
    b=pd.read_csv(out/'BDSTGNN_L14_NN_predictions.csv')
    m=pd.read_csv(out/'MepoGNN_active_derived_predictions.csv')
    keys=['origin_date','target_date','horizon','location']

    check(not b.duplicated(keys).any() and not m.duplicated(keys).any(), 'Predicciones duplicadas')

    joined=b.merge(m,on=keys,how='outer',suffixes=('_BDST','_Mepo'),indicator=True,validate='one_to_one')
    check((joined['_merge']=='both').all(), 'Los modelos no tienen identicas fechas, horizontes y regiones')
    check(np.allclose(joined.y_true_BDST,joined.y_true_Mepo,rtol=1e-7,atol=1e-4), 'Las verdades de activos no coinciden')

    joined=joined.drop(columns=['_merge','y_true_Mepo']).rename(columns={'y_true_BDST':'active_true'})
    joined.to_csv(out/'active_predictions_aligned.csv',index=False)

    rows=[]
    for name,col in [('BDSTGNN_direct','y_pred_BDST'),('MepoGNN_derived','y_pred_Mepo')]:
        for h in [None]+sorted(joined.horizon.unique().tolist()):
            a=joined if h is None else joined[joined.horizon==h]
            rows.append(dict(model=name,target='active_cases',horizon='all' if h is None else h, n_values=len(a),**metric_values(a[col].to_numpy(),a.active_true.to_numpy())))
    
    report=pd.DataFrame(rows)
    report.to_csv(out/'active_comparison_metrics.csv',index=False)
    print('COMPARACION DE ACTIVOS — mismas fechas, regiones y horizontes')
    print(report[report.horizon=='all'].to_string(index=False))
