import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--resultados', default='resultados_comparacion')
ap.add_argument('--prefecturas', nargs='*', default=['Tokyo', 'Osaka', 'Aichi', 'Hokkaido'])
a = ap.parse_args()
R = Path(a.resultados); G = R / 'graficas'; G.mkdir(exist_ok=True)

MODELOS = {'y_pred_BDST': 'BDSTGNN (directo)','y_pred_Mepo': 'MepoGNN (derivado de SIR)','y_pred_Mepo_supervised': 'MepoGNN corregido (activos)'}
COL = {'y_pred_BDST': '#d95f02', 'y_pred_Mepo': '#7570b3', 'y_pred_Mepo_supervised': '#1b9e77'}

met = pd.read_csv(R / 'active_comparison_tres.csv')
met.to_csv(R / 'comparacion_metricas.csv', index=False)

p = pd.read_csv(R / 'active_predictions_tres.csv', parse_dates=['origin_date', 'target_date'])
p = p.rename(columns={'active_true': 'real'})
p.to_csv(R / 'comparacion_diaria_ancha.csv', index=False)
nac = p.groupby(['target_date', 'horizon'], as_index=False)[['real', *MODELOS]].sum()
nac.to_csv(R / 'comparacion_diaria_nacional.csv', index=False)

def serie(ax, d, h, titulo):
    d = d[d.horizon == h].sort_values('target_date')
    ax.plot(d.target_date, d.real, color='black', lw=2.2, label='Real (dataset)')
    for c, n in MODELOS.items():
        ax.plot(d.target_date, d[c], color=COL[c], lw=1.4, label=n)
    ax.set_title(f'{titulo} — horizonte {h} día(s)', fontsize=11)
    ax.grid(alpha=.3); ax.ticklabel_format(axis='y', style='plain')
    ax.tick_params(axis='x', rotation=30)

for h in (1, 7, 14):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    serie(ax, nac, h, 'Casos activos, total Japon')
    ax.set_ylabel('Casos activos'); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(G / f'serie_nacional_h{h}.png', dpi=150); plt.close(fig)

pref = [x for x in a.prefecturas if x in set(p.location)]
for h in (1, 7, 14):
    fig, axs = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for ax, loc in zip(axs.ravel(), pref):
        serie(ax, p[p.location == loc], h, loc)
    axs[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(G / f'serie_prefecturas_h{h}.png', dpi=150); plt.close(fig)

m = met[met.horizon.astype(str) == 'all'].set_index('model')
orden = ['BDSTGNN_direct', 'MepoGNN_derived', 'MepoGNN_active_supervised']
etq = ['BDSTGNN\n(directo)', 'MepoGNN\n(derivado)', 'MepoGNN\n(corregido)']
cols = [COL['y_pred_BDST'], COL['y_pred_Mepo'], COL['y_pred_Mepo_supervised']]
fig, axs = plt.subplots(1, 5, figsize=(16, 3.8))
for ax, k in zip(axs, ['MAE', 'RMSE', 'MAPE_nonzero_pct', 'R2', 'CCC']):
    v = m.loc[orden, k]
    b = ax.bar(etq, v, color=cols)
    ax.set_title(k + (' (%)' if 'pct' in k else '')); ax.bar_label(b, fmt='%.2f' if k in ('R2', 'CCC') else '%.0f', fontsize=8)
    ax.tick_params(axis='x', labelsize=7); ax.axhline(0, color='k', lw=.6)
fig.suptitle('Metricas globales (todos los horizontes) - menor es mejor salvo R2/CCC', fontsize=11)
fig.tight_layout(); fig.savefig(G / 'barras_metricas.png', dpi=150); plt.close(fig)

mh = met[met.horizon.astype(str) != 'all'].copy(); mh['horizon'] = mh.horizon.astype(int)
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for mod, c in zip(orden, cols):
    d = mh[mh.model == mod].sort_values('horizon')
    axs[0].plot(d.horizon, d.MAE, marker='o', color=c, label=mod)
    axs[1].plot(d.horizon, d.MAPE_nonzero_pct, marker='o', color=c, label=mod)
axs[0].set_title('MAE por horizonte'); axs[1].set_title('MAPE (%) por horizonte')
for ax in axs: ax.set_xlabel('Horizonte (días)'); ax.grid(alpha=.3)
axs[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(G / 'error_por_horizonte.png', dpi=150); plt.close(fig)

p['ae'] = 0
top = p.groupby('location').real.mean().nlargest(10).index
fig, ax = plt.subplots(figsize=(12, 4))
w = .27
for i, (c, n) in enumerate(MODELOS.items()):
    e = (p[c] - p.real).abs().groupby(p.location).mean().loc[top]
    ax.bar([x + (i - 1) * w for x in range(len(top))], e, w, color=COL[c], label=n)
ax.set_xticks(range(len(top))); ax.set_xticklabels(top, rotation=30)
ax.set_title('MAE por prefectura (10 con mas casos activos)'); ax.legend(frameon=False); ax.grid(alpha=.3, axis='y')
fig.tight_layout(); fig.savefig(G / 'barras_mae_prefectura.png', dpi=150); plt.close(fig)
print('Listo:', sorted(x.name for x in G.iterdir()))
