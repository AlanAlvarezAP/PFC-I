import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn

class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class BDSTGNNWithDGL(nn.Module):
    def __init__(self, num_nodes: int, in_feats: int, hidden_dim: int, history_window: int, pred_window: int, kernel_size: int = 25):
        super().__init__()
        self.num_nodes = num_nodes
        self.history_window = history_window
        self.pred_window = pred_window
        
        self.decom = SeriesDecomp(kernel_size)
        self.fc_in = nn.Linear(in_feats, hidden_dim)
        
        # 1. Backbone Graph Learning (Rama Estática) -> Embeddings de tamaño 16
        self.nodevec = nn.Parameter(torch.randn(num_nodes, 16), requires_grad=True)
        
        # 2. Temporal Graph Learning (Rama Dinámica) -> Proyección temporal a dimensión 16 por nodo
        self.dyn_proj = nn.Linear(history_window * in_feats, 16)
        
        # Capa GCN de DGL
        self.gcn = dglnn.GraphConv(hidden_dim, hidden_dim, weight=True, bias=True)
        
        # Proyecciones finales de predicción
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x_seq):
        """
        x_seq: [Batch, History_Window, Nodes, Features]
        """
        B, T, N, C = x_seq.shape
        device = x_seq.device
        
        # Descomposición de series temporales
        x_flat = x_seq.reshape(B * N, T, C)
        res, trend = self.decom(x_flat)
        x_combined = (res + trend).reshape(B, T, N, C)
        
        # --- APRENDIZAJE DE GRAFO DINÁMICO (Alineado a Fig. 3) ---
        
        # Rama Estática (Backbone): [N, 16] x [16, N] -> [N, N]
        sta_adj = torch.softmax(torch.relu(torch.mm(self.nodevec, self.nodevec.T)), dim=1)
        
        # Rama Temporal: Procesar la historia de cada nodo y proyectarla a [N, 16]
        x_node_history = x_seq.permute(0, 2, 1, 3).reshape(B, N, T * C) # [B, N, T*C]
        h_temporal_b = self.dyn_proj(x_node_history) # [B, N, 16]
        h_temporal = h_temporal_b.mean(dim=0)        # Promedio en el batch -> [N, 16]
        
        # Multiplicación tensorial temporal: [N, 16] x [16, N] -> [N, N]
        dyn_score = torch.matmul(h_temporal, h_temporal.transpose(0, 1))
        dyn_adj = torch.softmax(torch.relu(dyn_score), dim=-1) # [N, N]
        
        # Fusión exacta de ambas matrices [N, N] + [N, N]
        fused_adj = sta_adj + dyn_adj
        fused_adj = torch.softmax(torch.relu(fused_adj), dim=-1)
        
        # Construcción del Grafo DGL utilizando la adyacencia fusionada
        src, dst = torch.nonzero(fused_adj, as_tuple=True)
        weights = fused_adj[src, dst]
        
        g = dgl.graph((src, dst), num_nodes=N).to(device)
        g.edata['weight'] = weights
        
        # --- PROYECCIÓN ESPACIAL (GCN) ---
        h = self.fc_in(x_combined[:, -1, :, :]) # [B, N, hidden_dim]
        
        gcn_outputs = []
        for b in range(B):
            res_gcn = self.gcn(g, h[b], edge_weight=g.edata['weight'])
            gcn_outputs.append(res_gcn)
            
        spatial_out = torch.stack(gcn_outputs, dim=0) # [B, N, hidden_dim]
        
        # --- PREDICCIÓN HACIA EL HORIZONTE ---
        preds = []
        curr = spatial_out
        for _ in range(self.pred_window):
            step_pred = self.predictor(curr) # [B, N, 1]
            preds.append(step_pred)
            
        return torch.stack(preds, dim=1) # [B, Pred_Len, N, 1]