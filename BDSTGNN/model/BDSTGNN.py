import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import dgl
import dgl.function as fn

class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        pad = (self.kernel_size - 1) // 2
        x = torch.cat([x[:, :1].repeat(1, pad, 1), x, x[:, -1:].repeat(1, pad, 1)], dim=1)
        return self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        return x - trend, trend


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(n_in, n_out, kernel_size, padding=pad, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(n_out, n_out, kernel_size, padding=pad, dilation=dilation))
        self.net = nn.Sequential(self.conv1, Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout), self.conv2, Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout))
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(self.net(x) + res)


class BDSTGNNWithDGL(nn.Module):
    def __init__(self, num_nodes, in_feats, hidden_dim, history_window, pred_window, kernel_size=15, tcn_channels=(8,), kernel_size_graph=3, dropout=0.0, use_dgl=True):
        super().__init__()
        self.num_nodes, self.T, self.pred_window, self.use_dgl = num_nodes, history_window, pred_window, use_dgl

        self.fc_in = nn.Linear(in_feats, hidden_dim)
        self.decom = SeriesDecomp(kernel_size)
        self.fc_res = nn.Linear(hidden_dim, hidden_dim)
        self.fc_trend = nn.Linear(hidden_dim, hidden_dim)

        self.nodevec = nn.Parameter(torch.empty(num_nodes, 10))
        nn.init.xavier_uniform_(self.nodevec)
        layers, prev = [], hidden_dim
        for i, c in enumerate(tcn_channels):
            layers.append(TemporalBlock(prev, c, kernel_size_graph, 2 ** i, dropout))
            prev = c
        self.tcn = nn.Sequential(*layers)

        self.gcn_w = nn.Parameter(torch.randn(history_window, hidden_dim, hidden_dim))
        self.intra = nn.Linear(hidden_dim * history_window, 2)
        self.predictor_st = nn.Linear(hidden_dim * history_window, pred_window)

        self._graphs = {}

    def _batched_graph(self, n_graphs, device):
        key = (n_graphs, str(device))
        if key not in self._graphs:
            N = self.num_nodes
            src = torch.arange(N).repeat_interleave(N)
            dst = torch.arange(N).repeat(N)
            g = dgl.graph((src, dst), num_nodes=N)
            self._graphs[key] = dgl.batch([g] * n_graphs).to(device)
        return self._graphs[key]

    def _support(self, adj, h):
        B, T, N, _ = adj.shape
        if self.use_dgl:
            g = self._batched_graph(B * T, adj.device)
            g.edata['w'] = adj.reshape(B * T, N * N).reshape(-1)
            g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 's'))
            colsum = g.ndata.pop('s').reshape(B, T, N)
            g.edata.pop('w')
        else:
            colsum = adj.sum(dim=2)
        return h * colsum.permute(0, 2, 1).unsqueeze(-1)

    def forward(self, x, data_true, I_min, I_max, N):
        B, Nn, T, _ = x.shape
        emb = self.fc_in(x)
        e2 = emb.reshape(B * Nn, T, -1)
        res, trend = self.decom(e2)
        h = (self.fc_res(res) + self.fc_trend(trend)).reshape(B, Nn, T, -1)

        sta = F.softmax(F.relu(self.nodevec @ self.nodevec.T), dim=1)
        d = self.tcn(e2.permute(0, 2, 1)).reshape(B, Nn, -1, T).permute(0, 3, 1, 2)
        dyn = F.softmax(F.relu(d @ d.transpose(2, 3)), dim=3)
        adj = F.softmax(F.relu(dyn + sta), dim=3)

        g = torch.einsum('bwtd,tdo->bwto', self._support(adj, h), self.gcn_w)
        flat = g.reshape(B, Nn, -1)

        pred_nn = self.predictor_st(flat)

        bg = torch.sigmoid(self.intra(flat))
        beta, gamma = bg[..., 0], bg[..., 1]
        I, R, S = data_true[:, :, -1, 0], data_true[:, :, -1, 1], data_true[:, :, -1, 2]
        outs = []
        for _ in range(self.pred_window):
            dI = (beta * S / N - gamma) * I
            dR = gamma * I
            I, R = I + dI, R + dR
            S = N - I - R
            outs.append((I - I_min) / (I_max - I_min))
        return pred_nn, torch.stack(outs, dim=2)


def check_dgl_parity(num_nodes=47, hist=5, pred=5, device='cpu'):
    torch.manual_seed(0)
    m = BDSTGNNWithDGL(num_nodes, 3, 32, hist, pred).to(device).eval()
    x = torch.rand(2, num_nodes, hist, 3, device=device)
    xt = torch.rand(2, num_nodes, hist, 3, device=device) * 1e4 + 1
    Imin, Imax = torch.zeros(1, num_nodes, device=device), torch.full((1, num_nodes), 2e4, device=device)
    Npop = torch.full((1, num_nodes), 1e6, device=device)
    with torch.no_grad():
        m.use_dgl = True;  a = m(x, xt, Imin, Imax, Npop)
        m.use_dgl = False; b = m(x, xt, Imin, Imax, Npop)
    ok = all(torch.allclose(u, v, atol=1e-4, rtol=1e-4) for u, v in zip(a, b))
    print('DGL vs denso:', ok)
    return ok