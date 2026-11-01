import torch
import torch.nn as nn
import torch.nn.functional as F

class nconv(nn.Module):
    def forward(self, x, A):
        if len(A.shape) == 2:
            x = torch.einsum('vw, ncwl->ncvl', A, x)
        else:
            x = torch.einsum('nvw, ncwl->ncvl', A, x)
        return x.contiguous()


class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=2, order=1):
        super().__init__()
        self.nconv = nconv()
        self.mlp = nn.Conv2d(order * support_len * c_in, c_out, kernel_size=(1, 1))
        self.dropout, self.order = dropout, order

    def forward(self, x, support):
        out = []
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = self.nconv(x1, a)
                out.append(x1)
        h = self.mlp(torch.cat(out, dim=1))
        return F.dropout(h, self.dropout, training=self.training)


class STCell(nn.Module):
    def __init__(self, num_nodes, dropout, in_dim, residual_channels, dilation_channels, skip_channels, kernel_size, blocks, layers):
        super().__init__()
        self.blocks, self.layers = blocks, layers
        self.filter_convs, self.gate_convs = nn.ModuleList(), nn.ModuleList()
        self.skip_convs, self.ln, self.gconv = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.start_conv = nn.Conv2d(in_dim, residual_channels, kernel_size=(1, 1))
        receptive_field = 1
        for _ in range(blocks):
            additional_scope, dilation = 1, 1
            for _ in range(layers):
                self.filter_convs.append(nn.Conv1d(residual_channels, dilation_channels, kernel_size, dilation=dilation))
                self.gate_convs.append(nn.Conv1d(residual_channels, dilation_channels, kernel_size, dilation=dilation))
                self.skip_convs.append(nn.Conv1d(dilation_channels, skip_channels, 1))
                dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                self.ln.append(nn.LayerNorm([residual_channels, num_nodes, (2 ** layers - 1) * blocks + 2 - receptive_field]))
                self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=2))
        self.receptive_field = receptive_field

    @staticmethod
    def _tconv(conv, x):
        B, C, N, T = x.shape
        y = conv(x.permute(0, 2, 1, 3).reshape(B * N, C, T))
        return y.reshape(B, N, y.shape[1], y.shape[2]).permute(0, 2, 1, 3)

    def forward(self, x, adp_g):
        if x.size(3) < self.receptive_field:
            x = F.pad(x, (self.receptive_field - x.size(3), 0, 0, 0))
        x = self.start_conv(x)
        skip, dense = 0, 0
        for i in range(self.blocks * self.layers):
            res = x
            x = torch.tanh(self._tconv(self.filter_convs[i], x)) * torch.sigmoid(self._tconv(self.gate_convs[i], x))
            s = self._tconv(self.skip_convs[i], x)
            skip = s if i == 0 else torch.cat((s, skip[:, :, :, -s.size(3):]), dim=1)
            x = self.gconv[i](x, adp_g)
            dense = res[:, :, :, -x.size(3):] + (dense if i == 0 else dense[:, :, :, -x.size(3):])
            gate = torch.sigmoid(x)
            x = self.ln[i](x * gate + dense * (1 - gate))
        return skip


class SIRcell(nn.Module):
    def forward(self, param_b, param_g, mob, SIR):
        if len(mob.shape) == 2:
            mob = mob.unsqueeze(0).expand(SIR.shape[0], -1, -1)
        n = SIR.shape[-2]
        S, I, R = SIR[..., [0]], SIR[..., [1]], SIR[..., [2]]
        pop = (S + I + R).expand(-1, n, n)
        Ie = I.expand(-1, n, n)
        propagation = ((mob / pop * Ie).sum(1) + (mob / pop * Ie.transpose(1, 2)).sum(2)).unsqueeze(2)
        I_new = param_b * propagation
        R_t = I * param_g + R
        I_t = I + I_new - I * param_g
        S_t = S - I_new
        return torch.cat((I_new, S_t, I_t, R_t), dim=-1)


class MepoGNNComplete(nn.Module):
    def __init__(self, num_nodes, adpinit=None, glm_type='Dynamic', dropout=0.5, in_dim=4, in_len=14, out_len=14,residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,kernel_size=2, blocks=2, layers=3):
        super().__init__()
        self.glm_type, self.out_len = glm_type, out_len
        self.stcell = STCell(num_nodes, dropout, in_dim, residual_channels, dilation_channels, skip_channels,kernel_size, blocks, layers)
        total_skip = skip_channels * blocks * layers
        self.end_conv_b1 = nn.Conv2d(total_skip, end_channels, 1)
        self.end_conv_b2 = nn.Conv2d(end_channels, out_len, 1)
        self.end_conv_g1 = nn.Conv2d(total_skip, end_channels, 1)
        self.end_conv_g2 = nn.Conv2d(end_channels, out_len, 1)
        self.SIRcell = SIRcell()

        if glm_type == 'Adaptive':
            log_g = torch.log(adpinit + 1.0)
            self.register_buffer('max_log', log_g.max())
            self.g_rescaled = nn.Parameter(log_g / self.max_log)
        elif glm_type == 'Dynamic':
            self.inc_init = nn.Parameter(torch.empty(out_len, in_len))
            nn.init.normal_(self.inc_init, 1, 0.01)
            self.od_scale_factor = 3
        else:
            raise NotImplementedError('glm_type debe ser Adaptive o Dynamic')

    def forward(self, x_node, SIR, od=None, max_od=None):
        x = x_node.permute(0, 3, 2, 1)
        if self.glm_type == 'Adaptive':
            mob = torch.exp(torch.relu(self.g_rescaled * self.max_log))
            skip = self.stcell(x, [mob / mob.sum(1, True), mob.T / mob.T.sum(1, True)])
            mob_at = lambda i: mob
        else:
            assert od is not None and max_od is not None, "Dynamic necesita od y max_od (el maximo de OD del train)"
            incidence = torch.softmax(self.inc_init, dim=1)
            mob = torch.einsum('kl,blnmc->bknmc', incidence, od).squeeze(-1)
            g = mob.mean(1)
            g_t = g.permute(0, 2, 1)
            skip = self.stcell(x, [g / g.sum(2, True), g_t / g_t.sum(2, True)])
            mob_at = lambda i: mob[:, i, ...] * max_od * self.od_scale_factor
        param_b = torch.sigmoid(self.end_conv_b2(F.relu(self.end_conv_b1(F.relu(skip)))))
        param_g = torch.sigmoid(self.end_conv_g2(F.relu(self.end_conv_g1(F.relu(skip)))))
        outs, cur = [], SIR[:, -1, ...]
        for i in range(self.out_len):
            nsir = self.SIRcell(param_b[:, i, ...], param_g[:, i, ...], mob_at(i), cur)
            cur = nsir[..., 1:]
            outs.append(nsir[..., [0]])
        return torch.stack(outs, dim=1)

MepoGNNCore = MepoGNNComplete