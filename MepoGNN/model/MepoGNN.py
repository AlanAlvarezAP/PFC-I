import torch
import torch.nn as nn
import torch.nn.functional as F

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        if len(A.shape) == 2:
            x = torch.einsum('vw, ncwl->ncvl', A, x)
        else:
            x = torch.einsum('nvw, ncwl->ncvl', A, x)
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=2, order=1):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = []
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class stcell_core(nn.Module):
    def __init__(self, num_nodes, dropout, in_dim, residual_channels, dilation_channels, skip_channels, kernel_size, blocks, layers):
        super(stcell_core, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.num_nodes = num_nodes
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.ln = nn.ModuleList()
        self.gconv = nn.ModuleList()
        
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1,1))
        receptive_field = 1
        self.supports_len = 2

        for b in range(blocks):
            additional_scope = 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(nn.Conv1d(residual_channels, dilation_channels, kernel_size=kernel_size, dilation=new_dilation))
                self.gate_convs.append(nn.Conv1d(residual_channels, dilation_channels, kernel_size=kernel_size, dilation=new_dilation))
                self.residual_convs.append(nn.Conv1d(dilation_channels, residual_channels, kernel_size=1))
                self.skip_convs.append(nn.Conv1d(dilation_channels, skip_channels, kernel_size=1))
                
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                
                self.ln.append(nn.LayerNorm([residual_channels, num_nodes, (2 ** layers - 1) * blocks + 2 - receptive_field]))
                self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.receptive_field = receptive_field

    def forward(self, input, adp_g):
        B, C, N, T = input.shape
        in_len = T
        
        if in_len < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
            
        x = self.start_conv(x)
        skip = 0

        for i in range(self.blocks * self.layers):
            res = x
            _, c_res, n_nodes, t_steps = x.shape
            x_flat = x.permute(0, 2, 1, 3).reshape(B * n_nodes, c_res, t_steps)

            filter_out = torch.tanh(self.filter_convs[i](x_flat))
            gate_out = torch.sigmoid(self.gate_convs[i](x_flat))
            x_conv = filter_out * gate_out

            t_out = x_conv.shape[-1]
            x_conv = x_conv.reshape(B, n_nodes, -1, t_out).permute(0, 2, 1, 3)

            s = x_conv
            s_flat = s.permute(0, 2, 1, 3).reshape(B * n_nodes, s.shape[1], t_out)
            s_skip = self.skip_convs[i](s_flat)
            s_skip = s_skip.reshape(B, n_nodes, -1, s_skip.shape[-1]).permute(0, 2, 1, 3)

            try:
                skip = torch.cat((s_skip, skip[:, :, :, -s_skip.size(3):]), dim=1)
            except:
                skip = s_skip

            x_gcn = self.gconv[i](x_conv, adp_g)

            try:
                dense = dense[:, :, :, -x_gcn.size(3):]
            except:
                dense = 0
            
            res_sliced = res[:, :, :, -x_gcn.size(3):]
            dense = res_sliced + dense

            gate = torch.sigmoid(x_gcn)
            x_next = x_gcn * gate + dense * (1 - gate)
            
            x = self.ln[i](x_next)

        return skip
    
class MepoGNNCore(nn.Module):
    def __init__(self, num_nodes, in_dim=4, in_len=14, out_len=14):
        super().__init__()
        self.stcell = stcell_core(num_nodes=num_nodes, dropout=0.5, in_dim=in_dim, skip_channels=256, kernel_size=2, blocks=2, layers=3, residual_channels=32, dilation_channels=32)
        self.inc_init = nn.Parameter(torch.empty(out_len, in_len), requires_grad=True)
        nn.init.normal_(self.inc_init, 1, 0.01)

    def forward(self, x_node, od):
        x_node_permuted = x_node.permute(0, 3, 2, 1)
        incidence = torch.softmax(self.inc_init, dim=1)
        mob = torch.einsum('kl,blnmc->bknmc', incidence, od).squeeze(-1)
        
        g = mob.mean(1)
        g_t = g.permute(0, 2, 1)
        
        g_dyn = [g / g.sum(2, True), g_t / g_t.sum(2, True)]

        h_st = self.stcell(x_node_permuted, g_dyn)
        return h_st