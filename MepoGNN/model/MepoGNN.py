import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn

class DGLGCNLayer(nn.Module):
    """Capa espacial GCN utilizando DGL y pesos de aristas dinámicos (movilidad/grafo)."""
    def __init__(self, c_in, c_out, dropout):
        super().__init__()
        self.gconv = dglnn.GraphConv(c_in, c_out, weight=True, bias=False)
        self.dropout = dropout

    def forward(self, x, num_nodes, mob_weights):
        """
        x: [Batch, Channels, Nodes, Time]
        mob_weights: [Batch, Nodes, Nodes] o matriz de adyacencia
        """
        B, C, N, T = x.shape
        device = x.device
        outputs = []

        for t in range(T):
            xt = x[:, :, :, t].permute(0, 2, 1) # [B, N, C]
            batch_res = []
            for b in range(B):
                # 1. Obtener la adyacencia del batch actual
                adj_b = mob_weights[b] if len(mob_weights.shape) == 3 else mob_weights
                
                # 2. Forzar de manera radical a que tenga exactamente 2 dimensiones [N, N]
                while adj_b.dim() > 2:
                    adj_b = adj_b.squeeze(0) # Elimina dimensiones de tamaño 1 si existieran
                
                if adj_b.shape != (num_nodes, num_nodes):
                    # Si por alguna razón la forma no coincide, recortar a los últimos N nodos
                    adj_b = adj_b[-num_nodes:, -num_nodes:]

                # En este punto adj_b es 100% una matriz de [N, N]
                src, dst = torch.nonzero(adj_b, as_tuple=True)
                weights = adj_b[src, dst]
                
                g_dyn = dgl.graph((src, dst), num_nodes=N).to(device)
                g_dyn.edata['weight'] = weights
                
                res = self.gconv(g_dyn, xt[b], edge_weight=g_dyn.edata['weight'])
                batch_res.append(res)
                
            out_t = torch.stack(batch_res, dim=0).permute(0, 2, 1) # [B, C_out, N]
            outputs.append(out_t)
            
        h = torch.stack(outputs, dim=-1) # [B, C_out, N, T]
        return F.dropout(h, self.dropout, training=self.training)
    
class STCellCore(nn.Module):
    """Bloque de Capas Espacio-Temporales adaptado para manejar tensores 4D con Conv2d/Conv1d."""
    def __init__(self, num_nodes, dropout, in_dim, residual_channels, dilation_channels, kernel_size, blocks, layers):
        super().__init__()
        self.num_nodes = num_nodes
        self.blocks = blocks
        self.layers = layers
        
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv = nn.ModuleList()
        
        # Usamos Conv2d con kernel_size=(1, kernel_size) tal como el Graph-WaveNet/MepoGNN original
        # para operar de forma segura sobre [Batch, Channels, Nodes, Time] sin errores de dimensiones.
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))

        for b in range(blocks):
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, kernel_size), dilation=(1, new_dilation), padding=(0, 0)))
                self.gate_convs.append(nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, kernel_size), dilation=(1, new_dilation), padding=(0, 0)))
                self.skip_convs.append(nn.Conv2d(dilation_channels, residual_channels, kernel_size=(1, 1)))
                
                new_dilation *= 2
                self.gconv.append(DGLGCNLayer(dilation_channels, residual_channels, dropout))

    def forward(self, input, mob_weights):
        """
        input: [B, C, N, T]
        mob_weights: [B, N, N]
        """
        x = self.start_conv(input)
        
        for i in range(self.blocks * self.layers):
            res = x
            
            # Manejo de padding temporal para evitar desajustes de tamaño en el tiempo (T)
            # Como kernel_size=2 y dilation varía, rellenamos a la izquierda en el eje temporal
            pad_len = x.size(3) - F.pad(x, (1, 0)).size(3) # simplificado o padding dinámico
            x_padded = F.pad(x, (1, 0)) # Padding a lo largo de la dimensión temporal T
            
            filter = torch.tanh(self.filter_convs[i](x_padded))
            gate = torch.sigmoid(self.gate_convs[i](x_padded))
            x_conv = filter * gate

            # Aplicar convolución espacial DGL-GCN utilizando los pesos del grafo
            x_gcn = self.gconv[i](x_conv, self.num_nodes, mob_weights)
            
            # Ajuste residual de dimensiones si es necesario
            if res.shape[-1] == x_gcn.shape[-1]:
                x = x_gcn + res[:, :, :, -x_gcn.size(3):]
            else:
                x = x_gcn

        return x  # Representaciones latentes de las capas ST

class MepoGNNCore(nn.Module):
    """Modelo base que solo procesa entrada, grafo y capas ST."""
    def __init__(self, num_nodes, in_dim=4, residual_channels=32, dilation_channels=32):
        super().__init__()
        self.st_cell = STCellCore(
            num_nodes=num_nodes, dropout=0.5, in_dim=in_dim, 
            residual_channels=residual_channels, dilation_channels=dilation_channels, 
            kernel_size=2, blocks=2, layers=3
        )

    def forward(self, x_node, od_matrix):
        """
        x_node: [Batch, T, N, Features] 
        od_matrix: [Batch, N, N] o [Batch, T, N, N]
        """
        # Adaptar dimensiones al formato requerido por Graph-WaveNet/MepoGNN: [B, C, N, T]
        x_in = x_node.permute(0, 3, 2, 1)
        
        if len(od_matrix.shape) == 4:
            mob_weights = od_matrix.mean(dim=1) # [B, N, N]
        else:
            mob_weights = od_matrix

        h_st = self.st_cell(x_in, mob_weights)
        return h_st