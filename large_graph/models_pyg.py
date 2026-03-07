import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, GCNConv


class GNN_PyG(nn.Module):
    def __init__(
        self,
        node_feats,
        edge_feats,
        n_classes,
        n_layers,
        n_heads,
        n_hidden,
        edge_emb,
        activation,
        dropout,
        input_drop,
        attn_drop,
        edge_drop,
        use_attn_dst=True,
        allow_zero_in_degree=False,
        mpnn='gat',
        jk=False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.mpnn = mpnn
        self.jk = jk
        self.edge_drop = edge_drop

        self.node_encoder = nn.Linear(node_feats, n_hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Only build an edge encoder for the 'gate' (GAT with edge features) mode
        if edge_emb > 0 and mpnn == 'gate':
            self.edge_encoder = nn.ModuleList()
        else:
            self.edge_encoder = None

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else n_hidden
            out_hidden = n_heads * n_hidden  # total output dimension after all heads

            if self.edge_encoder is not None:
                self.edge_encoder.append(nn.Linear(edge_feats, edge_emb))

            if mpnn == 'gat':
                self.convs.append(
                    GATConv(
                        in_hidden,
                        n_hidden,
                        heads=n_heads,
                        dropout=attn_drop,
                        concat=True,
                        add_self_loops=False,
                    )
                )
            elif mpnn == 'gate':
                self.convs.append(
                    GATConv(
                        in_hidden,
                        n_hidden,
                        heads=n_heads,
                        dropout=attn_drop,
                        edge_dim=edge_emb,
                        concat=True,
                        add_self_loops=False,
                    )
                )
            elif mpnn == 'sage':
                self.convs.append(SAGEConv(in_hidden, out_hidden))
            else:  # gcn
                self.convs.append(GCNConv(in_hidden, out_hidden, add_self_loops=False))

            self.norms.append(nn.BatchNorm1d(out_hidden))

        self.pred_linear = nn.Linear(n_heads * n_hidden, n_classes)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x, edge_index, edge_attr=None):
        # Apply edge drop during training by randomly masking edges
        if self.training and self.edge_drop > 0 and edge_index.shape[1] > 0:
            mask = torch.rand(edge_index.shape[1], device=edge_index.device) >= self.edge_drop
            edge_index = edge_index[:, mask]
            if edge_attr is not None:
                edge_attr = edge_attr[mask]

        h = self.node_encoder(x)
        h = F.relu(h)
        h = self.input_drop(h)

        h_local = []
        h_last = None

        for i in range(self.n_layers):
            if self.mpnn == 'gate' and self.edge_encoder is not None:
                efeat_emb = self.edge_encoder[i](edge_attr)
                efeat_emb = F.relu(efeat_emb)
            else:
                efeat_emb = None

            if self.mpnn in ('gat', 'gate'):
                h = self.convs[i](h, edge_index, efeat_emb)
            else:
                h = self.convs[i](h, edge_index)

            if h_last is not None:
                h = h + h_last[:h.shape[0], :]

            h_last = h
            h = self.norms[i](h)
            h = self.activation(h)
            h = self.dropout(h)
            h_local.append(h)

        if self.jk:
            for i in range(len(h_local)):
                h_local[i] = h_local[i][:h.shape[0], :]
            h = torch.sum(torch.stack(h_local), dim=0)

        h = self.pred_linear(h)
        return h
