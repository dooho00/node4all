import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch.conv import GraphConv

class GCN(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_layers,
                 dropout,
                 activation,
                 residual,
                 norm,
                 concat_out=False,
                 encoding=False
                 ):
        super(GCN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = dropout
        self.concat_out = concat_out

        last_activation = activation if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None
        
        if num_layers == 1:
            self.layers.append(GCNLayer(in_dim, out_dim, residual=last_residual, activation=last_activation, norm=last_norm, concat_out=concat_out))
        else:
            # input projection (no residual)
            self.layers.append(GCNLayer(in_dim, hidden_dim, residual=residual, activation=activation, norm=norm, concat_out=concat_out))
            # hidden layers
            for l in range(1, num_layers - 1):
                # due to multi-head, the in_dim = hidden_dim * num_heads
                self.layers.append(GCNLayer(hidden_dim, hidden_dim, residual=residual, activation=activation, norm=norm, concat_out=concat_out))
            # output projection
            self.layers.append(GCNLayer(hidden_dim, out_dim, residual=last_residual, activation=last_activation, norm=last_norm, concat_out=concat_out))

        self.head = nn.Identity()

    def forward(self, g, inputs, return_hidden=False):
        h = inputs
        hidden_list = []
        for l in range(self.num_layers):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.layers[l](g, h)
            hidden_list.append(h)
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def reset_classifier(self, num_classes):
        self.head = nn.Linear(self.out_dim, num_classes)


class GCNLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 residual=False,
                 activation=None,
                 norm=None,
                 concat_out=True,
                 allow_zero_in_degree=False):
        super(GCNLayer, self).__init__()
        self._in_feats = in_dim
        self._out_feats = out_dim
        self.activation = activation
        self.allow_zero_in_degree = allow_zero_in_degree
        self.concat_out = concat_out
        self.conv = GraphConv(in_dim, out_dim, allow_zero_in_degree=allow_zero_in_degree)
        if residual:
            if self._in_feats != self._out_feats:
                self.res_fc = nn.Linear(
                    self._in_feats, self._out_feats, bias=False)
                print("! Linear Residual !")
            else:
                print("Identity Residual ")
                self.res_fc = nn.Identity()
        else:
            self.register_buffer('res_fc', None)
        
        if norm is not None:
            self.norm = norm(out_dim)
        else:
            self.norm = None

    def forward(self, graph, feat):
        with graph.local_scope():
            rst = self.conv(graph, feat)

            if self.res_fc is not None:
                if isinstance(self.res_fc, nn.Identity):
                    resval = feat
                else:
                    resval = self.res_fc(feat)
                rst = rst + resval
            if self.activation:
                rst = self.activation(rst)
            if self.norm is not None:
                rst = self.norm(rst)

            if self.concat_out:
                rst = rst.unsqueeze(1)
            return rst
