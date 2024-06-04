import dgl.function as fn
import torch.nn as nn
import torch
import torch.nn.functional as F

class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
            name: nn.Linear(in_size, out_size) for name in etypes
        })
        self.edge_weight = nn.Linear(1, 1)

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Calculate W_r * h for each etype
            Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Message function copy_u: aggregates the features of the source node into 'm'; reduce function: assigns the mean of 'm' to 'h'
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype: G.nodes[ntype].data['h'] for ntype in G.ntypes}


class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size, num_layer):
        super(HeteroRGCN, self).__init__()
        # Use trainable node embeddings as featureless inputs.
        embed_dict = {ntype: G.nodes[ntype].data['h']
                      for ntype in G.ntypes}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.num_layer = num_layer
        self.embed = nn.ParameterDict(embed_dict)
        # create layers
        self.layer = HeteroRGCNLayer(in_size, hidden_size, G.etypes)

    def forward(self, G):
        h_dict = self.layer(G, self.embed)
        for i in range(self.num_layer - 1):
            h_dict = {k: F.leaky_relu(h) for k, h in h_dict.items()}
            h_dict = self.layer(G, h_dict)
        return h_dict

