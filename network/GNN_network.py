# -*- coding: utf-8 -*-
# @Time    : 2025/5/23 17:14
# @Author  : HaiqingSun
# @OriginalFileName: GNN_network
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc
import torch
from pytorch_tabnet.tab_network import initialize_non_glu
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, global_add_pool, GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, ASAPooling as asp

class GAT_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xnode=768, hidden1_dim=128,
                 hidden2_dim=800, dropout=0.2, dataset=None, asp_ratio=0.5, pe_dim=32,
                 graph_output_dim=800, n_heads=3, output_dim=1,
                 device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')):
        super(GAT_GCN, self).__init__()
        self.device = device
        self.dataset = dataset
        self.n_output = n_output
        self.pe_dim = pe_dim
        self.asp_ratio = asp_ratio
        self.graph_output_dim = graph_output_dim
        self.n_heads = n_heads
        self.conv1 = GATv2Conv(num_features_xnode + pe_dim, num_features_xnode + pe_dim, heads=self.n_heads)
        self.conv2 = GCNConv(self.n_heads * (num_features_xnode + pe_dim), self.n_heads * (num_features_xnode + pe_dim))
        conv_out_dim = self.n_heads * (num_features_xnode + pe_dim)
        self.fc_g1 = nn.Linear(2 * conv_out_dim, 4 * conv_out_dim)
        nn.init.xavier_uniform_(self.fc_g1.weight)
        self.graph_out = nn.Linear(4 * conv_out_dim, self.graph_output_dim)
        nn.init.xavier_uniform_(self.graph_out.weight)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.semi_final_mapping = Linear(self.graph_output_dim, self.graph_output_dim // 4)
        initialize_non_glu(self.semi_final_mapping, self.graph_output_dim, self.graph_output_dim // 4)
        self.final_mapping = Linear(self.graph_output_dim // 4, output_dim, bias=False)
        initialize_non_glu(self.final_mapping, self.graph_output_dim // 4, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_weight = getattr(data, 'edge_weight', None)
        pe_enc = data.pe_enc

        # relative abundance weighting (optional)
        # ab_weight = data.ab_weight
        # flat_weight = [w for graph_w in ab_weight for w in graph_w]  # flatten
        # flat_weight = torch.tensor(flat_weight, dtype=torch.float32, device=x.device).view(-1, 1)  # shape [N, 1]
        # x = x * flat_weight

        if edge_index.numel() == 0:
            edge_index = torch.arange(len(x), device=self.device)
            edge_index = torch.stack([edge_index, edge_index], dim=0)
            edge_weight = torch.ones(edge_index.size(1), device=self.device)

        # x = self.test_pre_fc1(x)
        # x = self.relu(x)

        x = torch.cat([x, pe_enc], dim=1)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        pooled_mean = global_mean_pool(x, batch)   # [num_graphs, F]
        pooled_max = global_max_pool(x, batch)    # [num_graphs, F]
        x = torch.cat([pooled_mean, pooled_max], dim=1)  # [num_graphs, 2F]
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.graph_out(x)
        x = self.semi_final_mapping(x)
        x = self.final_mapping(x)

        return x