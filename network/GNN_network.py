# -*- coding: utf-8 -*-
# @Time    : 2025/5/23 17:14
# @Author  : HaiqingSun
# @OriginalFileName: GNN_network
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc
import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, global_add_pool, GATv2Conv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, ASAPooling as asp

class GATGCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xnode=768, hidden1_dim=128,
                 hidden2_dim=128, output_dim=16, dropout=0.2, asp_ratio=0.5, pe_dim=32, device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')):
        super(GATGCN, self).__init__()
        self.device = device
        self.n_output = n_output
        self.pe_dim = pe_dim

        self.test_pre_fc1 = nn.Linear(num_features_xnode, hidden1_dim)
        nn.init.xavier_uniform_(self.test_pre_fc1.weight)

        self.conv1 = GATv2Conv(hidden1_dim + pe_dim, hidden1_dim + pe_dim, heads=5)  # 10
        self.conv2 = GCNConv(5 * (hidden1_dim + pe_dim), 5 * (hidden1_dim + pe_dim))  # 10
        self.fc_g1 = torch.nn.Linear(5 * (hidden1_dim + pe_dim) * 2, hidden2_dim)  # 10*2，
        nn.init.xavier_uniform_(self.fc_g1.weight)
        self.graph_out = torch.nn.Linear(hidden2_dim, output_dim)
        nn.init.xavier_uniform_(self.graph_out.weight)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        edge_attr, ab_weight = data.edge_attr, data.ab_weight
        # 添加pe_enc
        pe_enc = data.pe_enc

        if edge_index.numel() == 0:
            edge_index = torch.arange(len(x))
            edge_index = torch.stack([edge_index, edge_index], dim=0)
        x = self.test_pre_fc1(x)
        # x = self.bn(x)
        x = self.relu(x)

        x = torch.cat([x, pe_enc], dim=1)

        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        # x_asp, _, _, batch_asp, _ = self.asp1(x, edge_index, batch=batch)
        # x = torch.cat([gmp(x_asp, batch_asp), gap(x_asp, batch_asp), gmp(x, batch), gap(x, batch)], dim=1)

        # x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        flat_weight = [w for graph_w in ab_weight for w in graph_w]  # flatten
        flat_weight = torch.tensor(flat_weight, dtype=torch.float32, device=x.device).view(-1, 1)  # shape [N, 1]
        x_weighted = x * flat_weight
        pooled_sum = global_add_pool(x_weighted, batch)  # [num_graphs, F]
        weight_sum = global_add_pool(flat_weight, batch)  # [num_graphs, 1]
        weight_sum = weight_sum.clamp(min=1e-6)
        pooled_mean = pooled_sum / weight_sum  # [num_graphs, F]
        x = torch.cat([pooled_mean, gmp(x, batch)], dim=1)

        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)

        x = self.relu(self.graph_out(x))
        return x