# -*- coding: utf-8 -*-
# @Time    : 2025/6/3 13:01
# @Author  : HaiqingSun
# @OriginalFileName: step7_9_prepare_gnn_data
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc
import os

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path

from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_add

def build_graph_data_list(root,
                          edge_acc_emb=False,
                          p_threshold=0.3,
                          n_threshold=-0.3,
                          dna_emb_path='dna_embeddings.pt',
                          cor_path='cor_sparcc.out.tsv',
                          abundance_path='predict_abundance_4_seq.csv',
                          ori_abundance_path='../raw_data/mapped_ext_normalized_to_abu.csv'):
    script_path = Path(__file__).parent
    dna_emb_dict = torch.load(os.path.join(script_path, dna_emb_path))
    cor_df = pd.read_csv(os.path.join(script_path, cor_path), sep='\t', index_col=0)

    genes = cor_df.index.tolist()
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}

    # abundance_df = pd.read_csv(os.path.join(script_path, 'abundance_4_seq_v2.csv')) # for train
    # ori_abundance_df = pd.read_csv(os.path.join(script_path, 'abundance_with_age_taxid_update_drop0_v3.csv')) # for train

    # abundance_df = pd.read_csv(os.path.join(script_path, 'predict_abundance_4_seq.csv')) # for predict
    # ori_abundance_df = pd.read_csv(os.path.join(script_path, '../raw_data/mapped_ext_normalized_to_abu.csv')) # for predict

    abundance_df = pd.read_csv(os.path.join(script_path, abundance_path)) # for out train
    ori_abundance_df = pd.read_csv(os.path.join(script_path, ori_abundance_path)) # for out train

    feature_genes = abundance_df.columns[2:].tolist()
    # feature_genes = [name.split('|')[-1].split('__')[-1] for name in feature_genes if name.split('|')[-1].startswith('s__')]
    data_list = []

    for i, row in abundance_df.iterrows():
        y = torch.tensor([row['age']], dtype=torch.float)
        feat_values = row[feature_genes].values

        nonzero_mask = feat_values != 0
        nonzero_indices = np.where(nonzero_mask)[0]
        nonzero_feat_values = feat_values[nonzero_indices]
        num_nonzero = len(nonzero_indices)

        if num_nonzero == 0:  # create a graph with 0
            x = torch.zeros((0, 768))
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, fill_value=1.0, num_nodes=x.size(0))
            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, ab_weight=[], abundance=torch.tensor(ori_abundance_df.iloc[i, 4:].values, dtype=torch.float32)))
            continue

        nonzero_genes = [feature_genes[i] for i in nonzero_indices]

        x_list = []
        for gene in nonzero_genes:
            x_list.append(dna_emb_dict.get(gene, torch.zeros(768)))
        x = torch.stack(x_list)

        gene_indices = [gene_to_idx[gene] for gene in nonzero_genes]
        sub_cor = cor_df.iloc[gene_indices, gene_indices].values

        triu_indices = np.triu_indices_from(sub_cor, k=1)
        triu_values = sub_cor[triu_indices]

        if edge_acc_emb:
            pos_mask = triu_values > p_threshold
            neg_mask = triu_values < n_threshold
            selected_mask = pos_mask | neg_mask

            print(
                f"Sample {i}: total edges = {len(triu_values)}, pos = {pos_mask.sum()}, neg = {neg_mask.sum()}, selected = {selected_mask.sum()}")

            selected_edges = np.column_stack(triu_indices)[selected_mask]
            selected_values = triu_values[selected_mask]
        else:
            pos_mask = triu_values > 0
            neg_mask = triu_values < 0

            pos_values = triu_values[pos_mask]
            neg_values = triu_values[neg_mask]

            k = min(15 * num_nonzero, len(triu_values))
            k_pos = min(k, len(pos_values))
            k_neg = min(k, len(neg_values))

            pos_indices_sorted = np.argsort(-pos_values)[:k_pos] if k_pos > 0 else []
            neg_indices_sorted = np.argsort(neg_values)[:k_neg] if k_neg > 0 else []

            pos_edge_indices = np.column_stack(triu_indices)[pos_mask][pos_indices_sorted] if k_pos > 0 else np.empty((0, 2), dtype=int)
            neg_edge_indices = np.column_stack(triu_indices)[neg_mask][neg_indices_sorted] if k_neg > 0 else np.empty((0, 2), dtype=int)

            selected_edges = np.vstack([pos_edge_indices, neg_edge_indices])
            selected_values = np.concatenate([pos_values[pos_indices_sorted], neg_values[neg_indices_sorted]])

        if selected_edges.shape[0] == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
        else:
            edge_index = torch.tensor(selected_edges.T, dtype=torch.long)
            edge_attr = torch.tensor(selected_values.reshape(-1, 1), dtype=torch.float)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            ab_weight=nonzero_feat_values,
            abundance=torch.tensor(ori_abundance_df.iloc[i, 4:].values, dtype=torch.float32)
        )
        data_list.append(data)

    return data_list

def build_graph_data_list_ext(edge_acc_emb=False,
                          p_threshold=0.3,
                          n_threshold=-0.3,
                          dna_emb_path='dna_embeddings.pt',
                          cor_path='cor_sparcc.out.tsv',
                          abundance_path='predict_abundance_4_seq.csv',
                          ori_abundance_path='../raw_data/mapped_ext_normalized_to_abu.csv'):
    script_path = Path(__file__).parent
    dna_emb_dict = torch.load(os.path.join(script_path, dna_emb_path))
    cor_df = pd.read_csv(os.path.join(script_path, cor_path), sep='\t', index_col=0)

    genes = cor_df.index.tolist()
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}

    # abundance_df = pd.read_csv(os.path.join(script_path, 'abundance_4_seq_v2.csv')) # for train
    # ori_abundance_df = pd.read_csv(os.path.join(script_path, 'abundance_with_age_taxid_update_drop0_v3.csv')) # for train

    # abundance_df = pd.read_csv(os.path.join(script_path, 'predict_abundance_4_seq.csv')) # for predict
    # ori_abundance_df = pd.read_csv(os.path.join(script_path, '../raw_data/mapped_ext_normalized_to_abu.csv')) # for predict

    abundance_df = pd.read_csv(os.path.join(script_path, abundance_path), index_col=None) # for out train
    ori_abundance_df = pd.read_csv(os.path.join(script_path, ori_abundance_path)) # for out train

    feature_genes = abundance_df.columns.tolist()
    # feature_genes = [name.split('|')[-1].split('__')[-1] for name in feature_genes if name.split('|')[-1].startswith('s__')]
    data_list = []

    for i, row in abundance_df.iterrows():
        # y = torch.tensor([row['age']], dtype=torch.float)
        y = torch.tensor(ori_abundance_df.iloc[i, 2])
        feat_values = row[feature_genes].values

        nonzero_mask = feat_values != 0
        nonzero_indices = np.where(nonzero_mask)[0]
        nonzero_feat_values = feat_values[nonzero_indices]
        num_nonzero = len(nonzero_indices)

        if num_nonzero == 0:
            x = torch.zeros((0, 768))
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, fill_value=1.0, num_nodes=x.size(0))
            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, ab_weight=[], abundance=torch.tensor(ori_abundance_df.iloc[i, 4:].values, dtype=torch.float32)))
            continue

        nonzero_genes = [feature_genes[i] for i in nonzero_indices]

        x_list = []
        for gene in nonzero_genes:
            x_list.append(dna_emb_dict.get(gene, torch.zeros(768)))
        x = torch.stack(x_list)

        gene_indices = [gene_to_idx[gene] for gene in nonzero_genes]
        sub_cor = cor_df.iloc[gene_indices, gene_indices].values

        triu_indices = np.triu_indices_from(sub_cor, k=1)
        triu_values = sub_cor[triu_indices]

        if edge_acc_emb:
            pos_mask = triu_values > p_threshold
            neg_mask = triu_values < n_threshold
            selected_mask = pos_mask | neg_mask

            print(
                f"Sample {i}: total edges = {len(triu_values)}, pos = {pos_mask.sum()}, neg = {neg_mask.sum()}, selected = {selected_mask.sum()}")

            selected_edges = np.column_stack(triu_indices)[selected_mask]
            selected_values = triu_values[selected_mask]
        else:
            pos_mask = triu_values > 0
            neg_mask = triu_values < 0

            pos_values = triu_values[pos_mask]
            neg_values = triu_values[neg_mask]

            k = min(15 * num_nonzero, len(triu_values))
            k_pos = min(k, len(pos_values))
            k_neg = min(k, len(neg_values))

            pos_indices_sorted = np.argsort(-pos_values)[:k_pos] if k_pos > 0 else []
            neg_indices_sorted = np.argsort(neg_values)[:k_neg] if k_neg > 0 else []

            pos_edge_indices = np.column_stack(triu_indices)[pos_mask][pos_indices_sorted] if k_pos > 0 else np.empty((0, 2), dtype=int)
            neg_edge_indices = np.column_stack(triu_indices)[neg_mask][neg_indices_sorted] if k_neg > 0 else np.empty((0, 2), dtype=int)

            selected_edges = np.vstack([pos_edge_indices, neg_edge_indices])
            selected_values = np.concatenate([pos_values[pos_indices_sorted], neg_values[neg_indices_sorted]])

        if selected_edges.shape[0] == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
        else:
            edge_index = torch.tensor(selected_edges.T, dtype=torch.long)
            edge_attr = torch.tensor(selected_values.reshape(-1, 1), dtype=torch.float)

        values = pd.to_numeric(ori_abundance_df.iloc[i, 3:], errors='coerce').values.astype(np.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            ab_weight=nonzero_feat_values,
            abundance=torch.tensor(values, dtype=torch.float32)
        )
        data_list.append(data)

    return data_list



def process_graph_with_self_loops(edge_index, edge_weight, x):
    num_nodes = x.size(0)
    src, dst = edge_index
    mask = src != dst
    edge_index = edge_index[:, mask]
    edge_weight = edge_weight[mask].squeeze(-1)

    src = edge_index[0]
    node_weight = scatter_add(edge_weight, src, dim=0, dim_size=num_nodes)

    self_loop_index = torch.arange(num_nodes, device=edge_index.device)
    self_loop_edges = torch.stack([self_loop_index, self_loop_index], dim=0)
    self_loop_weights = node_weight

    new_edge_index = torch.cat([edge_index, self_loop_edges], dim=1)
    new_edge_weight = torch.cat([edge_weight, self_loop_weights], dim=0)

    return new_edge_index, new_edge_weight

if __name__ == '__main__':
    # graph_data_list = build_graph_data_list(True, 0.05, -0.05)
    graph_data_list = build_graph_data_list_ext(edge_acc_emb=True,
                                          p_threshold=0.05,
                                          n_threshold=-0.05,
                                          dna_emb_path='dna_embeddings_ext.pt',
                                          cor_path='cov_mat_SparCC_out_extv2.tsv',
                                          abundance_path='abundance_4_seq_ext_c_seqid.csv',
                                          ori_abundance_path='../raw_data/ext_id_update.csv')