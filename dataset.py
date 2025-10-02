# -*- coding: utf-8 -*-
# @Time    : 2025/5/25 20:43
# @Author  : HaiqingSun
# @OriginalFileName: dataset
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc
import torch
from torch_geometric.data import Data, InMemoryDataset
from utils.prepare_gnn_data import build_graph_data_list, build_graph_data_list_ext

import os
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import add_self_loops


class SpeciesNodePETabularDataset(InMemoryDataset):

    def __init__(self,
                 root,
                 age_label='host_age',
                 id_label='uid',
                 mode='all',
                 edge_acc_emb=True,
                 p_threshold=0.05,
                 n_threshold=-0.05,
                 transform=None,
                 pre_transform=None):
        self.age_label = age_label
        self.id_label = id_label
        self.mode = mode
        self.edge_acc_emb = edge_acc_emb
        self.p_threshold = p_threshold
        self.n_threshold = n_threshold

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            'pathway_embedding_lowdim.pt',
            'cor_sparcc.out.tsv',
            f'abundance_4_seq_{self.mode}.csv' if self.mode != 'all' else 'abundance_4_seq.csv',
            f'abundance_with_age_taxid_update_{self.mode}.csv' if self.mode != 'all' else 'abundance_with_age.csv',
            'matched_taxids_seqids.csv'
        ]

    @property
    def processed_file_names(self):
        edge_type = 'emb' if self.edge_acc_emb else 'topk'
        return [f'species_seq_data_{self.mode}_{edge_type}.pt'] if self.mode != 'all' else [f'species_seq_data_{self.mode}_{edge_type}.pt']

    def process(self):
        data_list = self._build_graph_data_list()
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _build_graph_data_list(self):
        emb_dict = torch.load(self.raw_paths[0])
        cor_df = pd.read_csv(self.raw_paths[1], sep='\t', index_col=0)
        abundance_df = pd.read_csv(self.raw_paths[2])
        ori_abundance_df = pd.read_csv(self.raw_paths[3])
        seqid_taxname_dict = pd.read_csv(self.raw_paths[4]).set_index("matched_seq_id")["tax_name"].to_dict()
        genes = cor_df.index.tolist()
        gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
        feature_cols = abundance_df.columns[2:]
        valid_cols = [col for col in feature_cols if col in emb_dict]
        abundance_df = abundance_df.iloc[:, :2].join(abundance_df[valid_cols])
        print(f"remaining features: {len(valid_cols)}")

        feature_genes = abundance_df.columns[2:].tolist()
        data_list = []
        for i, row in abundance_df.iterrows():
            data = self._process_sample(
                row, i, feature_genes, ori_abundance_df,
                emb_dict, gene_to_idx, cor_df, seqid_taxname_dict
            )
            data_list.append(data)

        return data_list

    def _process_sample(self, row, idx, feature_genes, ori_abundance_df,
                        emb_dict, gene_to_idx, cor_df, seqid_taxname_dict):
        y = torch.tensor([row[self.age_label]], dtype=torch.float)
        uid = str(row[self.id_label])
        feat_values = row[feature_genes].values

        nonzero_mask = feat_values != 0
        nonzero_indices = np.where(nonzero_mask)[0]
        nonzero_feat_values = feat_values[nonzero_indices]
        num_nonzero = len(nonzero_indices)

        if num_nonzero == 0:
            return self._create_empty_data(y, ori_abundance_df, idx)

        nonzero_genes = [feature_genes[i] for i in nonzero_indices]
        tax_names = [seqid_taxname_dict[gene] for gene in nonzero_genes]
        x_list = [emb_dict.get(gene, torch.zeros(200)) for gene in nonzero_genes]
        x = torch.stack(x_list)

        edge_index, edge_attr = self._build_edges(
            nonzero_genes, gene_to_idx, cor_df, num_nonzero
        )

        # # debug
        # cols = ori_abundance_df.columns[1:-1]
        # ori_abundance_df[cols] = ori_abundance_df[cols].apply(pd.to_numeric, errors='coerce')
        # n_nans = ori_abundance_df[cols].isna().sum().sum()
        # print(f"Converted abundance cols to numeric, total NaNs introduced: {n_nans}")

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            uid=uid,
            tax_names=tax_names,
            ab_weight=nonzero_feat_values,
            abundance=torch.from_numpy(ori_abundance_df.iloc[idx, 1:-1].to_numpy(dtype=np.float32))
        )

    def _process_common_sample(self, row, idx, feature_genes, ori_abundance_df,
                        emb_dict, gene_to_idx, cor_df, seqid_taxname_dict):
        y = torch.tensor([row[self.age_label]], dtype=torch.float)
        uid = str(int(row[self.id_label]))
        feat_values = row[feature_genes].values

        nonzero_mask = feat_values != 0
        nonzero_indices = np.where(nonzero_mask)[0]
        nonzero_feat_values = feat_values[nonzero_indices]
        num_nonzero = len(nonzero_indices)

        if num_nonzero == 0:
            return self._create_empty_data(y, ori_abundance_df, idx)

        # 构建节点特征
        nonzero_genes = [feature_genes[i] for i in nonzero_indices]
        tax_names = [seqid_taxname_dict[gene] for gene in nonzero_genes]
        x_list = [emb_dict.get(gene, torch.zeros(200)) for gene in nonzero_genes]
        x = torch.stack(x_list)

        # 构建边
        edge_index, edge_attr = self._build_edges(
            nonzero_genes, gene_to_idx, cor_df, num_nonzero
        )

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            uid=uid,
            tax_names=tax_names,
            ab_weight=nonzero_feat_values,
            abundance=torch.tensor(ori_abundance_df.iloc[idx, 4:].values, dtype=torch.float32)
        )

    def _create_empty_data(self, y, ori_abundance_df, idx):
        """创建空图数据"""
        x = torch.zeros((0, 200))
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr=edge_attr, fill_value=1.0, num_nodes=x.size(0)
        )
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            ab_weight=[],
            abundance=torch.tensor(ori_abundance_df.iloc[idx, 4:].values, dtype=torch.float32)
        )

    def _build_edges(self, nonzero_genes, gene_to_idx, cor_df, num_nonzero):
        """构建边索引和边属性"""
        gene_indices = [gene_to_idx[gene] for gene in nonzero_genes]
        sub_cor = cor_df.iloc[gene_indices, gene_indices].values

        triu_indices = np.triu_indices_from(sub_cor, k=1)
        triu_values = sub_cor[triu_indices]

        if self.edge_acc_emb:
            selected_edges, selected_values = self._threshold_edge_selection(
                triu_indices, triu_values
            )
        else:
            selected_edges, selected_values = self._topk_edge_selection(
                triu_indices, triu_values, num_nonzero
            )

        if selected_edges.shape[0] == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)
        else:
            edge_index = torch.tensor(selected_edges.T, dtype=torch.long)
            edge_attr = torch.tensor(selected_values.reshape(-1, 1), dtype=torch.float)

        return edge_index, edge_attr

    def _threshold_edge_selection(self, triu_indices, triu_values):
        """基于阈值选择边"""
        pos_mask = triu_values > self.p_threshold
        neg_mask = triu_values < self.n_threshold
        selected_mask = pos_mask | neg_mask

        selected_edges = np.column_stack(triu_indices)[selected_mask]
        selected_values = triu_values[selected_mask]

        return selected_edges, selected_values

    def _topk_edge_selection(self, triu_indices, triu_values, num_nonzero):
        """基于Top-K选择边"""
        pos_mask = triu_values > 0
        neg_mask = triu_values < 0

        pos_values = triu_values[pos_mask]
        neg_values = triu_values[neg_mask]

        k = min(15 * num_nonzero, len(triu_values))
        k_pos = min(k, len(pos_values))
        k_neg = min(k, len(neg_values))

        pos_indices_sorted = np.argsort(-pos_values)[:k_pos] if k_pos > 0 else []
        neg_indices_sorted = np.argsort(neg_values)[:k_neg] if k_neg > 0 else []

        pos_edge_indices = np.column_stack(triu_indices)[pos_mask][pos_indices_sorted] if k_pos > 0 else np.empty(
            (0, 2), dtype=int)
        neg_edge_indices = np.column_stack(triu_indices)[neg_mask][neg_indices_sorted] if k_neg > 0 else np.empty(
            (0, 2), dtype=int)

        selected_edges = np.vstack([pos_edge_indices, neg_edge_indices])
        selected_values = np.concatenate([pos_values[pos_indices_sorted], neg_values[neg_indices_sorted]])

        return selected_edges, selected_values


# 使用示例
if __name__ == "__main__":
    # # 训练集
    # train_dataset = SpeciesNodePETabularDataset(
    #     root='data/train',
    #     mode='train',
    #     edge_acc_emb=True,
    #     p_threshold=0.05,
    #     n_threshold=-0.05
    # )
    #
    # # 测试集
    # test_dataset = SpeciesNodePETabularDataset(
    #     root='data/test',
    #     mode='test',
    #     edge_acc_emb=True,
    #     p_threshold=0.05,
    #     n_threshold=-0.05
    # )

    dataset = SpeciesNodePETabularDataset(
        age_label='age',
        id_label='uid',
        root='data_ext',
        mode='all',
        edge_acc_emb=True,
        p_threshold=0.05,
        n_threshold=-0.05
    )

    # print(f"训练集样本数: {len(train_dataset)}")
    # print(f"测试集样本数: {len(test_dataset)}")
    print(f"测试集样本数: {len(dataset)}")