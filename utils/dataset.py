# -*- coding: utf-8 -*-
# @Time    : 2025/5/25 20:43
# @Author  : HaiqingSun
# @OriginalFileName: dataset
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import AddLaplacianEigenvectorPE

from utils.prepare_gnn_data import build_graph_data_list, build_graph_data_list_ext
class SpeciesNodeTabularDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # assert len(node_ids) == len(tabular_feats) == len(
        #     labels), f'Node ids and tabular and labels must have the same length!'
        # self.node_ids = torch.LongTensor(node_ids)
        # self.tabular_feats = torch.tensor(tabular_feats, dtype=torch.float32)
        # self.labels = torch.tensor(labels, dtype=torch.float32)

    @property
    def raw_file_names(self):
        return ["abundance_4_seq.csv"]

    @property
    def processed_file_names(self):
        return ["species_seq_data.pt"]

    def process(self):
        data_list = build_graph_data_list()
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class SpeciesNodePETabularDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # assert len(node_ids) == len(tabular_feats) == len(
        #     labels), f'Node ids and tabular and labels must have the same length!'
        # self.node_ids = torch.LongTensor(node_ids)
        # self.tabular_feats = torch.tensor(tabular_feats, dtype=torch.float32)
        # self.labels = torch.tensor(labels, dtype=torch.float32)

    @property
    def raw_file_names(self):
        return ["abundance_4_seq.csv"]

    @property
    def processed_file_names(self):
        return ["species_seq_data_pe_rw.pt"]

    def process(self):
        data_list = build_graph_data_list()
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SpeciesNodeEXTPETabularDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # assert len(node_ids) == len(tabular_feats) == len(
        #     labels), f'Node ids and tabular and labels must have the same length!'
        # self.node_ids = torch.LongTensor(node_ids)
        # self.tabular_feats = torch.tensor(tabular_feats, dtype=torch.float32)
        # self.labels = torch.tensor(labels, dtype=torch.float32)

    @property
    def raw_file_names(self):
        return ["abundance_4_seq_ext.csv"]

    @property
    def processed_file_names(self):
        return ["species_seq_data_pe_rw_ext.pt"]

    def process(self):
        # data_list = build_graph_data_list(edge_acc_emb=True, p_threshold=0.05, n_threshold=-0.05)
        data_list = build_graph_data_list_ext(edge_acc_emb=True,
                                          p_threshold=0.05,
                                          n_threshold=-0.05,
                                          dna_emb_path='dna_embeddings_ext.pt',
                                          cor_path='cov_mat_SparCC_out_extv2.tsv',
                                          abundance_path='abundance_4_seq_ext_c_seqid.csv',
                                          ori_abundance_path='../raw_data/ext_id_update.csv')
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
