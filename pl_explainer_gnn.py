# -*- coding: utf-8 -*-
# @Time    : 2025/9/13 11:18
# @Author  : HaiqingSun
# @OriginalFileName: pl_explainer_gnn
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc
from torch_geometric.explain import Explainer, ModelConfig, CaptumExplainer
from torch_geometric.explain.algorithm import GNNExplainer


import os
import random

import numpy as np
import pandas as pd
import pytorch_lightning
import shap
import torch
import torch_geometric
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer
from sklearn.model_selection import train_test_split, KFold
from torch import nn
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import MessagePassing
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE

from pl_module import GTABACLightningModule
from dataset import SpeciesNodePETabularDataset
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import matplotlib as mpl
from matplotlib.patches import Patch

base_dir = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed=6):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pytorch_lightning.seed_everything(seed)
    np.random.seed(seed)
    torch_geometric.seed_everything(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


batch_size = 5120
seed = 6
set_seed(seed)


def worker_init_fn(worker_id):
    np.random.seed(seed + worker_id)


graph_pe_encoding_dim = 32
transform = AddRandomWalkPE(walk_length=graph_pe_encoding_dim, attr_name='pe_enc')
# dataset = SpeciesNodePETabularDataset(os.path.join(base_dir, '../data'), pre_transform=transform)
dataset = SpeciesNodePETabularDataset(
    age_label='age',
    id_label='uid',
    root=os.path.join(base_dir, 'data_ext'),
    mode='all',
    edge_acc_emb=True,
    p_threshold=0.05,
    n_threshold=-0.05,
    pre_transform=transform,
)
predict_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, worker_init_fn=worker_init_fn)
input_dim = dataset[0].abundance.shape[0]
num_features_xnode = 200
graph_output_dim = 800
fold_epochs = 300
LOG_INTERVAL = 10
clip_value = 1  # 1 0.5
lambda_sparse = 9e-4
lr = 1e-4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

indices = list(range(len(dataset)))
train_valid_idx, test_idx = train_test_split(
    indices, test_size=0.15, random_state=seed, shuffle=True
)
test_dataset = dataset[test_idx]

all_fold_maes = []
current_fold = 0
best_test_mae = best_test_r2 = best_test_ci = 0
# model_file_name = 'saved_models/tabnet_model_exp3_graph_only_v3'  #   tabnet_model_exp3all_b128_seed6
model_file_name = 'saved_models/gtabac_model_exp_e80l1e-4s6'

model = GTABACLightningModule(input_dim=input_dim,
                              num_features_xnode=num_features_xnode,
                              graph_output_dim=graph_output_dim,
                              lambda_sparse=lambda_sparse,
                              lr=lr,
                              loss_fn=nn.SmoothL1Loss(beta=1.0),
                              graph_pe_encoding=graph_pe_encoding_dim)

# trainer = Trainer(
#     max_epochs=fold_epochs,
#     accelerator="gpu",
#     devices=1,
#     logger=False,
#     callbacks=[],
# )


model.load_state_dict(torch.load(model_file_name + '.pth', map_location=device), strict=False)
model = model.to(device)

class ModelWrapper(nn.Module):
    def __init__(self, original_model, original_data):
        super().__init__()
        self.original_model = original_model
        self.original_data = original_data

    def forward(self, x, edge_index, edge_weight=None, batch=None, **kwargs):
        self.original_model.eval()
        new_batch = Data()
        for key, value in self.original_data.to_dict().items():
            if key not in ['x', 'edge_index']:
                setattr(new_batch, key, value)
        new_batch.x = x
        new_batch.edge_index = edge_index
        if edge_weight is not None:
            new_batch.edge_weight = edge_weight
        if batch is not None:
            new_batch.batch = batch
        model_output = self.original_model(new_batch)
        if isinstance(model_output, tuple):
            prediction = model_output[0]
            return prediction
        else:
            return model_output


def explain_graph_predictions(model, graph_data, target_index=None):
    wrapped_model = ModelWrapper(model, graph_data)
    explainer = Explainer(
        model=wrapped_model,
        algorithm=GNNExplainer(epochs=100
                               ),

        explanation_type='model',
        model_config=ModelConfig(
            mode='regression',
            task_level='graph',
            return_type='raw',
        ),
        edge_mask_type='object',
        # node_mask_type='attributes',
    )

    explanation = explainer(
        x=graph_data.x,
        edge_index=graph_data.edge_index,
        edge_attr=graph_data.edge_weight,
        # target=graph_data.y,
        # index=target_index
    )


    return explanation


mpl.rcParams['svg.fonttype'] = 'none'

def visualize_explanation(graph_data, explanation, title="Graph Explanation", top_k=10, save_svg=True,
                          filename="gnn_explanation.svg"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    G = _build_networkx_graph(graph_data)
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
    _draw_original_graph(G, pos, ax1)
    legend_info = _draw_explanation_graph(G, pos, graph_data, explanation, ax2, top_k)
    _add_legend(fig, legend_info, ax1, ax2)
    ax1.set_title("Original Graph", fontsize=14, fontweight='bold')
    ax2.set_title(f"Explanation (Top-{top_k} Important Edges)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    num_nodes = len(legend_info['nodes']) if 'nodes' in legend_info and legend_info['nodes'] else 0
    if num_nodes <= 5:
        bottom_space = 0.25
    elif num_nodes <= 10:
        bottom_space = 0.35
    else:
        bottom_space = 0.45
    plt.subplots_adjust(bottom=bottom_space)
    if save_svg:
        plt.savefig(filename, format='svg', bbox_inches='tight', dpi=300,
                    metadata={'Description': 'GNN Explanation'})
        with open(filename, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        svg_content = svg_content.replace(
            '<svg',
            '<svg style="user-select: text; -webkit-user-select: text; -moz-user-select: text;"'
        )

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(svg_content)

        print(f"Saved to: {filename}")

    _print_statistics(explanation, G)


def _build_networkx_graph(graph_data):
    G = nx.Graph()
    edge_index = graph_data.edge_index.cpu().numpy()
    edges = [(int(edge_index[0, i]), int(edge_index[1, i]))
             for i in range(edge_index.shape[1])]
    G.add_edges_from(edges)
    return G


def _draw_original_graph(G, pos, ax):
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue',
                           node_size=400, alpha=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray',
                           width=1.5, alpha=0.6)
    ax.set_aspect('equal')


def _draw_explanation_graph(G, pos, graph_data, explanation, ax, top_k):
    legend_info = {'nodes': [], 'edges': []}

    if hasattr(explanation, 'edge_mask') and explanation.edge_mask is not None:
        edge_importance = explanation.edge_mask.cpu().numpy()
        top_edge_indices = np.argsort(edge_importance)[-top_k:]
        edge_index = graph_data.edge_index.cpu().numpy()
        involved_nodes = []
        important_edges = []
        for idx in top_edge_indices:
            if idx < edge_index.shape[1]:
                u, v = int(edge_index[0, idx]), int(edge_index[1, idx])
                if u not in involved_nodes:
                    involved_nodes.append(u)
                if v not in involved_nodes:
                    involved_nodes.append(v)
                important_edges.append(((u, v), edge_importance[idx]))
        subgraph_nodes = sorted(involved_nodes)
        G_sub = G.subgraph(subgraph_nodes)
        actual_nodes_in_subgraph = [node for node in subgraph_nodes if node in G_sub.nodes()]
        node_colors, legend_info['nodes'] = _get_node_colors_with_legend(
            explanation.node_mask if hasattr(explanation, 'node_mask') else None,
            actual_nodes_in_subgraph, graph_data)
        nx.draw_networkx_nodes(G_sub, pos, nodelist=actual_nodes_in_subgraph,
                               node_color=node_colors, node_size=500, alpha=0.8, ax=ax)
        nx.draw_networkx_labels(G_sub, pos, labels={node: str(node) for node in actual_nodes_in_subgraph},
                                ax=ax, font_size=8)
        _draw_edges_with_importance(G_sub, pos, important_edges, ax)

    else:
        all_nodes = sorted(list(G.nodes()))
        node_colors, legend_info['nodes'] = _get_node_colors_with_legend(
            explanation.node_mask if hasattr(explanation, 'node_mask') else None,
            all_nodes, graph_data)
        nx.draw_networkx_nodes(G, pos, nodelist=all_nodes, node_color=node_colors,
                               node_size=400, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray',
                               width=1.5, alpha=0.6)
        nx.draw_networkx_labels(G, pos, labels={node: str(node) for node in all_nodes},
                                ax=ax, font_size=8)

    ax.set_aspect('equal')
    return legend_info


def _get_node_colors_with_legend(node_mask, nodes, graph_data):
    node_importance = None
    if node_mask is not None:
        node_importance = node_mask.cpu().numpy()
        if node_importance.ndim > 1:
            node_importance = node_importance.mean(axis=1)
    colors = []
    legend_items = []
    color_sources = [
        plt.cm.Set3, plt.cm.tab20, plt.cm.Set1, plt.cm.Set2,
        plt.cm.Pastel1, plt.cm.Pastel2, plt.cm.Accent, plt.cm.Dark2
    ]

    for i, node in enumerate(nodes):
        cmap = color_sources[i % len(color_sources)]
        color_index = (i // len(color_sources)) % cmap.N
        color = cmap(color_index)
        colors.append(color)
        node_name = ""
        if hasattr(graph_data, 'tax_names') and node < len(graph_data.tax_names):
            node_name = graph_data.tax_names[node]
        if node_importance is not None and node < len(node_importance):
            importance = node_importance[node]
            label = f"{node}:{node_name} ({importance:.3f})" if node_name else f"Node {node} ({importance:.3f})"
        else:
            label = f"{node}:{node_name}" if node_name else f"Node {node}"

        legend_items.append((color, label))
    return colors, legend_items


def _draw_edges_with_importance(G, pos, important_edges, ax):
    edge_scores = [score for _, score in important_edges]
    if edge_scores:
        min_score, max_score = min(edge_scores), max(edge_scores)
        score_range = max_score - min_score if max_score > min_score else 1

    for (u, v), importance in important_edges:
        normalized_importance = (importance - min_score) / score_range if score_range > 0 else 0.5
        color_intensity = 0.3 + 0.7 * normalized_importance
        edge_color = plt.cm.Reds(color_intensity)
        edge_width = 2 + 4 * normalized_importance
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               edge_color=[edge_color], width=edge_width,
                               alpha=0.8, ax=ax)
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, f'{importance:.3f}',
                fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                fontweight='bold')


def _add_legend(fig, legend_info, ax1, ax2):
    legend_elements = []
    if legend_info['nodes']:
        legend_elements.append(Patch(color='white', label='Nodes:'))
        for color, label in legend_info['nodes']:
            legend_elements.append(Patch(color=color, label=label))
    if legend_info['nodes']:
        legend_elements.append(Patch(color='white', label=''))
    legend_elements.append(Patch(color='white', label='Edge Importance:'))
    legend_elements.append(Patch(color=plt.cm.Reds(0.3), label='Low importance'))
    legend_elements.append(Patch(color=plt.cm.Reds(0.7), label='High importance'))
    legend_elements.append(Patch(color='white', label='(Numbers on edges = exact scores)'))
    num_nodes = len(legend_info['nodes']) if legend_info['nodes'] else 0
    if num_nodes <= 5:
        ncol = 2
    elif num_nodes <= 10:
        ncol = 3
    else:
        ncol = 4
    fig.legend(handles=legend_elements, loc='lower center',
               bbox_to_anchor=(0.5, 0.02), ncol=ncol, fontsize=9,
               columnspacing=1.5, handletextpad=0.5)


def _print_statistics(explanation, G):
    print("\n" + "=" * 50)
    print("GNN Explanation Statistic")
    print("=" * 50)

    if hasattr(explanation, 'edge_mask') and explanation.edge_mask is not None:
        edge_scores = explanation.edge_mask.cpu().numpy()
        print(f"Num of edges: {len(edge_scores)}")
        print(f"Edge score range: [{edge_scores.min():.4f}, {edge_scores.max():.4f}]")
        print(f"Mean edge score: {edge_scores.mean():.4f}")
        print(f"std dev: {edge_scores.std():.4f}")

    if hasattr(explanation, 'node_mask') and explanation.node_mask is not None:
        node_scores = explanation.node_mask.cpu().numpy()
        if node_scores.ndim > 1:
            node_scores = node_scores.mean(axis=1)
        print(f"Num of nodes: {len(node_scores)}")
        print(f"Nodes score range: [{node_scores.min():.4f}, {node_scores.max():.4f}]")
        print(f"Mean nodes score: {node_scores.mean():.4f}")
        print(f"std dev: {node_scores.std():.4f}")

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("=" * 50)


def check_graph_validation(explanation):
    edge_mask = getattr(explanation, "edge_mask", None)
    node_mask = getattr(explanation, "node_mask", None)
    if edge_mask is None and node_mask is None:
        return False
    elif edge_mask is None:
        if node_mask.numel() == 0:
            return False
    elif node_mask is None:
        if edge_mask.numel() == 0:
            return False
    return True



for i in range(0, len(dataset)): #2134 3336 ?3322
    model.eval()
    test_graph = dataset[i].to(device)
    uid = test_graph.uid
    num_edges = test_graph.edge_index.size(1)
    edge_weight = torch.ones(num_edges, device=device, requires_grad=True)
    test_graph.edge_weight = edge_weight

    prediction = model(test_graph)
    # # try:
    explanation = explain_graph_predictions(model, test_graph)
    os.makedirs('explanations/pt/', exist_ok=True)
    os.makedirs('explanations/svg/', exist_ok=True)
    torch.save(explanation, f"explanations/pt/explanation_{uid}.pt")
    print(f"Explanations saved to explanation_{uid}.pt")
    if not check_graph_validation(explanation):
        print("No valid explanations")
        continue
    G = to_networkx(test_graph, to_undirected=True)
    print(f"Num {i+1} explanation generated.")
    print("\nVisualization...")
    visualize_explanation(
        test_graph.cpu(),
        explanation,
        # title="Explanation",
        top_k=8,
        filename=f"explanations/svg/explanation_uid_{uid}.svg"
    )
    print(f"Num {i+1} explanation successfully visualized.")