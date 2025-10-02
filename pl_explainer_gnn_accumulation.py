# -*- coding: utf-8 -*-
# @Time    : 2025/9/18 1:11
# @Author  : HaiqingSun
# @OriginalFileName: pl_explainer_gnn_accumulation
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc
import os
import random
from itertools import combinations

import numpy as np
import networkx as nx
from collections import defaultdict, Counter
import pandas as pd
from typing import List, Dict, Tuple, Any

import pytorch_lightning
import torch
import torch_geometric
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from torch_geometric.transforms import AddRandomWalkPE
import matplotlib.patches as mpatches
from dataset import SpeciesNodePETabularDataset


class ExplanationAnalyzer:

    def __init__(self, explanations: List[Any], graph_data_list: List[Any]):
        self.explanations = explanations
        self.graph_data_list = graph_data_list

    def _filter_valid_explanations(self) -> Tuple[List[Any], List[Any]]:
        valid_explanations = []
        valid_graph_data = []
        for explanation, graph_data in zip(self.explanations, self.graph_data_list):
            if self._check_graph_validation(explanation):
                valid_explanations.append(explanation)
                valid_graph_data.append(graph_data)
        return valid_explanations, valid_graph_data

    @staticmethod
    def _check_graph_validation(explanation) -> bool:
        edge_mask = getattr(explanation, "edge_mask", None)
        node_mask = getattr(explanation, "node_mask", None)
        edge_valid = edge_mask is not None and edge_mask.numel() > 0
        node_valid = node_mask is not None and node_mask.numel() > 0
        return edge_valid or node_valid

    def find_representative_elements(self, top_k: int = 10, threshold: float = 0.1) -> Dict[str, Any]:
        print("Starting to find representative elements...")
        valid_explanations, valid_graph_data = self._filter_valid_explanations()
        if not valid_explanations:
            print("Warning: no valid explanations!")
            return {'representative_nodes': [], 'representative_edges': [],
                    'representative_subgraphs': [], 'summary_statistics': {}}

        print(f"Valid explanations: {len(valid_explanations)}/{len(self.explanations)}")
        original_explanations = self.explanations
        original_graph_data = self.graph_data_list
        self.explanations = valid_explanations
        self.graph_data_list = valid_graph_data
        node_stats = self._analyze_nodes(top_k)
        edge_stats = self._analyze_edges(top_k)
        subgraph_stats = self._analyze_subgraphs(top_k, threshold)
        self.explanations = original_explanations
        self.graph_data_list = original_graph_data
        results = {
            'representative_nodes': node_stats,
            'representative_edges': edge_stats,
            'representative_subgraphs': subgraph_stats,
            'summary_statistics': self._compute_summary_stats(valid_explanations)
        }

        self._print_analysis_results(results, top_k)

        return results

    def _analyze_nodes(self, top_k: int) -> List[Dict]:
        print("Analyzing nodes...")
        node_degree_sum = defaultdict(float)
        node_occurrence_count = defaultdict(int)
        node_names = {}

        for i, graph_data in enumerate(self.graph_data_list):
            G = self._build_graph(graph_data)
            degrees = dict(G.degree())
            sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
            top_node_indices = [node_idx for node_idx, _ in sorted_nodes[:top_k]]
            for node_idx in top_node_indices:
                node_name = self._get_node_name(graph_data, node_idx)
                node_names[node_name] = node_name
                node_degree_sum[node_name] += degrees.get(node_idx, 0)
                node_occurrence_count[node_name] += 1
        node_scores = []
        for node_name in node_degree_sum.keys():
            avg_degree = node_degree_sum[node_name] / node_occurrence_count[node_name]
            occurrence_ratio = node_occurrence_count[node_name] / len(self.graph_data_list)
            composite_score = avg_degree * occurrence_ratio * np.log1p(avg_degree)

            node_scores.append({
                'node_name': node_name,
                'avg_degree': avg_degree,
                'occurrence_count': node_occurrence_count[node_name],
                'occurrence_ratio': occurrence_ratio,
                'composite_score': composite_score
            })

        node_scores.sort(key=lambda x: x['composite_score'], reverse=True)
        return node_scores[:top_k]

    def _analyze_edges(self, top_k: int) -> List[Dict]:
        print("Analyzing edge score...")

        edge_importance_sum = defaultdict(float)
        edge_occurrence_count = defaultdict(int)
        edge_names = {}

        for explanation, graph_data in zip(self.explanations, self.graph_data_list):
            if hasattr(explanation, 'edge_mask') and explanation.edge_mask is not None:
                edge_importance = explanation.edge_mask.cpu().numpy()
                edge_index = graph_data.edge_index.cpu().numpy()
                top_edge_indices = np.argsort(edge_importance)[-top_k:]

                for edge_idx in top_edge_indices:
                    if edge_idx < edge_index.shape[1]:
                        u, v = int(edge_index[0, edge_idx]), int(edge_index[1, edge_idx])
                        u_name = self._get_node_name(graph_data, u)
                        v_name = self._get_node_name(graph_data, v)
                        lst = [u_name, v_name]
                        lst.sort()
                        u_name, v_name = lst
                        # 创建边的唯一标识（保证有向边的一致性）!!!!!!!
                        edge_key = f"({u_name})-({v_name})"

                        edge_names[edge_key] = (u_name, v_name)
                        edge_importance_sum[edge_key] += edge_importance[edge_idx]
                        edge_occurrence_count[edge_key] += 1

        edge_scores = []
        for edge_key in edge_importance_sum.keys():
            avg_importance = edge_importance_sum[edge_key] / edge_occurrence_count[edge_key]
            occurrence_ratio = edge_occurrence_count[edge_key] / len(self.explanations)
            composite_score = avg_importance * occurrence_ratio

            edge_scores.append({
                'edge_key': edge_key,
                'edge_names': edge_names[edge_key],
                'avg_importance': avg_importance,
                'occurrence_count': edge_occurrence_count[edge_key],
                'occurrence_ratio': occurrence_ratio,
                'composite_score': composite_score
            })

        edge_scores.sort(key=lambda x: x['composite_score'], reverse=True)

        return edge_scores[:top_k]

    def _analyze_subgraphs(self, top_k: int, threshold: float, draw_pattern: bool = True) -> dict:
        print("Analyzing subgraphs...")
        results = {'representative_subgraphs': [], 'per_k': {}}

        num_explanations = len(self.explanations)
        if num_explanations == 0:
            print("No explanation。")
            return results
        if top_k < 3:
            print("top_k too low.")
            return {}

        overall_patterns = []
        for k_nodes in range(3, top_k + 1):
            patterns_k = defaultdict(list)
            for explanation, graph_data in zip(self.explanations, self.graph_data_list):
                if not hasattr(explanation, 'edge_mask') or explanation.edge_mask is None:
                    continue
                edge_importance = explanation.edge_mask.detach().cpu().numpy()
                edge_index = graph_data.edge_index.detach().cpu().numpy()
                if edge_importance.size == 0 or edge_index.size == 0:
                    continue
                adj_full = defaultdict(set)
                for u, v in edge_index.T:
                    u_i, v_i = int(u), int(v)
                    adj_full[u_i].add(v_i)
                    adj_full[v_i].add(u_i)
                sorted_edges = np.argsort(edge_importance)[::-1]
                top_edges_count = min(len(sorted_edges), max(10, k_nodes * 3))
                top_edge_indices = sorted_edges[:top_edges_count]
                node_imp = defaultdict(float)
                edges_list = []
                for eidx in top_edge_indices:
                    u_idx = int(edge_index[0, eidx])
                    v_idx = int(edge_index[1, eidx])
                    imp = float(edge_importance[eidx])
                    node_imp[u_idx] += imp
                    node_imp[v_idx] += imp
                    edges_list.append((u_idx, v_idx, imp))
                if not node_imp:
                    continue
                N_candidate = min(len(node_imp), 12)
                candidate_nodes = sorted(node_imp.keys(),
                                         key=lambda n: node_imp[n],
                                         reverse=True)[:N_candidate]
                if len(candidate_nodes) < k_nodes:
                    continue
                for comb in combinations(candidate_nodes, k_nodes):
                    comb_set = set(comb)
                    comb_edges = [(u, v, imp) for (u, v, imp) in edges_list
                                  if (u in comb_set and v in comb_set)]
                    if len(comb_edges) < k_nodes - 1:
                        continue
                    visited = set()
                    stack = [next(iter(comb_set))]
                    while stack:
                        n = stack.pop()
                        if n in visited:
                            continue
                        visited.add(n)
                        for nb in adj_full.get(n, ()):
                            if nb in comb_set and nb not in visited:
                                stack.append(nb)
                    if len(visited) != k_nodes:
                        continue
                    avg_imp = float(np.mean([e[2] for e in comb_edges]))
                    try:
                        node_names = [self._get_node_name(graph_data, n) for n in comb]
                    except Exception:
                        node_names = [f"n{n}" for n in comb]
                    node_names_sorted = tuple(sorted(node_names))
                    key = (k_nodes, node_names_sorted)
                    edge_imp_dict = defaultdict(list)
                    for u, v, imp in comb_edges:
                        try:
                            uu = self._get_node_name(graph_data, u)
                            vv = self._get_node_name(graph_data, v)
                        except:
                            uu, vv = f"n{u}", f"n{v}"
                        a, b = sorted([uu, vv])
                        edge_imp_dict[(a, b)].append(imp)

                    patterns_k[key].append({
                        'avg_importance': avg_imp,
                        'edge_imp_dict': edge_imp_dict,
                        'source_graph': getattr(graph_data, 'id', None)
                    })

            stats_k = []
            for (k_val, node_names_tuple), instances in patterns_k.items():
                frequency = len(instances)
                frequency_ratio = frequency / num_explanations
                if frequency_ratio < threshold:
                    continue

                # 平均节点组合重要性
                avg_importance = float(np.mean([inst['avg_importance'] for inst in instances]))

                # 计算每条边的平均重要性
                edge_to_vals = defaultdict(list)
                for inst in instances:
                    for edge_key, vals in inst['edge_imp_dict'].items():
                        edge_to_vals[edge_key].extend(vals)
                edge_avg_importance = {k: float(np.mean(v)) for k, v in edge_to_vals.items()}

                # 代表实例：这里我们直接存聚合好的边重要性
                representative_instance = {
                    'node_names': list(node_names_tuple),
                    'edges_avg_importance': edge_avg_importance  # ⬅️ 新增
                }

                pattern_name = f"{k_val}_nodes: {', '.join(node_names_tuple)}"
                stats_k.append({
                    'pattern': pattern_name,
                    'frequency': frequency,
                    'frequency_ratio': frequency_ratio,
                    'avg_importance': avg_importance,
                    'composite_score': avg_importance * frequency_ratio,
                    'representative_instance': representative_instance
                })

            stats_k.sort(key=lambda x: x['composite_score'], reverse=True)
            results['per_k'][k_nodes] = stats_k

            print(f"\n{k_nodes} nodes subgraph — patterns (Top 3): ")
            if stats_k:
                for i, subgraph in enumerate(stats_k[:3], 1):
                    print(f"  {i}. pattern: {subgraph['pattern']}")
                    print(f"     frequency: {subgraph['frequency']}次 "
                          f"({subgraph['frequency_ratio']:.2%}), "
                          f"mean importance score: {subgraph['avg_importance']:.4f}")
            else:
                print(f" No representative pattern (or frequency under threshold {threshold:.2f}).")
                break

            overall_patterns.extend(stats_k)

        overall_patterns.sort(key=lambda x: x['composite_score'], reverse=True)
        results['representative_subgraphs'] = overall_patterns[:top_k]

        if draw_pattern and results['per_k']:
            patterns_to_draw = []
            for k_nodes in sorted(results['per_k'].keys()):
                patterns_to_draw.extend(results['per_k'][k_nodes][:3])
            self._draw_representative_subgraphs(patterns_to_draw)

        print("\n" + "=" * 60)
        return results


    def _draw_representative_subgraphs(self, representative_subgraphs: list):
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        output_dir = "sub_graph_out"
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nStarting plot {len(representative_subgraphs)} subgraph(s)...")

        # 颜色映射
        all_species = set()
        for subgraph in representative_subgraphs:
            all_species.update(subgraph['representative_instance']['node_names'])
        colors = plt.cm.Set3(np.linspace(0, 1, len(all_species)))
        species_color_map = dict(zip(sorted(all_species), colors))

        for idx, subgraph in enumerate(representative_subgraphs):
            instance = subgraph['representative_instance']
            node_names = instance['node_names']
            edge_avg_importance = instance['edges_avg_importance']

            G = nx.Graph()
            node_id_map = {name: i for i, name in enumerate(node_names)}
            for name in node_names:
                G.add_node(node_id_map[name], species=name)

            edge_importances = {}
            for (u_name, v_name), imp in edge_avg_importance.items():
                if u_name in node_id_map and v_name in node_id_map:
                    u_id, v_id = node_id_map[u_name], node_id_map[v_name]
                    G.add_edge(u_id, v_id)
                    edge_importances[(u_id, v_id)] = imp
                    edge_importances[(v_id, u_id)] = imp

            plt.figure(figsize=(12, 9))
            n_nodes = len(G.nodes())
            k_layout = 0.4 if n_nodes <= 5 else 0.5
            pos = nx.spring_layout(G, k=k_layout, iterations=200, seed=42)

            node_colors = [species_color_map[node_names[node]] for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                   node_size=800, alpha=0.9, edgecolors='black', linewidths=1.5)
            nx.draw_networkx_labels(G, pos, {i: str(i) for i in G.nodes()},
                                    font_size=14, font_weight='bold', font_color='white')

            if edge_importances:
                imp_values = list(edge_importances.values())
                min_imp, max_imp = min(imp_values), max(imp_values)
                edge_cmap = LinearSegmentedColormap.from_list("edge_colors",
                                                              ["lightblue", "orange", "red"])
                for (u, v), importance in edge_importances.items():
                    if max_imp > min_imp:
                        norm_imp = (importance - min_imp) / (max_imp - min_imp)
                    else:
                        norm_imp = 0.5
                    color = edge_cmap(norm_imp)
                    width = 2 + 6 * norm_imp
                    nx.draw_networkx_edges(G, pos, [(u, v)], edge_color=[color],
                                           width=width, alpha=0.8)
                    x = (pos[u][0] + pos[v][0]) / 2
                    y = (pos[u][1] + pos[v][1]) / 2
                    plt.text(x, y, f'{importance:.3f}', fontsize=9,
                             bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9, edgecolor='gray'),
                             ha='center', va='center', weight='bold')
            else:
                nx.draw_networkx_edges(G, pos, edge_color='gray', width=2, alpha=0.6)

            legend_elements = [mpatches.Patch(color=species_color_map[sp],
                                              label=f'{i}: {sp}')
                               for i, sp in enumerate(node_names)]
            plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                       fontsize=12, frameon=True, fancybox=True, shadow=True)

            title_text = (f'Representative Subgraph #{idx + 1}\n'
                          f'Frequency: {subgraph["frequency"]} times '
                          f'({subgraph["frequency_ratio"]:.2%}), '
                          f'Avg Importance: {subgraph["avg_importance"]:.4f}')
            plt.title(title_text, fontsize=14, pad=20, weight='bold')
            plt.axis('off')
            plt.tight_layout()

            filename = f"subgraph_{idx + 1:03d}_{len(node_names)}nodes_score{subgraph['composite_score']:.4f}.svg"
            filepath = os.path.join(output_dir, filename)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plt.savefig(filepath, format='svg', dpi=300, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
            plt.close()
            print(f" Saved: {filename}")

        print(f"All subgraphs saved to '{output_dir}' ")

    def _build_graph(self, graph_data) -> nx.Graph:
        G = nx.Graph()
        edge_index = graph_data.edge_index.cpu().numpy()
        edges = [(int(edge_index[0, i]), int(edge_index[1, i]))
                 for i in range(edge_index.shape[1])]
        G.add_edges_from(edges)
        return G

    def _get_node_name(self, graph_data, node_idx: int) -> str:
        if hasattr(graph_data, 'tax_names') and node_idx < len(graph_data.tax_names):
            return graph_data.tax_names[node_idx]
        return f"Node_{node_idx}"

    def _compute_summary_stats(self, valid_explanations: List[Any] = None) -> Dict:
        explanations_to_use = valid_explanations if valid_explanations is not None else self.explanations
        total_samples = len(explanations_to_use)

        node_importance_stats = []
        edge_importance_stats = []

        for explanation in self.explanations:
            if hasattr(explanation, 'node_mask') and explanation.node_mask is not None:
                node_scores = explanation.node_mask.cpu().numpy()
                if node_scores.ndim > 1:
                    node_scores = node_scores.mean(axis=1)
                node_importance_stats.extend(node_scores.tolist())

            if hasattr(explanation, 'edge_mask') and explanation.edge_mask is not None:
                edge_scores = explanation.edge_mask.cpu().numpy()
                edge_importance_stats.extend(edge_scores.tolist())

        return {
            'total_samples': total_samples,
            'node_importance_mean': np.mean(node_importance_stats) if node_importance_stats else 0,
            'node_importance_std': np.std(node_importance_stats) if node_importance_stats else 0,
            'edge_importance_mean': np.mean(edge_importance_stats) if edge_importance_stats else 0,
            'edge_importance_std': np.std(edge_importance_stats) if edge_importance_stats else 0,
        }

    def _print_analysis_results(self, results: Dict, top_k: int):
        """Print analysis results"""
        print(f"\n{'=' * 60}")
        print(f"GNN Explanation Comprehensive Analysis Report (Top-{top_k})")
        print(f"{'=' * 60}")

        print(f"\n📊 Summary Statistics:")
        stats = results['summary_statistics']
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Mean node importance: {stats['node_importance_mean']:.4f} ± {stats['node_importance_std']:.4f}")
        print(f"  Mean edge importance: {stats['edge_importance_mean']:.4f} ± {stats['edge_importance_std']:.4f}")

        print(f"\n🎯 Most Representative Nodes:")
        for i, node in enumerate(results['representative_nodes'][:5], 1):
            print(f"  {i}. {node['node_name']}")
            print(
                f"     Occurrence count: {node['occurrence_count']:.2f}, Occurrence ratio: {node['occurrence_ratio']:.2%}, "
                f"Average degree: {node['avg_degree']:.1f}, Composite score: {node['composite_score']:.4f}")

        print(f"\n🔗 Most Representative Edges:")
        for i, edge in enumerate(results['representative_edges'][:5], 1):
            print(f"  {i}. {edge['edge_key']}")
            print(f"     Importance: {edge['avg_importance']:.4f}, Occurrence ratio: {edge['occurrence_ratio']:.2%}, "
                  f"Composite score: {edge['composite_score']:.4f}")

        print(f"\n{'=' * 60}")


# 使用示例函数
def analyze_multiple_explanations(explanations: List[Any],
                                  graph_data_list: List[Any],
                                  top_k: int = 10, threshold: float = 0.1) -> Dict[str, Any]:
    analyzer = ExplanationAnalyzer(explanations, graph_data_list)
    return analyzer.find_representative_elements(top_k, threshold)


if __name__ == '__main__':
    pts = []
    l_uids = []
    task = 'all'
    # task = 'test'
    root_dir = 'explanations/pt'
    for file in os.listdir(root_dir):
        if file.endswith('.pt'):
            pt_file = torch.load(f'{root_dir}/{file}')
            luid = file.replace('explanation_', '').replace('.pt', '')
            l_uids.append(luid)
            pts.append(pt_file)

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


    batch_size = 512
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
        p_threshold=0.1,
        n_threshold=-0.1,
        pre_transform=transform,
    )
    indices = list(range(len(dataset)))

    train_valid_idx, test_idx = train_test_split(
        indices, test_size=0.15, random_state=seed, shuffle=True
    )

    if task == 'test':
        dataset = dataset[test_idx]
    sorted_datasets = []
    for luid in l_uids:
        for data in dataset:
            if str(luid) == str(data.uid):
                sorted_datasets.append(data)
    assert len(sorted_datasets) == len(l_uids)
    analyze_multiple_explanations(pts, sorted_datasets, top_k=10, threshold=0.05)
