# -*- coding: utf-8 -*-
# @Time    : 2025/5/17 11:59
# @Author  : HaiqingSun
# @OriginalFileName: utils
# @Software: PyCharm
# @AIMedicineWebPage: https://www.ilovemyhome.cc

import csv
import json
import pickle
import os
import re
from math import sqrt

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.stats import stats
from sklearn.metrics import r2_score
import seaborn as sns


PERSIST_FOLDER = './cache'
if not os.path.exists(PERSIST_FOLDER):
    os.makedirs(PERSIST_FOLDER)
PERSIST_FILE = PERSIST_FOLDER + '/' + "taxonomy_ko_mapping.pkl"

def _build_and_persist_mapping(org_ko_file='./step5.fetch_pathways/org_ko_mapping.csv', taxid_mapping_file='./step5.fetch_pathways/taxid_T_org_mapping.csv'):
    """
    Construct a mapping of TaxonomyID to KO_Pathway_ID and save it to a pickle file.
    """
    org_ko_map = {}
    with open(org_ko_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            org = row['Org_code']
            ko = row['KO_Pathway_ID']
            org_ko_map.setdefault(org, []).append(ko)

    taxonomy_ko_map = {}
    with open(taxid_mapping_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            org = row['Org_code']
            taxid = row['TaxonomyID']
            ko_list = org_ko_map.get(org, [])
            taxonomy_ko_map[taxid] = ko_list

    with open(PERSIST_FILE, 'wb') as f:
        pickle.dump(taxonomy_ko_map, f)

    print("The persistent mapping is built.")

def get_ko_pathways_by_taxid(taxid):
    """
    Enter a TaxonomyID and return a list of corresponding KO_Pathway_IDs.
    """
    if not os.path.exists(PERSIST_FILE):
        _build_and_persist_mapping()

    with open(PERSIST_FILE, 'rb') as f:
        taxonomy_ko_map = pickle.load(f)

    return taxonomy_ko_map.get(str(taxid), [])




def _extract_brite_ids_from_file(file_path):
    """
    Extract all BRITE field pure numeric numbers (5 digits) from a single file.
    """
    brite_ids = set()
    in_brite_section = False
    pattern = re.compile(r'^\s*(\d{5})\s')

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("BRITE"):
                in_brite_section = True
            elif line.strip() == "" and in_brite_section:
                break
            elif in_brite_section:
                match = pattern.match(line)
                if match:
                    brite_ids.add(match.group(1))
    return brite_ids

def count_all_brite_ids(directory):
    """
    Count the number of BRITE numbers (5-digit numbers only) that appear in all files.
    """
    all_ids = set()
    for filename in os.listdir(directory):
        if filename.startswith("K") and filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            brite_ids = _extract_brite_ids_from_file(file_path)
            all_ids.update(brite_ids)
    return all_ids

def generate_onehot_vectors(directory, cache_dir='cache', rebuild_cache=False):
    """
        Generate a corresponding BRITE one-hot vector for each K number file and save the mapping to the cache directory.
        Returns:
        brite_list: all numbers (sequential)
        vectors: dict, key = K number, value = one-hot list
    """
    os.makedirs(cache_dir, exist_ok=True)
    onehot_path = os.path.join(cache_dir, 'k_id_onehot_vectors.json')
    brite_index_path = os.path.join(cache_dir, 'brite_id_index.json')


    if os.path.exists(onehot_path) and os.path.exists(brite_index_path) and not rebuild_cache:
        vectors = json.load(open(onehot_path, 'r'))
        all_ids = len(json.load(open(brite_index_path, 'r')))
        return all_ids, vectors
    all_ids = sorted(count_all_brite_ids(directory))
    id_index = {brite_id: idx for idx, brite_id in enumerate(all_ids)}
    vectors = {}

    for filename in os.listdir(directory):
        if filename.startswith("K") and filename.endswith(".txt"):
            k_id = filename.replace(".txt", "")
            file_path = os.path.join(directory, filename)
            brite_ids = _extract_brite_ids_from_file(file_path)
            onehot = [0] * len(all_ids)
            for bid in brite_ids:
                if bid in id_index:
                    onehot[id_index[bid]] = 1
            vectors[k_id] = onehot


    with open(onehot_path, 'w') as f:
        json.dump(vectors, f)


    with open(brite_index_path, 'w') as f:
        json.dump(id_index, f)

    return all_ids, vectors


def load_csv_indices(csv_path):
    mapping = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row['name']] = int(row['index'])
    return mapping

def mae(G, P):
    G = np.asarray(G)
    P = np.asarray(P)

    assert G.shape == P.shape, f"Shape mismatch: G{G.shape} vs P{P.shape}"
    return np.mean(np.abs(G - P))

def cal_r2_score(y, f):
    r2 = r2_score(y, f)
    n_iterations = 100
    r2_values = np.zeros(n_iterations)
    for i in range(n_iterations):
        random_pred = np.random.rand(len(y))
        r2_values[i] = r2_score(y, random_pred)

    r2_std = r2, np.std(r2_values)
    return r2_std

def rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse


def mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci

def visualize_and_save_masks(M_explain, masks, output_dir="tabnet_masks_output", feature_names=None):
    os.makedirs(output_dir, exist_ok=True)

    M_explain = M_explain.detach().cpu()
    M_explain_norm = M_explain / (M_explain.sum(dim=1, keepdim=True) + 1e-8)
    df_explain = pd.DataFrame(M_explain_norm.numpy(),
                              columns=feature_names if feature_names else [f"feature_{i}" for i in
                                                                           range(M_explain.shape[1])])
    df_explain.index.name = "sample_idx"
    df_explain.to_csv(os.path.join(output_dir, "M_explain_normalized.csv"))
    print("Saved normalized M_explain to CSV")
    plt.figure(figsize=(14, 8))
    sns.heatmap(M_explain_norm[:, :], cmap="YlGnBu")
    plt.title("M_explain (normalized) - Top 100 samples × 20 features")
    plt.xlabel("Feature index")
    plt.ylabel("Sample index")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "M_explain_heatmap.svg"))
    plt.close()
    print("Saved M_explain heatmap")

    for step, mask_tensor in masks.items():
        mask_tensor = mask_tensor.detach().cpu()
        df_mask = pd.DataFrame(mask_tensor.numpy(), columns=feature_names if feature_names else [f"feature_{i}" for i in
                                                                                                 range(
                                                                                                     mask_tensor.shape[
                                                                                                         1])])
        df_mask.index.name = "sample_idx"
        df_mask.to_csv(os.path.join(output_dir, f"mask_step_{step}.csv"))
        print(f"Saved mask for step {step} to CSV")

        plt.figure(figsize=(14, 8))
        sns.heatmap(mask_tensor[:, :], cmap="YlGnBu")
        plt.title(f"Step {step} mask - Top 100 samples × 20 features")
        plt.xlabel("Feature index")
        plt.ylabel("Sample index")
        plt.tight_layout()
        # plt.savefig(os.path.join(output_dir, f"mask_step_{step}_heatmap.png"))
        plt.savefig(os.path.join(output_dir, f"mask_step_{step}_heatmap.svg"))
        plt.close()
        print(f"Saved heatmap for mask step {step}")