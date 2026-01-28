# -*- coding: utf-8 -*- 
# @Time : 2026/1/15 22:58 
# @Author : DirtyBoy 
# @File : dataset_reconstruction.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os, json
from collections import defaultdict
import random

random.seed(1234)
import re


def safe_name(s):
    return re.sub(r"[^\w\-\.]", "_", s)


def write_to_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            line = json.dumps(item, ensure_ascii=False)
            f.write(line + '\n')


def read_from_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def malcertain_data_processing(data_name="vulcure_vd_random", data_type=None,
                               llm_name="Qwen/Qwen3-4B-Instruct-2507"):
    if len(data_type) == 1:
        source = pd.read_csv(
            f"feature_csv/{data_name}/{data_type[0]}/{llm_name}.csv"
        )
    else:
        dfs = [
            pd.read_csv(f"feature_csv/{data_name}/{dt}/{llm_name}.csv")
            for dt in data_type
        ]
        source = pd.concat(dfs, axis=0, ignore_index=True)
    gt_labels = np.array(source["gt_label"].tolist())

    uc_method_list = ["blob", "mcdropout", "deepensemble"]
    metrics_name = ["aleatoric", "epistemic", "pred_gap", "pred_mean", "probs", "label_kld", "wd", "md", "ed", "cd"]

    columns_name = [f"{m}_{n}" for m in uc_method_list for n in metrics_name]

    X = source[columns_name].to_numpy()
    random_state = 1234
    n_clusters = 2
    results = {}
    for gt in [0, 1]:
        idx = (gt_labels == gt)
        X_sub = X[idx]

        if X_sub.shape[0] == 0:
            raise ValueError(f"No samples for gt={gt}")

        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X_sub)

        pca = PCA(n_components=10, random_state=random_state)
        X_pca = pca.fit_transform(X_norm)

        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=random_state
        )
        labels = kmeans.fit_predict(X_pca)

        results[gt] = {
            "indices": np.where(idx)[0],
            "cluster_labels": labels,
            "X_pca": X_pca,
            "scaler": scaler,
            "pca": pca,
            "kmeans": kmeans,
            "explained_variance": pca.explained_variance_ratio_
        }

    def uncertainty_score(x):
        return np.mean(x)

    def identify_low_uncertainty_cluster(results, X):
        low_uncertainty_cluster = {}

        for gt in [0, 1]:
            indices = results[gt]["indices"]
            cluster_labels = results[gt]["cluster_labels"]

            X_sub = X[indices]

            cluster_scores = {}

            for c in np.unique(cluster_labels):
                X_c = X_sub[cluster_labels == c]
                scores = np.array([uncertainty_score(x) for x in X_c])
                cluster_scores[c] = scores.mean()
            low_cluster = min(cluster_scores, key=cluster_scores.get)

            low_uncertainty_cluster[gt] = {
                "low_cluster": low_cluster,
                "cluster_scores": cluster_scores
            }

        return low_uncertainty_cluster

    cluster_semantics = identify_low_uncertainty_cluster(results, X)

    return results, cluster_semantics


if __name__ == '__main__':
    llm_name = "bigcode/starcoder2-7b"  #"Salesforce/codegen-2B-multi"
    data_name = "vulcure_vd_random"
    data_type = ["train"]
    my_data = []
    for item in data_type:
        tmp = read_from_jsonl(f"../../Database/dataset/{data_name}_{item}_800.jsonl")
        my_data = my_data + tmp

    results, cluster_semantics = malcertain_data_processing(data_type=data_type, data_name=data_name, llm_name=llm_name)
    gt_labels = np.array([item['target'] for item in my_data])
    sample_uncertainty_type = np.empty(len(my_data), dtype=int)

    for gt in [0, 1]:
        indices = results[gt]["indices"]
        cluster_labels = results[gt]["cluster_labels"]
        low_cluster = cluster_semantics[gt]["low_cluster"]

        for i, c in zip(indices, cluster_labels):
            sample_uncertainty_type[i] = 0 if c == low_cluster else 1

    groups = defaultdict(list)

    for idx, (gt, unc) in enumerate(zip(gt_labels, sample_uncertainty_type)):
        groups[(gt, unc)].append(idx)

    vul_high_idx = groups[(1, 1)]
    vul_low_idx = groups[(1, 0)]
    nonvul_high_idx = groups[(0, 1)]
    nonvul_low_idx = groups[(0, 0)]


    def rebalance_nonvul(
            vul_high_idx,
            nonvul_high_idx,
            nonvul_low_idx
    ):
        vul_high_idx = list(vul_high_idx)
        nonvul_high_idx = list(nonvul_high_idx)
        nonvul_low_idx = list(nonvul_low_idx)

        target = len(vul_high_idx)
        cur = len(nonvul_high_idx)

        if cur < target:
            need = target - cur

            sampled = random.sample(nonvul_low_idx, need)
            nonvul_high_idx.extend(sampled)
            nonvul_low_idx = [i for i in nonvul_low_idx if i not in sampled]

        elif cur > target:
            excess = cur - target
            sampled = random.sample(nonvul_high_idx, excess)
            nonvul_low_idx.extend(sampled)
            nonvul_high_idx = [i for i in nonvul_high_idx if i not in sampled]

        return nonvul_high_idx, nonvul_low_idx


    nonvul_high_idx, nonvul_low_idx = rebalance_nonvul(
        vul_high_idx,
        nonvul_high_idx,
        nonvul_low_idx
    )

    easy_samples = [my_data[i] for i in (nonvul_low_idx + vul_low_idx)]
    random.shuffle(easy_samples)
    print(len(easy_samples))
    diff_samples = [my_data[i] for i in (nonvul_high_idx + vul_high_idx)]
    random.shuffle(diff_samples)
    print(len(diff_samples))

    write_to_jsonl(easy_samples, f"../../bayesian_peft/database/{data_name}_easy_{safe_name(llm_name)}.jsonl")
    write_to_jsonl(diff_samples, f"../../bayesian_peft/database/{data_name}_diff_{safe_name(llm_name)}.jsonl")
