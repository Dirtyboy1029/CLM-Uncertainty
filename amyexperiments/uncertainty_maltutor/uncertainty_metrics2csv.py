# -*- coding: utf-8 -*- 
# @Time : 2026/1/13 12:23 
# @Author : DirtyBoy 
# @File : uncertainty_metrics2csv.py
import pandas as pd
import numpy as np
from utils import *
from scipy import special
import os

llm_7b = ["deepseek-AI/deepseek-coder-6.7b-base", "bigcode/starcoder2-7b", "Salesforce/codegen-6B-multi"]


def main_(llm_name="Qwen/Qwen3-4B-Instruct-2507",
          dataset_name="vulcure_vd_dataset",
          data_type="train"):
    if llm_name in llm_7b:
        epoch = 3
    else:
        epoch = 4
    if "vd" in dataset_name:
        suffix = "_800"
    else:
        suffix = ""

    uc_metrics = []
    uc_metrics_name = []
    source_data = read_from_jsonl(f'../../Database/dataset/{dataset_name}_{data_type}{suffix}.jsonl')

    gt_labels = np.array([1 if item['target'] == 1 else 0 for item in source_data])

    # probs = read_joblib(
    #     f'../../bayesian_peft/output/causallm/mle/{dataset_name}_train{suffix}/{dataset_name}_{data_type}{suffix}/epoch{epoch}/{llm_name}-1.data')
    # probs = special.softmax(probs, axis=-1)
    # probs = np.array([item[1] for item in probs])
    uc_metrics.append(gt_labels)
    # uc_metrics.append(probs)

    uc_metrics_name.append("gt_label")
    # uc_metrics_name.append("mle_prob")

    uc_types = ['blob', "mcdropout", "deepensemble"]

    for uc_type in uc_types:
        logits = np.array([read_joblib(
            f'../../bayesian_peft/output/causallm/{uc_type}/{dataset_name}_train{suffix}/{dataset_name}_{data_type}{suffix}/epoch{epoch}/{llm_name}-{i + 1}.data')
            for i in range(10)]).transpose(1, 0, 2)

        log_probs = special.softmax(logits, axis=-1)
        mean_log_probs = log_probs.mean(axis=1)

        uc_prob = np.array([item[1] for item in mean_log_probs])
        aleatoric_uncertaintys = np.array([np.trace(compute_variance(item)[0]) for item in log_probs])
        epistemic_uncertaintys = np.array([np.trace(compute_variance(item)[1]) for item in log_probs])
        Pred_GAP = np.array([pred_gap(np.transpose(item, (1, 0))[1]) for item in log_probs])
        Pred_Mean = np.array([pred_mead(np.transpose(item, (1, 0))[1]) for item in log_probs])

        uc_metrics_name.append(uc_type + '_probs')
        uc_metrics_name.append(uc_type + '_aleatoric')
        uc_metrics_name.append(uc_type + '_epistemic')
        uc_metrics_name.append(uc_type + '_pred_gap')
        uc_metrics_name.append(uc_type + '_pred_mean')

        uc_metrics.append(uc_prob)
        uc_metrics.append(aleatoric_uncertaintys)
        uc_metrics.append(epistemic_uncertaintys)
        uc_metrics.append(Pred_GAP)
        uc_metrics.append(Pred_Mean)

        label_kld = np.array(
            [prob_label_kld(p=np.transpose(item, (1, 0))[1], label=gt_labels[i]) for i, item in enumerate(log_probs)])

        wd = np.array(
            [Wasserstein_distance(p=np.transpose(item, (1, 0))[1], label=gt_labels[i]) for i, item in
             enumerate(log_probs)])

        ed = np.array(
            [Euclidean_distance(p=np.transpose(item, (1, 0))[1], label=gt_labels[i]) for i, item in
             enumerate(log_probs)])

        md = np.array(
            [Manhattan_distance(p=np.transpose(item, (1, 0))[1], label=gt_labels[i]) for i, item in
             enumerate(log_probs)])

        cd = np.array(
            [Chebyshev_distance(p=np.transpose(item, (1, 0))[1], label=gt_labels[i]) for i, item in
             enumerate(log_probs)])

        uc_metrics.append(label_kld)
        uc_metrics.append(wd)
        uc_metrics.append(ed)
        uc_metrics.append(md)
        uc_metrics.append(cd)

        uc_metrics_name.append(uc_type + '_label_kld')
        uc_metrics_name.append(uc_type + '_wd')
        uc_metrics_name.append(uc_type + '_ed')
        uc_metrics_name.append(uc_type + '_md')
        uc_metrics_name.append(uc_type + '_cd')

    return uc_metrics, uc_metrics_name


if __name__ == '__main__':
    llm_name = "bigcode/starcoder2-7b"  # "Salesforce/codegen-2B-multi"  # "Qwen/Qwen3-4B-Instruct-2507"
    for dataset_name in [ "vulcure_vd_random"]:
        for data_type in ["train"]:
            if data_type == 'test':
                pass
            else:
                if not os.path.isfile(
                        os.path.join("feature_csv", f"{dataset_name}/{data_type}/{llm_name}.csv")):
                    uc_metrics, uc_metrics_name = main_(llm_name=llm_name, dataset_name=dataset_name,
                                                        data_type=data_type)

                    df = pd.DataFrame(
                        {col: arr for col, arr in zip(uc_metrics_name, uc_metrics)}
                    )

                    os.makedirs(os.path.dirname(
                        os.path.join("feature_csv", f"{dataset_name}/{data_type}/{llm_name}.csv")
                    ), exist_ok=True)

                    df.to_csv(os.path.join("feature_csv", f"{dataset_name}/{data_type}/{llm_name}.csv"), index=False)
                    print(f"file save to {dataset_name}/{data_type}/{llm_name}.csv ")
                else:
                    print(f"{dataset_name}/{data_type}/{llm_name}.csv exist!!!")
