# -*- coding: utf-8 -*- 
# @Time : 2025/12/12 0:46 
# @Author : DirtyBoy 
# @File : evaluate_uncertainty_calibration.py
import json, os, joblib
import numpy as np
from scipy import special
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

model_type = 'causallm'

from sklearn.metrics import f1_score, confusion_matrix


def predictive_kld(p, number=10, w=None, base=2, eps=1e-10):
    if number <= 1:
        return np.zeros_like(p)

    p_arr = np.asarray(p).reshape((-1, number))
    q_arr = np.tile(np.mean(p_arr, axis=-1, keepdims=True), [1, number])
    if w is None:
        w_arr = np.ones(shape=(number, 1), dtype=float) / number
    else:
        w_arr = np.asarray(w).reshape((number, 1))

    kld_elem = p_arr * np.log((p_arr + eps) / (q_arr + eps)) + (1. - p_arr) * np.log(
        (1. - p_arr + eps) / (1. - q_arr + eps))
    if base is not None:
        kld_elem = kld_elem / np.log(base)
    kld = np.matmul(kld_elem, w_arr)
    return kld[0][0]


def max_max2(end_list):
    max2 = np.sort(end_list)[-2]
    max = np.max(end_list)
    return max - max2


def max_min(end_list):
    min = np.min(end_list)
    max = np.max(end_list)
    return max - min


def mean_med(end_list):
    mean = np.mean(end_list)
    med = np.median(end_list)
    return mean


def sym_auc(auc):
    return 1 - auc if auc < 0.5 else auc


def confusion_index_dict(gt_labels, preds):
    return {
        "TP": [i for i, (g, p) in enumerate(zip(gt_labels, preds)) if g == 1 and p == 1],
        "FP": [i for i, (g, p) in enumerate(zip(gt_labels, preds)) if g == 0 and p == 1],
        "TN": [i for i, (g, p) in enumerate(zip(gt_labels, preds)) if g == 0 and p == 0],
        "FN": [i for i, (g, p) in enumerate(zip(gt_labels, preds)) if g == 1 and p == 0],
    }


def compute_variance(pt_samples):
    pt_mean = np.mean(pt_samples, axis=0)
    diag_terms = np.array([np.diag(pt) - np.outer(pt, pt) for pt in pt_samples])
    aleatoric_uncertainty = np.mean(diag_terms, axis=0)
    centered_terms = np.array([np.outer(pt - pt_mean, pt - pt_mean) for pt in pt_samples])
    epistemic_uncertainty = np.mean(centered_terms, axis=0)
    return aleatoric_uncertainty, epistemic_uncertainty


def read_joblib(path):
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


def read_from_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def evaluate_binary_logits(
        logits,
        labels,
        threshold=0.5
):
    import numpy as np
    from sklearn.metrics import brier_score_loss
    from scipy.special import softmax

    def brier_score(p, q, pos_label=1):
        return brier_score_loss(q, p, pos_label=pos_label)

    def nll(p, q, eps=1e-10, base=2):
        nll = -(q * np.log(p + eps) + (1. - q) * np.log(1. - p + eps))
        if base is not None:
            nll = np.clip(nll / np.log(base), a_min=0., a_max=1000)
        return np.mean(nll)

    def compute_ece(probs, labels, n_bins=15):
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        N = len(probs)

        for i in range(n_bins):
            bin_lower = bins[i]
            bin_upper = bins[i + 1]

            mask = (probs > bin_lower) & (probs <= bin_upper)
            if np.sum(mask) == 0:
                continue

            bin_confidence = np.mean(probs[mask])
            bin_accuracy = np.mean(labels[mask])
            ece += np.abs(bin_confidence - bin_accuracy) * np.sum(mask) / N

        return float(ece)

    probs = softmax(logits, axis=1)[:, 1]

    preds = (probs >= threshold).astype(np.int64)

    TP = np.sum((preds == 1) & (labels == 1))
    TN = np.sum((preds == 0) & (labels == 0))
    FP = np.sum((preds == 1) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    fnr = FN / (FN + TP)
    fpr = FP / (FP + TN)

    precision_pos = TP / (TP + FP)
    recall_pos = TP / (TP + FN)
    npv = TN / (TN + FN)
    f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos)

    precision_neg = TN / (TN + FN)
    recall_neg = TN / (TN + FP)

    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg)

    macro_f1 = 0.5 * (f1_pos + f1_neg)

    NLL = nll(probs, labels)
    brier = brier_score(probs, labels)
    ece = compute_ece(probs, labels)

    metrics = {
        "accuracy": accuracy,
        "fnr": fnr,
        "fpr": fpr,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "f1_pos": f1_pos,
        "f1_neg": f1_neg,
        "macro_f1": macro_f1,
        "nll": NLL,
        "brier": brier,
        "ece": ece
    }

    print("Accuracy: {:.5f}%".format(accuracy * 100))
    print("FNR: {:.5f}%, FPR: {:.5f}%".format(fnr * 100, fpr * 100))
    # print("Precision(+): {:.5f}%, Recall(+): {:.5f}%, F1(+): {:.5f}%".format(
    #     precision_pos * 100,
    #     recall_pos * 100,
    #     f1_pos * 100
    # ))
    # print("F1(-): {:.5f}%, Macro-F1: {:.5f}%".format(
    #     f1_neg * 100,
    #     macro_f1 * 100
    # ))
    print(" Macro-F1: {:.5f}%".format(
        macro_f1 * 100
    ))
   # print("MCC: {:.5f}".format(mcc))
    print("ECE is {:.6f}".format(metrics["ece"]))
    print("NLL is {:.6f}".format(metrics["nll"]))
    print("Brier Score is {:.6f}".format(metrics["brier"]))



def eval_uncertainty(uc_type="mcdropout", n_samples=3):
    print(f'============================={uc_type} Model================================')
    logits = np.array([read_joblib(
        f'../bayesian_peft/output/{model_type}/{uc_type}/{dataset_name}_train{suffix}/{dataset_name}_test{suffix}/epoch{epoch}/{llm_name}-{i + 1}.data')
        for i in range(n_samples)]).transpose(1, 0, 2)

    log_probs = special.softmax(logits, axis=-1)  # [N, 10, C]
    mean_log_probs = logits.mean(axis=1)  # [N, C]
    # uc_preds = np.argmax(mean_log_probs, axis=-1)
    evaluate_binary_logits(mean_log_probs, gt_labels[0:len(mean_log_probs)])
    print(' =================================================================================')

    aleatoric_uncertaintys = np.array([np.trace(compute_variance(item)[0]) for item in log_probs])
    epistemic_uncertaintys = np.array([np.trace(compute_variance(item)[1]) for item in log_probs])
    # PCS = np.array([max_max2(np.transpose(item, (1, 0))[1]) for item in log_probs])
    Pred_GAP = np.array([max_min(np.transpose(item, (1, 0))[1]) for item in log_probs])
    MM = np.array([mean_med(np.transpose(item, (1, 0))[1]) for item in log_probs])
    # D_KL = np.array([predictive_kld(np.transpose(item, (1, 0))[1], number=n_samples) for item in log_probs])

    uncertainties = [
        ("Aleatoric", aleatoric_uncertaintys),
        ("Epistemic", epistemic_uncertaintys),
        # ("D_KL", D_KL),
        # ("PCS", PCS),
        ("Pred_GAP", Pred_GAP),
        ("Mean", MM)
    ]

    print("| Uncertainty | Incorrect vs Correct | FN vs TN | FP vs TP | Positive vs Negative |")
    print("|-------------|----------------------|----------|----------|----------------------|")

    RED = "\033[31m"
    RESET = "\033[0m"

    col_widths = [20, 8, 8, 20]

    for name, uncertainty in uncertainties:
        row_values = []
        row_values_float = []

        for _, pos_idx, neg_idx in groups:
            y_true = np.array([1] * len(pos_idx) + [0] * len(neg_idx))
            y_score = np.concatenate([
                uncertainty[pos_idx],
                uncertainty[neg_idx]
            ])
            auc = sym_auc(roc_auc_score(y_true, y_score))

            row_values.append(f"{auc:.4f}")
            row_values_float.append(auc)

        max_idx = int(np.argmax(row_values_float))

        aligned = [
            f"{v:<{w}}"
            for v, w in zip(row_values, col_widths)
        ]
        aligned[max_idx] = f"{RED}{aligned[max_idx]}{RESET}"

        print(
            f"| {name:<11} | "
            f"{aligned[0]} | "
            f"{aligned[1]} | "
            f"{aligned[2]} | "
            f"{aligned[3]} | "
        )
    return aleatoric_uncertaintys


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-llm_name', '-ln', type=str, default="Salesforce/codegen-2B-multi")
    parser.add_argument('-dataset_name', '-dn', type=str, default="vulcure_vd_random")
    parser.add_argument('-epoch', '-e', type=int, default=4)
    args = parser.parse_args()
    llm_name = args.llm_name
    epoch = args.epoch
    dataset_name = args.dataset_name
    n_samples = 10

    # llm_list = ["codellama/CodeLlama-7b-hf", "bigcode/starcoder2-3b", "Qwen/Qwen2.5-Coder-3B", "Salesforce/codegen-2B-multi
    #             "Qwen/Qwen3-4B-Instruct-2507", "bigcode/starcoder2-7b", "microsoft/codebert-base"] deepseek-AI/deepseek-coder-6.7b-base

    # test_data_list = ["code_type_binary_python_java_##tmp##", "vulcure_vd_random_##tmp##_800",
    #                   "vulcure_vd_dataset_##tmp##_800"]

    if "vd" in dataset_name:
        suffix = "_800"
    else:
        suffix = ""

    print(f'=========================={dataset_name}_test{suffix}=============================')
    source_data = read_from_jsonl(f'../Database/dataset/{dataset_name}_test{suffix}.jsonl')

    gt_labels = np.array([1 if item['target'] == 1 else 0 for item in source_data])

    zero_shot = read_joblib(
        f'../bayesian_peft/output/{model_type}/zero_shot/{dataset_name}_test{suffix}/epoch{epoch}/{llm_name}-1.data')
    print(zero_shot.shape)

    print('=============================Zero Shot================================')
    evaluate_binary_logits(zero_shot, gt_labels)

    probs = read_joblib(
        f'../bayesian_peft/output/{model_type}/mle/{dataset_name}_train{suffix}/{dataset_name}_test{suffix}/epoch{epoch}/{llm_name}-1.data')

    preds = [np.argmax(item) for item in probs]

    samples_indexs = confusion_index_dict(gt_labels, preds)

    groups = [
        ("Incorrect vs Correct",
         samples_indexs['FP'] + samples_indexs['FN'],
         samples_indexs['TP'] + samples_indexs['TN']),

        ("FN vs TN",
         samples_indexs['FN'],
         samples_indexs['TN']),

        ("FP vs TP",
         samples_indexs['FP'],
         samples_indexs['TP']),

        ("Positive vs Negative",
         samples_indexs['TP'] + samples_indexs['FN'],
         samples_indexs['TN'] + samples_indexs['FP']),
    ]

    print('=============================MLE Model================================')
    evaluate_binary_logits(probs, gt_labels)

    # aug_probs = read_joblib(
    #     f'../bayesian_peft/output/{model_type}/mle/{dataset_name}_trainplus{suffix}/{dataset_name}_test{suffix}/epoch{epoch}/{llm_name}-1.data')
    #
    # print('=============================MLE aug Model================================')
    # evaluate_binary_logits(aug_probs, gt_labels)

    uc_type = "mcdropout"
    try:
        eval_uncertainty(uc_type, n_samples=n_samples)
    except Exception as e:
        print(e)

    uc_type = "blob"
    try:
        eval_uncertainty(uc_type, n_samples=n_samples)
    except Exception as e:
        print(e)

    uc_type = "deepensemble"
    try:
        eval_uncertainty(uc_type, n_samples=n_samples)
    except Exception as e:
        print(e)
