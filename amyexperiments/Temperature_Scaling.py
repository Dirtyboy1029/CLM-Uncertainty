# -*- coding: utf-8 -*- 
# @Time : 2026/1/7 14:43 
# @Author : DirtyBoy 
# @File : Temperature_Scaling.py
import torch
import torch.nn as nn
import os, joblib, json
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import brier_score_loss
from scipy.special import softmax





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

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-12)

    fnr = FN / (FN + TP + 1e-12)
    fpr = FP / (FP + TN + 1e-12)

    precision = TP / (TP + FP + 1e-12)
    recall = TP / (TP + FN + 1e-12)

    NPV = TN / float(TN + FN)

    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    # ---------- NLL ----------
    NLL = nll(probs, labels)

    # ---------- Brier Score ----------
    brier = brier_score(probs, labels)

    # ---------- ECE ----------
    ece = compute_ece(probs, labels)

    metrics = {
        "accuracy": accuracy,
        "fnr": fnr,
        "fpr": fpr,
        "npv": NPV,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "nll": NLL,
        "brier": brier,
        "ece": ece
    }

    print("=" * 40)
    print("Classification Metrics")
    print("=" * 40)

    print("Accuracy is {:.5f}%".format(metrics["accuracy"] * 100))
    print("FNR is {:.5f}%, FPR is {:.5f}%".format(
        metrics["fnr"] * 100,
        metrics["fpr"] * 100
    ))
    print("NPV is {:.5f}%".format(metrics["npv"] * 100))
    print("Precision is {:.5f}%, Recall is {:.5f}%, F1 is {:.5f}%".format(
        metrics["precision"] * 100,
        metrics["recall"] * 100,
        metrics["f1"] * 100
    ))

    print("=" * 40)
    print("Uncertainty / Calibration Metrics")
    print("=" * 40)

    print("NLL is {:.6f}".format(metrics["nll"]))
    print("Brier Score is {:.6f}".format(metrics["brier"]))
    print("ECE is {:.6f}".format(metrics["ece"]))


def read_from_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def read_joblib(path):
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


class BinaryTemperatureScaler(nn.Module):

    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self):
        return torch.exp(self.log_temperature)

    def forward(self, logits):
        return logits / self.temperature


def fit_temperature(logits_calib, labels_calib, max_iter=500):
    scaler = BinaryTemperatureScaler()
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.LBFGS(
        scaler.parameters(),
        lr=0.01,
        max_iter=max_iter
    )

    def closure():
        optimizer.zero_grad()
        scaled_logits = scaler(logits_calib)
        loss = criterion(scaled_logits, labels_calib)
        loss.backward()
        return loss

    optimizer.step(closure)

    return scaler


@torch.no_grad()
def calibrated_probability(scaler, logits):
    scaled_logits = scaler(logits)
    return scaled_logits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-llm_name', '-ln', type=str, default="Qwen/Qwen2.5-Coder-3B")
    parser.add_argument('-dataset_name', '-dn', type=str, default="code_type_binary_python_java")
    parser.add_argument('-epoch', '-e', type=int, default=1)
    args = parser.parse_args()
    llm_name = args.llm_name
    epoch = args.epoch
    dataset_name = args.dataset_name

    if "vd" in dataset_name:
        suffix = "_800"
    else:
        suffix = ""

    correct_set = read_from_jsonl(
        f'../bayesian_peft/database/{dataset_name}_correct{suffix}.jsonl')
    correct_set_labels = [item['target'] for item in correct_set]
    correct_set_logits = read_joblib(
        f"../bayesian_peft/output/causallm/mle/{dataset_name}_train{suffix}/{dataset_name}_correct{suffix}/epoch{epoch}/{llm_name}-1.data")

    test_set = read_from_jsonl(
        f'../bayesian_peft/database/{dataset_name}_test{suffix}.jsonl')
    test_set_labels = [item['target'] for item in test_set]
    test_set_logits = read_joblib(
        f"../bayesian_peft/output/causallm/mle/{dataset_name}_train{suffix}/{dataset_name}_test{suffix}/epoch{epoch}/{llm_name}-1.data")

    correct_set_logits = torch.tensor(correct_set_logits)
    correct_set_labels = torch.tensor(correct_set_labels)

    test_set_logits = torch.tensor(test_set_logits)
    test_set_labels = torch.tensor(test_set_labels)

    scaler = fit_temperature(correct_set_logits, correct_set_labels)

    print(f"Learned temperature T = {scaler.temperature.item():.4f}")

    cal_logits = calibrated_probability(scaler, test_set_logits)
    print("+++++++++++++++++++++ before Calibration +++++++++++++++++++++")
    evaluate_binary_logits(test_set_logits.detach().cpu().numpy(),
                           test_set_labels.detach().cpu().numpy())
    print("+++++++++++++++++++++ after Calibration +++++++++++++++++++++")
    evaluate_binary_logits(cal_logits.detach().cpu().numpy(), test_set_labels.detach().cpu().numpy())
