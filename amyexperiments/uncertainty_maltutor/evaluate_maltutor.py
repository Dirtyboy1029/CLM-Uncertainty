# -*- coding: utf-8 -*- 
# @Time : 2026/1/16 14:40 
# @Author : DirtyBoy 
# @File : evaluate_maltutor.py
import json, os, joblib
import numpy as np

suffix = "_800"
model_type = "causallm"
llm_7b = ["deepseek-AI/deepseek-coder-6.7b-base", "bigcode/starcoder2-7b", "Salesforce/codegen-6B-multi"]


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


def evaluate_binary_logits(
        logits,
        labels,
        threshold=0.5
):
    from scipy.special import softmax

    probs = softmax(logits, axis=1)[:, 1]

    preds = (probs >= threshold).astype(np.int64)

    TP = np.sum((preds == 1) & (labels == 1))
    TN = np.sum((preds == 0) & (labels == 0))
    FP = np.sum((preds == 1) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))

    # ------------------------
    # 基础指标
    # ------------------------
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    fnr = FN / (FN + TP)
    fpr = FP / (FP + TN)

    precision_pos = TP / (TP + FP)
    recall_pos = TP / (TP + FN)

    npv = TN / (TN + FN)

    # ------------------------
    # F1+（positive class）
    # ------------------------
    f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos)

    # ------------------------
    # F1-（negative class）
    # ------------------------
    precision_neg = TN / (TN + FN)  # NPV
    recall_neg = TN / (TN + FP)  # TNR / specificity

    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg)

    # ------------------------
    # Macro-F1
    # ------------------------
    macro_f1 = 0.5 * (f1_pos + f1_neg)

    # ------------------------
    # MCC
    # ------------------------
    mcc = (TP * TN - FP * FN) / np.sqrt(
        (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    )

    # ------------------------
    # 汇总
    # ------------------------
    metrics = {
        "accuracy": accuracy,
        "fnr": fnr,
        "fpr": fpr,
        "npv": npv,
        "precision_pos": precision_pos,
        "recall_pos": recall_pos,
        "f1_pos": f1_pos,
        "f1_neg": f1_neg,
        "macro_f1": macro_f1,
        "mcc": mcc
    }

    print("Accuracy: {:.5f}%".format(accuracy * 100))
    print("FNR: {:.5f}%, FPR: {:.5f}%".format(fnr * 100, fpr * 100))
    print("Precision(+): {:.5f}%, Recall(+): {:.5f}%, F1(+): {:.5f}%".format(
        precision_pos * 100,
        recall_pos * 100,
        f1_pos * 100
    ))
    print("F1(-): {:.5f}%, Macro-F1: {:.5f}%".format(
        f1_neg * 100,
        macro_f1 * 100
    ))
    print("MCC: {:.5f}".format(mcc))


if __name__ == '__main__':
    dataset_name = "vulcure_vd_random"
    llm_name = "Salesforce/codegen-2B-multi"

    # llm_list = ["codellama/CodeLlama-7b-hf", "bigcode/starcoder2-3b", "Qwen/Qwen2.5-Coder-3B", "Salesforce/codegen-2B-multi
    #             "Qwen/Qwen3-4B-Instruct-2507", "bigcode/starcoder2-7b", "microsoft/codebert-base"] deepseek-AI/deepseek-coder-6.7b-base

    # "Qwen/Qwen3-4B-Instruct-2507"
    source_data = read_from_jsonl(f"../../Database/dataset/{dataset_name}_test_800.jsonl")
    gt_labels = np.array([1 if item['target'] == 1 else 0 for item in source_data])
    if llm_name in llm_7b:
        epoch = 1
        E=3
    else:
        epoch = 2
        E=4

    probs = read_joblib(
        f'../../bayesian_peft/output/{model_type}/mle/{dataset_name}_train{suffix}/{dataset_name}_test{suffix}/epoch{E}/{llm_name}-1.data')

    print('=============================MLE Model================================')
    evaluate_binary_logits(probs, gt_labels)

    aug_probs = read_joblib(
        f'../../bayesian_peft/output/{model_type}/mle/{dataset_name}_trainplus{suffix}/{dataset_name}_test{suffix}/epoch{E}/{llm_name}-1.data')

    print('=============================MLE aug Model================================')
    evaluate_binary_logits(aug_probs, gt_labels)
    try:
        maltutor_easy_prob = read_joblib(
            f'../../bayesian_peft/output/{model_type}/mle/{dataset_name}_easy_{llm_name.replace("/", "_")}/{dataset_name}_test{suffix}/epoch{epoch}/{llm_name}-maltutor-easy.data')

        print('=============================MalTutor Easy Model================================')
        evaluate_binary_logits(maltutor_easy_prob, gt_labels)
    except Exception as e:
        print(e)

    try:
        maltutor_diff_prob = read_joblib(
            f'../../bayesian_peft/output/{model_type}/mle/{dataset_name}_diff_{llm_name.replace("/", "_")}/{dataset_name}_test{suffix}/epoch1/{llm_name}-maltutor-easy-diff.data')

        print('=============================MalTutor Diff Model================================')
        evaluate_binary_logits(maltutor_diff_prob, gt_labels)
    except Exception as e:
        print(e)
    try:
        maltutor_correct_prob = read_joblib(
            f'../../bayesian_peft/output/{model_type}/mle/{dataset_name}_correct{suffix}/{dataset_name}_test{suffix}/epoch1/{llm_name}-maltutor-easy-diff-correct.data')

        print('=============================MalTutor Correct Model================================')
        evaluate_binary_logits(maltutor_correct_prob, gt_labels)
    except Exception as e:
        print(e)

    try:
        maltutor_train_prob = read_joblib(
            f'../../bayesian_peft/output/{model_type}/mle/{dataset_name}_train_800/{dataset_name}_test{suffix}/epoch1/{llm_name}-maltutor-easy-train.data')

        print('=============================MalTutor Train Model================================')
        evaluate_binary_logits(maltutor_train_prob, gt_labels)
    except Exception as e:
        print(e)
    try:
        maltutor_trainplus_prob = read_joblib(
            f'../../bayesian_peft/output/{model_type}/mle/{dataset_name}_trainplus_800/{dataset_name}_test{suffix}/epoch1/{llm_name}-maltutor-easy-train-trainplus.data')

        print('=============================MalTutor Trainplus Model================================')
        evaluate_binary_logits(maltutor_trainplus_prob, gt_labels)
    except Exception as e:
        print(e)
