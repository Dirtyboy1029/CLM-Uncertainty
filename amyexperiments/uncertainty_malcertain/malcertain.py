# -*- coding: utf-8 -*- 
# @Time : 2026/1/14 23:52 
# @Author : DirtyBoy 
# @File : malcertain.py
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score


def malcertain_data_processing(data_name="vulcure_vd_dataset", data_type="correct",
                               llm_name="deepseek-AI/deepseek-coder-6.7b-base", pn=False):
    source = pd.read_csv(f"feature_csv/{data_name}/{data_type}/{llm_name}.csv")
    vanilla_prob = np.array(source["mle_prob"].tolist())
    gt_labels = np.array(source["gt_label"].tolist())
    vanilla_pred = (vanilla_prob >= 0.5).astype(np.int32)

    uc_method_list = ["deepensemble"]  #"blob", "mcdropout",
    metrics_name = ["aleatoric", "epistemic", "pred_gap", "pred_mean", "probs"]

    columns_name = [f"{m}_{n}" for m in uc_method_list for n in metrics_name]
    X = source[columns_name].to_numpy()
    y = (vanilla_pred != gt_labels).astype(int)
    if pn:
        idx_pred_p = np.where(vanilla_pred == 1)[0]
        idx_pred_n = np.where(vanilla_pred == 0)[0]

        X_P = X[idx_pred_p, :]
        X_N = X[idx_pred_n, :]

        y_p = y[idx_pred_p]
        y_n = y[idx_pred_n]
        return X_P, y_p, X_N, y_n
    else:
        return X, y


if __name__ == '__main__':
    llm_name = "bigcode/starcoder2-3b"
    data_name = "vulcure_vd_random"


    source = pd.read_csv(f"feature_csv/{data_name}/test/{llm_name}.csv")
    vanilla_prob = np.array(source["mle_prob"].tolist())
    gt_labels = np.array(source["gt_label"].tolist())
    vanilla_pred = (vanilla_prob >= 0.5).astype(np.int32)
    for pn in [False,True]:
        if pn:
            print('+++++++++++++++MalCertain MC++++++++++++++++++++')
            idx_pred_p = list(np.where(vanilla_pred == 1)[0])
            idx_pred_n = list(np.where(vanilla_pred == 0)[0])

            X_train_P, y_train_p, X_train_N, y_train_n = malcertain_data_processing(data_type="correct", pn=pn,
                                                                                    llm_name=llm_name, data_name=data_name)
            X_test_P, y_test_p, X_test_N, y_test_n = malcertain_data_processing(data_type="test", pn=pn,
                                                                                llm_name=llm_name, data_name=data_name)

            P_clf = svm.SVC(C=1000, gamma=0.01, kernel="linear")
            N_clf = svm.SVC(C=1000, gamma=0.01, kernel="linear")

            P_clf.fit(X=X_train_P, y=y_train_p)
            N_clf.fit(X=X_train_N, y=y_train_n)

            y_test_p_mask = P_clf.predict(X_test_P)
            y_test_n_mask = N_clf.predict(X_test_N)

            print(accuracy_score(y_test_n, y_test_n_mask))
            print(accuracy_score(y_test_p, y_test_p_mask))

            malcertain_mask = []

            for i in range(len(gt_labels)):
                if i in idx_pred_p:
                    malcertain_mask.append(y_test_p_mask[idx_pred_p.index(i)])
                else:
                    malcertain_mask.append(y_test_n_mask[idx_pred_n.index(i)])
        else:
            print("++++++++++++++++++MalCertain++++++++++++++++++++++")
            X_train, y_train = malcertain_data_processing(data_type="correct", pn=pn,
                                                          llm_name=llm_name, data_name=data_name)
            X_test, y_test = malcertain_data_processing(data_type="test", pn=pn,
                                                        llm_name=llm_name, data_name=data_name)

            clf = svm.SVC(C=1000, gamma=0.01, kernel="linear")

            clf.fit(X=X_train, y=y_train)

            malcertain_mask = clf.predict(X_test)

        malcertain_label = []
        for j, mask in enumerate(malcertain_mask):
            if mask == 1:
                malcertain_label.append((vanilla_pred[j] + 1) % 2)
            else:
                malcertain_label.append(vanilla_pred[j])
        malcertain_label = np.array(malcertain_label)

        TP = np.sum((malcertain_label == 1) & (gt_labels == 1))
        TN = np.sum((malcertain_label == 0) & (gt_labels == 0))
        FP = np.sum((malcertain_label == 1) & (gt_labels == 0))
        FN = np.sum((malcertain_label == 0) & (gt_labels == 1))

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