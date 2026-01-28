# -*- coding: utf-8 -*- 
# @Time : 2026/1/13 12:23 
# @Author : DirtyBoy 
# @File : utils.py
import os, json, joblib
import numpy as np


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

def pred_gap(end_list):
    min = np.min(end_list)
    max = np.max(end_list)
    return max - min


def pred_mead(end_list):
    mean = np.mean(end_list)
    return mean
