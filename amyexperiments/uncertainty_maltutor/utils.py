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

def nll(p, label, eps=1e-10, base=2):
    p = np.array(p)
    q = np.full(len(p), label)
    nll = -(q * np.log(p + eps) + (1. - q) * np.log(1. - p + eps))
    if base is not None:
        nll = np.clip(nll / np.log(base), a_min=0., a_max=1000)
    return np.mean(nll)

def prob_label_kld(p, label, number=10, w=None, base=2, eps=1e-10):
    if number <= 1:
        return np.zeros_like(p)

    p_arr = np.asarray(p).reshape((-1, number))
    q_arr = np.full(number, label)
    if w is None:
        w_arr = np.ones(shape=(number, 1), dtype=float) / number
    else:
        w_arr = np.asarray(w).reshape((number, 1))

    kld_elem = p_arr * np.log((p_arr + eps) / (q_arr + eps)) + (1. - p_arr) * np.log(
        (1. - p_arr + eps) / (1. - q_arr + eps))
    if base is not None:
        kld_elem = kld_elem / np.log(base)
    kld = np.matmul(kld_elem, w_arr)
    return (kld / number)[0][0]

def Wasserstein_distance(p, label):
    from scipy.stats import wasserstein_distance
    p = np.array(p)
    q = np.full(len(p), label)
    emd = wasserstein_distance(p, q)

    return emd


def Euclidean_distance(p, label):
    p = np.array(p)
    q = np.full(len(p), label)

    v1 = np.array(p)
    v2 = np.array(q)

    distance = np.linalg.norm(v1 - v2)

    return distance


def Manhattan_distance(p, label):
    p = np.array(p)
    q = np.full(len(p), label)
    v1 = np.array(p)
    v2 = np.array(q)
    distance = np.sum(np.abs(v1 - v2)) / len(p)

    return distance


def Chebyshev_distance(p, label):
    p = np.array(p)
    q = np.full(len(p), label)
    v1 = np.array(p)
    v2 = np.array(q)
    distance = np.max(np.abs(v1 - v2))

    return distance
