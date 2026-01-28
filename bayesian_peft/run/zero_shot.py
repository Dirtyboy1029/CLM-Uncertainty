import numpy  # needed (don't change it)
import importlib
import os
import socket
import sys
from ipdb import iex
import numpy as np
from tqdm import tqdm

project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
sys.path.append(project_path + '/datasets')
sys.path.append(project_path + '/backbones')
sys.path.append(project_path + '/models')
sys.path.append(project_path + '/main')

from transformers import AutoTokenizer
import datetime
import uuid
from argparse import ArgumentParser

import setproctitle
import torch
from utils.args import add_management_args, add_experiment_args
from utils import create_if_not_exists
# from utils.continual_training import train as ctrain
from run.ood_eval import ood_eval
from run.laplace_train import laplace_train_old
from run.laplace_ood_eval import laplace_ood_eval
from run.laplace_ood_vis import laplace_ood_vis
from run import *

from accelerate.utils import set_seed
from accelerate import Accelerator

try:
    import wandb
except ImportError:
    wandb = None

import numpy as np


def dump_joblib(data, path):
    try:
        import joblib
        with open(path, 'wb') as wr:
            joblib.dump(data, wr)
        return
    except IOError:
        raise IOError("Dump data failed.")


def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib  # pyright: ignore
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def parse_args():
    parser = ArgumentParser(description='Bayesian LoRA', allow_abbrev=False)
    add_management_args(parser)
    add_experiment_args(parser)
    args = parser.parse_known_args()[0]

    # add model-specific arguments
    mod = importlib.import_module('modelwrappers.' + args.modelwrapper)
    get_parser = getattr(mod, 'get_parser')
    parser = get_parser()  # the real parsing happens.
    args = parser.parse_args()

    # set random seed
    if args.seed is not None:
        set_seed(args.seed)
    return args


# @iex
def main(args=None):
    from tqdm import tqdm

    lecun_fix()
    if args is None:
        args = parse_args()

    os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    os.putenv("NPY_MKL_FORCE_INTEL", "1")

    # Add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    accelerator = Accelerator()
    ood_ori_dataset = None
    if args.ood_ori_dataset is not None:
        dataset = args.dataset
        args.dataset = args.ood_ori_dataset
        ood_ori_dataset = get_dataset(args.dataset_type, accelerator, args)
        ood_ori_dataset.get_loaders()
        args.ood_ori_outdim = ood_ori_dataset.num_labels  # should be careful to use in evaluate_all
        args.ood_ori_num_samples = ood_ori_dataset.num_samples
        args.dataset = dataset
    args.outdim = ood_ori_dataset.num_labels

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.load_model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.load_model_path,
        device_map="auto"
    )

    print("Load model from ", args.load_model_path)

    model.eval()

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    print('main.py print \
            target_ids:', ood_ori_dataset.target_ids, )

    args.num_samples = ood_ori_dataset.num_samples
    if args.ood_ori_dataset is not None:

        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        print("CUDA_VISIBLE_DEVICES:", cuda_visible)
        device = torch.device('cuda:0')
        model.to(device)
        model.eval()
        target_ids = ood_ori_dataset.target_ids
        if 'multi' in args.ood_ori_dataset:
            output_dim = 4
        else:
            output_dim = 2
        ending = np.empty((0, output_dim))

        save_path = f'output/{args.model_type}/zero_shot/{args.ood_ori_dataset}/epoch{args.n_epochs}/{args.model}-tmp_index.data'

        if not os.path.isdir(os.path.dirname(save_path)):
            print(os.path.dirname(save_path))
            os.makedirs(os.path.dirname(save_path))
        print('save to ', os.path.dirname(save_path))

        device = next(model.parameters()).device

        for step, batch in enumerate(tqdm(ood_ori_dataset.test_dataloader, desc="Predict Progress")):
            inputs, labels, extra = batch

            inputs = inputs.to(device)

            labels = labels.to(device)
            extra = extra.to(device)

            with torch.no_grad():
                outputs = model(**inputs).logits

            output = outputs.detach().cpu().numpy()
            print(output.shape)
            output = np.squeeze(output[:, -1, target_ids])
            if args.batch_size == 1:
                output = output.reshape((1, args.outdim))
            print(output)
            ending = np.concatenate([ending, output], axis=0)
        dump_joblib(ending, save_path.replace('tmp_index', str(args.bayes_eval_index)))


if __name__ == '__main__':
    main()
