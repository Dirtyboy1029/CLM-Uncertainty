from transformers import AutoTokenizer
from dataset.utils import dsets
from dataset.utils.datasetbase import DatasetBase


class S2SDataset_Classification(DatasetBase):
    NAME = 'mcdataset'  # mutil-choice Source_Data
    task_info = {
        'winogrande_s': {
            'num_labels': 2,
        },
        'winogrande_m': {
            'num_labels': 2,
        },
        'boolq': {
            'num_labels': 2,
        },
        'obqa': {
            'num_labels': 4,
        },
        'ARC-Easy': {
            'num_labels': 5,
        },
        'ARC-Challenge': {
            'num_labels': 5,
        },
        'malicious_code': {
            'num_labels': 2,
        },
        'bookmia': {
            'num_labels': 2,
        },
        'bookmia_onlyseen_10': {
            'num_labels': 2,
        },
        'bookmia_onlyseen_30': {
            'num_labels': 2,
        },
        'bookmia_onlyseen_50': {
            'num_labels': 2,
        },
        'bookmia_onlyseen_70': {
            'num_labels': 2,
        },
        'bookmia_onlyseen_90': {
            'num_labels': 2,
        },
        'bookmia_bonlyseen_10': {
            'num_labels': 2,
        },
        'bookmia_bonlyseen_30': {
            'num_labels': 2,
        },
        'bookmia_bonlyseen_50': {
            'num_labels': 2,
        },
        'bookmia_bonlyseen_20': {
            'num_labels': 2,
        },
        'bookmia_bonlyseen_40': {
            'num_labels': 2,
        },
        'bookmia_random_10': {
            'num_labels': 2,
        },
        'bookmia_random_20': {
            'num_labels': 2,
        },
        'bookmia_random_30': {
            'num_labels': 2,
        },
        'bookmia_random_40': {
            'num_labels': 2,
        },
        'bookmia_random_50': {
            'num_labels': 2,
        },
        'bookmia_both10_20': {
            'num_labels': 2,
        },
        'bookmia_both10_40': {
            'num_labels': 2,
        },
        'bookmia_both10_60': {
            'num_labels': 2,
        },
        'bookmia_both10_80': {
            'num_labels': 2,
        },
        'bookmia_both50_20': {
            'num_labels': 2,
        },
        'bookmia_both50_40': {
            'num_labels': 2,
        },
        'bookmia_both50_60': {
            'num_labels': 2,
        },
        'bookmia_both50_80': {
            'num_labels': 2,
        },
        'bookmia_both90_20': {
            'num_labels': 2,
        },
        'bookmia_both90_40': {
            'num_labels': 2,
        },
        'bookmia_both90_60': {
            'num_labels': 2,
        },
        'bookmia_both90_80': {
            'num_labels': 2,
        },
        'bookmia_base10_test0': {
            'num_labels': 2,
        },
        'bookmia_base10_test100': {
            'num_labels': 2,
        },
        'bookmia_bonlyseen_0': {
            'num_labels': 2,
        },
        'bookmia_bonlyseen_20_arxiv': {
            'num_labels': 2,
        },
        'bookmia_bonlyseen_10_arxiv': {
            'num_labels': 2,
        },
        'bookmia_bonlyseen_30_arxiv': {
            'num_labels': 2,
        },
        'bookmia_bonlyseen_40_arxiv': {
            'num_labels': 2,
        },
        'bookmia_test_seen10': {
            'num_labels': 2,
        },
        'bookmia_test_seen20': {
            'num_labels': 2,
        },
        'bookmia_test_seen30': {
            'num_labels': 2,
        },
        'bookmia_test_seen40': {
            'num_labels': 2,
        },
        'bookmia_bonlyseen_20_arxiv2223': {
            'num_labels': 2,
        },
        'bookmia_bonlyseen_10_arxiv2223': {
            'num_labels': 2,
        },
        'bookmia_bonlyseen_30_arxiv2223': {
            'num_labels': 2,
        },
        'bookmia_bonlyseen_40_arxiv2223': {
            'num_labels': 2,
        },
        'bookmia_bonlyseen_10_wiki24': {
            'num_labels': 2,
        },
        'bookmia_bonlyseen_20_wiki24': {
            'num_labels': 2,
        },
        'bookmia_bonlyseen_30_wiki24': {
            'num_labels': 2,
        },
        'bookmia_bonlyseen_40_wiki24': {
            'num_labels': 2,
        },
        'bookmia_bonlyseen_40_arxiv2223mc': {
            'num_labels': 2,
        },
        'bookmia_bonlyseen_30_arxiv2223mc': {
            'num_labels': 2,
        },
        'bookmia_bonlyseen_20_arxiv2223mc': {
            'num_labels': 2,
        },
        'bookmia_bonlyseen_10_arxiv2223mc': {
            'num_labels': 2,
        },
        'proglang_c_python_classifiction': {
            'num_labels': 2,
        },
        'proglang_c_python_classifiction_flip': {
            'num_labels': 2,
        },
        'code_smell_train': {
            'num_labels': 2,
        },
        'code_smell_correct': {
            'num_labels': 2,
        },
        'code_smell_test': {
            'num_labels': 2,
        },
        'my_primevul_test_paired': {
            'num_labels': 2,
        },
        'my_primevul_valid': {
            'num_labels': 2,
        },
        'my_primevul_valid_paired': {
            'num_labels': 2,
        },
        'my_primevul_800_train': {
            'num_labels': 2,
        },
        'my_primevul_800_train_paired': {
            'num_labels': 2,
        },
        'my_primevul_800_test': {
            'num_labels': 2,
        },
        'my_primevul_800_test_paired': {
            'num_labels': 2,
        },
        'my_primevul_800_valid': {
            'num_labels': 2,
        },
        'my_primevul_1024_test_val': {
            'num_labels': 2,
        },
        'my_primevul_800_valid_paired': {
            'num_labels': 2
        },
        'code_clone_train': {
            'num_labels': 2
        },
        'code_clone_test': {
            'num_labels': 2
        },
        'code_clone_correct': {
            'num_labels': 2
        },
        'cwe_binary_train': {
            'num_labels': 2
        },
        'cwe_binary_test': {
            'num_labels': 2
        },
        'cwe_binary_correct': {
            'num_labels': 2
        },
        'vulcure_vd_dataset_train_800_tfblora': {
            'num_labels': 2
        },
        'vulcure_vd_random_train_800_tfblora': {
            'num_labels': 2
        },
        'vulcure_ood_random_unpaired_train_800': {
            'num_labels': 2
        },
        'vulcure_ood_random_unpaired_test_800': {
            'num_labels': 2
        },
        'vulcure_vd_random_train_800': {
            'num_labels': 2
        },
        'vulcure_vd_random_test_800': {
            'num_labels': 2
        },
        'vulcure_vd_random_correct_800': {
            'num_labels': 2
        },
        'vulcure_vd_random_patch_test_800': {
            'num_labels': 2
        },
        'vulcure_vd_random_patch_correct_800': {
            'num_labels': 2
        },
        'vulcure_vd_cwe_train_800': {
            'num_labels': 2
        },
        'vulcure_vd_cwe_test_800': {
            'num_labels': 2
        },
        'vulcure_vd_cwe_correct_800': {
            'num_labels': 2
        },
        'vulcure_vd_cwe_patch_test_800': {
            'num_labels': 2
        },
        'vulcure_vd_cwe_patch_correct_800': {
            'num_labels': 2
        },
        'vulcure_vd_project_train_800': {
            'num_labels': 2
        },
        'vulcure_vd_project_test_800': {
            'num_labels': 2
        },
        'vulcure_vd_project_correct_800': {
            'num_labels': 2
        },
        'vulcure_vd_project_patch_test_800': {
            'num_labels': 2
        },
        'vulcure_vd_project_patch_correct_800': {
            'num_labels': 2
        },
        'vulcure_vd_dataset_train_800': {
            'num_labels': 2
        },
        'vulcure_vd_dataset_test_800': {
            'num_labels': 2
        },
        'vulcure_vd_dataset_correct_800': {
            'num_labels': 2
        },
        'vulcure_vd_dataset_patch_test_800': {
            'num_labels': 2
        },
        'vulcure_vd_dataset_patch_correct_800': {
            'num_labels': 2
        },
        "code_type_binary_python_java_train": {
            'num_labels': 2
        },
        "code_type_binary_python_java_test": {
            'num_labels': 2
        },
        "code_type_binary_python_java_correct": {
            'num_labels': 2
        },

    }

    def __init__(self, accelerator, args):
        super().__init__()

        self.args = args
        self.accelerator = accelerator

        accelerator.wait_for_everyone()
        import os
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join("../my_llm", args.model),
                                                       trust_remote_code=True)
        self.tokenizer.padding_side = 'left'
        if args.dataset in ['boolq', 'winogrande_m', 'winogrande_s']:
            self.tokenizer.add_eos_token = True
        self.tokenizer.pad_token = self.tokenizer.bos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if args.dataset in self.task_info:
            self.num_labels = self.task_info[args.dataset]['num_labels']
        elif args.dataset.startswith('MMLU'):
            self.num_labels = 4
        else:
            raise NotImplementedError

        if args.dataset.startswith('winogrande'):
            dset_class: dsets.ClassificationDataset = getattr(dsets, 'winogrande')
            self.dset = dset_class(self.tokenizer, add_space=args.add_space, name=args.dataset,
                                   max_seq_len=args.max_seq_len)
        elif args.dataset.startswith('ARC'):
            dset_class: dsets.ClassificationDataset = getattr(dsets, 'arc')
            self.dset = dset_class(self.tokenizer, add_space=args.add_space, name=args.dataset,
                                   max_seq_len=args.max_seq_len)
        elif args.dataset.startswith('bookmia'):
            dset_class: dsets.ClassificationDataset = getattr(dsets, 'bookmia')
            self.dset = dset_class(self.tokenizer, add_space=args.add_space, file_path=args.dataset,
                                   max_seq_len=args.max_seq_len)
        elif args.dataset.startswith('code_type_binary'):
            dset_class: dsets.ClassificationDataset = getattr(dsets, 'prog_lang')
            self.dset = dset_class(self.tokenizer, add_space=args.add_space, file_path=args.dataset,
                                   max_seq_len=args.max_seq_len)
        elif args.dataset.startswith('code_clone'):
            dset_class: dsets.ClassificationDataset = getattr(dsets, 'code_clone_dataset')
            self.dset = dset_class(self.tokenizer, add_space=args.add_space, file_path=args.dataset,
                                   max_seq_len=args.max_seq_len)
        elif args.dataset.startswith('cwe_binary'):
            dset_class: dsets.ClassificationDataset = getattr(dsets, 'binary_cwe_dataset')
            self.dset = dset_class(self.tokenizer, add_space=args.add_space, file_path=args.dataset,
                                   max_seq_len=args.max_seq_len)
        elif args.dataset.startswith('vulcure_vd'):
            dset_class: dsets.ClassificationDataset = getattr(dsets, 'primevul_dataset')
            self.dset = dset_class(self.tokenizer, add_space=args.add_space, file_path=args.dataset,
                                   max_seq_len=args.max_seq_len)
        elif args.dataset.startswith('code_smell'):
            dset_class: dsets.ClassificationDataset = getattr(dsets, 'codesmell_dataset')
            self.dset = dset_class(self.tokenizer, add_space=args.add_space, file_path=args.dataset,
                                   max_seq_len=args.max_seq_len)
        elif args.dataset.startswith('my_multi_cve'):
            dset_class: dsets.ClassificationDataset = getattr(dsets, 'my_multi_cve_primevul_dataset')
            self.dset = dset_class(self.tokenizer, add_space=args.add_space, file_path=args.dataset,
                                   max_seq_len=args.max_seq_len)
        elif args.dataset.startswith('my_multi_cwe'):
            dset_class: dsets.ClassificationDataset = getattr(dsets, 'my_multi_cwe_primevul_dataset')
            self.dset = dset_class(self.tokenizer, add_space=args.add_space, file_path=args.dataset,
                                   max_seq_len=args.max_seq_len)
        elif args.dataset.startswith('MMLU'):
            dset_class: dsets.ClassificationDataset = getattr(dsets, 'mmlu')
            self.dset = dset_class(self.tokenizer, add_space=args.add_space, name=args.dataset[5:],
                                   max_seq_len=args.max_seq_len)
        else:
            dset_class: dsets.ClassificationDataset = getattr(dsets, args.dataset)
            self.dset = dset_class(self.tokenizer, add_space=args.add_space, max_seq_len=args.max_seq_len)

        if accelerator.is_local_main_process:
            print("=====================================")
            print(f"Loaded {args.dataset} dataset.")
            print("=====================================")

    def get_loaders(self):
        """
        Returns the train and test data loaders.
        """

        self.target_ids = self.dset.target_ids
        if self.args.dataset.startswith('MMLU'):
            self.train_dataloader = self.dset.loader(
                is_s2s=self.args.is_s2s,  # sequence to sequence model?
                batch_size=self.args.batch_size,  # training batch size
                split="test",  # training split name in dset
                subset_size=-1,  # train on subset? (-1 = no subset)
            )
            total_data_count = 0
            for batch in self.train_dataloader:
                total_data_count += batch[1].size(0)
            self.num_samples = total_data_count
            self.test_dataloader = self.dset.loader(
                is_s2s=self.args.is_s2s,  # sequence to sequence model?
                batch_size=self.args.batch_size,  # training batch size
                split="test",  # training split name in dset
                subset_size=-1,  # train on subset? (-1 = no subset)
            )
            return

        if self.args.dataset.startswith('malicious'):
            self.train_dataloader = self.dset.loader(
                is_s2s=self.args.is_s2s,  # sequence to sequence model?
                batch_size=self.args.batch_size,  # training batch size
                split="train",  # training split name in dset
                subset_size=-1,  # train on subset? (-1 = no subset)
            )
            total_data_count = 0
            for batch in self.train_dataloader:
                total_data_count += batch[1].size(0)
            self.num_samples = total_data_count

            self.test_dataloader = self.dset.loader(
                is_s2s=self.args.is_s2s,  # sequence to sequence model?
                batch_size=self.args.batch_size,  # training batch size
                split="test",  # training split name in dset
                subset_size=-1,  # train on subset? (-1 = no subset)
            )
            return
        elif self.args.dataset.startswith('bookmia') or self.args.dataset.startswith(
                'my_primevul') or self.args.dataset.startswith('my_multi_c')  or self.args.dataset.startswith('cwe_binary')\
                or self.args.dataset.startswith('vulcure') or self.args.dataset.startswith('code_smell')\
                or self.args.dataset.startswith('code_clone'):
            self.train_dataloader = self.dset.loader(
                is_s2s=self.args.is_s2s,  # sequence to sequence model?
                batch_size=self.args.batch_size,  # training batch size
                split=None,  # training split name in dset
                subset_size=-1,  # train on subset? (-1 = no subset)
            )
            total_data_count = 0
            for batch in self.train_dataloader:
                total_data_count += batch[1].size(0)
                # total_data_count += len(batch["input_ids"])

            self.num_samples = total_data_count
            self.test_dataloader = self.dset.loader(
                is_s2s=self.args.is_s2s,  # sequence to sequence model?
                batch_size=self.args.batch_size,  # training batch size
                split=None,  # training split name in dset
                subset_size=-1,  # train on subset? (-1 = no subset)
            )
            return
        elif self.args.dataset.startswith('code_type_binary'):
            self.train_dataloader = self.dset.loader(
                is_s2s=self.args.is_s2s,  # sequence to sequence model?
                batch_size=self.args.batch_size,  # training batch size
                split=None,  # training split name in dset
                subset_size=-1,  # train on subset? (-1 = no subset)
            )
            total_data_count = 0
            for batch in self.train_dataloader:
                total_data_count += batch[1].size(0)
            self.num_samples = total_data_count

            self.test_dataloader = self.dset.loader(
                is_s2s=self.args.is_s2s,  # sequence to sequence model?
                batch_size=self.args.batch_size,  # training batch size
                split=None,  # training split name in dset
                subset_size=-1,  # train on subset? (-1 = no subset)
            )
            return
        else:

            self.train_dataloader = self.dset.loader(
                is_s2s=self.args.is_s2s,  # sequence to sequence model?
                batch_size=self.args.batch_size,  # training batch size
                split="train",  # training split name in dset
                subset_size=-1,  # train on subset? (-1 = no subset)
            )
            total_data_count = 0
            for batch in self.train_dataloader:
                total_data_count += batch[1].size(0)
            self.num_samples = total_data_count

            self.test_dataloader = self.dset.loader(
                is_s2s=self.args.is_s2s,  # sequence to sequence model?
                batch_size=self.args.batch_size,  # training batch size
                split="validation",  # training split name in dset
                subset_size=-1,  # train on subset? (-1 = no subset)
            )

            if self.args.testing_set != 'val':
                raise NotImplementedError('Only validation set is supported for now.')
