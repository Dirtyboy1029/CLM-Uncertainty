from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
)
from dataset.utils.datasetbase import DatasetBase
import os
from dataset.utils import dsets


class S2ClassDataset(DatasetBase):
    NAME = "bertds"  # used for loading the Source_Data
    task_info = {
        "sst2": {"num_labels": 2, "tokenize_keys": ("sentence", None)},
        "imdb": {"num_labels": 2, "tokenize_keys": ("text", None)},
        "ag_news": {"num_labels": 4, "tokenize_keys": ("text", None)},
        "mnli": {"num_labels": 3, "tokenize_keys": ("premise", "hypothesis")},
        "yelp": {"num_labels": 2, "tokenize_keys": ("text", None)},
        "wnli": {"num_labels": 2, "tokenize_keys": ("sentence1", "sentence2")},
        "mrpc": {"num_labels": 2, "tokenize_keys": ("sentence1", "sentence2")},
        "winogrande_s": {"num_labels": 2, "tokenize_keys": ("sentence1", "sentence2")},
        "rte": {"num_labels": 2, "tokenize_keys": ("sentence1", "sentence2")},
        "boolq": {"num_labels": 2, "tokenize_keys": ("passage", "question")},
        "wic": {"num_labels": 2, "tokenize_keys": ("sentence1", "sentence2")},
        "cola": {"num_labels": 2, "tokenize_keys": ("sentence", None)},
        "wiki_bonlyseen_20": {"num_labels": 2, "tokenize_keys": ("sentence", None)},
        "wiki_bonlyseen_40": {"num_labels": 2, "tokenize_keys": ("sentence", None)},
        "wiki_bonlyseen_60": {"num_labels": 2, "tokenize_keys": ("sentence", None)},
        "wiki_bonlyseen_80": {"num_labels": 2, "tokenize_keys": ("sentence", None)},
        'vulcure_vd_cwe_correct_800': {
            'num_labels': 2, "tokenize_keys": ("func", 'target')
        },
        'vulcure_vd_cwe_test_800': {
            'num_labels': 2, "tokenize_keys": ("func", 'target')
        },
        'vulcure_vd_cwe_train_800': {
            'num_labels': 2, "tokenize_keys": ("func", 'target')
        },
        'vulcure_vd_dataset_correct_800_basic_blocks': {
            'num_labels': 2, "tokenize_keys": ("func", 'target')
        },
        'vulcure_vd_dataset_test_800_basic_blocks': {
            'num_labels': 2, "tokenize_keys": ("func", 'target')
        },
        'vulcure_vd_dataset_correct_800': {
            'num_labels': 2, "tokenize_keys": ("func", 'target')
        },
        'vulcure_vd_dataset_test_800': {
            'num_labels': 2, "tokenize_keys": ("func", 'target')
        },
        'vulcure_vd_dataset_train_800': {
            'num_labels': 2, "tokenize_keys": ("func", 'target')
        },
        'vulcure_vd_random_correct_800': {
            'num_labels': 2, "tokenize_keys": ("func", 'target')
        },
        'vulcure_vd_random_test_800': {
            'num_labels': 2, "tokenize_keys": ("func", 'target')
        },
        'vulcure_vd_random_train_800': {
            'num_labels': 2, "tokenize_keys": ("func", 'target')
        },
        'vulcure_vd_dataset_correct_800_normalized': {
            'num_labels': 2, "tokenize_keys": ("func", 'target')
        },
        'vulcure_vd_dataset_test_800_normalized': {
            'num_labels': 2, "tokenize_keys": ("func", 'target')
        },
        'vulcure_vd_dataset_train_800_normalized': {
            'num_labels': 2, "tokenize_keys": ("func", 'target')
        },
        'vulcure_vd_random_correct_800_normalized': {
            'num_labels': 2, "tokenize_keys": ("func", 'target')
        },
        'vulcure_vd_random_test_800_normalized': {
            'num_labels': 2, "tokenize_keys": ("func", 'target')
        },
        'vulcure_vd_random_train_800_normalized': {
            'num_labels': 2, "tokenize_keys": ("func", 'target')
        },
        'code_clone_train': {
            'num_labels': 2, "tokenize_keys": ("func1", "func2", 'target')
        },
        'code_clone_test': {
            'num_labels': 2, "tokenize_keys": ("func1", "func2", 'target')
        },
        'code_clone_correct': {
            'num_labels': 2, "tokenize_keys": ("func1", "func2", 'target')
        },
        'code_type_binary_python_java_train': {
            'num_labels': 2, "tokenize_keys": ("code", 'target')
        },
        "code_ai_detection_train": {
            'num_labels': 2, "tokenize_keys": ("code", 'target')
        },
        "code_ai_detection_test": {
            'num_labels': 2, "tokenize_keys": ("code", 'target')
        },
        "code_ai_detection_correct": {
            'num_labels': 2, "tokenize_keys": ("code", 'target')
        },
        'code_type_binary_python_java_test': {
            'num_labels': 2, "tokenize_keys": ("code", 'target')
        },
        'code_type_binary_python_java_correct': {
            'num_labels': 2, "tokenize_keys": ("code", 'target')
        },
        'code_type_binary_php_go_train': {
            'num_labels': 2, "tokenize_keys": ("code", 'target')
        },
        'code_type_binary_php_go_test': {
            'num_labels': 2, "tokenize_keys": ("code", 'target')
        },
        'code_type_binary_php_go_correct': {
            'num_labels': 2, "tokenize_keys": ("code", 'target')
        },
        'code_type_multi_train': {
            'num_labels': 4, "tokenize_keys": ("code", 'target')
        },
        'code_type_multi_test': {
            'num_labels': 4, "tokenize_keys": ("code", 'target')
        },
        'code_type_multi_correct': {
            'num_labels': 4, "tokenize_keys": ("code", 'target')
        },
        'cwe_binary_train': {
            'num_labels': 2, "tokenize_keys": ("code", 'label')
        },
        'cwe_binary_test': {
            'num_labels': 2, "tokenize_keys": ("code", 'label')
        },
        'cwe_binary_correct': {
            'num_labels': 2, "tokenize_keys": ("code", 'label')
        },
        'cwe_c_binary_train': {
            'num_labels': 2, "tokenize_keys": ("code", 'target')
        },
        'cwe_c_binary_test': {
            'num_labels': 2, "tokenize_keys": ("code", 'target')
        },
        'cwe_c_binary_correct': {
            'num_labels': 2, "tokenize_keys": ("code", 'target')
        },
        'cwe_c_multi_train': {
            'num_labels': 4, "tokenize_keys": ("code", 'target')
        },
        'cwe_c_multi_test': {
            'num_labels': 4, "tokenize_keys": ("code", 'target')
        },
        'cwe_c_multi_correct': {
            'num_labels': 4, "tokenize_keys": ("code", 'target')
        },
        "diversevul_vd_500_train": {
            'num_labels': 2, "tokenize_keys": ("func", 'target')
        },
        "diversevul_vd_500_test": {
            'num_labels': 2, "tokenize_keys": ("func", 'target')
        },
        "diversevul_vd_500_correct": {
            'num_labels': 2, "tokenize_keys": ("func", 'target')
        },
    }

    def __init__(self, accelerator, args):
        super().__init__()

        self.args = args
        self.accelerator = accelerator
        # dataset_path = os.path.join(base_path_data(), 'wnli')
        # create_if_not_exists(dataset_path)
        if args.dataset in [
            "wnli",
            "rte",
            "mrpc",
            "cola",
            "sst2",
            "qnli",
            "qqp",
            "mnli",
            "ax",
        ]:
            self.raw_dataset = load_dataset("glue", args.dataset)
        elif args.dataset in ["cb", "wic", "boolq"]:
            self.raw_dataset = load_dataset("super_glue", args.dataset)
        elif "ARC" in args.dataset:
            self.raw_dataset = load_dataset("ai2_arc", args.dataset)
        elif "winogrande" in args.dataset:
            self.raw_dataset = load_dataset("winogrande", args.dataset)
        elif "vulcure" in args.dataset or "diversevul" in args.dataset or "code_ai_detection" in args.dataset:
            # dset_class: dsets.ClassificationDataset = getattr(dsets, 'primevul_dataset')
            # self.raw_dataset = dset_class(self.tokenizer, add_space=args.add_space, file_path=args.Source_Data,
            #                        max_seq_len=args.max_seq_len)
            self.raw_dataset = load_dataset(
                "json",
                data_files={
                    "train": os.path.join('./database', args.dataset + '.jsonl'),
                    "validation": os.path.join('./database', args.dataset + '.jsonl'),
                    "test": os.path.join('./database', args.dataset + '.jsonl')
                },
                split=None
            )
            print('load data from ', os.path.join('./database', args.dataset + '.jsonl'))
        elif "code_clone" in args.dataset:
            # dset_class: dsets.ClassificationDataset = getattr(dsets, 'primevul_dataset')
            # self.raw_dataset = dset_class(self.tokenizer, add_space=args.add_space, file_path=args.Source_Data,
            #                        max_seq_len=args.max_seq_len)
            self.raw_dataset = load_dataset(
                "json",
                data_files={
                    "train": os.path.join('./database', args.dataset + '.jsonl'),
                    "test": os.path.join('./database', args.dataset + '.jsonl'),
                    "validation": os.path.join('./database', args.dataset + '.jsonl'),
                },
                split=None
            )
            print('load data from ', os.path.join('./database', args.dataset + '.jsonl'))
        elif "cwe_binary" in args.dataset or "cwe_c_binary" in args.dataset or "cwe_c_multi" in args.dataset:
            # dset_class: dsets.ClassificationDataset = getattr(dsets, 'primevul_dataset')
            # self.raw_dataset = dset_class(self.tokenizer, add_space=args.add_space, file_path=args.Source_Data,
            #                        max_seq_len=args.max_seq_len)
            self.raw_dataset = load_dataset(
                "json",
                data_files={
                    "train": os.path.join('./database', args.dataset + '.jsonl'),
                    "test": os.path.join('./database', args.dataset + '.jsonl'),
                    "validation": os.path.join('./database', args.dataset + '.jsonl'),
                },
                split=None
            )
            print('load data from ', os.path.join('./database', args.dataset + '.jsonl'))
        elif "code_type" in args.dataset:
            # dset_class: dsets.ClassificationDataset = getattr(dsets, 'primevul_dataset')
            # self.raw_dataset = dset_class(self.tokenizer, add_space=args.add_space, file_path=args.Source_Data,
            #                        max_seq_len=args.max_seq_len)
            self.raw_dataset = load_dataset(
                "json",
                data_files={
                    "train": os.path.join('./database', args.dataset + '.jsonl'),
                    "test": os.path.join('./database', args.dataset + '.jsonl'),
                    "validation": os.path.join('./database', args.dataset + '.jsonl'),
                },
                split=None
            )
            print('load data from ', os.path.join('./database', args.dataset + '.jsonl'))
        else:
            self.raw_dataset = load_dataset(args.dataset)

        if "ARC" in args.dataset or "openbookqa" in args.dataset:
            self.deal_with_specical_datasets()

        # self.raw_dataset = load_dataset('glue', args.Source_Data)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"

        if accelerator is not None:
            accelerator.wait_for_everyone()

        # if accelerator.is_main_process:
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join('../my_llm', args.model), use_fast=not args.use_slow_tokenizer,
            trust_remote_code=True
        )

        self.num_labels = self.task_info[args.dataset]["num_labels"]
        self.tokenize_keys = self.task_info[args.dataset]["tokenize_keys"]

        self.num_samples = len(self.raw_dataset["train"])
        self.num_tests = len(self.raw_dataset["validation"])
        if accelerator is not None:
            if accelerator.is_local_main_process:
                print("=====================================")
                print(f"Loaded {args.dataset} Source_Data.")
                print(f"Number of samples: {self.num_samples}")
                print("=====================================")

        self.padding = "max_length" if args.pad_to_max_length else False

    def _tokenize(self, examples):
        if self.args.dataset == "boolq" and "llama" in self.args.model:
            texts = [
                f"Answer the question with only True or False: {question} Context: {passage}"
                for passage, question in zip(examples["passage"], examples["question"])
            ]
            result = self.tokenizer(
                texts,
                padding=self.padding,
                max_length=self.args.max_length,
                truncation=True,
            )
            result["labels"] = examples["label"]

        elif 'my_primevul' in self.args.dataset or 'vulcure' in self.args.dataset:
            if self.tokenizer.pad_token == None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            texts = [f"{func}" for func in examples['func']]
            result = self.tokenizer(
                texts,
                padding="longest",
                #max_length=self.args.max_seq_len,
                truncation=True,
            )
            map_dict = {"no": 0, "yes": 1}
            result["labels"] = [map_dict[label] for label in examples["target"]]
        elif 'diversevul_vd' in self.args.dataset:
            if self.tokenizer.pad_token == None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            texts = [f"{func}" for func in examples['func']]
            result = self.tokenizer(
                texts,
                padding=True,
                max_length=self.args.max_seq_len,
                truncation=True,
            )
            result["labels"] = [label for label in examples["target"]]
        elif 'code_ai_detection' in self.args.dataset:
            if self.tokenizer.pad_token == None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            texts = [f"{func}" for func in examples['code']]
            result = self.tokenizer(
                texts,
                padding="longest",
                truncation=True,
            )
            result["labels"] = [label for label in examples["target"]]
        elif 'code_clone' in self.args.dataset:
            if self.tokenizer.pad_token == None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            texts = []
            for func_a, func_b in zip(examples["func1"], examples["func2"]):
                text = f"function A:\n{func_a}\n" \
                       f"function B:\n{func_b}"
                texts.append(text)
            result = self.tokenizer(
                texts,
                padding=True,
                max_length=self.args.max_seq_len,
                truncation=True,
            )
            result["labels"] = [1 if label else 0 for label in examples["target"]]
        elif 'code_type' in self.args.dataset or 'cwe_c' in self.args.dataset:
            if self.tokenizer.pad_token == None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            texts = [f"{func}" for func in examples['code']]
            result = self.tokenizer(
                texts,
                padding=True,
                max_length=self.args.max_seq_len,
                truncation=True,
            )
            result["labels"] = [label for label in examples["target"]]
        elif 'cwe_binary' in self.args.dataset:
            if self.tokenizer.pad_token == None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            texts = [f"{func}" for func in examples['code']]
            result = self.tokenizer(
                texts,
                padding=True,
                max_length=self.args.max_seq_len,
                truncation=True,
            )
            result["labels"] = [label for label in examples["label"]]

        else:
            sentence1_key, sentence2_key = self.tokenize_keys
            if sentence2_key is not None:
                result = self.tokenizer(
                    examples[sentence1_key],
                    examples[sentence2_key],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.args.max_seq_len,
                )
                result["labels"] = examples["label"]
            else:
                result = self.tokenizer(
                    examples[sentence1_key],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.args.max_seq_len,
                )
                result["labels"] = examples["label"]

        return result

    def get_loaders(self):
        """
        Returns the train and test data loaders.
        """
        if self.accelerator is not None:
            with self.accelerator.main_process_first():
                processed_datasets = self.raw_dataset.map(
                    self._tokenize,
                    batched=True,
                    remove_columns=self.raw_dataset["train"].column_names,
                    desc="Running tokenizer on Source_Data",
                )
        else:
            processed_datasets = self.raw_dataset.map(
                self._tokenize,
                batched=True,
                remove_columns=self.raw_dataset["train"].column_names,
                desc="Running tokenizer on Source_Data",
            )
        # print('====train data====')
        train_dataset = processed_datasets["train"]
        # print('====validation data====')
        processed_dataset = processed_datasets[
            "validation_matched" if self.args.dataset == "mnli" else "validation"
        ]

        if self.args.testing_set == "test":
            ds = processed_dataset.train_test_split(
                test_size=0.5, seed=self.args.seed, shuffle=False
            )
            val_dataset, eval_dataset = ds["train"], ds["test"]
        elif self.args.testing_set == "train_val":
            ds = train_dataset.train_test_split(
                test_size=0.2, seed=self.args.seed, shuffle=False
            )
            train_dataset, val_dataset = ds["train"], ds["test"]
            eval_dataset = processed_dataset
        elif self.args.testing_set == "val":
            eval_dataset = processed_dataset

        # DataLoaders creation:
        if self.args.pad_to_max_length:
            # If padding was already done ot max length, we use the default data collator that will just convert everything
            # to tensors.
            data_collator = default_data_collator
        else:
            # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
            # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
            # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
            if self.accelerator is not None:
                data_collator = DataCollatorWithPadding(
                    self.tokenizer,
                    pad_to_multiple_of=8,
                )
            else:
                data_collator = DataCollatorWithPadding(
                    self.tokenizer, pad_to_multiple_of=None
                )

        self.train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=data_collator,
            batch_size=self.args.batch_size,
            num_workers=16,
        )
        self.test_dataloader = DataLoader(
            eval_dataset,
            shuffle=False,
            collate_fn=data_collator,
            batch_size=self.args.batch_size,
            num_workers=16,
        )

        if self.args.testing_set != "val":
            self.val_dataloader = DataLoader(
                val_dataset,
                shuffle=False,
                collate_fn=data_collator,
                batch_size=self.args.batch_size,
                num_workers=16,
            )

    def deal_with_specical_datasets(self):

        # Initialize counters
        count_3_choices_train = 0
        count_5_choices_train = 0
        count_3_choices_valid = 0
        count_5_choices_valid = 0

        # Count in the training Source_Data
        for example in self.raw_datasets["train"]:
            if len(example["choices"]["label"]) == 3:
                count_3_choices_train += 1
            elif len(example["choices"]["label"]) == 5:
                count_5_choices_train += 1

        # Count in the validation Source_Data
        for example in self.raw_datasets["validation"]:
            if len(example["choices"]["label"]) == 3:
                count_3_choices_valid += 1
            elif len(example["choices"]["label"]) == 5:
                count_5_choices_valid += 1

        # Get total counts
        total_train = len(self.raw_datasets["train"])
        total_valid = len(self.raw_datasets["validation"])

        # Print counts
        print("====counts train====")
        print(f"Total number of training examples: {total_train}")
        print(f"Number of training questions with 3 choices: {count_3_choices_train}")
        print(f"Number of training questions with 5 choices: {count_5_choices_train}")

        print("====counts valid====")
        print(f"Total number of validation examples: {total_valid}")
        print(f"Number of validation questions with 3 choices: {count_3_choices_valid}")
        print(f"Number of validation questions with 5 choices: {count_5_choices_valid}")

        # Filter the examples in the training Source_Data
        filtered_train = self.raw_datasets["train"].filter(
            lambda example: len(example["choices"]["label"]) == 4
        )

        # Filter the examples in the validation Source_Data
        filtered_valid = self.raw_datasets["validation"].filter(
            lambda example: len(example["choices"]["label"]) == 4
        )

        # Filter the examples in the test Source_Data
        filtered_test = self.raw_datasets["test"].filter(
            lambda example: len(example["choices"]["label"]) == 4
        )

        # Replace the original datasets with the filtered datasets
        self.raw_datasets["train"] = filtered_train
        self.raw_datasets["validation"] = filtered_valid
        self.raw_datasets["test"] = filtered_test

        print("====counts train====")
        print(f"Total number of training examples: {len(self.raw_datasets['train'])}")
        print("====counts valid====")
        print(
            f"Total number of validation examples: {len(self.raw_datasets['validation'])}"
        )

        def convert_choices_to_alpha(example):
            # Define a mapping from numerical to alphabetical labels
            mapping = {"1": "A", "2": "B", "3": "C", "4": "D"}

            # Convert the 'label' field in 'choices'
            example["choices"]["label"] = [
                mapping.get(label, label) for label in example["choices"]["label"]
            ]

            # Convert the 'answerKey' field
            example["answerKey"] = mapping.get(
                example["answerKey"], example["answerKey"]
            )

            example["choices"]["text"] = [
                text if text.endswith(".") else text + "."
                for text in example["choices"]["text"]
            ]
            example["choices"]["text"] = [
                text[0].upper() + text[1:] if text else text
                for text in example["choices"]["text"]
            ]

            return example

        # Apply the conversion to the training, validation, and test datasets
        self.raw_datasets["train"] = self.raw_datasets["train"].map(
            convert_choices_to_alpha
        )
        self.raw_datasets["validation"] = self.raw_datasets["validation"].map(
            convert_choices_to_alpha
        )
        self.raw_datasets["test"] = self.raw_datasets["test"].map(
            convert_choices_to_alpha
        )

        print("====train data====")
        from collections import Counter

        # Initialize counters for training and validation datasets
        counter_train = Counter()
        counter_valid = Counter()

        # Count in the training Source_Data
        for example in self.raw_datasets["train"]:
            counter_train.update(example["answerKey"])

        # Count in the validation Source_Data
        for example in self.raw_datasets["validation"]:
            counter_valid.update(example["answerKey"])

        # Print the results
        print("Training Source_Data counts:")
        for choice, count in counter_train.items():
            print(f"Choice {choice}: {count} occurrences")

        print("Validation Source_Data counts:")
        for choice, count in counter_valid.items():
            print(f"Choice {choice}: {count} occurrences")
