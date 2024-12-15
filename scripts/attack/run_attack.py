"""
+-------------------------------+--------+
| Attack Results                |        |
+-------------------------------+--------+
| Number of successful attacks: | 128    |
| Number of failed attacks:     | 29     |
| Number of skipped attacks:    | 43     |
| Original accuracy:            | 78.5%  |
| Accuracy under attack:        | 14.5%  |
| Attack success rate:          | 81.53% |
| Average perturbed word %:     | 15.52% |
| Average num. words per input: | 18.97  |
| Avg num queries:              | 32.68  |
+-------------------------------+--------+

These results come from using TextAttack to run the DeepWordBug attack on an LSTM trained on the 
Rotten Tomatoes Movie Review sentiment classification dataset, using 200 total examples. These 
results come from using TextAttack to run the DeepWordBug attack on an LSTM trained on the Rotten 
Tomatoes Movie Review sentiment classification dataset, using 200 total examples.

This attack was run on 200 examples. Out of those 200, the model initially predicted 43 of them 
incorrectly; this leads to an accuracy of 157/200 or 78.5%. TextAttack ran the adversarial attack 
process on the remaining 157 examples to try to find a valid adversarial perturbation for each one. 
Out of those 157, 29 attacks failed, leading to a success rate of 128/157 or 81.5%. Another way to 
articulate this is that the model correctly predicted and resisted attacks for 29 out of 200 total 
samples, leading to an accuracy under attack (or “after-attack accuracy”) of 29/200 or 14.5%.

TextAttack also logged some other helpful statistics for this attack. Among the 157 successful attacks, 
on average, the attack changed 15.5% of words to alter the prediction, and made 32.7 queries to find 
a successful perturbation. Across all 200 inputs, the average number of words was 18.97.
"""
import os
import json
import torch
import argparse
import textattack
import numpy as np
import torch.nn as nn
from rich import print
from datasets import load_dataset
from typing import List
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    default_data_collator, 
    DataCollatorWithPadding
)

task_to_keys = {
    "sst2": ("text", ), 
    "mr": ("text", ), 
    "mrpc": ("text_a", "text_b"), 
    "scitail": ("premise", "hypothesis"), 
}
task_to_name = {
    'sst2': 'DefenceLab/sst2', 
    'mr': 'DefenceLab/mr', 
    'mrpc': 'DefenceLab/mrpc', 
    'scitail': 'DefenceLab/scitail'
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the custom dataset to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default='default',
        help="The name of the custom dataset configuration.",
    )
    parser.add_argument(
        "--eval_dataset_split",
        type=str,
        default='test',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to save attack result.",
        required=True,
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Where do you want to store the pretrained models downloaded from huggingface.co",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        action="store_true",
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.",
    )
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--num_devices",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_workers_per_device",
        type=int,
        default=1,
    )
    parser.add_argument(
        '--attackers', 
        default=['textfooler', 'textbugger', 'pwws'], 
        nargs='+', 
        type=str,
        help=''
    )
    parser.add_argument(
        "--save_attack_results",
        action="store_true", 
    )
    args = parser.parse_args()
    return args


class HuggingFaceModelWrapper(textattack.models.wrappers.HuggingFaceModelWrapper):

    def __init__(self, model, tokenizer, max_length=64, pad_to_max_length=False, is_single_text=True):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length
        self.is_single_text = is_single_text

    def __call__(self, text_input_list):
        """Passes inputs to HuggingFace models as keyword arguments.

        (Regular PyTorch ``nn.Module`` models typically take inputs as
        positional arguments.)
        """
        if not self.is_single_text:
            text_input_list = ([line[0] for line in text_input_list], [line[1] for line in text_input_list])
        else:
            text_input_list = ([line for line in text_input_list], )

        padding = "max_length" if self.pad_to_max_length else False
        inputs_dict = self.tokenizer(
            *text_input_list, 
            padding=padding, 
            max_length=self.max_length, 
            truncation=True, 
            return_tensors="pt",
        )

        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs_dict)
            logits = outputs.logits

        return logits


def main():
    args = parse_args()

    dataset_name = args.dataset_name
    if args.dataset_name is not None:
        text_columns = task_to_keys[args.dataset_name]
        args.dataset_name = task_to_name[args.dataset_name]
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name, 
            args.dataset_config_name, 
            cache_dir=args.cache_dir, 
        )
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = (args.train_file if args.train_file is not None else args.validation_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

    # Labels
    if args.dataset_name is not None:
        is_regression = args.dataset_name == "stsb"
        if not is_regression:
            try:
                label_list = raw_datasets["train"].features["label"].names
            except:
                label_list = sorted(set(raw_datasets["train"]["label"]))
                # raw_datasets = raw_datasets.cast_column("isDif", Sequence(ClassLabel(names=label_names)))
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    if args.dataset_config_name == '':
        args.dataset_config_name = None

    if dataset_name in ['sst2', 'mr', 'mrpc', 'scitail']:
        label_map = None
    else:
        label_map = {str(l): i for i, l in enumerate(label_list)}
        print(label_map)
        
    dataset = textattack.datasets.HuggingFaceDataset(
        args.dataset_name, 
        subset=args.dataset_config_name, 
        split=args.eval_dataset_split, 
        dataset_columns=[text_columns, 'label'], 
        label_map=label_map, 
    )

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.dataset_name,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        trust_remote_code=args.trust_remote_code,
    )

    model_wrapper = HuggingFaceModelWrapper(
        model, 
        tokenizer=tokenizer, 
        max_length=args.max_length, 
        pad_to_max_length=args.pad_to_max_length, 
        is_single_text=(len(text_columns)==1)
    )

    report = {}
    report['model'] = args.model_name_or_path
    report['dataset'] = args.dataset_name

    attackers = args.attackers
    for attacker in attackers:
        attacker_name = attacker
        if attacker == 'textfooler':
            attack = textattack.attack_recipes.textfooler_jin_2019.TextFoolerJin2019.build(model_wrapper)
        elif attacker == 'textbugger':
            attack = textattack.attack_recipes.textbugger_li_2018.TextBuggerLi2018.build(model_wrapper)
        elif attacker == 'bae':
            attack = textattack.attack_recipes.bae_garg_2019.BAEGarg2019.build(model_wrapper)
        elif attacker == 'pwws':
            attack = textattack.attack_recipes.pwws_ren_2019.PWWSRen2019.build(model_wrapper)
        elif attacker == 'iga':
            attack = textattack.attack_recipes.iga_wang_2019.IGAWang2019.build(model_wrapper)
        attack_args = textattack.AttackArgs(
            num_examples=args.num_examples, 
            disable_stdout=True, 
            parallel=(True if args.num_devices>1 else False), 
            num_workers_per_device=args.num_workers_per_device, 
            random_seed=914
        )
        attacker = textattack.Attacker(attack, dataset, attack_args)
        attack_result = attacker.attack_dataset()
        # attack_report[attacker] = attack_result

        attack_success_stats = textattack.metrics.attack_metrics.AttackSuccessRate().calculate(attack_result)
        attack_query_stats = textattack.metrics.attack_metrics.AttackQueries().calculate(attack_result)
        attack_dict = {
            'ACC': str(attack_success_stats["original_accuracy"]) + "%", 
            'AUA': str(attack_success_stats["attack_accuracy_perc"]) + "%", 
            'ASR': str(attack_success_stats["attack_success_rate"]) + "%", 
            'NOQ': str(attack_query_stats["avg_num_queries"])
        }
        report[attacker_name] = attack_dict

    # report['attack'] = attack_report
    print(report)

    if args.save_attack_results:
        with open(os.path.join(args.output_dir, "attack_results.json"), "w") as f:
            json.dump(report, f, indent=4)


if __name__ == '__main__':
    main()