"""
Fine-tuning a 🤗 Transformers model on token classification tasks (NER, POS, CHUNKS) relying on the accelerate library
without using a Trainer.
"""
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
import datasets
import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import ClassLabel, load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import CONFIG_MAPPING, MODEL_MAPPING, AutoConfig, AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, PretrainedConfig, SchedulerType, default_data_collator, get_scheduler
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
check_min_version('4.36.0.dev0')
logger = get_logger(__name__)
require_version('datasets>=1.8.0', 'To fix: pip install -r examples/pytorch/token-classification/requirements.txt')
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple((conf.model_type for conf in MODEL_CONFIG_CLASSES))

def parse_args():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description='Finetune a transformers model on a text classification task (NER) with accelerate library')
    parser.add_argument('--dataset_name', type=str, default=None, help='The name of the dataset to use (via the datasets library).')
    parser.add_argument('--dataset_config_name', type=str, default=None, help='The configuration name of the dataset to use (via the datasets library).')
    parser.add_argument('--train_file', type=str, default=None, help='A csv or a json file containing the training data.')
    parser.add_argument('--validation_file', type=str, default=None, help='A csv or a json file containing the validation data.')
    parser.add_argument('--text_column_name', type=str, default=None, help='The column name of text to input in the file (a csv or JSON file).')
    parser.add_argument('--label_column_name', type=str, default=None, help='The column name of label to input in the file (a csv or JSON file).')
    parser.add_argument('--max_length', type=int, default=128, help='The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded if `--pad_to_max_length` is passed.')
    parser.add_argument('--pad_to_max_length', action='store_true', help='If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.')
    parser.add_argument('--model_name_or_path', type=str, help='Path to pretrained model or model identifier from huggingface.co/models.', required=False)
    parser.add_argument('--config_name', type=str, default=None, help='Pretrained config name or path if not the same as model_name')
    parser.add_argument('--tokenizer_name', type=str, default=None, help='Pretrained tokenizer name or path if not the same as model_name')
    parser.add_argument('--per_device_train_batch_size', type=int, default=8, help='Batch size (per device) for the training dataloader.')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8, help='Batch size (per device) for the evaluation dataloader.')
    parser.add_argument('--learning_rate', type=float, default=5e-05, help='Initial learning rate (after the potential warmup period) to use.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay to use.')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Total number of training epochs to perform.')
    parser.add_argument('--max_train_steps', type=int, default=None, help='Total number of training steps to perform. If provided, overrides num_train_epochs.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--lr_scheduler_type', type=SchedulerType, default='linear', help='The scheduler type to use.', choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'])
    parser.add_argument('--num_warmup_steps', type=int, default=0, help='Number of steps for the warmup in the lr scheduler.')
    parser.add_argument('--output_dir', type=str, default=None, help='Where to store the final model.')
    parser.add_argument('--seed', type=int, default=None, help='A seed for reproducible training.')
    parser.add_argument('--model_type', type=str, default=None, help='Model type to use if training from scratch.', choices=MODEL_TYPES)
    parser.add_argument('--label_all_tokens', action='store_true', help='Setting labels of all special tokens to -100 and thus PyTorch will ignore them.')
    parser.add_argument('--return_entity_level_metrics', action='store_true', help='Indication whether entity level metrics are to be returner.')
    parser.add_argument('--task_name', type=str, default='ner', choices=['ner', 'pos', 'chunk'], help='The name of the task.')
    parser.add_argument('--debug', action='store_true', help='Activate debug mode and run training only with a subset of data.')
    parser.add_argument('--push_to_hub', action='store_true', help='Whether or not to push the model to the Hub.')
    parser.add_argument('--hub_model_id', type=str, help='The name of the repository to keep in sync with the local `output_dir`.')
    parser.add_argument('--hub_token', type=str, help='The token to use to push to the Model Hub.')
    parser.add_argument('--trust_remote_code', type=bool, default=False, help='Whether or not to allow for custom models defined on the Hub in their own modeling files. This optionshould only be set to `True` for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.')
    parser.add_argument('--checkpointing_steps', type=str, default=None, help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='If the training should continue from a checkpoint folder.')
    parser.add_argument('--with_tracking', action='store_true', help='Whether to enable experiment trackers for logging.')
    parser.add_argument('--report_to', type=str, default='all', help='The integration to report the results and logs to. Supported platforms are `"tensorboard"`, `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. Only applicable when `--with_tracking` is passed.')
    parser.add_argument('--ignore_mismatched_sizes', action='store_true', help='Whether or not to enable to load a pretrained model whose head dimensions are different.')
    args = parser.parse_args()
    if args.task_name is None and args.train_file is None and (args.validation_file is None):
        raise ValueError('Need either a task name or a training/validation file.')
    else:
        if args.train_file is not None:
            extension = args.train_file.split('.')[-1]
            assert extension in ['csv', 'json'], '`train_file` should be a csv or a json file.'
        if args.validation_file is not None:
            extension = args.validation_file.split('.')[-1]
            assert extension in ['csv', 'json'], '`validation_file` should be a csv or a json file.'
    if args.push_to_hub:
        assert args.output_dir is not None, 'Need an `output_dir` to create a repo when `--push_to_hub` is passed.'
    return args

def main():
    if False:
        print('Hello World!')
    args = parse_args()
    send_example_telemetry('run_ner_no_trainer', args)
    accelerator = Accelerator(log_with=args.report_to, project_dir=args.output_dir) if args.with_tracking else Accelerator()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        if args.push_to_hub:
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            repo_id = create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id
            repo = Repository(args.output_dir, clone_from=repo_id, token=args.hub_token)
            with open(os.path.join(args.output_dir, '.gitignore'), 'w+') as gitignore:
                if 'step_*' not in gitignore:
                    gitignore.write('step_*\n')
                if 'epoch_*' not in gitignore:
                    gitignore.write('epoch_*\n')
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if args.dataset_name is not None:
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files['train'] = args.train_file
        if args.validation_file is not None:
            data_files['validation'] = args.validation_file
        extension = args.train_file.split('.')[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))
    if raw_datasets['train'] is not None:
        column_names = raw_datasets['train'].column_names
        features = raw_datasets['train'].features
    else:
        column_names = raw_datasets['validation'].column_names
        features = raw_datasets['validation'].features
    if args.text_column_name is not None:
        text_column_name = args.text_column_name
    elif 'tokens' in column_names:
        text_column_name = 'tokens'
    else:
        text_column_name = column_names[0]
    if args.label_column_name is not None:
        label_column_name = args.label_column_name
    elif f'{args.task_name}_tags' in column_names:
        label_column_name = f'{args.task_name}_tags'
    else:
        label_column_name = column_names[1]

    def get_label_list(labels):
        if False:
            while True:
                i = 10
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list
    labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
    if labels_are_int:
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(raw_datasets['train'][label_column_name])
        label_to_id = {l: i for (i, l) in enumerate(label_list)}
    num_labels = len(label_list)
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, num_labels=num_labels, trust_remote_code=args.trust_remote_code)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, trust_remote_code=args.trust_remote_code)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning('You are instantiating a new config instance from scratch.')
    tokenizer_name_or_path = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    if not tokenizer_name_or_path:
        raise ValueError('You are instantiating a new tokenizer from scratch. This is not supported by this script. You can do it from another script, save it, and load it from here, using --tokenizer_name.')
    if config.model_type in {'bloom', 'gpt2', 'roberta'}:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, add_prefix_space=True, trust_remote_code=args.trust_remote_code)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True, trust_remote_code=args.trust_remote_code)
    if args.model_name_or_path:
        model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config, ignore_mismatched_sizes=args.ignore_mismatched_sizes, trust_remote_code=args.trust_remote_code)
    else:
        logger.info('Training new model from scratch')
        model = AutoModelForTokenClassification.from_config(config, trust_remote_code=args.trust_remote_code)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id:
        if sorted(model.config.label2id.keys()) == sorted(label_list):
            if labels_are_int:
                label_to_id = {i: int(model.config.label2id[l]) for (i, l) in enumerate(label_list)}
                label_list = [model.config.id2label[i] for i in range(num_labels)]
            else:
                label_list = [model.config.id2label[i] for i in range(num_labels)]
                label_to_id = {l: i for (i, l) in enumerate(label_list)}
        else:
            logger.warning("Your model seems to have been trained with labels, but they don't match the dataset: ", f'model labels: {sorted(model.config.label2id.keys())}, dataset labels: {sorted(label_list)}.\nIgnoring the model labels as a result.')
    model.config.label2id = {l: i for (i, l) in enumerate(label_list)}
    model.config.id2label = dict(enumerate(label_list))
    b_to_i_label = []
    for (idx, label) in enumerate(label_list):
        if label.startswith('B-') and label.replace('B-', 'I-') in label_list:
            b_to_i_label.append(label_list.index(label.replace('B-', 'I-')))
        else:
            b_to_i_label.append(idx)
    padding = 'max_length' if args.pad_to_max_length else False

    def tokenize_and_align_labels(examples):
        if False:
            for i in range(10):
                print('nop')
        tokenized_inputs = tokenizer(examples[text_column_name], max_length=args.max_length, padding=padding, truncation=True, is_split_into_words=True)
        labels = []
        for (i, label) in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                elif args.label_all_tokens:
                    label_ids.append(b_to_i_label[label_to_id[label[word_idx]]])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs['labels'] = labels
        return tokenized_inputs
    with accelerator.main_process_first():
        processed_raw_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True, remove_columns=raw_datasets['train'].column_names, desc='Running tokenizer on dataset')
    train_dataset = processed_raw_datasets['train']
    eval_dataset = processed_raw_datasets['validation']
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f'Sample {index} of the training set: {train_dataset[index]}.')
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if accelerator.use_fp16 else None)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{'params': [p for (n, p) in model.named_parameters() if not any((nd in n for nd in no_decay))], 'weight_decay': args.weight_decay}, {'params': [p for (n, p) in model.named_parameters() if any((nd in n for nd in no_decay))], 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    device = accelerator.device
    model.to(device)
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=args.max_train_steps)
    (model, optimizer, train_dataloader, eval_dataloader, lr_scheduler) = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)
    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config['lr_scheduler_type'] = experiment_config['lr_scheduler_type'].value
        accelerator.init_trackers('ner_no_trainer', experiment_config)
    metric = evaluate.load('seqeval')

    def get_labels(predictions, references):
        if False:
            return 10
        if device.type == 'cpu':
            y_pred = predictions.detach().clone().numpy()
            y_true = references.detach().clone().numpy()
        else:
            y_pred = predictions.detach().cpu().clone().numpy()
            y_true = references.detach().cpu().clone().numpy()
        true_predictions = [[label_list[p] for (p, l) in zip(pred, gold_label) if l != -100] for (pred, gold_label) in zip(y_pred, y_true)]
        true_labels = [[label_list[l] for (p, l) in zip(pred, gold_label) if l != -100] for (pred, gold_label) in zip(y_pred, y_true)]
        return (true_predictions, true_labels)

    def compute_metrics():
        if False:
            return 10
        results = metric.compute()
        if args.return_entity_level_metrics:
            final_results = {}
            for (key, value) in results.items():
                if isinstance(value, dict):
                    for (n, v) in value.items():
                        final_results[f'{key}_{n}'] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {'precision': results['overall_precision'], 'recall': results['overall_recall'], 'f1': results['overall_f1'], 'accuracy': results['overall_accuracy']}
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(train_dataset)}')
    logger.info(f'  Num Epochs = {args.num_train_epochs}')
    logger.info(f'  Instantaneous batch size per device = {args.per_device_train_batch_size}')
    logger.info(f'  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}')
    logger.info(f'  Gradient Accumulation steps = {args.gradient_accumulation_steps}')
    logger.info(f'  Total optimization steps = {args.max_train_steps}')
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != '':
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)
        accelerator.print(f'Resumed from checkpoint: {checkpoint_path}')
        accelerator.load_state(checkpoint_path)
        training_difference = os.path.splitext(path)[0]
        if 'epoch' in training_difference:
            starting_epoch = int(training_difference.replace('epoch_', '')) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = int(training_difference.replace('step_', '')) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)
    progress_bar.update(completed_steps)
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if args.with_tracking:
            total_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and (resume_step is not None):
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for (step, batch) in enumerate(active_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            if args.with_tracking:
                total_loss += loss.detach().float()
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f'step_{completed_steps}'
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= args.max_train_steps:
                break
        model.eval()
        samples_seen = 0
        for (step, batch) in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            labels = batch['labels']
            if not args.pad_to_max_length:
                predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
                labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            (predictions_gathered, labels_gathered) = accelerator.gather((predictions, labels))
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions_gathered = predictions_gathered[:len(eval_dataloader.dataset) - samples_seen]
                    labels_gathered = labels_gathered[:len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += labels_gathered.shape[0]
            (preds, refs) = get_labels(predictions_gathered, labels_gathered)
            metric.add_batch(predictions=preds, references=refs)
        eval_metric = compute_metrics()
        accelerator.print(f'epoch {epoch}:', eval_metric)
        if args.with_tracking:
            accelerator.log({'seqeval': eval_metric, 'train_loss': total_loss.item() / len(train_dataloader), 'epoch': epoch, 'step': completed_steps}, step=completed_steps)
        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(commit_message=f'Training in progress epoch {epoch}', blocking=False, auto_lfs_prune=True)
        if args.checkpointing_steps == 'epoch':
            output_dir = f'epoch_{epoch}'
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
    if args.with_tracking:
        accelerator.end_training()
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message='End of training', auto_lfs_prune=True)
            all_results = {f'eval_{k}': v for (k, v) in eval_metric.items()}
            if args.with_tracking:
                all_results.update({'train_loss': total_loss.item() / len(train_dataloader)})
            with open(os.path.join(args.output_dir, 'all_results.json'), 'w') as f:
                for (key, value) in all_results.items():
                    if isinstance(value, np.float64):
                        all_results[key] = float(value)
                    elif isinstance(value, np.int64):
                        all_results[key] = int(value)
                json.dump(all_results, f)
if __name__ == '__main__':
    main()