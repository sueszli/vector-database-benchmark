"""
Fine-tuning a 🤗 Transformers model on summarization.
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
import nltk
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from filelock import FileLock
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import CONFIG_MAPPING, MODEL_MAPPING, AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, SchedulerType, get_scheduler
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version
check_min_version('4.36.0.dev0')
logger = get_logger(__name__)
require_version('datasets>=1.8.0', 'To fix: pip install -r examples/pytorch/summarization/requirements.txt')
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple((conf.model_type for conf in MODEL_CONFIG_CLASSES))
try:
    nltk.data.find('tokenizers/punkt')
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError('Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files')
    with FileLock('.lock') as lock:
        nltk.download('punkt', quiet=True)
summarization_name_mapping = {'amazon_reviews_multi': ('review_body', 'review_title'), 'big_patent': ('description', 'abstract'), 'cnn_dailymail': ('article', 'highlights'), 'orange_sum': ('text', 'summary'), 'pn_summary': ('article', 'summary'), 'psc': ('extract_text', 'summary_text'), 'samsum': ('dialogue', 'summary'), 'thaisum': ('body', 'summary'), 'xglue': ('news_body', 'news_title'), 'xsum': ('document', 'summary'), 'wiki_summary': ('article', 'highlights')}

def parse_args():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser(description='Finetune a transformers model on a summarization task')
    parser.add_argument('--dataset_name', type=str, default=None, help='The name of the dataset to use (via the datasets library).')
    parser.add_argument('--dataset_config_name', type=str, default=None, help='The configuration name of the dataset to use (via the datasets library).')
    parser.add_argument('--train_file', type=str, default=None, help='A csv or a json file containing the training data.')
    parser.add_argument('--validation_file', type=str, default=None, help='A csv or a json file containing the validation data.')
    parser.add_argument('--ignore_pad_token_for_loss', type=bool, default=True, help='Whether to ignore the tokens corresponding to padded labels in the loss computation or not.')
    parser.add_argument('--max_source_length', type=int, default=1024, help='The maximum total input sequence length after tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.')
    parser.add_argument('--source_prefix', type=str, default=None, help='A prefix to add before every source text (useful for T5 models).')
    parser.add_argument('--preprocessing_num_workers', type=int, default=None, help='The number of processes to use for the preprocessing.')
    parser.add_argument('--overwrite_cache', action='store_true', help='Overwrite the cached training and evaluation sets')
    parser.add_argument('--max_target_length', type=int, default=128, help='The maximum total sequence length for target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. during ``evaluate`` and ``predict``.')
    parser.add_argument('--val_max_target_length', type=int, default=None, help='The maximum total sequence length for validation target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` param of ``model.generate``, which is used during ``evaluate`` and ``predict``.')
    parser.add_argument('--num_beams', type=int, default=None, help='Number of beams to use for evaluation. This argument will be passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.')
    parser.add_argument('--pad_to_max_length', action='store_true', help='If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.')
    parser.add_argument('--model_name_or_path', type=str, help='Path to pretrained model or model identifier from huggingface.co/models.', required=False)
    parser.add_argument('--config_name', type=str, default=None, help='Pretrained config name or path if not the same as model_name')
    parser.add_argument('--tokenizer_name', type=str, default=None, help='Pretrained tokenizer name or path if not the same as model_name')
    parser.add_argument('--text_column', type=str, default=None, help='The name of the column in the datasets containing the full texts (for summarization).')
    parser.add_argument('--summary_column', type=str, default=None, help='The name of the column in the datasets containing the summaries (for summarization).')
    parser.add_argument('--use_slow_tokenizer', action='store_true', help='If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).')
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
    parser.add_argument('--push_to_hub', action='store_true', help='Whether or not to push the model to the Hub.')
    parser.add_argument('--hub_model_id', type=str, help='The name of the repository to keep in sync with the local `output_dir`.')
    parser.add_argument('--hub_token', type=str, help='The token to use to push to the Model Hub.')
    parser.add_argument('--trust_remote_code', type=bool, default=False, help='Whether or not to allow for custom models defined on the Hub in their own modeling files. This optionshould only be set to `True` for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.')
    parser.add_argument('--checkpointing_steps', type=str, default=None, help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='If the training should continue from a checkpoint folder.')
    parser.add_argument('--with_tracking', action='store_true', help='Whether to enable experiment trackers for logging.')
    parser.add_argument('--report_to', type=str, default='all', help='The integration to report the results and logs to. Supported platforms are `"tensorboard"`, `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. Only applicable when `--with_tracking` is passed.')
    args = parser.parse_args()
    if args.dataset_name is None and args.train_file is None and (args.validation_file is None):
        raise ValueError('Need either a dataset name or a training/validation file.')
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
    send_example_telemetry('run_summarization_no_trainer', args)
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs['log_with'] = args.report_to
        accelerator_log_kwargs['project_dir'] = args.output_dir
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    if args.source_prefix is None and args.model_name_or_path in ['t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b']:
        logger.warning("You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with `--source_prefix 'summarize: ' `")
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
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=args.trust_remote_code)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning('You are instantiating a new config instance from scratch.')
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code)
    else:
        raise ValueError('You are instantiating a new tokenizer from scratch. This is not supported by this script. You can do it from another script, save it, and load it from here, using --tokenizer_name.')
    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config, trust_remote_code=args.trust_remote_code)
    else:
        logger.info('Training new model from scratch')
        model = AutoModelForSeq2SeqLM.from_config(config, trust_remote_code=args.trust_remote_code)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError('Make sure that `config.decoder_start_token_id` is correctly defined')
    prefix = args.source_prefix if args.source_prefix is not None else ''
    column_names = raw_datasets['train'].column_names
    dataset_columns = summarization_name_mapping.get(args.dataset_name, None)
    if args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = args.text_column
        if text_column not in column_names:
            raise ValueError(f"--text_column' value '{args.text_column}' needs to be one of: {', '.join(column_names)}")
    if args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = args.summary_column
        if summary_column not in column_names:
            raise ValueError(f"--summary_column' value '{args.summary_column}' needs to be one of: {', '.join(column_names)}")
    if args.val_max_target_length is None:
        args.val_max_target_length = args.max_target_length
    max_target_length = args.max_target_length
    padding = 'max_length' if args.pad_to_max_length else False

    def preprocess_function(examples):
        if False:
            while True:
                i = 10
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)
        if padding == 'max_length' and args.ignore_pad_token_for_loss:
            labels['input_ids'] = [[l if l != tokenizer.pad_token_id else -100 for l in label] for label in labels['input_ids']]
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    with accelerator.main_process_first():
        train_dataset = raw_datasets['train'].map(preprocess_function, batched=True, num_proc=args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not args.overwrite_cache, desc='Running tokenizer on dataset')
        max_target_length = args.val_max_target_length
        eval_dataset = raw_datasets['validation'].map(preprocess_function, batched=True, num_proc=args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not args.overwrite_cache, desc='Running tokenizer on dataset')
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f'Sample {index} of the training set: {train_dataset[index]}.')
    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8 if accelerator.use_fp16 else None)

    def postprocess_text(preds, labels):
        if False:
            i = 10
            return i + 15
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        preds = ['\n'.join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ['\n'.join(nltk.sent_tokenize(label)) for label in labels]
        return (preds, labels)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    optimizer_grouped_parameters = [{'params': [p for (n, p) in model.named_parameters() if not any((nd in n for nd in no_decay))], 'weight_decay': args.weight_decay}, {'params': [p for (n, p) in model.named_parameters() if any((nd in n for nd in no_decay))], 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps, num_training_steps=args.max_train_steps * args.gradient_accumulation_steps)
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
        accelerator.init_trackers('summarization_no_trainer', experiment_config)
    metric = evaluate.load('rouge')
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
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                if args.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
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
        gen_kwargs = {'max_length': args.val_max_target_length, 'num_beams': args.num_beams}
        for (step, batch) in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(batch['input_ids'], attention_mask=batch['attention_mask'], **gen_kwargs)
                generated_tokens = accelerator.pad_across_processes(generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)
                labels = batch['labels']
                if not args.pad_to_max_length:
                    labels = accelerator.pad_across_processes(batch['labels'], dim=1, pad_index=tokenizer.pad_token_id)
                (generated_tokens, labels) = accelerator.gather_for_metrics((generated_tokens, labels))
                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()
                if args.ignore_pad_token_for_loss:
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                (decoded_preds, decoded_labels) = postprocess_text(decoded_preds, decoded_labels)
                metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        result = metric.compute(use_stemmer=True)
        result = {k: round(v * 100, 4) for (k, v) in result.items()}
        logger.info(result)
        if args.with_tracking:
            result['train_loss'] = total_loss.item() / len(train_dataloader)
            result['epoch'] = epoch
            result['step'] = completed_steps
            accelerator.log(result, step=completed_steps)
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
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message='End of training', auto_lfs_prune=True)
            all_results = {f'eval_{k}': v for (k, v) in result.items()}
            with open(os.path.join(args.output_dir, 'all_results.json'), 'w') as f:
                json.dump(all_results, f)
if __name__ == '__main__':
    main()