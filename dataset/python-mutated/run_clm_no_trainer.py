"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from transformers import CONFIG_MAPPING, MODEL_MAPPING, AutoConfig, AutoModelForCausalLM, AutoTokenizer, SchedulerType, default_data_collator, get_scheduler
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
check_min_version('4.36.0.dev0')
logger = get_logger(__name__)
require_version('datasets>=1.8.0', 'To fix: pip install -r examples/pytorch/language-modeling/requirements.txt')
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple((conf.model_type for conf in MODEL_CONFIG_CLASSES))

def parse_args():
    if False:
        return 10
    parser = argparse.ArgumentParser(description='Finetune a transformers model on a causal language modeling task')
    parser.add_argument('--dataset_name', type=str, default=None, help='The name of the dataset to use (via the datasets library).')
    parser.add_argument('--dataset_config_name', type=str, default=None, help='The configuration name of the dataset to use (via the datasets library).')
    parser.add_argument('--train_file', type=str, default=None, help='A csv, txt or a json file containing the training data.')
    parser.add_argument('--validation_file', type=str, default=None, help='A csv, txt or a json file containing the validation data.')
    parser.add_argument('--validation_split_percentage', default=5, help="The percentage of the train set used as validation set in case there's no validation split")
    parser.add_argument('--model_name_or_path', type=str, help='Path to pretrained model or model identifier from huggingface.co/models.', required=False)
    parser.add_argument('--config_name', type=str, default=None, help='Pretrained config name or path if not the same as model_name')
    parser.add_argument('--tokenizer_name', type=str, default=None, help='Pretrained tokenizer name or path if not the same as model_name')
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
    parser.add_argument('--block_size', type=int, default=None, help='Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).')
    parser.add_argument('--preprocessing_num_workers', type=int, default=None, help='The number of processes to use for the preprocessing.')
    parser.add_argument('--overwrite_cache', action='store_true', help='Overwrite the cached training and evaluation sets')
    parser.add_argument('--no_keep_linebreaks', action='store_true', help='Do not keep line breaks when using TXT files.')
    parser.add_argument('--push_to_hub', action='store_true', help='Whether or not to push the model to the Hub.')
    parser.add_argument('--hub_model_id', type=str, help='The name of the repository to keep in sync with the local `output_dir`.')
    parser.add_argument('--hub_token', type=str, help='The token to use to push to the Model Hub.')
    parser.add_argument('--trust_remote_code', type=bool, default=False, help='Whether or not to allow for custom models defined on the Hub in their own modeling files. This optionshould only be set to `True` for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.')
    parser.add_argument('--checkpointing_steps', type=str, default=None, help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='If the training should continue from a checkpoint folder.')
    parser.add_argument('--with_tracking', action='store_true', help='Whether to enable experiment trackers for logging.')
    parser.add_argument('--report_to', type=str, default='all', help='The integration to report the results and logs to. Supported platforms are `"tensorboard"`, `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. Only applicable when `--with_tracking` is passed.')
    parser.add_argument('--low_cpu_mem_usage', action='store_true', help='It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. If passed, LLM loading time and RAM consumption will be benefited.')
    args = parser.parse_args()
    if args.dataset_name is None and args.train_file is None and (args.validation_file is None):
        raise ValueError('Need either a dataset name or a training/validation file.')
    else:
        if args.train_file is not None:
            extension = args.train_file.split('.')[-1]
            if extension not in ['csv', 'json', 'txt']:
                raise ValueError('`train_file` should be a csv, json or txt file.')
        if args.validation_file is not None:
            extension = args.validation_file.split('.')[-1]
            if extension not in ['csv', 'json', 'txt']:
                raise ValueError('`validation_file` should be a csv, json or txt file.')
    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError('Need an `output_dir` to create a repo when `--push_to_hub` is passed.')
    return args

def main():
    if False:
        return 10
    args = parse_args()
    send_example_telemetry('run_clm_no_trainer', args)
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs['log_with'] = args.report_to
        accelerator_log_kwargs['project_dir'] = args.output_dir
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
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
        if 'validation' not in raw_datasets.keys():
            raw_datasets['validation'] = load_dataset(args.dataset_name, args.dataset_config_name, split=f'train[:{args.validation_split_percentage}%]')
            raw_datasets['train'] = load_dataset(args.dataset_name, args.dataset_config_name, split=f'train[{args.validation_split_percentage}%:]')
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files['train'] = args.train_file
        if args.validation_file is not None:
            data_files['validation'] = args.validation_file
        extension = args.train_file.split('.')[-1]
        if extension == 'txt':
            extension = 'text'
            dataset_args['keep_linebreaks'] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        if 'validation' not in raw_datasets.keys():
            raw_datasets['validation'] = load_dataset(extension, data_files=data_files, split=f'train[:{args.validation_split_percentage}%]', **dataset_args)
            raw_datasets['train'] = load_dataset(extension, data_files=data_files, split=f'train[{args.validation_split_percentage}%:]', **dataset_args)
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
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config, low_cpu_mem_usage=args.low_cpu_mem_usage, trust_remote_code=args.trust_remote_code)
    else:
        logger.info('Training new model from scratch')
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    column_names = raw_datasets['train'].column_names
    text_column_name = 'text' if 'text' in column_names else column_names[0]

    def tokenize_function(examples):
        if False:
            return 10
        return tokenizer(examples[text_column_name])
    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, num_proc=args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not args.overwrite_cache, desc='Running tokenizer on dataset')
    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            logger.warning(f'The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx.')
            block_size = min(1024, config.max_position_embeddings)
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(f'The block_size passed ({args.block_size}) is larger than the maximum length for the model ({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}.')
        block_size = min(args.block_size, tokenizer.model_max_length)

    def group_texts(examples):
        if False:
            i = 10
            return i + 15
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = total_length // block_size * block_size
        result = {k: [t[i:i + block_size] for i in range(0, total_length, block_size)] for (k, t) in concatenated_examples.items()}
        result['labels'] = result['input_ids'].copy()
        return result
    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=args.preprocessing_num_workers, load_from_cache_file=not args.overwrite_cache, desc=f'Grouping texts in chunks of {block_size}')
    train_dataset = lm_datasets['train']
    eval_dataset = lm_datasets['validation']
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f'Sample {index} of the training set: {train_dataset[index]}.')
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size)
    no_decay = ['bias', 'layer_norm.weight']
    optimizer_grouped_parameters = [{'params': [p for (n, p) in model.named_parameters() if not any((nd in n for nd in no_decay))], 'weight_decay': args.weight_decay}, {'params': [p for (n, p) in model.named_parameters() if any((nd in n for nd in no_decay))], 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer, num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps, num_training_steps=args.max_train_steps * args.gradient_accumulation_steps)
    (model, optimizer, train_dataloader, eval_dataloader, lr_scheduler) = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()
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
        accelerator.init_trackers('clm_no_trainer', experiment_config)
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
        losses = []
        for (step, batch) in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float('inf')
        logger.info(f'epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}')
        if args.with_tracking:
            accelerator.log({'perplexity': perplexity, 'eval_loss': eval_loss, 'train_loss': total_loss.item() / len(train_dataloader), 'epoch': epoch, 'step': completed_steps}, step=completed_steps)
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
            with open(os.path.join(args.output_dir, 'all_results.json'), 'w') as f:
                json.dump({'perplexity': perplexity}, f)
if __name__ == '__main__':
    main()