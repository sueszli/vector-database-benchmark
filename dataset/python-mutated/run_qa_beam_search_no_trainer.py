"""
Fine-tuning XLNet for question answering with beam search using 🤗 Accelerate.
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
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils_qa import postprocess_qa_predictions_with_beam_search
import transformers
from transformers import AdamW, DataCollatorWithPadding, EvalPrediction, SchedulerType, XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizerFast, default_data_collator, get_scheduler
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
check_min_version('4.36.0.dev0')
require_version('datasets>=1.8.0', 'To fix: pip install -r examples/pytorch/question-answering/requirements.txt')
logger = get_logger(__name__)

def save_prefixed_metrics(results, output_dir, file_name: str='all_results.json', metric_key_prefix: str='eval'):
    if False:
        i = 10
        return i + 15
    '\n    Save results while prefixing metric names.\n\n    Args:\n        results: (:obj:`dict`):\n            A dictionary of results.\n        output_dir: (:obj:`str`):\n            An output directory.\n        file_name: (:obj:`str`, `optional`, defaults to :obj:`all_results.json`):\n            An output file name.\n        metric_key_prefix: (:obj:`str`, `optional`, defaults to :obj:`eval`):\n            A metric name prefix.\n    '
    for key in list(results.keys()):
        if not key.startswith(f'{metric_key_prefix}_'):
            results[f'{metric_key_prefix}_{key}'] = results.pop(key)
    with open(os.path.join(output_dir, file_name), 'w') as f:
        json.dump(results, f, indent=4)

def parse_args():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(description='Finetune a transformers model on a Question Answering task')
    parser.add_argument('--dataset_name', type=str, default=None, help='The name of the dataset to use (via the datasets library).')
    parser.add_argument('--dataset_config_name', type=str, default=None, help='The configuration name of the dataset to use (via the datasets library).')
    parser.add_argument('--train_file', type=str, default=None, help='A csv or a json file containing the training data.')
    parser.add_argument('--preprocessing_num_workers', type=int, default=1, help='A csv or a json file containing the training data.')
    parser.add_argument('--do_predict', action='store_true', help='Eval the question answering model')
    parser.add_argument('--validation_file', type=str, default=None, help='A csv or a json file containing the validation data.')
    parser.add_argument('--test_file', type=str, default=None, help='A csv or a json file containing the Prediction data.')
    parser.add_argument('--max_seq_length', type=int, default=384, help='The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded if `--pad_to_max_lengh` is passed.')
    parser.add_argument('--pad_to_max_length', action='store_true', help='If passed, pad all samples to `max_seq_length`. Otherwise, dynamic padding is used.')
    parser.add_argument('--model_name_or_path', type=str, help='Path to pretrained model or model identifier from huggingface.co/models.', required=True)
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
    parser.add_argument('--doc_stride', type=int, default=128, help='When splitting up a long document into chunks how much stride to take between chunks.')
    parser.add_argument('--n_best_size', type=int, default=20, help='The total number of n-best predictions to generate when looking for an answer.')
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0, help='The threshold used to select the null answer: if the best answer has a score that is less than the score of the null answer minus this threshold, the null answer is selected for this example. Only useful when `version_2_with_negative=True`.')
    parser.add_argument('--version_2_with_negative', action='store_true', help='If true, some of the examples do not have an answer.')
    parser.add_argument('--max_answer_length', type=int, default=30, help='The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.')
    parser.add_argument('--max_train_samples', type=int, default=None, help='For debugging purposes or quicker training, truncate the number of training examples to this value if set.')
    parser.add_argument('--max_eval_samples', type=int, default=None, help='For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set.')
    parser.add_argument('--overwrite_cache', action='store_true', help='Overwrite the cached training and evaluation sets')
    parser.add_argument('--max_predict_samples', type=int, default=None, help='For debugging purposes or quicker training, truncate the number of prediction examples to this')
    parser.add_argument('--push_to_hub', action='store_true', help='Whether or not to push the model to the Hub.')
    parser.add_argument('--hub_model_id', type=str, help='The name of the repository to keep in sync with the local `output_dir`.')
    parser.add_argument('--hub_token', type=str, help='The token to use to push to the Model Hub.')
    parser.add_argument('--checkpointing_steps', type=str, default=None, help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='If the training should continue from a checkpoint folder.')
    parser.add_argument('--with_tracking', action='store_true', help='Whether to load in all available experiment trackers from the environment and use them for logging.')
    args = parser.parse_args()
    if args.dataset_name is None and args.train_file is None and (args.validation_file is None) and (args.test_file is None):
        raise ValueError('Need either a dataset name or a training/validation/test file.')
    else:
        if args.train_file is not None:
            extension = args.train_file.split('.')[-1]
            assert extension in ['csv', 'json'], '`train_file` should be a csv or a json file.'
        if args.validation_file is not None:
            extension = args.validation_file.split('.')[-1]
            assert extension in ['csv', 'json'], '`validation_file` should be a csv or a json file.'
        if args.test_file is not None:
            extension = args.test_file.split('.')[-1]
            assert extension in ['csv', 'json'], '`test_file` should be a csv or a json file.'
    if args.push_to_hub:
        assert args.output_dir is not None, 'Need an `output_dir` to create a repo when `--push_to_hub` is passed.'
    return args

def main():
    if False:
        while True:
            i = 10
    args = parse_args()
    send_example_telemetry('run_qa_beam_search_no_trainer', args)
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
    else:
        data_files = {}
        if args.train_file is not None:
            data_files['train'] = args.train_file
        if args.validation_file is not None:
            data_files['validation'] = args.validation_file
        if args.test_file is not None:
            data_files['test'] = args.test_file
        extension = args.train_file.split('.')[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, field='data')
    config = XLNetConfig.from_pretrained(args.model_name_or_path)
    tokenizer = XLNetTokenizerFast.from_pretrained(args.model_name_or_path)
    model = XLNetForQuestionAnswering.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
    column_names = raw_datasets['train'].column_names
    question_column_name = 'question' if 'question' in column_names else column_names[0]
    context_column_name = 'context' if 'context' in column_names else column_names[1]
    answer_column_name = 'answers' if 'answers' in column_names else column_names[2]
    pad_on_right = tokenizer.padding_side == 'right'
    if args.max_seq_length > tokenizer.model_max_length:
        logger.warning(f'The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}.')
    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    def prepare_train_features(examples):
        if False:
            return 10
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]
        tokenized_examples = tokenizer(examples[question_column_name if pad_on_right else context_column_name], examples[context_column_name if pad_on_right else question_column_name], truncation='only_second' if pad_on_right else 'only_first', max_length=max_seq_length, stride=args.doc_stride, return_overflowing_tokens=True, return_offsets_mapping=True, return_special_tokens_mask=True, return_token_type_ids=True, padding='max_length')
        sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
        offset_mapping = tokenized_examples.pop('offset_mapping')
        special_tokens = tokenized_examples.pop('special_tokens_mask')
        tokenized_examples['start_positions'] = []
        tokenized_examples['end_positions'] = []
        tokenized_examples['is_impossible'] = []
        tokenized_examples['cls_index'] = []
        tokenized_examples['p_mask'] = []
        for (i, offsets) in enumerate(offset_mapping):
            input_ids = tokenized_examples['input_ids'][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            tokenized_examples['cls_index'].append(cls_index)
            sequence_ids = tokenized_examples['token_type_ids'][i]
            for (k, s) in enumerate(special_tokens[i]):
                if s:
                    sequence_ids[k] = 3
            context_idx = 1 if pad_on_right else 0
            tokenized_examples['p_mask'].append([0.0 if not special_tokens[i][k] and s == context_idx or k == cls_index else 1.0 for (k, s) in enumerate(sequence_ids)])
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            if len(answers['answer_start']) == 0:
                tokenized_examples['start_positions'].append(cls_index)
                tokenized_examples['end_positions'].append(cls_index)
                tokenized_examples['is_impossible'].append(1.0)
            else:
                start_char = answers['answer_start'][0]
                end_char = start_char + len(answers['text'][0])
                token_start_index = 0
                while sequence_ids[token_start_index] != context_idx:
                    token_start_index += 1
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != context_idx:
                    token_end_index -= 1
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples['start_positions'].append(cls_index)
                    tokenized_examples['end_positions'].append(cls_index)
                    tokenized_examples['is_impossible'].append(1.0)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples['start_positions'].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples['end_positions'].append(token_end_index + 1)
                    tokenized_examples['is_impossible'].append(0.0)
        return tokenized_examples
    if 'train' not in raw_datasets:
        raise ValueError('--do_train requires a train dataset')
    train_dataset = raw_datasets['train']
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    with accelerator.main_process_first():
        train_dataset = train_dataset.map(prepare_train_features, batched=True, num_proc=args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not args.overwrite_cache, desc='Running tokenizer on train dataset')
    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(args.max_train_samples))

    def prepare_validation_features(examples):
        if False:
            print('Hello World!')
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]
        tokenized_examples = tokenizer(examples[question_column_name if pad_on_right else context_column_name], examples[context_column_name if pad_on_right else question_column_name], truncation='only_second' if pad_on_right else 'only_first', max_length=max_seq_length, stride=args.doc_stride, return_overflowing_tokens=True, return_offsets_mapping=True, return_special_tokens_mask=True, return_token_type_ids=True, padding='max_length')
        sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
        special_tokens = tokenized_examples.pop('special_tokens_mask')
        tokenized_examples['example_id'] = []
        tokenized_examples['cls_index'] = []
        tokenized_examples['p_mask'] = []
        for (i, input_ids) in enumerate(tokenized_examples['input_ids']):
            cls_index = input_ids.index(tokenizer.cls_token_id)
            tokenized_examples['cls_index'].append(cls_index)
            sequence_ids = tokenized_examples['token_type_ids'][i]
            for (k, s) in enumerate(special_tokens[i]):
                if s:
                    sequence_ids[k] = 3
            context_idx = 1 if pad_on_right else 0
            tokenized_examples['p_mask'].append([0.0 if not special_tokens[i][k] and s == context_idx or k == cls_index else 1.0 for (k, s) in enumerate(sequence_ids)])
            sample_index = sample_mapping[i]
            tokenized_examples['example_id'].append(examples['id'][sample_index])
            tokenized_examples['offset_mapping'][i] = [o if sequence_ids[k] == context_idx else None for (k, o) in enumerate(tokenized_examples['offset_mapping'][i])]
        return tokenized_examples
    if 'validation' not in raw_datasets:
        raise ValueError('--do_eval requires a validation dataset')
    eval_examples = raw_datasets['validation']
    if args.max_eval_samples is not None:
        eval_examples = eval_examples.select(range(args.max_eval_samples))
    with accelerator.main_process_first():
        eval_dataset = eval_examples.map(prepare_validation_features, batched=True, num_proc=args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not args.overwrite_cache, desc='Running tokenizer on validation dataset')
    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))
    if args.do_predict:
        if 'test' not in raw_datasets:
            raise ValueError('--do_predict requires a test dataset')
        predict_examples = raw_datasets['test']
        if args.max_predict_samples is not None:
            predict_examples = predict_examples.select(range(args.max_predict_samples))
        with accelerator.main_process_first():
            predict_dataset = predict_examples.map(prepare_validation_features, batched=True, num_proc=args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not args.overwrite_cache, desc='Running tokenizer on prediction dataset')
            if args.max_predict_samples is not None:
                predict_dataset = predict_dataset.select(range(args.max_predict_samples))
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f'Sample {index} of the training set: {train_dataset[index]}.')
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if accelerator.use_fp16 else None)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    eval_dataset_for_model = eval_dataset.remove_columns(['example_id', 'offset_mapping'])
    eval_dataloader = DataLoader(eval_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    if args.do_predict:
        predict_dataset_for_model = predict_dataset.remove_columns(['example_id', 'offset_mapping'])
        predict_dataloader = DataLoader(predict_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    def post_processing_function(examples, features, predictions, stage='eval'):
        if False:
            i = 10
            return i + 15
        (predictions, scores_diff_json) = postprocess_qa_predictions_with_beam_search(examples=examples, features=features, predictions=predictions, version_2_with_negative=args.version_2_with_negative, n_best_size=args.n_best_size, max_answer_length=args.max_answer_length, start_n_top=model.config.start_n_top, end_n_top=model.config.end_n_top, output_dir=args.output_dir, prefix=stage)
        if args.version_2_with_negative:
            formatted_predictions = [{'id': k, 'prediction_text': v, 'no_answer_probability': scores_diff_json[k]} for (k, v) in predictions.items()]
        else:
            formatted_predictions = [{'id': k, 'prediction_text': v} for (k, v) in predictions.items()]
        references = [{'id': ex['id'], 'answers': ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)
    metric = evaluate.load('squad_v2' if args.version_2_with_negative else 'squad')

    def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
        if False:
            return 10
        '\n        Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor\n\n        Args:\n            start_or_end_logits(:obj:`tensor`):\n                This is the output predictions of the model. We can only enter either start or end logits.\n            eval_dataset: Evaluation dataset\n            max_len(:obj:`int`):\n                The maximum length of the output tensor. ( See the model.eval() part for more details )\n        '
        step = 0
        logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float32)
        for (i, output_logit) in enumerate(start_or_end_logits):
            batch_size = output_logit.shape[0]
            cols = output_logit.shape[1]
            if step + batch_size < len(dataset):
                logits_concat[step:step + batch_size, :cols] = output_logit
            else:
                logits_concat[step:, :cols] = output_logit[:len(dataset) - step]
            step += batch_size
        return logits_concat
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{'params': [p for (n, p) in model.named_parameters() if not any((nd in n for nd in no_decay))], 'weight_decay': args.weight_decay}, {'params': [p for (n, p) in model.named_parameters() if any((nd in n for nd in no_decay))], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
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
        accelerator.init_trackers('qa_beam_search_no_trainer', experiment_config)
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
                    accelerator.save_state(f'step_{completed_steps}')
            if completed_steps >= args.max_train_steps:
                break
        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(commit_message=f'Training in progress epoch {epoch}', blocking=False, auto_lfs_prune=True)
    all_start_top_log_probs = []
    all_start_top_index = []
    all_end_top_log_probs = []
    all_end_top_index = []
    all_cls_logits = []
    model.eval()
    for (step, batch) in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            start_top_log_probs = outputs.start_top_log_probs
            start_top_index = outputs.start_top_index
            end_top_log_probs = outputs.end_top_log_probs
            end_top_index = outputs.end_top_index
            cls_logits = outputs.cls_logits
            if not args.pad_to_max_length:
                start_top_log_probs = accelerator.pad_across_processes(start_top_log_probs, dim=1, pad_index=-100)
                start_top_index = accelerator.pad_across_processes(start_top_index, dim=1, pad_index=-100)
                end_top_log_probs = accelerator.pad_across_processes(end_top_log_probs, dim=1, pad_index=-100)
                end_top_index = accelerator.pad_across_processes(end_top_index, dim=1, pad_index=-100)
                cls_logits = accelerator.pad_across_processes(cls_logits, dim=1, pad_index=-100)
            all_start_top_log_probs.append(accelerator.gather_for_metrics(start_top_log_probs).cpu().numpy())
            all_start_top_index.append(accelerator.gather_for_metrics(start_top_index).cpu().numpy())
            all_end_top_log_probs.append(accelerator.gather_for_metrics(end_top_log_probs).cpu().numpy())
            all_end_top_index.append(accelerator.gather_for_metrics(end_top_index).cpu().numpy())
            all_cls_logits.append(accelerator.gather_for_metrics(cls_logits).cpu().numpy())
    max_len = max([x.shape[1] for x in all_end_top_log_probs])
    start_top_log_probs_concat = create_and_fill_np_array(all_start_top_log_probs, eval_dataset, max_len)
    start_top_index_concat = create_and_fill_np_array(all_start_top_index, eval_dataset, max_len)
    end_top_log_probs_concat = create_and_fill_np_array(all_end_top_log_probs, eval_dataset, max_len)
    end_top_index_concat = create_and_fill_np_array(all_end_top_index, eval_dataset, max_len)
    cls_logits_concat = np.concatenate(all_cls_logits, axis=0)
    del start_top_log_probs
    del start_top_index
    del end_top_log_probs
    del end_top_index
    del cls_logits
    outputs_numpy = (start_top_log_probs_concat, start_top_index_concat, end_top_log_probs_concat, end_top_index_concat, cls_logits_concat)
    prediction = post_processing_function(eval_examples, eval_dataset, outputs_numpy)
    eval_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
    logger.info(f'Evaluation metrics: {eval_metric}')
    if args.do_predict:
        all_start_top_log_probs = []
        all_start_top_index = []
        all_end_top_log_probs = []
        all_end_top_index = []
        all_cls_logits = []
        model.eval()
        for (step, batch) in enumerate(predict_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
                start_top_log_probs = outputs.start_top_log_probs
                start_top_index = outputs.start_top_index
                end_top_log_probs = outputs.end_top_log_probs
                end_top_index = outputs.end_top_index
                cls_logits = outputs.cls_logits
                if not args.pad_to_max_length:
                    start_top_log_probs = accelerator.pad_across_processes(start_top_log_probs, dim=1, pad_index=-100)
                    start_top_index = accelerator.pad_across_processes(start_top_index, dim=1, pad_index=-100)
                    end_top_log_probs = accelerator.pad_across_processes(end_top_log_probs, dim=1, pad_index=-100)
                    end_top_index = accelerator.pad_across_processes(end_top_index, dim=1, pad_index=-100)
                    cls_logits = accelerator.pad_across_processes(cls_logits, dim=1, pad_index=-100)
                all_start_top_log_probs.append(accelerator.gather_for_metrics(start_top_log_probs).cpu().numpy())
                all_start_top_index.append(accelerator.gather_for_metrics(start_top_index).cpu().numpy())
                all_end_top_log_probs.append(accelerator.gather_for_metrics(end_top_log_probs).cpu().numpy())
                all_end_top_index.append(accelerator.gather_for_metrics(end_top_index).cpu().numpy())
                all_cls_logits.append(accelerator.gather_for_metrics(cls_logits).cpu().numpy())
        max_len = max([x.shape[1] for x in all_end_top_log_probs])
        start_top_log_probs_concat = create_and_fill_np_array(all_start_top_log_probs, predict_dataset, max_len)
        start_top_index_concat = create_and_fill_np_array(all_start_top_index, predict_dataset, max_len)
        end_top_log_probs_concat = create_and_fill_np_array(all_end_top_log_probs, predict_dataset, max_len)
        end_top_index_concat = create_and_fill_np_array(all_end_top_index, predict_dataset, max_len)
        cls_logits_concat = np.concatenate(all_cls_logits, axis=0)
        del start_top_log_probs
        del start_top_index
        del end_top_log_probs
        del end_top_index
        del cls_logits
        outputs_numpy = (start_top_log_probs_concat, start_top_index_concat, end_top_log_probs_concat, end_top_index_concat, cls_logits_concat)
        prediction = post_processing_function(predict_examples, predict_dataset, outputs_numpy)
        predict_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
        logger.info(f'Predict metrics: {predict_metric}')
    if args.with_tracking:
        log = {'squad_v2' if args.version_2_with_negative else 'squad': eval_metric, 'train_loss': total_loss, 'epoch': epoch, 'step': completed_steps}
        if args.do_predict:
            log['squad_v2_predict' if args.version_2_with_negative else 'squad_predict'] = predict_metric
        accelerator.log(log)
    if args.checkpointing_steps == 'epoch':
        accelerator.save_state(f'epoch_{epoch}')
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message='End of training', auto_lfs_prune=True)
            logger.info(json.dumps(eval_metric, indent=4))
            save_prefixed_metrics(eval_metric, args.output_dir)
if __name__ == '__main__':
    main()