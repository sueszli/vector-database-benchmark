"""
Fine-tuning the library models for question answering.
"""
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import datasets
import quant_trainer
from datasets import load_dataset, load_metric
from trainer_quant_qa import QuestionAnsweringTrainer
from utils_qa import postprocess_qa_predictions
import transformers
from transformers import AutoTokenizer, DataCollatorWithPadding, EvalPrediction, HfArgumentParser, PreTrainedTokenizerFast, QDQBertConfig, QDQBertForQuestionAnswering, TrainingArguments, default_data_collator, set_seed
from transformers.trainer_utils import SchedulerType, get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
check_min_version('4.9.0')
require_version('datasets>=1.8.0', 'To fix: pip install -r examples/pytorch/question-answering/requirements.txt')
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'})
    config_name: Optional[str] = field(default=None, metadata={'help': 'Pretrained config name or path if not the same as model_name'})
    tokenizer_name: Optional[str] = field(default=None, metadata={'help': 'Pretrained tokenizer name or path if not the same as model_name'})
    cache_dir: Optional[str] = field(default=None, metadata={'help': 'Path to directory to store the pretrained models downloaded from huggingface.co'})
    model_revision: str = field(default='main', metadata={'help': 'The specific model version to use (can be a branch name, tag name or commit id).'})
    use_auth_token: bool = field(default=False, metadata={'help': 'Will use the token generated when running `huggingface-cli login` (necessary to use this script with private models).'})
    do_calib: bool = field(default=False, metadata={'help': 'Whether to run calibration of quantization ranges.'})
    num_calib_batch: int = field(default=4, metadata={'help': 'Number of batches for calibration. 0 will disable calibration '})
    save_onnx: bool = field(default=False, metadata={'help': 'Whether to save model to onnx.'})

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(default=None, metadata={'help': 'The name of the dataset to use (via the datasets library).'})
    dataset_config_name: Optional[str] = field(default=None, metadata={'help': 'The configuration name of the dataset to use (via the datasets library).'})
    train_file: Optional[str] = field(default=None, metadata={'help': 'The input training data file (a text file).'})
    validation_file: Optional[str] = field(default=None, metadata={'help': 'An optional input evaluation data file to evaluate the perplexity on (a text file).'})
    test_file: Optional[str] = field(default=None, metadata={'help': 'An optional input test data file to evaluate the perplexity on (a text file).'})
    overwrite_cache: bool = field(default=False, metadata={'help': 'Overwrite the cached training and evaluation sets'})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={'help': 'The number of processes to use for the preprocessing.'})
    max_seq_length: int = field(default=384, metadata={'help': 'The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.'})
    pad_to_max_length: bool = field(default=True, metadata={'help': 'Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when batching to the maximum length in the batch (which can be faster on GPU but will be slower on TPU).'})
    max_train_samples: Optional[int] = field(default=None, metadata={'help': 'For debugging purposes or quicker training, truncate the number of training examples to this value if set.'})
    max_eval_samples: Optional[int] = field(default=None, metadata={'help': 'For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set.'})
    max_predict_samples: Optional[int] = field(default=None, metadata={'help': 'For debugging purposes or quicker training, truncate the number of prediction examples to this value if set.'})
    version_2_with_negative: bool = field(default=False, metadata={'help': 'If true, some of the examples do not have an answer.'})
    null_score_diff_threshold: float = field(default=0.0, metadata={'help': 'The threshold used to select the null answer: if the best answer has a score that is less than the score of the null answer minus this threshold, the null answer is selected for this example. Only useful when `version_2_with_negative=True`.'})
    doc_stride: int = field(default=128, metadata={'help': 'When splitting up a long document into chunks, how much stride to take between chunks.'})
    n_best_size: int = field(default=20, metadata={'help': 'The total number of n-best predictions to generate when looking for an answer.'})
    max_answer_length: int = field(default=30, metadata={'help': 'The maximum length of an answer that can be generated. This is needed because the start and end predictions are not conditioned on one another.'})

    def __post_init__(self):
        if False:
            return 10
        if self.dataset_name is None and self.train_file is None and (self.validation_file is None) and (self.test_file is None):
            raise ValueError('Need either a dataset name or a training/validation file/test_file.')
        else:
            if self.train_file is not None:
                extension = self.train_file.split('.')[-1]
                assert extension in ['csv', 'json'], '`train_file` should be a csv or a json file.'
            if self.validation_file is not None:
                extension = self.validation_file.split('.')[-1]
                assert extension in ['csv', 'json'], '`validation_file` should be a csv or a json file.'
            if self.test_file is not None:
                extension = self.test_file.split('.')[-1]
                assert extension in ['csv', 'json'], '`test_file` should be a csv or a json file.'

def main():
    if False:
        print('Hello World!')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    quant_trainer.add_arguments(parser)
    (model_args, data_args, training_args, quant_trainer_args) = parser.parse_args_into_dataclasses()
    training_args.lr_scheduler_type = SchedulerType.COSINE
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', handlers=[logging.StreamHandler(sys.stdout)])
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}' + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}')
    logger.info(f'Training/evaluation parameters {training_args}')
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and (not training_args.overwrite_output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f'Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.')
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch.')
    set_seed(training_args.seed)
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files['train'] = data_args.train_file
            extension = data_args.train_file.split('.')[-1]
        if data_args.validation_file is not None:
            data_files['validation'] = data_args.validation_file
            extension = data_args.validation_file.split('.')[-1]
        if data_args.test_file is not None:
            data_files['test'] = data_args.test_file
            extension = data_args.test_file.split('.')[-1]
        raw_datasets = load_dataset(extension, data_files=data_files, field='data', cache_dir=model_args.cache_dir)
    quant_trainer.set_default_quantizers(quant_trainer_args)
    config = QDQBertConfig.from_pretrained(model_args.config_name if model_args.config_name else model_args.model_name_or_path, cache_dir=model_args.cache_dir, revision=model_args.model_revision, token=True if model_args.use_auth_token else None)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=True, revision=model_args.model_revision, token=True if model_args.use_auth_token else None)
    model = QDQBertForQuestionAnswering.from_pretrained(model_args.model_name_or_path, from_tf=bool('.ckpt' in model_args.model_name_or_path), config=config, cache_dir=model_args.cache_dir, revision=model_args.model_revision, token=True if model_args.use_auth_token else None)
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError('This example script only works for models that have a fast tokenizer. Checkout the big table of models at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this requirement')
    if training_args.do_train or model_args.do_calib:
        column_names = raw_datasets['train'].column_names
    elif training_args.do_eval or model_args.save_onnx:
        column_names = raw_datasets['validation'].column_names
    else:
        column_names = raw_datasets['test'].column_names
    question_column_name = 'question' if 'question' in column_names else column_names[0]
    context_column_name = 'context' if 'context' in column_names else column_names[1]
    answer_column_name = 'answers' if 'answers' in column_names else column_names[2]
    pad_on_right = tokenizer.padding_side == 'right'
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(f'The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}.')
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def prepare_train_features(examples):
        if False:
            return 10
        tokenized_examples = tokenizer(examples[question_column_name if pad_on_right else context_column_name], examples[context_column_name if pad_on_right else question_column_name], truncation='only_second' if pad_on_right else 'only_first', max_length=max_seq_length, stride=data_args.doc_stride, return_overflowing_tokens=True, return_offsets_mapping=True, padding='max_length' if data_args.pad_to_max_length else False)
        sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
        offset_mapping = tokenized_examples.pop('offset_mapping')
        tokenized_examples['start_positions'] = []
        tokenized_examples['end_positions'] = []
        for (i, offsets) in enumerate(offset_mapping):
            input_ids = tokenized_examples['input_ids'][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            if len(answers['answer_start']) == 0:
                tokenized_examples['start_positions'].append(cls_index)
                tokenized_examples['end_positions'].append(cls_index)
            else:
                start_char = answers['answer_start'][0]
                end_char = start_char + len(answers['text'][0])
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples['start_positions'].append(cls_index)
                    tokenized_examples['end_positions'].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples['start_positions'].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples['end_positions'].append(token_end_index + 1)
        return tokenized_examples
    if training_args.do_train or model_args.do_calib:
        if 'train' not in raw_datasets:
            raise ValueError('--do_train requires a train dataset')
        train_dataset = raw_datasets['train']
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc='train dataset map pre-processing'):
            train_dataset = train_dataset.map(prepare_train_features, batched=True, num_proc=data_args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not data_args.overwrite_cache, desc='Running tokenizer on train dataset')
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    def prepare_validation_features(examples):
        if False:
            for i in range(10):
                print('nop')
        tokenized_examples = tokenizer(examples[question_column_name if pad_on_right else context_column_name], examples[context_column_name if pad_on_right else question_column_name], truncation='only_second' if pad_on_right else 'only_first', max_length=max_seq_length, stride=data_args.doc_stride, return_overflowing_tokens=True, return_offsets_mapping=True, padding='max_length' if data_args.pad_to_max_length else False)
        sample_mapping = tokenized_examples.pop('overflow_to_sample_mapping')
        tokenized_examples['example_id'] = []
        for i in range(len(tokenized_examples['input_ids'])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
            sample_index = sample_mapping[i]
            tokenized_examples['example_id'].append(examples['id'][sample_index])
            tokenized_examples['offset_mapping'][i] = [o if sequence_ids[k] == context_index else None for (k, o) in enumerate(tokenized_examples['offset_mapping'][i])]
        return tokenized_examples
    if training_args.do_eval or model_args.save_onnx:
        if 'validation' not in raw_datasets:
            raise ValueError('--do_eval requires a validation dataset')
        eval_examples = raw_datasets['validation']
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_examples), data_args.max_eval_samples)
            eval_examples = eval_examples.select(range(max_eval_samples))
        with training_args.main_process_first(desc='validation dataset map pre-processing'):
            eval_dataset = eval_examples.map(prepare_validation_features, batched=True, num_proc=data_args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not data_args.overwrite_cache, desc='Running tokenizer on validation dataset')
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
    if training_args.do_predict:
        if 'test' not in raw_datasets:
            raise ValueError('--do_predict requires a test dataset')
        predict_examples = raw_datasets['test']
        if data_args.max_predict_samples is not None:
            predict_examples = predict_examples.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc='prediction dataset map pre-processing'):
            predict_dataset = predict_examples.map(prepare_validation_features, batched=True, num_proc=data_args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not data_args.overwrite_cache, desc='Running tokenizer on prediction dataset')
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
    data_collator = default_data_collator if data_args.pad_to_max_length else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

    def post_processing_function(examples, features, predictions, stage='eval'):
        if False:
            return 10
        predictions = postprocess_qa_predictions(examples=examples, features=features, predictions=predictions, version_2_with_negative=data_args.version_2_with_negative, n_best_size=data_args.n_best_size, max_answer_length=data_args.max_answer_length, null_score_diff_threshold=data_args.null_score_diff_threshold, output_dir=training_args.output_dir, log_level=log_level, prefix=stage)
        if data_args.version_2_with_negative:
            formatted_predictions = [{'id': k, 'prediction_text': v, 'no_answer_probability': 0.0} for (k, v) in predictions.items()]
        else:
            formatted_predictions = [{'id': k, 'prediction_text': v} for (k, v) in predictions.items()]
        references = [{'id': ex['id'], 'answers': ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)
    metric = load_metric('squad_v2' if data_args.version_2_with_negative else 'squad')

    def compute_metrics(p: EvalPrediction):
        if False:
            for i in range(10):
                print('nop')
        return metric.compute(predictions=p.predictions, references=p.label_ids)
    trainer = QuestionAnsweringTrainer(model=model, args=training_args, train_dataset=train_dataset if training_args.do_train or model_args.do_calib else None, eval_dataset=eval_dataset if training_args.do_eval or model_args.save_onnx else None, eval_examples=eval_examples if training_args.do_eval or model_args.save_onnx else None, tokenizer=tokenizer, data_collator=data_collator, post_process_function=post_processing_function, compute_metrics=compute_metrics, quant_trainer_args=quant_trainer_args)
    if model_args.do_calib:
        logger.info('*** Calibrate ***')
        results = trainer.calibrate()
        trainer.save_model()
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        quant_trainer.configure_model(trainer.model, quant_trainer_args)
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        metrics['train_samples'] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()
    if training_args.do_eval:
        logger.info('*** Evaluate ***')
        quant_trainer.configure_model(trainer.model, quant_trainer_args, eval=True)
        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics['eval_samples'] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)
    if training_args.do_predict:
        logger.info('*** Predict ***')
        results = trainer.predict(predict_dataset, predict_examples)
        metrics = results.metrics
        max_predict_samples = data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        metrics['predict_samples'] = min(max_predict_samples, len(predict_dataset))
        trainer.log_metrics('predict', metrics)
        trainer.save_metrics('predict', metrics)
    if training_args.push_to_hub:
        kwargs = {'finetuned_from': model_args.model_name_or_path, 'tasks': 'question-answering'}
        if data_args.dataset_name is not None:
            kwargs['dataset_tags'] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs['dataset_args'] = data_args.dataset_config_name
                kwargs['dataset'] = f'{data_args.dataset_name} {data_args.dataset_config_name}'
            else:
                kwargs['dataset'] = data_args.dataset_name
        trainer.push_to_hub(**kwargs)
    if model_args.save_onnx:
        logger.info('Exporting model to onnx')
        results = trainer.save_onnx(output_dir=training_args.output_dir)

def _mp_fn(index):
    if False:
        return 10
    main()
if __name__ == '__main__':
    main()