"""
Fine-tuning the library models for sequence to sequence.
"""
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional
import datasets
import evaluate
import numpy as np
from datasets import load_dataset
import transformers
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, HfArgumentParser, M2M100Tokenizer, MBart50Tokenizer, MBart50TokenizerFast, MBartTokenizer, MBartTokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator, set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
check_min_version('4.36.0.dev0')
require_version('datasets>=1.8.0', 'To fix: pip install -r examples/pytorch/translation/requirements.txt')
logger = logging.getLogger(__name__)
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'})
    config_name: Optional[str] = field(default=None, metadata={'help': 'Pretrained config name or path if not the same as model_name'})
    tokenizer_name: Optional[str] = field(default=None, metadata={'help': 'Pretrained tokenizer name or path if not the same as model_name'})
    cache_dir: Optional[str] = field(default=None, metadata={'help': 'Where to store the pretrained models downloaded from huggingface.co'})
    use_fast_tokenizer: bool = field(default=True, metadata={'help': 'Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.'})
    model_revision: str = field(default='main', metadata={'help': 'The specific model version to use (can be a branch name, tag name or commit id).'})
    token: str = field(default=None, metadata={'help': 'The token to use as HTTP bearer authorization for remote files. If not specified, will use the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).'})
    use_auth_token: bool = field(default=None, metadata={'help': 'The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.'})
    trust_remote_code: bool = field(default=False, metadata={'help': 'Whether or not to allow for custom models defined on the Hub in their own modeling files. This optionshould only be set to `True` for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.'})

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    source_lang: str = field(default=None, metadata={'help': 'Source language id for translation.'})
    target_lang: str = field(default=None, metadata={'help': 'Target language id for translation.'})
    dataset_name: Optional[str] = field(default=None, metadata={'help': 'The name of the dataset to use (via the datasets library).'})
    dataset_config_name: Optional[str] = field(default=None, metadata={'help': 'The configuration name of the dataset to use (via the datasets library).'})
    train_file: Optional[str] = field(default=None, metadata={'help': 'The input training data file (a jsonlines).'})
    validation_file: Optional[str] = field(default=None, metadata={'help': 'An optional input evaluation data file to evaluate the metrics (sacrebleu) on a jsonlines file.'})
    test_file: Optional[str] = field(default=None, metadata={'help': 'An optional input test data file to evaluate the metrics (sacrebleu) on a jsonlines file.'})
    overwrite_cache: bool = field(default=False, metadata={'help': 'Overwrite the cached training and evaluation sets'})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={'help': 'The number of processes to use for the preprocessing.'})
    max_source_length: Optional[int] = field(default=1024, metadata={'help': 'The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.'})
    max_target_length: Optional[int] = field(default=128, metadata={'help': 'The maximum total sequence length for target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.'})
    val_max_target_length: Optional[int] = field(default=None, metadata={'help': 'The maximum total sequence length for validation target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`. This argument is also used to override the ``max_length`` param of ``model.generate``, which is used during ``evaluate`` and ``predict``.'})
    pad_to_max_length: bool = field(default=False, metadata={'help': 'Whether to pad all samples to model maximum sentence length. If False, will pad the samples dynamically when batching to the maximum length in the batch. More efficient on GPU but very bad for TPU.'})
    max_train_samples: Optional[int] = field(default=None, metadata={'help': 'For debugging purposes or quicker training, truncate the number of training examples to this value if set.'})
    max_eval_samples: Optional[int] = field(default=None, metadata={'help': 'For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set.'})
    max_predict_samples: Optional[int] = field(default=None, metadata={'help': 'For debugging purposes or quicker training, truncate the number of prediction examples to this value if set.'})
    num_beams: Optional[int] = field(default=1, metadata={'help': 'Number of beams to use for evaluation. This argument will be passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.'})
    ignore_pad_token_for_loss: bool = field(default=True, metadata={'help': 'Whether to ignore the tokens corresponding to padded labels in the loss computation or not.'})
    source_prefix: Optional[str] = field(default=None, metadata={'help': 'A prefix to add before every source text (useful for T5 models).'})
    forced_bos_token: Optional[str] = field(default=None, metadata={'help': 'The token to force as the first generated token after the :obj:`decoder_start_token_id`.Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token needs to be the target language token.(Usually it is the target language token)'})

    def __post_init__(self):
        if False:
            return 10
        if self.dataset_name is None and self.train_file is None and (self.validation_file is None):
            raise ValueError('Need either a dataset name or a training/validation file.')
        elif self.source_lang is None or self.target_lang is None:
            raise ValueError('Need to specify the source language and the target language.')
        valid_extensions = ['json', 'jsonl']
        if self.train_file is not None:
            extension = self.train_file.split('.')[-1]
            assert extension in valid_extensions, '`train_file` should be a jsonlines file.'
        if self.validation_file is not None:
            extension = self.validation_file.split('.')[-1]
            assert extension in valid_extensions, '`validation_file` should be a jsonlines file.'
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

def main():
    if False:
        print('Hello World!')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        (model_args, data_args, training_args) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (model_args, data_args, training_args) = parser.parse_args_into_dataclasses()
    if model_args.use_auth_token is not None:
        warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.', FutureWarning)
        if model_args.token is not None:
            raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
        model_args.token = model_args.use_auth_token
    send_example_telemetry('run_translation', model_args, data_args)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', handlers=[logging.StreamHandler(sys.stdout)])
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, ' + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}")
    logger.info(f'Training/evaluation parameters {training_args}')
    if data_args.source_prefix is None and model_args.model_name_or_path in ['t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b']:
        logger.warning("You're running a t5 model but didn't provide a source prefix, which is expected, e.g. with `--source_prefix 'translate English to German: ' `")
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and (not training_args.overwrite_output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f'Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.')
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch.')
    set_seed(training_args.seed)
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir, token=model_args.token)
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
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir, token=model_args.token)
    config = AutoConfig.from_pretrained(model_args.config_name if model_args.config_name else model_args.model_name_or_path, cache_dir=model_args.cache_dir, revision=model_args.model_revision, token=model_args.token, trust_remote_code=model_args.trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer, revision=model_args.model_revision, token=model_args.token, trust_remote_code=model_args.trust_remote_code)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path, from_tf=bool('.ckpt' in model_args.model_name_or_path), config=config, cache_dir=model_args.cache_dir, revision=model_args.model_revision, token=model_args.token, trust_remote_code=model_args.trust_remote_code)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.target_lang)
    if model.config.decoder_start_token_id is None:
        raise ValueError('Make sure that `config.decoder_start_token_id` is correctly defined')
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ''
    if training_args.do_train:
        column_names = raw_datasets['train'].column_names
    elif training_args.do_eval:
        column_names = raw_datasets['validation'].column_names
    elif training_args.do_predict:
        column_names = raw_datasets['test'].column_names
    else:
        logger.info('There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.')
        return
    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert data_args.target_lang is not None and data_args.source_lang is not None, f'{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --source_lang and --target_lang arguments.'
        tokenizer.src_lang = data_args.source_lang
        tokenizer.tgt_lang = data_args.target_lang
        forced_bos_token_id = tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        model.config.forced_bos_token_id = forced_bos_token_id
    source_lang = data_args.source_lang.split('_')[0]
    target_lang = data_args.target_lang.split('_')[0]
    max_target_length = data_args.max_target_length
    padding = 'max_length' if data_args.pad_to_max_length else False
    if training_args.label_smoothing_factor > 0 and (not hasattr(model, 'prepare_decoder_input_ids_from_labels')):
        logger.warning(f'label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for `{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory')

    def preprocess_function(examples):
        if False:
            i = 10
            return i + 15
        inputs = [ex[source_lang] for ex in examples['translation']]
        targets = [ex[target_lang] for ex in examples['translation']]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)
        if padding == 'max_length' and data_args.ignore_pad_token_for_loss:
            labels['input_ids'] = [[l if l != tokenizer.pad_token_id else -100 for l in label] for label in labels['input_ids']]
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    if training_args.do_train:
        if 'train' not in raw_datasets:
            raise ValueError('--do_train requires a train dataset')
        train_dataset = raw_datasets['train']
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc='train dataset map pre-processing'):
            train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=data_args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not data_args.overwrite_cache, desc='Running tokenizer on train dataset')
    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if 'validation' not in raw_datasets:
            raise ValueError('--do_eval requires a validation dataset')
        eval_dataset = raw_datasets['validation']
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc='validation dataset map pre-processing'):
            eval_dataset = eval_dataset.map(preprocess_function, batched=True, num_proc=data_args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not data_args.overwrite_cache, desc='Running tokenizer on validation dataset')
    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if 'test' not in raw_datasets:
            raise ValueError('--do_predict requires a test dataset')
        predict_dataset = raw_datasets['test']
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc='prediction dataset map pre-processing'):
            predict_dataset = predict_dataset.map(preprocess_function, batched=True, num_proc=data_args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not data_args.overwrite_cache, desc='Running tokenizer on prediction dataset')
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=label_pad_token_id, pad_to_multiple_of=8 if training_args.fp16 else None)
    metric = evaluate.load('sacrebleu')

    def postprocess_text(preds, labels):
        if False:
            i = 10
            return i + 15
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return (preds, labels)

    def compute_metrics(eval_preds):
        if False:
            return 10
        (preds, labels) = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        (decoded_preds, decoded_labels) = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {'bleu': result['score']}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result['gen_len'] = np.mean(prediction_lens)
        result = {k: round(v, 4) for (k, v) in result.items()}
        return result
    trainer = Seq2SeqTrainer(model=model, args=training_args, train_dataset=train_dataset if training_args.do_train else None, eval_dataset=eval_dataset if training_args.do_eval else None, tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics if training_args.predict_with_generate else None)
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        metrics = train_result.metrics
        max_train_samples = data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        metrics['train_samples'] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()
    results = {}
    max_length = training_args.generation_max_length if training_args.generation_max_length is not None else data_args.val_max_target_length
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info('*** Evaluate ***')
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix='eval')
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics['eval_samples'] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)
    if training_args.do_predict:
        logger.info('*** Predict ***')
        predict_results = trainer.predict(predict_dataset, metric_key_prefix='predict', max_length=max_length, num_beams=num_beams)
        metrics = predict_results.metrics
        max_predict_samples = data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        metrics['predict_samples'] = min(max_predict_samples, len(predict_dataset))
        trainer.log_metrics('predict', metrics)
        trainer.save_metrics('predict', metrics)
        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = predict_results.predictions
                predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
                predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, 'generated_predictions.txt')
                with open(output_prediction_file, 'w', encoding='utf-8') as writer:
                    writer.write('\n'.join(predictions))
    kwargs = {'finetuned_from': model_args.model_name_or_path, 'tasks': 'translation'}
    if data_args.dataset_name is not None:
        kwargs['dataset_tags'] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs['dataset_args'] = data_args.dataset_config_name
            kwargs['dataset'] = f'{data_args.dataset_name} {data_args.dataset_config_name}'
        else:
            kwargs['dataset'] = data_args.dataset_name
    languages = [l for l in [data_args.source_lang, data_args.target_lang] if l is not None]
    if len(languages) > 0:
        kwargs['language'] = languages
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)
    return results

def _mp_fn(index):
    if False:
        print('Hello World!')
    main()
if __name__ == '__main__':
    main()