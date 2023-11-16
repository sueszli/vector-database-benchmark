"""
Fine-tuning the library models for permutation language modeling.
"""
import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import datasets
from datasets import load_dataset
import transformers
from transformers import AutoConfig, AutoTokenizer, DataCollatorForPermutationLanguageModeling, HfArgumentParser, Trainer, TrainingArguments, XLNetConfig, XLNetLMHeadModel, set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
check_min_version('4.36.0.dev0')
require_version('datasets>=1.8.0', 'To fix: pip install -r examples/pytorch/language-modeling/requirements.txt')
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(default=None, metadata={'help': "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."})
    config_name: Optional[str] = field(default=None, metadata={'help': 'Pretrained config name or path if not the same as model_name'})
    config_overrides: Optional[str] = field(default=None, metadata={'help': 'Override some existing default config settings when a model is trained from scratch. Example: n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index'})
    tokenizer_name: Optional[str] = field(default=None, metadata={'help': 'Pretrained tokenizer name or path if not the same as model_name'})
    cache_dir: Optional[str] = field(default=None, metadata={'help': 'Where do you want to store the pretrained models downloaded from huggingface.co'})
    use_fast_tokenizer: bool = field(default=True, metadata={'help': 'Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.'})
    model_revision: str = field(default='main', metadata={'help': 'The specific model version to use (can be a branch name, tag name or commit id).'})
    token: str = field(default=None, metadata={'help': 'The token to use as HTTP bearer authorization for remote files. If not specified, will use the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).'})
    use_auth_token: bool = field(default=None, metadata={'help': 'The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.'})
    low_cpu_mem_usage: bool = field(default=False, metadata={'help': 'It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. set True will benefit LLM loading time and RAM consumption.'})

    def __post_init__(self):
        if False:
            i = 10
            return i + 15
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError("--config_overrides can't be used in combination with --config_name or --model_name_or_path")

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(default=None, metadata={'help': 'The name of the dataset to use (via the datasets library).'})
    dataset_config_name: Optional[str] = field(default=None, metadata={'help': 'The configuration name of the dataset to use (via the datasets library).'})
    train_file: Optional[str] = field(default=None, metadata={'help': 'The input training data file (a text file).'})
    validation_file: Optional[str] = field(default=None, metadata={'help': 'An optional input evaluation data file to evaluate the perplexity on (a text file).'})
    overwrite_cache: bool = field(default=False, metadata={'help': 'Overwrite the cached training and evaluation sets'})
    validation_split_percentage: Optional[int] = field(default=5, metadata={'help': "The percentage of the train set used as validation set in case there's no validation split"})
    max_seq_length: int = field(default=512, metadata={'help': 'The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.'})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={'help': 'The number of processes to use for the preprocessing.'})
    plm_probability: float = field(default=1 / 6, metadata={'help': 'Ratio of length of a span of masked tokens to surrounding context length for permutation language modeling.'})
    max_span_length: int = field(default=5, metadata={'help': 'Maximum length of a span of masked tokens for permutation language modeling.'})
    line_by_line: bool = field(default=False, metadata={'help': 'Whether distinct lines of text in the dataset are to be handled as distinct sequences.'})
    pad_to_max_length: bool = field(default=False, metadata={'help': 'Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when batching to the maximum length in the batch.'})
    max_train_samples: Optional[int] = field(default=None, metadata={'help': 'For debugging purposes or quicker training, truncate the number of training examples to this value if set.'})
    max_eval_samples: Optional[int] = field(default=None, metadata={'help': 'For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set.'})

    def __post_init__(self):
        if False:
            print('Hello World!')
        if self.dataset_name is None and self.train_file is None and (self.validation_file is None):
            raise ValueError('Need either a dataset name or a training/validation file.')
        else:
            if self.train_file is not None:
                extension = self.train_file.split('.')[-1]
                assert extension in ['csv', 'json', 'txt'], '`train_file` should be a csv, a json or a txt file.'
            if self.validation_file is not None:
                extension = self.validation_file.split('.')[-1]
                assert extension in ['csv', 'json', 'txt'], '`validation_file` should be a csv, a json or a txt file.'

def main():
    if False:
        i = 10
        return i + 15
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        (model_args, data_args, training_args) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (model_args, data_args, training_args) = parser.parse_args_into_dataclasses()
    if model_args.use_auth_token is not None:
        warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.', FutureWarning)
        if model_args.token is not None:
            raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
        model_args.token = model_args.use_auth_token
    send_example_telemetry('run_plm', model_args, data_args)
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
        if 'validation' not in raw_datasets.keys():
            raw_datasets['validation'] = load_dataset(data_args.dataset_name, data_args.dataset_config_name, split=f'train[:{data_args.validation_split_percentage}%]', cache_dir=model_args.cache_dir, token=model_args.token)
            raw_datasets['train'] = load_dataset(data_args.dataset_name, data_args.dataset_config_name, split=f'train[{data_args.validation_split_percentage}%:]', cache_dir=model_args.cache_dir, token=model_args.token)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files['train'] = data_args.train_file
        if data_args.validation_file is not None:
            data_files['validation'] = data_args.validation_file
        extension = data_args.train_file.split('.')[-1]
        if extension == 'txt':
            extension = 'text'
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
        if 'validation' not in raw_datasets.keys():
            raw_datasets['validation'] = load_dataset(extension, data_files=data_files, split=f'train[:{data_args.validation_split_percentage}%]', cache_dir=model_args.cache_dir, token=model_args.token)
            raw_datasets['train'] = load_dataset(extension, data_files=data_files, split=f'train[{data_args.validation_split_percentage}%:]', cache_dir=model_args.cache_dir, token=model_args.token)
    config_kwargs = {'cache_dir': model_args.cache_dir, 'revision': model_args.model_revision, 'token': model_args.token}
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = XLNetConfig()
        logger.warning('You are instantiating a new config instance from scratch.')
        if model_args.config_overrides is not None:
            logger.info(f'Overriding config: {model_args.config_overrides}')
            config.update_from_string(model_args.config_overrides)
            logger.info(f'New config: {config}')
    tokenizer_kwargs = {'cache_dir': model_args.cache_dir, 'use_fast': model_args.use_fast_tokenizer, 'revision': model_args.model_revision, 'token': model_args.token}
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError('You are instantiating a new tokenizer from scratch. This is not supported by this script. You can do it from another script, save it, and load it from here, using --tokenizer_name.')
    if model_args.model_name_or_path:
        model = XLNetLMHeadModel.from_pretrained(model_args.model_name_or_path, from_tf=bool('.ckpt' in model_args.model_name_or_path), config=config, cache_dir=model_args.cache_dir, revision=model_args.model_revision, token=model_args.token, low_cpu_mem_usage=model_args.low_cpu_mem_usage)
    else:
        logger.info('Training new model from scratch')
        model = XLNetLMHeadModel(config)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    if training_args.do_train:
        column_names = raw_datasets['train'].column_names
    else:
        column_names = raw_datasets['validation'].column_names
    text_column_name = 'text' if 'text' in column_names else column_names[0]
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(f'The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}.')
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    if data_args.line_by_line:
        padding = 'max_length' if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            if False:
                return 10
            examples['text'] = [line for line in examples['text'] if len(line) > 0 and (not line.isspace())]
            return tokenizer(examples['text'], padding=padding, truncation=True, max_length=max_seq_length)
        with training_args.main_process_first(desc='dataset map tokenization'):
            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, num_proc=data_args.preprocessing_num_workers, remove_columns=[text_column_name], load_from_cache_file=not data_args.overwrite_cache, desc='Running tokenizer on dataset line_by_line')
    else:

        def tokenize_function(examples):
            if False:
                while True:
                    i = 10
            return tokenizer(examples[text_column_name])
        with training_args.main_process_first(desc='dataset map tokenization'):
            tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, num_proc=data_args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not data_args.overwrite_cache, desc='Running tokenizer on every text in dataset')

        def group_texts(examples):
            if False:
                print('Hello World!')
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = total_length // max_seq_length * max_seq_length
            result = {k: [t[i:i + max_seq_length] for i in range(0, total_length, max_seq_length)] for (k, t) in concatenated_examples.items()}
            return result
        with training_args.main_process_first(desc='grouping texts together'):
            tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=data_args.preprocessing_num_workers, load_from_cache_file=not data_args.overwrite_cache, desc=f'Grouping texts in chunks of {max_seq_length}')
    if training_args.do_train:
        if 'train' not in tokenized_datasets:
            raise ValueError('--do_train requires a train dataset')
        train_dataset = tokenized_datasets['train']
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
    if training_args.do_eval:
        if 'validation' not in tokenized_datasets:
            raise ValueError('--do_eval requires a validation dataset')
        eval_dataset = tokenized_datasets['validation']
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
    data_collator = DataCollatorForPermutationLanguageModeling(tokenizer=tokenizer, plm_probability=data_args.plm_probability, max_span_length=data_args.max_span_length)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset if training_args.do_train else None, eval_dataset=eval_dataset if training_args.do_eval else None, tokenizer=tokenizer, data_collator=data_collator)
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
    if training_args.do_eval:
        logger.info('*** Evaluate ***')
        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics['eval_samples'] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics['eval_loss'])
        except OverflowError:
            perplexity = float('inf')
        metrics['perplexity'] = perplexity
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)
    kwargs = {'finetuned_from': model_args.model_name_or_path, 'tasks': 'language-modeling'}
    if data_args.dataset_name is not None:
        kwargs['dataset_tags'] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs['dataset_args'] = data_args.dataset_config_name
            kwargs['dataset'] = f'{data_args.dataset_name} {data_args.dataset_config_name}'
        else:
            kwargs['dataset'] = data_args.dataset_name
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

def _mp_fn(index):
    if False:
        while True:
            i = 10
    main()
if __name__ == '__main__':
    main()