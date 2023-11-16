"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) with whole word masking on a
text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from datasets import Dataset, load_dataset
import transformers
from transformers import CONFIG_MAPPING, MODEL_FOR_MASKED_LM_MAPPING, AutoConfig, AutoModelForMaskedLM, AutoTokenizer, DataCollatorForWholeWordMask, HfArgumentParser, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint, is_main_process
logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple((conf.model_type for conf in MODEL_CONFIG_CLASSES))

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(default=None, metadata={'help': "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."})
    model_type: Optional[str] = field(default=None, metadata={'help': 'If training from scratch, pass a model type from the list: ' + ', '.join(MODEL_TYPES)})
    config_overrides: Optional[str] = field(default=None, metadata={'help': 'Override some existing default config settings when a model is trained from scratch. Example: n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index'})
    config_name: Optional[str] = field(default=None, metadata={'help': 'Pretrained config name or path if not the same as model_name'})
    tokenizer_name: Optional[str] = field(default=None, metadata={'help': 'Pretrained tokenizer name or path if not the same as model_name'})
    cache_dir: Optional[str] = field(default=None, metadata={'help': 'Where do you want to store the pretrained models downloaded from huggingface.co'})
    use_fast_tokenizer: bool = field(default=True, metadata={'help': 'Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.'})
    model_revision: str = field(default='main', metadata={'help': 'The specific model version to use (can be a branch name, tag name or commit id).'})
    use_auth_token: bool = field(default=False, metadata={'help': 'Will use the token generated when running `huggingface-cli login` (necessary to use this script with private models).'})

    def __post_init__(self):
        if False:
            return 10
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
    train_ref_file: Optional[str] = field(default=None, metadata={'help': 'An optional input train ref data file for whole word masking in Chinese.'})
    validation_ref_file: Optional[str] = field(default=None, metadata={'help': 'An optional input validation ref data file for whole word masking in Chinese.'})
    overwrite_cache: bool = field(default=False, metadata={'help': 'Overwrite the cached training and evaluation sets'})
    validation_split_percentage: Optional[int] = field(default=5, metadata={'help': "The percentage of the train set used as validation set in case there's no validation split"})
    max_seq_length: Optional[int] = field(default=None, metadata={'help': 'The maximum total input sequence length after tokenization. Sequences longer than this will be truncated. Default to the max input length of the model.'})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={'help': 'The number of processes to use for the preprocessing.'})
    mlm_probability: float = field(default=0.15, metadata={'help': 'Ratio of tokens to mask for masked language modeling loss'})
    pad_to_max_length: bool = field(default=False, metadata={'help': 'Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when batching to the maximum length in the batch.'})

    def __post_init__(self):
        if False:
            while True:
                i = 10
        if self.train_file is not None:
            extension = self.train_file.split('.')[-1]
            assert extension in ['csv', 'json', 'txt'], '`train_file` should be a csv, a json or a txt file.'
        if self.validation_file is not None:
            extension = self.validation_file.split('.')[-1]
            assert extension in ['csv', 'json', 'txt'], '`validation_file` should be a csv, a json or a txt file.'

def add_chinese_references(dataset, ref_file):
    if False:
        return 10
    with open(ref_file, 'r', encoding='utf-8') as f:
        refs = [json.loads(line) for line in f.read().splitlines() if len(line) > 0 and (not line.isspace())]
    assert len(dataset) == len(refs)
    dataset_dict = {c: dataset[c] for c in dataset.column_names}
    dataset_dict['chinese_ref'] = refs
    return Dataset.from_dict(dataset_dict)

def main():
    if False:
        while True:
            i = 10
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        (model_args, data_args, training_args) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (model_args, data_args, training_args) = parser.parse_args_into_dataclasses()
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and (not training_args.overwrite_output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f'Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.')
        elif last_checkpoint is not None:
            logger.info(f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch.')
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', handlers=[logging.StreamHandler(sys.stdout)])
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    logger.warning(f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}' + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}')
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info('Training/evaluation parameters %s', training_args)
    set_seed(training_args.seed)
    if data_args.dataset_name is not None:
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
        if 'validation' not in datasets.keys():
            datasets['validation'] = load_dataset(data_args.dataset_name, data_args.dataset_config_name, split=f'train[:{data_args.validation_split_percentage}%]')
            datasets['train'] = load_dataset(data_args.dataset_name, data_args.dataset_config_name, split=f'train[{data_args.validation_split_percentage}%:]')
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files['train'] = data_args.train_file
        if data_args.validation_file is not None:
            data_files['validation'] = data_args.validation_file
        extension = data_args.train_file.split('.')[-1]
        if extension == 'txt':
            extension = 'text'
        datasets = load_dataset(extension, data_files=data_files)
    config_kwargs = {'cache_dir': model_args.cache_dir, 'revision': model_args.model_revision, 'use_auth_token': True if model_args.use_auth_token else None}
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning('You are instantiating a new config instance from scratch.')
        if model_args.config_overrides is not None:
            logger.info(f'Overriding config: {model_args.config_overrides}')
            config.update_from_string(model_args.config_overrides)
            logger.info(f'New config: {config}')
    tokenizer_kwargs = {'cache_dir': model_args.cache_dir, 'use_fast': model_args.use_fast_tokenizer, 'revision': model_args.model_revision, 'use_auth_token': True if model_args.use_auth_token else None}
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError('You are instantiating a new tokenizer from scratch. This is not supported by this script. You can do it from another script, save it, and load it from here, using --tokenizer_name.')
    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, from_tf=bool('.ckpt' in model_args.model_name_or_path), config=config, cache_dir=model_args.cache_dir, revision=model_args.model_revision, token=True if model_args.use_auth_token else None)
    else:
        logger.info('Training new model from scratch')
        model = AutoModelForMaskedLM.from_config(config)
    model.resize_token_embeddings(len(tokenizer))
    if training_args.do_train:
        column_names = datasets['train'].column_names
    else:
        column_names = datasets['validation'].column_names
    text_column_name = 'text' if 'text' in column_names else column_names[0]
    padding = 'max_length' if data_args.pad_to_max_length else False

    def tokenize_function(examples):
        if False:
            i = 10
            return i + 15
        examples['text'] = [line for line in examples['text'] if len(line) > 0 and (not line.isspace())]
        return tokenizer(examples['text'], padding=padding, truncation=True, max_length=data_args.max_seq_length)
    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=data_args.preprocessing_num_workers, remove_columns=[text_column_name], load_from_cache_file=not data_args.overwrite_cache)
    if data_args.train_ref_file is not None:
        tokenized_datasets['train'] = add_chinese_references(tokenized_datasets['train'], data_args.train_ref_file)
    if data_args.validation_ref_file is not None:
        tokenized_datasets['validation'] = add_chinese_references(tokenized_datasets['validation'], data_args.validation_ref_file)
    has_ref = data_args.train_ref_file or data_args.validation_ref_file
    if has_ref:
        training_args.remove_unused_columns = False
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability)
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_datasets['train'] if training_args.do_train else None, eval_dataset=tokenized_datasets['validation'] if training_args.do_eval else None, tokenizer=tokenizer, data_collator=data_collator)
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        output_train_file = os.path.join(training_args.output_dir, 'train_results.txt')
        if trainer.is_world_process_zero():
            with open(output_train_file, 'w') as writer:
                logger.info('***** Train results *****')
                for (key, value) in sorted(train_result.metrics.items()):
                    logger.info(f'  {key} = {value}')
                    writer.write(f'{key} = {value}\n')
            trainer.state.save_to_json(os.path.join(training_args.output_dir, 'trainer_state.json'))
    results = {}
    if training_args.do_eval:
        logger.info('*** Evaluate ***')
        eval_output = trainer.evaluate()
        perplexity = math.exp(eval_output['eval_loss'])
        results['perplexity'] = perplexity
        output_eval_file = os.path.join(training_args.output_dir, 'eval_results_mlm_wwm.txt')
        if trainer.is_world_process_zero():
            with open(output_eval_file, 'w') as writer:
                logger.info('***** Eval results *****')
                for (key, value) in sorted(results.items()):
                    logger.info(f'  {key} = {value}')
                    writer.write(f'{key} = {value}\n')
    return results

def _mp_fn(index):
    if False:
        while True:
            i = 10
    main()
if __name__ == '__main__':
    main()