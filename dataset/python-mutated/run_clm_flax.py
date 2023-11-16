"""
Pre-training/Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
import json
import logging
import math
import os
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Callable, Optional
import datasets
import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import Dataset, load_dataset
from flax import jax_utils, traverse_util
from flax.jax_utils import pad_shard_unpad, unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from huggingface_hub import Repository, create_repo
from tqdm import tqdm
import transformers
from transformers import CONFIG_MAPPING, FLAX_MODEL_FOR_CAUSAL_LM_MAPPING, AutoConfig, AutoTokenizer, FlaxAutoModelForCausalLM, HfArgumentParser, is_tensorboard_available, set_seed
from transformers.testing_utils import CaptureLogger
from transformers.utils import send_example_telemetry
logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple((conf.model_type for conf in MODEL_CONFIG_CLASSES))

@dataclass
class TrainingArguments:
    output_dir: str = field(metadata={'help': 'The output directory where the model predictions and checkpoints will be written.'})
    overwrite_output_dir: bool = field(default=False, metadata={'help': 'Overwrite the content of the output directory. Use this to continue training if output_dir points to a checkpoint directory.'})
    do_train: bool = field(default=False, metadata={'help': 'Whether to run training.'})
    do_eval: bool = field(default=False, metadata={'help': 'Whether to run eval on the dev set.'})
    per_device_train_batch_size: int = field(default=8, metadata={'help': 'Batch size per GPU/TPU core/CPU for training.'})
    per_device_eval_batch_size: int = field(default=8, metadata={'help': 'Batch size per GPU/TPU core/CPU for evaluation.'})
    learning_rate: float = field(default=5e-05, metadata={'help': 'The initial learning rate for AdamW.'})
    weight_decay: float = field(default=0.0, metadata={'help': 'Weight decay for AdamW if we apply some.'})
    adam_beta1: float = field(default=0.9, metadata={'help': 'Beta1 for AdamW optimizer'})
    adam_beta2: float = field(default=0.999, metadata={'help': 'Beta2 for AdamW optimizer'})
    adam_epsilon: float = field(default=1e-08, metadata={'help': 'Epsilon for AdamW optimizer.'})
    adafactor: bool = field(default=False, metadata={'help': 'Whether or not to replace AdamW by Adafactor.'})
    num_train_epochs: float = field(default=3.0, metadata={'help': 'Total number of training epochs to perform.'})
    warmup_steps: int = field(default=0, metadata={'help': 'Linear warmup over warmup_steps.'})
    logging_steps: int = field(default=500, metadata={'help': 'Log every X updates steps.'})
    save_steps: int = field(default=500, metadata={'help': 'Save checkpoint every X updates steps.'})
    eval_steps: int = field(default=None, metadata={'help': 'Run an evaluation every X steps.'})
    seed: int = field(default=42, metadata={'help': 'Random seed that will be set at the beginning of training.'})
    push_to_hub: bool = field(default=False, metadata={'help': 'Whether or not to upload the trained model to the model hub after training.'})
    hub_model_id: str = field(default=None, metadata={'help': 'The name of the repository to keep in sync with the local `output_dir`.'})
    hub_token: str = field(default=None, metadata={'help': 'The token to use to push to the Model Hub.'})

    def __post_init__(self):
        if False:
            print('Hello World!')
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

    def to_dict(self):
        if False:
            while True:
                i = 10
        '\n        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates\n        the token values by removing their value.\n        '
        d = asdict(self)
        for (k, v) in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith('_token'):
                d[k] = f'<{k.upper()}>'
        return d

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(default=None, metadata={'help': "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."})
    model_type: Optional[str] = field(default=None, metadata={'help': 'If training from scratch, pass a model type from the list: ' + ', '.join(MODEL_TYPES)})
    config_name: Optional[str] = field(default=None, metadata={'help': 'Pretrained config name or path if not the same as model_name'})
    tokenizer_name: Optional[str] = field(default=None, metadata={'help': 'Pretrained tokenizer name or path if not the same as model_name'})
    cache_dir: Optional[str] = field(default=None, metadata={'help': 'Where do you want to store the pretrained models downloaded from s3'})
    use_fast_tokenizer: bool = field(default=True, metadata={'help': 'Whether to use one of the fast tokenizer (backed by the tokenizers library) or not.'})
    dtype: Optional[str] = field(default='float32', metadata={'help': 'Floating-point format in which the model weights should be initialized and trained. Choose one of `[float32, float16, bfloat16]`.'})
    token: str = field(default=None, metadata={'help': 'The token to use as HTTP bearer authorization for remote files. If not specified, will use the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).'})
    use_auth_token: bool = field(default=None, metadata={'help': 'The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.'})
    trust_remote_code: bool = field(default=False, metadata={'help': 'Whether or not to allow for custom models defined on the Hub in their own modeling files. This optionshould only be set to `True` for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.'})

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(default=None, metadata={'help': 'The name of the dataset to use (via the datasets library).'})
    dataset_config_name: Optional[str] = field(default=None, metadata={'help': 'The configuration name of the dataset to use (via the datasets library).'})
    train_file: Optional[str] = field(default=None, metadata={'help': 'The input training data file (a text file).'})
    validation_file: Optional[str] = field(default=None, metadata={'help': 'An optional input evaluation data file to evaluate the perplexity on (a text file).'})
    max_train_samples: Optional[int] = field(default=None, metadata={'help': 'For debugging purposes or quicker training, truncate the number of training examples to this value if set.'})
    max_eval_samples: Optional[int] = field(default=None, metadata={'help': 'For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set.'})
    overwrite_cache: bool = field(default=False, metadata={'help': 'Overwrite the cached training and evaluation sets'})
    validation_split_percentage: Optional[int] = field(default=5, metadata={'help': "The percentage of the train set used as validation set in case there's no validation split"})
    block_size: Optional[int] = field(default=None, metadata={'help': 'Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).'})
    overwrite_cache: bool = field(default=False, metadata={'help': 'Overwrite the cached training and evaluation sets'})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={'help': 'The number of processes to use for the preprocessing.'})
    keep_linebreaks: bool = field(default=True, metadata={'help': 'Whether to keep line breaks when using TXT files or not.'})

    def __post_init__(self):
        if False:
            print('Hello World!')
        if self.dataset_name is None and self.train_file is None and (self.validation_file is None):
            raise ValueError('Need either a dataset name or a training/validation file.')
        else:
            if self.train_file is not None:
                extension = self.train_file.split('.')[-1]
                if extension not in ['csv', 'json', 'txt']:
                    raise ValueError('train_file` should be a csv, json or text file.')
            if self.validation_file is not None:
                extension = self.validation_file.split('.')[-1]
                if extension not in ['csv', 'json', 'txt']:
                    raise ValueError('`validation_file` should be a csv, json or text file.')

class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

    def replicate(self):
        if False:
            i = 10
            return i + 15
        return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))

def data_loader(rng: jax.random.PRNGKey, dataset: Dataset, batch_size: int, shuffle: bool=False, drop_last=True):
    if False:
        return 10
    '\n    Returns batches of size `batch_size` from `dataset`. If `drop_last` is set to `False`, the final batch may be incomplete,\n    and range in size from 1 to `batch_size`. Shuffle batches if `shuffle` is `True`.\n    '
    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
        batch_idx = np.asarray(batch_idx)
    else:
        batch_idx = np.arange(len(dataset))
    if drop_last:
        steps_per_epoch = len(dataset) // batch_size
        batch_idx = batch_idx[:steps_per_epoch * batch_size]
        batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))
    else:
        steps_per_epoch = math.ceil(len(dataset) / batch_size)
        batch_idx = np.array_split(batch_idx, steps_per_epoch)
    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: np.array(v) for (k, v) in batch.items()}
        yield batch

def write_train_metric(summary_writer, train_metrics, train_time, step):
    if False:
        print('Hello World!')
    summary_writer.scalar('train_time', train_time, step)
    train_metrics = get_metrics(train_metrics)
    for (key, vals) in train_metrics.items():
        tag = f'train_{key}'
        for (i, val) in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)

def write_eval_metric(summary_writer, eval_metrics, step):
    if False:
        i = 10
        return i + 15
    for (metric_name, value) in eval_metrics.items():
        summary_writer.scalar(f'eval_{metric_name}', value, step)

def create_learning_rate_fn(train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int, learning_rate: float) -> Callable[[int], jnp.ndarray]:
    if False:
        while True:
            i = 10
    'Returns a linear warmup, linear_decay learning rate function.'
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn

def main():
    if False:
        while True:
            i = 10
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
    send_example_telemetry('run_clm', model_args, data_args, framework='flax')
    if os.path.exists(training_args.output_dir) and os.listdir(training_args.output_dir) and training_args.do_train and (not training_args.overwrite_output_dir):
        raise ValueError(f'Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.')
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    logger.info(f'Training/evaluation parameters {training_args}')
    set_seed(training_args.seed)
    if training_args.push_to_hub:
        repo_name = training_args.hub_model_id
        if repo_name is None:
            repo_name = Path(training_args.output_dir).absolute().name
        repo_id = create_repo(repo_name, exist_ok=True, token=training_args.hub_token).repo_id
        repo = Repository(training_args.output_dir, clone_from=repo_id, token=training_args.hub_token)
    if data_args.dataset_name is not None:
        dataset = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir, keep_in_memory=False, token=model_args.token, num_proc=data_args.preprocessing_num_workers)
        if 'validation' not in dataset.keys():
            dataset['validation'] = load_dataset(data_args.dataset_name, data_args.dataset_config_name, split=f'train[:{data_args.validation_split_percentage}%]', cache_dir=model_args.cache_dir, token=model_args.token, num_proc=data_args.preprocessing_num_workers)
            dataset['train'] = load_dataset(data_args.dataset_name, data_args.dataset_config_name, split=f'train[{data_args.validation_split_percentage}%:]', cache_dir=model_args.cache_dir, token=model_args.token, num_proc=data_args.preprocessing_num_workers)
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files['train'] = data_args.train_file
        if data_args.validation_file is not None:
            data_files['validation'] = data_args.validation_file
        extension = data_args.train_file.split('.')[-1]
        if extension == 'txt':
            extension = 'text'
            dataset_args['keep_linebreaks'] = data_args.keep_linebreaks
        dataset = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir, **dataset_args, token=model_args.token, num_proc=data_args.preprocessing_num_workers)
        if 'validation' not in dataset.keys():
            dataset['validation'] = load_dataset(extension, data_files=data_files, split=f'train[:{data_args.validation_split_percentage}%]', cache_dir=model_args.cache_dir, **dataset_args, token=model_args.token, num_proc=data_args.preprocessing_num_workers)
            dataset['train'] = load_dataset(extension, data_files=data_files, split=f'train[{data_args.validation_split_percentage}%:]', cache_dir=model_args.cache_dir, **dataset_args, token=model_args.token, num_proc=data_args.preprocessing_num_workers)
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir, token=model_args.token, trust_remote_code=model_args.trust_remote_code)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir, token=model_args.token, trust_remote_code=model_args.trust_remote_code)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning('You are instantiating a new config instance from scratch.')
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer, token=model_args.token, trust_remote_code=model_args.trust_remote_code)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer, token=model_args.token, trust_remote_code=model_args.trust_remote_code)
    else:
        raise ValueError('You are instantiating a new tokenizer from scratch. This is not supported by this script. You can do it from another script, save it, and load it from here, using --tokenizer_name.')
    if model_args.model_name_or_path:
        model = FlaxAutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype), token=model_args.token, trust_remote_code=model_args.trust_remote_code)
    else:
        model = FlaxAutoModelForCausalLM.from_config(config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype), trust_remote_code=model_args.trust_remote_code)
    if training_args.do_train:
        column_names = dataset['train'].column_names
    else:
        column_names = dataset['validation'].column_names
    text_column_name = 'text' if 'text' in column_names else column_names[0]
    tok_logger = transformers.utils.logging.get_logger('transformers.tokenization_utils_base')

    def tokenize_function(examples):
        if False:
            while True:
                i = 10
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        if 'Token indices sequence length is longer than the' in cl.out:
            tok_logger.warning('^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model.')
        return output
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=data_args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not data_args.overwrite_cache)
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            logger.warning(f'The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx.')
            block_size = min(1024, config.max_position_embeddings)
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(f'The block_size passed ({data_args.block_size}) is larger than the maximum length for the model ({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}.')
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    def group_texts(examples):
        if False:
            print('Hello World!')
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = total_length // block_size * block_size
        result = {k: [t[i:i + block_size] for i in range(0, total_length, block_size)] for (k, t) in concatenated_examples.items()}
        result['labels'] = result['input_ids'].copy()
        return result
    lm_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=data_args.preprocessing_num_workers, load_from_cache_file=not data_args.overwrite_cache)
    if training_args.do_train:
        if 'train' not in tokenized_datasets:
            raise ValueError('--do_train requires a train dataset')
        train_dataset = lm_datasets['train']
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
    if training_args.do_eval:
        if 'validation' not in tokenized_datasets:
            raise ValueError('--do_eval requires a validation dataset')
        eval_dataset = lm_datasets['validation']
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter
            summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir))
        except ImportError as ie:
            has_tensorboard = False
            logger.warning(f'Unable to display metrics through TensorBoard because some package are not installed: {ie}')
    else:
        logger.warning('Unable to display metrics through TensorBoard because the package is not installed: Please run pip install tensorboard to enable.')
    rng = jax.random.PRNGKey(training_args.seed)
    (rng, dropout_rng) = jax.random.split(rng)
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
    eval_batch_size = per_device_eval_batch_size * jax.device_count()
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs
    linear_decay_lr_schedule_fn = create_learning_rate_fn(len(train_dataset), train_batch_size, training_args.num_train_epochs, training_args.warmup_steps, training_args.learning_rate)

    def decay_mask_fn(params):
        if False:
            return 10
        flat_params = traverse_util.flatten_dict(params)
        layer_norm_candidates = ['layernorm', 'layer_norm', 'ln']
        layer_norm_named_params = {layer[-2:] for layer_norm_name in layer_norm_candidates for layer in flat_params.keys() if layer_norm_name in ''.join(layer).lower()}
        flat_mask = {path: path[-1] != 'bias' and path[-2:] not in layer_norm_named_params for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)
    if training_args.adafactor:
        optimizer = optax.adafactor(learning_rate=linear_decay_lr_schedule_fn)
    else:
        optimizer = optax.adamw(learning_rate=linear_decay_lr_schedule_fn, b1=training_args.adam_beta1, b2=training_args.adam_beta2, eps=training_args.adam_epsilon, weight_decay=training_args.weight_decay, mask=decay_mask_fn)
    state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=optimizer, dropout_rng=dropout_rng)

    def loss_fn(logits, labels):
        if False:
            while True:
                i = 10
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        loss = optax.softmax_cross_entropy(shift_logits, onehot(shift_labels, shift_logits.shape[-1]))
        return loss.mean()

    def train_step(state, batch):
        if False:
            for i in range(10):
                print('nop')
        (dropout_rng, new_dropout_rng) = jax.random.split(state.dropout_rng)

        def compute_loss(params):
            if False:
                i = 10
                return i + 15
            labels = batch.pop('labels')
            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
            loss = loss_fn(logits, labels)
            return loss
        grad_fn = jax.value_and_grad(compute_loss)
        (loss, grad) = grad_fn(state.params)
        grad = jax.lax.pmean(grad, 'batch')
        new_state = state.apply_gradients(grads=grad, dropout_rng=new_dropout_rng)
        metrics = {'loss': loss, 'learning_rate': linear_decay_lr_schedule_fn(state.step)}
        metrics = jax.lax.pmean(metrics, axis_name='batch')
        return (new_state, metrics)

    def eval_step(params, batch):
        if False:
            i = 10
            return i + 15
        labels = batch.pop('labels')
        logits = model(**batch, params=params, train=False)[0]
        loss = loss_fn(logits, labels)
        metrics = {'loss': loss}
        metrics = jax.lax.pmean(metrics, axis_name='batch')
        return metrics
    p_train_step = jax.pmap(train_step, 'batch', donate_argnums=(0,))
    p_eval_step = jax.pmap(eval_step, 'batch')
    state = state.replicate()
    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {len(train_dataset)}')
    logger.info(f'  Num Epochs = {num_epochs}')
    logger.info(f'  Instantaneous batch size per device = {training_args.per_device_train_batch_size}')
    logger.info(f'  Total train batch size (w. parallel & distributed) = {train_batch_size}')
    logger.info(f'  Total optimization steps = {total_train_steps}')
    train_time = 0
    train_metrics = []
    epochs = tqdm(range(num_epochs), desc='Epoch ... ', position=0)
    for epoch in epochs:
        train_start = time.time()
        (rng, input_rng) = jax.random.split(rng)
        train_loader = data_loader(input_rng, train_dataset, train_batch_size, shuffle=True)
        steps_per_epoch = len(train_dataset) // train_batch_size
        for step in tqdm(range(steps_per_epoch), desc='Training...', position=1, leave=False):
            batch = next(train_loader)
            batch = shard(batch)
            (state, train_metric) = p_train_step(state, batch)
            train_metrics.append(train_metric)
            cur_step = epoch * (len(train_dataset) // train_batch_size) + step
            if cur_step % training_args.logging_steps == 0 and cur_step > 0:
                train_metric = unreplicate(train_metric)
                train_time += time.time() - train_start
                if has_tensorboard and jax.process_index() == 0:
                    write_train_metric(summary_writer, train_metrics, train_time, cur_step)
                epochs.write(f"Step... ({cur_step} | Loss: {train_metric['loss'].mean()}, Learning Rate: {train_metric['learning_rate'].mean()})")
                train_metrics = []
            if cur_step % training_args.eval_steps == 0 and cur_step > 0:
                eval_metrics = []
                eval_loader = data_loader(input_rng, eval_dataset, eval_batch_size, drop_last=False)
                eval_steps = math.ceil(len(eval_dataset) / eval_batch_size)
                for _ in tqdm(range(eval_steps), desc='Evaluating...', position=2, leave=False):
                    batch = next(eval_loader)
                    metrics = pad_shard_unpad(p_eval_step, static_return=True)(state.params, batch, min_device_batch=per_device_eval_batch_size)
                    eval_metrics.append(metrics)
                eval_metrics = get_metrics(eval_metrics)
                eval_metrics = jax.tree_util.tree_map(jnp.mean, eval_metrics)
                try:
                    eval_metrics['perplexity'] = math.exp(eval_metrics['loss'])
                except OverflowError:
                    eval_metrics['perplexity'] = float('inf')
                desc = f"Step... ({cur_step} | Eval Loss: {eval_metrics['loss']} | Eval Perplexity: {eval_metrics['perplexity']})"
                epochs.write(desc)
                epochs.desc = desc
                if has_tensorboard and jax.process_index() == 0:
                    write_eval_metric(summary_writer, eval_metrics, cur_step)
            if cur_step % training_args.save_steps == 0 and cur_step > 0:
                if jax.process_index() == 0:
                    params = jax.device_get(unreplicate(state.params))
                    model.save_pretrained(training_args.output_dir, params=params)
                    tokenizer.save_pretrained(training_args.output_dir)
                    if training_args.push_to_hub:
                        repo.push_to_hub(commit_message=f'Saving weights and logs of step {cur_step}', blocking=False)
    if training_args.do_eval:
        eval_metrics = []
        eval_loader = data_loader(input_rng, eval_dataset, eval_batch_size, drop_last=False)
        eval_steps = math.ceil(len(eval_dataset) / eval_batch_size)
        for _ in tqdm(range(eval_steps), desc='Evaluating...', position=2, leave=False):
            batch = next(eval_loader)
            metrics = pad_shard_unpad(p_eval_step, static_return=True)(state.params, batch, min_device_batch=per_device_eval_batch_size)
            eval_metrics.append(metrics)
        eval_metrics = get_metrics(eval_metrics)
        eval_metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x).item(), eval_metrics)
        try:
            eval_metrics['perplexity'] = math.exp(eval_metrics['loss'])
        except OverflowError:
            eval_metrics['perplexity'] = float('inf')
        if jax.process_index() == 0:
            eval_metrics = {f'eval_{metric_name}': value for (metric_name, value) in eval_metrics.items()}
            path = os.path.join(training_args.output_dir, 'eval_results.json')
            with open(path, 'w') as f:
                json.dump(eval_metrics, f, indent=4, sort_keys=True)
if __name__ == '__main__':
    main()