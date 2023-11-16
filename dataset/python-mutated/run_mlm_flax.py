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
import time
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import load_dataset
from flax import jax_utils, traverse_util
from flax.jax_utils import pad_shard_unpad
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard
from huggingface_hub import Repository, create_repo
from tqdm import tqdm
from transformers import CONFIG_MAPPING, FLAX_MODEL_FOR_MASKED_LM_MAPPING, AutoConfig, AutoTokenizer, FlaxAutoModelForMaskedLM, HfArgumentParser, PreTrainedTokenizerBase, TensorType, is_tensorboard_available, set_seed
from transformers.utils import send_example_telemetry
MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_MASKED_LM_MAPPING.keys())
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
    gradient_checkpointing: bool = field(default=False, metadata={'help': 'If True, use gradient checkpointing to save memory at the expense of slower backward pass.'})

    def __post_init__(self):
        if False:
            i = 10
            return i + 15
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

    def to_dict(self):
        if False:
            for i in range(10):
                print('nop')
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
    train_ref_file: Optional[str] = field(default=None, metadata={'help': 'An optional input train ref data file for whole word masking in Chinese.'})
    validation_ref_file: Optional[str] = field(default=None, metadata={'help': 'An optional input validation ref data file for whole word masking in Chinese.'})
    overwrite_cache: bool = field(default=False, metadata={'help': 'Overwrite the cached training and evaluation sets'})
    validation_split_percentage: Optional[int] = field(default=5, metadata={'help': "The percentage of the train set used as validation set in case there's no validation split"})
    max_seq_length: Optional[int] = field(default=None, metadata={'help': 'The maximum total input sequence length after tokenization. Sequences longer than this will be truncated. Default to the max input length of the model.'})
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={'help': 'The number of processes to use for the preprocessing.'})
    mlm_probability: float = field(default=0.15, metadata={'help': 'Ratio of tokens to mask for masked language modeling loss'})
    pad_to_max_length: bool = field(default=False, metadata={'help': 'Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when batching to the maximum length in the batch.'})
    line_by_line: bool = field(default=False, metadata={'help': 'Whether distinct lines of text in the dataset are to be handled as distinct sequences.'})

    def __post_init__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.dataset_name is None and self.train_file is None and (self.validation_file is None):
            raise ValueError('Need either a dataset name or a training/validation file.')
        else:
            if self.train_file is not None:
                extension = self.train_file.split('.')[-1]
                assert extension in ['csv', 'json', 'txt'], '`train_file` should be a csv, a json or a txt file.'
            if self.validation_file is not None:
                extension = self.validation_file.split('.')[-1]
                assert extension in ['csv', 'json', 'txt'], '`validation_file` should be a csv, a json or a txt file.'

@flax.struct.dataclass
class FlaxDataCollatorForLanguageModeling:
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mlm_probability (:obj:`float`, `optional`, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input.

    .. note::

        For best performance, this data collator should be used with a dataset having items that are dictionaries or
        BatchEncoding, with the :obj:`"special_tokens_mask"` key, as returned by a
        :class:`~transformers.PreTrainedTokenizer` or a :class:`~transformers.PreTrainedTokenizerFast` with the
        argument :obj:`return_special_tokens_mask=True`.
    """
    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15

    def __post_init__(self):
        if False:
            i = 10
            return i + 15
        if self.tokenizer.mask_token is None:
            raise ValueError('This tokenizer does not have a mask token which is necessary for masked language modeling. You should pass `mlm=False` to train on causal language modeling instead.')

    def __call__(self, examples: List[Dict[str, np.ndarray]], pad_to_multiple_of: int) -> Dict[str, np.ndarray]:
        if False:
            for i in range(10):
                print('nop')
        batch = self.tokenizer.pad(examples, pad_to_multiple_of=pad_to_multiple_of, return_tensors=TensorType.NUMPY)
        special_tokens_mask = batch.pop('special_tokens_mask', None)
        (batch['input_ids'], batch['labels']) = self.mask_tokens(batch['input_ids'], special_tokens_mask=special_tokens_mask)
        return batch

    def mask_tokens(self, inputs: np.ndarray, special_tokens_mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            i = 10
            return i + 15
        '\n        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.\n        '
        labels = inputs.copy()
        probability_matrix = np.full(labels.shape, self.mlm_probability)
        special_tokens_mask = special_tokens_mask.astype('bool')
        probability_matrix[special_tokens_mask] = 0.0
        masked_indices = np.random.binomial(1, probability_matrix).astype('bool')
        labels[~masked_indices] = -100
        indices_replaced = np.random.binomial(1, np.full(labels.shape, 0.8)).astype('bool') & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        indices_random = np.random.binomial(1, np.full(labels.shape, 0.5)).astype('bool')
        indices_random &= masked_indices & ~indices_replaced
        random_words = np.random.randint(self.tokenizer.vocab_size, size=labels.shape, dtype='i4')
        inputs[indices_random] = random_words[indices_random]
        return (inputs, labels)

def generate_batch_splits(samples_idx: np.ndarray, batch_size: int, drop_last=True) -> np.ndarray:
    if False:
        return 10
    'Generate batches of data for a specified batch size from sample indices. If the dataset size is not divisible by\n    the batch size and `drop_last` is `True`, the last incomplete batch is dropped. Else, it is returned.'
    num_samples = len(samples_idx)
    if drop_last:
        samples_to_remove = num_samples % batch_size
        if samples_to_remove != 0:
            samples_idx = samples_idx[:-samples_to_remove]
        sections_split = num_samples // batch_size
        samples_idx = samples_idx.reshape((sections_split, batch_size))
    else:
        sections_split = math.ceil(num_samples / batch_size)
        samples_idx = np.array_split(samples_idx, sections_split)
    return samples_idx

def write_train_metric(summary_writer, train_metrics, train_time, step):
    if False:
        return 10
    summary_writer.scalar('train_time', train_time, step)
    train_metrics = get_metrics(train_metrics)
    for (key, vals) in train_metrics.items():
        tag = f'train_{key}'
        for (i, val) in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)

def write_eval_metric(summary_writer, eval_metrics, step):
    if False:
        while True:
            i = 10
    for (metric_name, value) in eval_metrics.items():
        summary_writer.scalar(f'eval_{metric_name}', value, step)

def main():
    if False:
        print('Hello World!')
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
    send_example_telemetry('run_mlm', model_args, data_args, framework='flax')
    if os.path.exists(training_args.output_dir) and os.listdir(training_args.output_dir) and training_args.do_train and (not training_args.overwrite_output_dir):
        raise ValueError(f'Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.')
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO, datefmt='[%X]')
    logger = logging.getLogger(__name__)
    logger.info(f'Training/evaluation parameters {training_args}')
    set_seed(training_args.seed)
    if training_args.push_to_hub:
        repo_name = training_args.hub_model_id
        if repo_name is None:
            repo_name = Path(training_args.output_dir).absolute().name
        repo_id = create_repo(repo_name, exist_ok=True, token=training_args.hub_token).repo_id
        repo = Repository(training_args.output_dir, clone_from=repo_id, token=training_args.hub_token)
    if data_args.dataset_name is not None:
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir, token=model_args.token, num_proc=data_args.preprocessing_num_workers)
        if 'validation' not in datasets.keys():
            datasets['validation'] = load_dataset(data_args.dataset_name, data_args.dataset_config_name, split=f'train[:{data_args.validation_split_percentage}%]', cache_dir=model_args.cache_dir, token=model_args.token, num_proc=data_args.preprocessing_num_workers)
            datasets['train'] = load_dataset(data_args.dataset_name, data_args.dataset_config_name, split=f'train[{data_args.validation_split_percentage}%:]', cache_dir=model_args.cache_dir, token=model_args.token, num_proc=data_args.preprocessing_num_workers)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files['train'] = data_args.train_file
        if data_args.validation_file is not None:
            data_files['validation'] = data_args.validation_file
        extension = data_args.train_file.split('.')[-1]
        if extension == 'txt':
            extension = 'text'
        datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir, token=model_args.token, num_proc=data_args.preprocessing_num_workers)
        if 'validation' not in datasets.keys():
            datasets['validation'] = load_dataset(extension, data_files=data_files, split=f'train[:{data_args.validation_split_percentage}%]', cache_dir=model_args.cache_dir, token=model_args.token, num_proc=data_args.preprocessing_num_workers)
            datasets['train'] = load_dataset(extension, data_files=data_files, split=f'train[{data_args.validation_split_percentage}%:]', cache_dir=model_args.cache_dir, token=model_args.token, num_proc=data_args.preprocessing_num_workers)
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
    if training_args.do_train:
        column_names = datasets['train'].column_names
    else:
        column_names = datasets['validation'].column_names
    text_column_name = 'text' if 'text' in column_names else column_names[0]
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    if data_args.line_by_line:
        padding = 'max_length' if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            if False:
                while True:
                    i = 10
            examples = [line for line in examples if len(line) > 0 and (not line.isspace())]
            return tokenizer(examples, return_special_tokens_mask=True, padding=padding, truncation=True, max_length=max_seq_length)
        tokenized_datasets = datasets.map(tokenize_function, input_columns=[text_column_name], batched=True, num_proc=data_args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not data_args.overwrite_cache)
    else:

        def tokenize_function(examples):
            if False:
                return 10
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)
        tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=data_args.preprocessing_num_workers, remove_columns=column_names, load_from_cache_file=not data_args.overwrite_cache)

        def group_texts(examples):
            if False:
                for i in range(10):
                    print('nop')
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length >= max_seq_length:
                total_length = total_length // max_seq_length * max_seq_length
            result = {k: [t[i:i + max_seq_length] for i in range(0, total_length, max_seq_length)] for (k, t) in concatenated_examples.items()}
            return result
        tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=data_args.preprocessing_num_workers, load_from_cache_file=not data_args.overwrite_cache)
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
    data_collator = FlaxDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability)
    rng = jax.random.PRNGKey(training_args.seed)
    dropout_rngs = jax.random.split(rng, jax.local_device_count())
    if model_args.model_name_or_path:
        model = FlaxAutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, config=config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype), token=model_args.token, trust_remote_code=model_args.trust_remote_code)
    else:
        model = FlaxAutoModelForMaskedLM.from_config(config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype), trust_remote_code=model_args.trust_remote_code)
    if training_args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
    eval_batch_size = per_device_eval_batch_size * jax.device_count()
    num_train_steps = len(tokenized_datasets['train']) // train_batch_size * num_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=training_args.learning_rate, transition_steps=training_args.warmup_steps)
    decay_fn = optax.linear_schedule(init_value=training_args.learning_rate, end_value=0, transition_steps=num_train_steps - training_args.warmup_steps)
    linear_decay_lr_schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[training_args.warmup_steps])

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
    state = train_state.TrainState.create(apply_fn=model.__call__, params=model.params, tx=optimizer)

    def train_step(state, batch, dropout_rng):
        if False:
            i = 10
            return i + 15
        (dropout_rng, new_dropout_rng) = jax.random.split(dropout_rng)

        def loss_fn(params):
            if False:
                i = 10
                return i + 15
            labels = batch.pop('labels')
            logits = state.apply_fn(**batch, params=params, dropout_rng=dropout_rng, train=True)[0]
            label_mask = jnp.where(labels > 0, 1.0, 0.0)
            loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1])) * label_mask
            loss = loss.sum()
            num_labels = label_mask.sum()
            return (loss, num_labels)
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        ((loss, num_labels), grad) = grad_fn(state.params)
        num_labels = jax.lax.psum(num_labels, 'batch')
        loss = jax.lax.psum(loss, 'batch')
        loss = jax.tree_util.tree_map(lambda x: x / num_labels, loss)
        grad = jax.lax.psum(grad, 'batch')
        grad = jax.tree_util.tree_map(lambda x: x / num_labels, grad)
        new_state = state.apply_gradients(grads=grad)
        metrics = {'loss': loss, 'learning_rate': linear_decay_lr_schedule_fn(state.step)}
        return (new_state, metrics, new_dropout_rng)
    p_train_step = jax.pmap(train_step, 'batch', donate_argnums=(0,))

    def eval_step(params, batch):
        if False:
            i = 10
            return i + 15
        labels = batch.pop('labels')
        logits = model(**batch, params=params, train=False)[0]
        label_mask = jnp.where(labels > 0, 1.0, 0.0)
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1])) * label_mask
        accuracy = jnp.equal(jnp.argmax(logits, axis=-1), labels) * label_mask
        metrics = {'loss': loss.sum(), 'accuracy': accuracy.sum(), 'normalizer': label_mask.sum()}
        metrics = jax.lax.psum(metrics, axis_name='batch')
        return metrics
    p_eval_step = jax.pmap(eval_step, 'batch', donate_argnums=(0,))
    state = jax_utils.replicate(state)
    train_time = 0
    epochs = tqdm(range(num_epochs), desc=f'Epoch ... (1/{num_epochs})', position=0)
    for epoch in epochs:
        train_start = time.time()
        train_metrics = []
        (rng, input_rng) = jax.random.split(rng)
        num_train_samples = len(tokenized_datasets['train'])
        train_samples_idx = np.random.permutation(np.arange(num_train_samples))
        train_batch_idx = generate_batch_splits(train_samples_idx, train_batch_size)
        for (step, batch_idx) in enumerate(tqdm(train_batch_idx, desc='Training...', position=1)):
            samples = [tokenized_datasets['train'][int(idx)] for idx in batch_idx]
            model_inputs = data_collator(samples, pad_to_multiple_of=16)
            model_inputs = shard(model_inputs.data)
            (state, train_metric, dropout_rngs) = p_train_step(state, model_inputs, dropout_rngs)
            train_metrics.append(train_metric)
            cur_step = epoch * (num_train_samples // train_batch_size) + step
            if cur_step % training_args.logging_steps == 0 and cur_step > 0:
                train_metric = jax_utils.unreplicate(train_metric)
                train_time += time.time() - train_start
                if has_tensorboard and jax.process_index() == 0:
                    write_train_metric(summary_writer, train_metrics, train_time, cur_step)
                epochs.write(f"Step... ({cur_step} | Loss: {train_metric['loss']}, Learning Rate: {train_metric['learning_rate']})")
                train_metrics = []
            if cur_step % training_args.eval_steps == 0 and cur_step > 0:
                num_eval_samples = len(tokenized_datasets['validation'])
                eval_samples_idx = np.arange(num_eval_samples)
                eval_batch_idx = generate_batch_splits(eval_samples_idx, eval_batch_size, drop_last=False)
                eval_metrics = []
                for (i, batch_idx) in enumerate(tqdm(eval_batch_idx, desc='Evaluating ...', position=2)):
                    samples = [tokenized_datasets['validation'][int(idx)] for idx in batch_idx]
                    model_inputs = data_collator(samples, pad_to_multiple_of=16)
                    metrics = pad_shard_unpad(p_eval_step, static_return=True)(state.params, model_inputs.data, min_device_batch=per_device_eval_batch_size)
                    eval_metrics.append(metrics)
                eval_metrics = get_metrics(eval_metrics)
                eval_metrics = jax.tree_util.tree_map(jnp.sum, eval_metrics)
                eval_normalizer = eval_metrics.pop('normalizer')
                eval_metrics = jax.tree_util.tree_map(lambda x: x / eval_normalizer, eval_metrics)
                epochs.desc = f"Step... ({cur_step} | Loss: {eval_metrics['loss']}, Acc: {eval_metrics['accuracy']})"
                if has_tensorboard and jax.process_index() == 0:
                    write_eval_metric(summary_writer, eval_metrics, cur_step)
            if cur_step % training_args.save_steps == 0 and cur_step > 0:
                if jax.process_index() == 0:
                    params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params))
                    model.save_pretrained(training_args.output_dir, params=params)
                    tokenizer.save_pretrained(training_args.output_dir)
                    if training_args.push_to_hub:
                        repo.push_to_hub(commit_message=f'Saving weights and logs of step {cur_step}', blocking=False)
    if training_args.do_eval:
        num_eval_samples = len(tokenized_datasets['validation'])
        eval_samples_idx = np.arange(num_eval_samples)
        eval_batch_idx = generate_batch_splits(eval_samples_idx, eval_batch_size, drop_last=False)
        eval_metrics = []
        for (_, batch_idx) in enumerate(tqdm(eval_batch_idx, desc='Evaluating ...', position=2)):
            samples = [tokenized_datasets['validation'][int(idx)] for idx in batch_idx]
            model_inputs = data_collator(samples, pad_to_multiple_of=16)
            metrics = pad_shard_unpad(p_eval_step, static_return=True)(state.params, model_inputs.data, min_device_batch=per_device_eval_batch_size)
            eval_metrics.append(metrics)
        eval_metrics = get_metrics(eval_metrics)
        eval_metrics = jax.tree_util.tree_map(lambda metric: jnp.sum(metric).item(), eval_metrics)
        eval_normalizer = eval_metrics.pop('normalizer')
        eval_metrics = jax.tree_util.tree_map(lambda x: x / eval_normalizer, eval_metrics)
        try:
            perplexity = math.exp(eval_metrics['loss'])
        except OverflowError:
            perplexity = float('inf')
        eval_metrics['perplexity'] = perplexity
        if jax.process_index() == 0:
            eval_metrics = {f'eval_{metric_name}': value for (metric_name, value) in eval_metrics.items()}
            path = os.path.join(training_args.output_dir, 'eval_results.json')
            with open(path, 'w') as f:
                json.dump(eval_metrics, f, indent=4, sort_keys=True)
if __name__ == '__main__':
    main()