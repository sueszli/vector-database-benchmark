import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional
import evaluate
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Lambda, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor
import transformers
from transformers import MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING, AutoConfig, AutoImageProcessor, AutoModelForImageClassification, HfArgumentParser, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
' Fine-tuning a 🤗 Transformers model for image classification'
logger = logging.getLogger(__name__)
check_min_version('4.36.0.dev0')
require_version('datasets>=1.8.0', 'To fix: pip install -r examples/pytorch/image-classification/requirements.txt')
MODEL_CONFIG_CLASSES = list(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING.keys())
MODEL_TYPES = tuple((conf.model_type for conf in MODEL_CONFIG_CLASSES))

def pil_loader(path: str):
    if False:
        while True:
            i = 10
    with open(path, 'rb') as f:
        im = Image.open(f)
        return im.convert('RGB')

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify
    them on the command line.
    """
    dataset_name: Optional[str] = field(default=None, metadata={'help': 'Name of a dataset from the hub (could be your own, possibly private dataset hosted on the hub).'})
    dataset_config_name: Optional[str] = field(default=None, metadata={'help': 'The configuration name of the dataset to use (via the datasets library).'})
    train_dir: Optional[str] = field(default=None, metadata={'help': 'A folder containing the training data.'})
    validation_dir: Optional[str] = field(default=None, metadata={'help': 'A folder containing the validation data.'})
    train_val_split: Optional[float] = field(default=0.15, metadata={'help': 'Percent to split off of train for validation.'})
    max_train_samples: Optional[int] = field(default=None, metadata={'help': 'For debugging purposes or quicker training, truncate the number of training examples to this value if set.'})
    max_eval_samples: Optional[int] = field(default=None, metadata={'help': 'For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set.'})

    def __post_init__(self):
        if False:
            print('Hello World!')
        if self.dataset_name is None and (self.train_dir is None and self.validation_dir is None):
            raise ValueError('You must specify either a dataset name from the hub or a train and/or validation directory.')

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(default='google/vit-base-patch16-224-in21k', metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'})
    model_type: Optional[str] = field(default=None, metadata={'help': 'If training from scratch, pass a model type from the list: ' + ', '.join(MODEL_TYPES)})
    config_name: Optional[str] = field(default=None, metadata={'help': 'Pretrained config name or path if not the same as model_name'})
    cache_dir: Optional[str] = field(default=None, metadata={'help': 'Where do you want to store the pretrained models downloaded from s3'})
    model_revision: str = field(default='main', metadata={'help': 'The specific model version to use (can be a branch name, tag name or commit id).'})
    image_processor_name: str = field(default=None, metadata={'help': 'Name or path of preprocessor config.'})
    token: str = field(default=None, metadata={'help': 'The token to use as HTTP bearer authorization for remote files. If not specified, will use the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).'})
    use_auth_token: bool = field(default=None, metadata={'help': 'The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.'})
    trust_remote_code: bool = field(default=False, metadata={'help': 'Whether or not to allow for custom models defined on the Hub in their own modeling files. This optionshould only be set to `True` for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.'})
    ignore_mismatched_sizes: bool = field(default=False, metadata={'help': 'Will enable to load a pretrained model whose head dimensions are different.'})

def collate_fn(examples):
    if False:
        for i in range(10):
            print('nop')
    pixel_values = torch.stack([example['pixel_values'] for example in examples])
    labels = torch.tensor([example['labels'] for example in examples])
    return {'pixel_values': pixel_values, 'labels': labels}

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
    send_example_telemetry('run_image_classification', model_args, data_args)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', handlers=[logging.StreamHandler(sys.stdout)])
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
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
        dataset = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir, task='image-classification', token=model_args.token)
    else:
        data_files = {}
        if data_args.train_dir is not None:
            data_files['train'] = os.path.join(data_args.train_dir, '**')
        if data_args.validation_dir is not None:
            data_files['validation'] = os.path.join(data_args.validation_dir, '**')
        dataset = load_dataset('imagefolder', data_files=data_files, cache_dir=model_args.cache_dir, task='image-classification')
    data_args.train_val_split = None if 'validation' in dataset.keys() else data_args.train_val_split
    if isinstance(data_args.train_val_split, float) and data_args.train_val_split > 0.0:
        split = dataset['train'].train_test_split(data_args.train_val_split)
        dataset['train'] = split['train']
        dataset['validation'] = split['test']
    labels = dataset['train'].features['labels'].names
    (label2id, id2label) = ({}, {})
    for (i, label) in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    metric = evaluate.load('accuracy')

    def compute_metrics(p):
        if False:
            for i in range(10):
                print('nop')
        'Computes accuracy on a batch of predictions'
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
    config = AutoConfig.from_pretrained(model_args.config_name or model_args.model_name_or_path, num_labels=len(labels), label2id=label2id, id2label=id2label, finetuning_task='image-classification', cache_dir=model_args.cache_dir, revision=model_args.model_revision, token=model_args.token, trust_remote_code=model_args.trust_remote_code)
    model = AutoModelForImageClassification.from_pretrained(model_args.model_name_or_path, from_tf=bool('.ckpt' in model_args.model_name_or_path), config=config, cache_dir=model_args.cache_dir, revision=model_args.model_revision, token=model_args.token, trust_remote_code=model_args.trust_remote_code, ignore_mismatched_sizes=model_args.ignore_mismatched_sizes)
    image_processor = AutoImageProcessor.from_pretrained(model_args.image_processor_name or model_args.model_name_or_path, cache_dir=model_args.cache_dir, revision=model_args.model_revision, token=model_args.token, trust_remote_code=model_args.trust_remote_code)
    if 'shortest_edge' in image_processor.size:
        size = image_processor.size['shortest_edge']
    else:
        size = (image_processor.size['height'], image_processor.size['width'])
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std) if hasattr(image_processor, 'image_mean') and hasattr(image_processor, 'image_std') else Lambda(lambda x: x)
    _train_transforms = Compose([RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize])
    _val_transforms = Compose([Resize(size), CenterCrop(size), ToTensor(), normalize])

    def train_transforms(example_batch):
        if False:
            i = 10
            return i + 15
        'Apply _train_transforms across a batch.'
        example_batch['pixel_values'] = [_train_transforms(pil_img.convert('RGB')) for pil_img in example_batch['image']]
        return example_batch

    def val_transforms(example_batch):
        if False:
            while True:
                i = 10
        'Apply _val_transforms across a batch.'
        example_batch['pixel_values'] = [_val_transforms(pil_img.convert('RGB')) for pil_img in example_batch['image']]
        return example_batch
    if training_args.do_train:
        if 'train' not in dataset:
            raise ValueError('--do_train requires a train dataset')
        if data_args.max_train_samples is not None:
            dataset['train'] = dataset['train'].shuffle(seed=training_args.seed).select(range(data_args.max_train_samples))
        dataset['train'].set_transform(train_transforms)
    if training_args.do_eval:
        if 'validation' not in dataset:
            raise ValueError('--do_eval requires a validation dataset')
        if data_args.max_eval_samples is not None:
            dataset['validation'] = dataset['validation'].shuffle(seed=training_args.seed).select(range(data_args.max_eval_samples))
        dataset['validation'].set_transform(val_transforms)
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset['train'] if training_args.do_train else None, eval_dataset=dataset['validation'] if training_args.do_eval else None, compute_metrics=compute_metrics, tokenizer=image_processor, data_collator=collate_fn)
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics('train', train_result.metrics)
        trainer.save_metrics('train', train_result.metrics)
        trainer.save_state()
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics('eval', metrics)
        trainer.save_metrics('eval', metrics)
    kwargs = {'finetuned_from': model_args.model_name_or_path, 'tasks': 'image-classification', 'dataset': data_args.dataset_name, 'tags': ['image-classification', 'vision']}
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)
if __name__ == '__main__':
    main()