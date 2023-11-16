""" Finetuning any 🤗 Transformers model supported by AutoModelForSemanticSegmentation for semantic segmentation."""
import argparse
import json
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
from huggingface_hub import Repository, create_repo, hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional
from tqdm.auto import tqdm
import transformers
from transformers import AutoConfig, AutoImageProcessor, AutoModelForSemanticSegmentation, SchedulerType, default_data_collator, get_scheduler
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
check_min_version('4.36.0.dev0')
logger = get_logger(__name__)
require_version('datasets>=2.0.0', 'To fix: pip install -r examples/pytorch/semantic-segmentation/requirements.txt')

def pad_if_smaller(img, size, fill=0):
    if False:
        print('Hello World!')
    min_size = min(img.size)
    if min_size < size:
        (original_width, original_height) = img.size
        pad_height = size - original_height if original_height < size else 0
        pad_width = size - original_width if original_width < size else 0
        img = functional.pad(img, (0, 0, pad_width, pad_height), fill=fill)
    return img

class Compose:

    def __init__(self, transforms):
        if False:
            for i in range(10):
                print('nop')
        self.transforms = transforms

    def __call__(self, image, target):
        if False:
            i = 10
            return i + 15
        for t in self.transforms:
            (image, target) = t(image, target)
        return (image, target)

class Identity:

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def __call__(self, image, target):
        if False:
            while True:
                i = 10
        return (image, target)

class Resize:

    def __init__(self, size):
        if False:
            while True:
                i = 10
        self.size = size

    def __call__(self, image, target):
        if False:
            i = 10
            return i + 15
        image = functional.resize(image, self.size)
        target = functional.resize(target, self.size, interpolation=transforms.InterpolationMode.NEAREST)
        return (image, target)

class RandomResize:

    def __init__(self, min_size, max_size=None):
        if False:
            print('Hello World!')
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        if False:
            while True:
                i = 10
        size = random.randint(self.min_size, self.max_size)
        image = functional.resize(image, size)
        target = functional.resize(target, size, interpolation=transforms.InterpolationMode.NEAREST)
        return (image, target)

class RandomCrop:

    def __init__(self, size):
        if False:
            while True:
                i = 10
        self.size = size

    def __call__(self, image, target):
        if False:
            return 10
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = transforms.RandomCrop.get_params(image, (self.size, self.size))
        image = functional.crop(image, *crop_params)
        target = functional.crop(target, *crop_params)
        return (image, target)

class RandomHorizontalFlip:

    def __init__(self, flip_prob):
        if False:
            print('Hello World!')
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if False:
            i = 10
            return i + 15
        if random.random() < self.flip_prob:
            image = functional.hflip(image)
            target = functional.hflip(target)
        return (image, target)

class PILToTensor:

    def __call__(self, image, target):
        if False:
            return 10
        image = functional.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return (image, target)

class ConvertImageDtype:

    def __init__(self, dtype):
        if False:
            i = 10
            return i + 15
        self.dtype = dtype

    def __call__(self, image, target):
        if False:
            print('Hello World!')
        image = functional.convert_image_dtype(image, self.dtype)
        return (image, target)

class Normalize:

    def __init__(self, mean, std):
        if False:
            print('Hello World!')
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        if False:
            for i in range(10):
                print('nop')
        image = functional.normalize(image, mean=self.mean, std=self.std)
        return (image, target)

class ReduceLabels:

    def __call__(self, image, target):
        if False:
            i = 10
            return i + 15
        if not isinstance(target, np.ndarray):
            target = np.array(target).astype(np.uint8)
        target[target == 0] = 255
        target = target - 1
        target[target == 254] = 255
        target = Image.fromarray(target)
        return (image, target)

def parse_args():
    if False:
        return 10
    parser = argparse.ArgumentParser(description='Finetune a transformers model on a text classification task')
    parser.add_argument('--model_name_or_path', type=str, help='Path to a pretrained model or model identifier from huggingface.co/models.', default='nvidia/mit-b0')
    parser.add_argument('--dataset_name', type=str, help='Name of the dataset on the hub.', default='segments/sidewalk-semantic')
    parser.add_argument('--reduce_labels', action='store_true', help='Whether or not to reduce all labels by 1 and replace background by 255.')
    parser.add_argument('--train_val_split', type=float, default=0.15, help='Fraction of the dataset to be used for validation.')
    parser.add_argument('--cache_dir', type=str, help='Path to a folder in which the model and dataset will be cached.')
    parser.add_argument('--use_auth_token', action='store_true', help='Whether to use an authentication token to access the model repository.')
    parser.add_argument('--per_device_train_batch_size', type=int, default=8, help='Batch size (per device) for the training dataloader.')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8, help='Batch size (per device) for the evaluation dataloader.')
    parser.add_argument('--learning_rate', type=float, default=5e-05, help='Initial learning rate (after the potential warmup period) to use.')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='Beta1 for AdamW optimizer')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='Beta2 for AdamW optimizer')
    parser.add_argument('--adam_epsilon', type=float, default=1e-08, help='Epsilon for AdamW optimizer')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Total number of training epochs to perform.')
    parser.add_argument('--max_train_steps', type=int, default=None, help='Total number of training steps to perform. If provided, overrides num_train_epochs.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--lr_scheduler_type', type=SchedulerType, default='polynomial', help='The scheduler type to use.', choices=['linear', 'cosine', 'cosine_with_restarts', 'polynomial', 'constant', 'constant_with_warmup'])
    parser.add_argument('--num_warmup_steps', type=int, default=0, help='Number of steps for the warmup in the lr scheduler.')
    parser.add_argument('--output_dir', type=str, default=None, help='Where to store the final model.')
    parser.add_argument('--seed', type=int, default=None, help='A seed for reproducible training.')
    parser.add_argument('--push_to_hub', action='store_true', help='Whether or not to push the model to the Hub.')
    parser.add_argument('--hub_model_id', type=str, help='The name of the repository to keep in sync with the local `output_dir`.')
    parser.add_argument('--hub_token', type=str, help='The token to use to push to the Model Hub.')
    parser.add_argument('--trust_remote_code', type=bool, default=False, help='Whether or not to allow for custom models defined on the Hub in their own modeling files. This optionshould only be set to `True` for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.')
    parser.add_argument('--checkpointing_steps', type=str, default=None, help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='If the training should continue from a checkpoint folder.')
    parser.add_argument('--with_tracking', required=False, action='store_true', help='Whether to enable experiment trackers for logging.')
    parser.add_argument('--report_to', type=str, default='all', help='The integration to report the results and logs to. Supported platforms are `"tensorboard"`, `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. Only applicable when `--with_tracking` is passed.')
    args = parser.parse_args()
    if args.push_to_hub or args.with_tracking:
        if args.output_dir is None:
            raise ValueError('Need an `output_dir` to create a repo when `--push_to_hub` or `with_tracking` is specified.')
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    return args

def main():
    if False:
        i = 10
        return i + 15
    args = parse_args()
    send_example_telemetry('run_semantic_segmentation_no_trainer', args)
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs['log_with'] = args.report_to
        accelerator_log_kwargs['project_dir'] = args.output_dir
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        set_seed(args.seed, device_specific=True)
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
    dataset = load_dataset(args.dataset_name, cache_dir=args.cache_dir)
    if 'pixel_values' in dataset['train'].column_names:
        dataset = dataset.rename_columns({'pixel_values': 'image'})
    if 'annotation' in dataset['train'].column_names:
        dataset = dataset.rename_columns({'annotation': 'label'})
    args.train_val_split = None if 'validation' in dataset.keys() else args.train_val_split
    if isinstance(args.train_val_split, float) and args.train_val_split > 0.0:
        split = dataset['train'].train_test_split(args.train_val_split)
        dataset['train'] = split['train']
        dataset['validation'] = split['test']
    if args.dataset_name == 'scene_parse_150':
        repo_id = 'huggingface/label-files'
        filename = 'ade20k-id2label.json'
    else:
        repo_id = args.dataset_name
        filename = 'id2label.json'
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type='dataset'), 'r'))
    id2label = {int(k): v for (k, v) in id2label.items()}
    label2id = {v: k for (k, v) in id2label.items()}
    config = AutoConfig.from_pretrained(args.model_name_or_path, id2label=id2label, label2id=label2id, trust_remote_code=args.trust_remote_code)
    image_processor = AutoImageProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model_name_or_path, config=config, trust_remote_code=args.trust_remote_code)
    if 'shortest_edge' in image_processor.size:
        size = (image_processor.size['shortest_edge'], image_processor.size['shortest_edge'])
    else:
        size = (image_processor.size['height'], image_processor.size['width'])
    train_transforms = Compose([ReduceLabels() if args.reduce_labels else Identity(), RandomCrop(size=size), RandomHorizontalFlip(flip_prob=0.5), PILToTensor(), ConvertImageDtype(torch.float), Normalize(mean=image_processor.image_mean, std=image_processor.image_std)])
    val_transforms = Compose([ReduceLabels() if args.reduce_labels else Identity(), Resize(size=size), PILToTensor(), ConvertImageDtype(torch.float), Normalize(mean=image_processor.image_mean, std=image_processor.image_std)])

    def preprocess_train(example_batch):
        if False:
            return 10
        pixel_values = []
        labels = []
        for (image, target) in zip(example_batch['image'], example_batch['label']):
            (image, target) = train_transforms(image.convert('RGB'), target)
            pixel_values.append(image)
            labels.append(target)
        encoding = {}
        encoding['pixel_values'] = torch.stack(pixel_values)
        encoding['labels'] = torch.stack(labels)
        return encoding

    def preprocess_val(example_batch):
        if False:
            return 10
        pixel_values = []
        labels = []
        for (image, target) in zip(example_batch['image'], example_batch['label']):
            (image, target) = val_transforms(image.convert('RGB'), target)
            pixel_values.append(image)
            labels.append(target)
        encoding = {}
        encoding['pixel_values'] = torch.stack(pixel_values)
        encoding['labels'] = torch.stack(labels)
        return encoding
    with accelerator.main_process_first():
        train_dataset = dataset['train'].with_transform(preprocess_train)
        eval_dataset = dataset['validation'].with_transform(preprocess_val)
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size)
    optimizer = torch.optim.AdamW(list(model.parameters()), lr=args.learning_rate, betas=[args.adam_beta1, args.adam_beta2], eps=args.adam_epsilon)
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)
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
    metric = evaluate.load('mean_iou')
    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config['lr_scheduler_type'] = experiment_config['lr_scheduler_type'].value
        accelerator.init_trackers('semantic_segmentation_no_trainer', experiment_config)
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
                    if args.push_to_hub and epoch < args.num_train_epochs - 1:
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
                        if accelerator.is_main_process:
                            image_processor.save_pretrained(args.output_dir)
                            repo.push_to_hub(commit_message=f'Training in progress {completed_steps} steps', blocking=False, auto_lfs_prune=True)
            if completed_steps >= args.max_train_steps:
                break
        logger.info('***** Running evaluation *****')
        model.eval()
        for (step, batch) in enumerate(tqdm(eval_dataloader, disable=not accelerator.is_local_main_process)):
            with torch.no_grad():
                outputs = model(**batch)
            upsampled_logits = torch.nn.functional.interpolate(outputs.logits, size=batch['labels'].shape[-2:], mode='bilinear', align_corners=False)
            predictions = upsampled_logits.argmax(dim=1)
            (predictions, references) = accelerator.gather_for_metrics((predictions, batch['labels']))
            metric.add_batch(predictions=predictions, references=references)
        eval_metrics = metric.compute(num_labels=len(id2label), ignore_index=255, reduce_labels=False)
        logger.info(f'epoch {epoch}: {eval_metrics}')
        if args.with_tracking:
            accelerator.log({'mean_iou': eval_metrics['mean_iou'], 'mean_accuracy': eval_metrics['mean_accuracy'], 'overall_accuracy': eval_metrics['overall_accuracy'], 'train_loss': total_loss.item() / len(train_dataloader), 'epoch': epoch, 'step': completed_steps}, step=completed_steps)
        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
            if accelerator.is_main_process:
                image_processor.save_pretrained(args.output_dir)
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
            image_processor.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message='End of training', auto_lfs_prune=True)
            all_results = {f'eval_{k}': v.tolist() if isinstance(v, np.ndarray) else v for (k, v) in eval_metrics.items()}
            with open(os.path.join(args.output_dir, 'all_results.json'), 'w') as f:
                json.dump(all_results, f)
if __name__ == '__main__':
    main()