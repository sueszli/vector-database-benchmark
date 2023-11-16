""" Finetuning any 🤗 Transformers model for image classification leveraging 🤗 Accelerate."""
import argparse
import json
import logging
import math
import os
from pathlib import Path
import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, Compose, Lambda, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor
from tqdm.auto import tqdm
import transformers
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification, SchedulerType, get_scheduler
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
check_min_version('4.36.0.dev0')
logger = get_logger(__name__)
require_version('datasets>=2.0.0', 'To fix: pip install -r examples/pytorch/image-classification/requirements.txt')

def parse_args():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(description='Fine-tune a Transformers model on an image classification dataset')
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private, dataset).')
    parser.add_argument('--train_dir', type=str, default=None, help='A folder containing the training data.')
    parser.add_argument('--validation_dir', type=str, default=None, help='A folder containing the validation data.')
    parser.add_argument('--max_train_samples', type=int, default=None, help='For debugging purposes or quicker training, truncate the number of training examples to this value if set.')
    parser.add_argument('--max_eval_samples', type=int, default=None, help='For debugging purposes or quicker training, truncate the number of evaluation examples to this value if set.')
    parser.add_argument('--train_val_split', type=float, default=0.15, help='Percent to split off of train for validation')
    parser.add_argument('--model_name_or_path', type=str, help='Path to pretrained model or model identifier from huggingface.co/models.', default='google/vit-base-patch16-224-in21k')
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
    parser.add_argument('--push_to_hub', action='store_true', help='Whether or not to push the model to the Hub.')
    parser.add_argument('--hub_model_id', type=str, help='The name of the repository to keep in sync with the local `output_dir`.')
    parser.add_argument('--hub_token', type=str, help='The token to use to push to the Model Hub.')
    parser.add_argument('--trust_remote_code', type=bool, default=False, help='Whether or not to allow for custom models defined on the Hub in their own modeling files. This optionshould only be set to `True` for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.')
    parser.add_argument('--checkpointing_steps', type=str, default=None, help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='If the training should continue from a checkpoint folder.')
    parser.add_argument('--with_tracking', action='store_true', help='Whether to enable experiment trackers for logging.')
    parser.add_argument('--report_to', type=str, default='all', help='The integration to report the results and logs to. Supported platforms are `"tensorboard"`, `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. Only applicable when `--with_tracking` is passed.')
    parser.add_argument('--ignore_mismatched_sizes', action='store_true', help='Whether or not to enable to load a pretrained model whose head dimensions are different.')
    args = parser.parse_args()
    if args.dataset_name is None and args.train_dir is None and (args.validation_dir is None):
        raise ValueError('Need either a dataset name or a training/validation folder.')
    if args.push_to_hub or args.with_tracking:
        if args.output_dir is None:
            raise ValueError('Need an `output_dir` to create a repo when `--push_to_hub` or `with_tracking` is specified.')
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    return args

def main():
    if False:
        for i in range(10):
            print('nop')
    args = parse_args()
    send_example_telemetry('run_image_classification_no_trainer', args)
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs['log_with'] = args.report_to
        accelerator_log_kwargs['project_dir'] = args.output_dir
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    logger.info(accelerator.state)
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
        dataset = load_dataset(args.dataset_name, task='image-classification')
    else:
        data_files = {}
        if args.train_dir is not None:
            data_files['train'] = os.path.join(args.train_dir, '**')
        if args.validation_dir is not None:
            data_files['validation'] = os.path.join(args.validation_dir, '**')
        dataset = load_dataset('imagefolder', data_files=data_files, cache_dir=args.cache_dir, task='image-classification')
    args.train_val_split = None if 'validation' in dataset.keys() else args.train_val_split
    if isinstance(args.train_val_split, float) and args.train_val_split > 0.0:
        split = dataset['train'].train_test_split(args.train_val_split)
        dataset['train'] = split['train']
        dataset['validation'] = split['test']
    labels = dataset['train'].features['labels'].names
    label2id = {label: str(i) for (i, label) in enumerate(labels)}
    id2label = {str(i): label for (i, label) in enumerate(labels)}
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=len(labels), i2label=id2label, label2id=label2id, finetuning_task='image-classification', trust_remote_code=args.trust_remote_code)
    image_processor = AutoImageProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    model = AutoModelForImageClassification.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config, ignore_mismatched_sizes=args.ignore_mismatched_sizes, trust_remote_code=args.trust_remote_code)
    if 'shortest_edge' in image_processor.size:
        size = image_processor.size['shortest_edge']
    else:
        size = (image_processor.size['height'], image_processor.size['width'])
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std) if hasattr(image_processor, 'image_mean') and hasattr(image_processor, 'image_std') else Lambda(lambda x: x)
    train_transforms = Compose([RandomResizedCrop(size), RandomHorizontalFlip(), ToTensor(), normalize])
    val_transforms = Compose([Resize(size), CenterCrop(size), ToTensor(), normalize])

    def preprocess_train(example_batch):
        if False:
            return 10
        'Apply _train_transforms across a batch.'
        example_batch['pixel_values'] = [train_transforms(image.convert('RGB')) for image in example_batch['image']]
        return example_batch

    def preprocess_val(example_batch):
        if False:
            for i in range(10):
                print('nop')
        'Apply _val_transforms across a batch.'
        example_batch['pixel_values'] = [val_transforms(image.convert('RGB')) for image in example_batch['image']]
        return example_batch
    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset['train'] = dataset['train'].shuffle(seed=args.seed).select(range(args.max_train_samples))
        train_dataset = dataset['train'].with_transform(preprocess_train)
        if args.max_eval_samples is not None:
            dataset['validation'] = dataset['validation'].shuffle(seed=args.seed).select(range(args.max_eval_samples))
        eval_dataset = dataset['validation'].with_transform(preprocess_val)

    def collate_fn(examples):
        if False:
            while True:
                i = 10
        pixel_values = torch.stack([example['pixel_values'] for example in examples])
        labels = torch.tensor([example['labels'] for example in examples])
        return {'pixel_values': pixel_values, 'labels': labels}
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_fn, batch_size=args.per_device_eval_batch_size)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{'params': [p for (n, p) in model.named_parameters() if not any((nd in n for nd in no_decay))], 'weight_decay': args.weight_decay}, {'params': [p for (n, p) in model.named_parameters() if any((nd in n for nd in no_decay))], 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
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
        accelerator.init_trackers('image_classification_no_trainer', experiment_config)
    metric = evaluate.load('accuracy')
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
        model.eval()
        for (step, batch) in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            (predictions, references) = accelerator.gather_for_metrics((predictions, batch['labels']))
            metric.add_batch(predictions=predictions, references=references)
        eval_metric = metric.compute()
        logger.info(f'epoch {epoch}: {eval_metric}')
        if args.with_tracking:
            accelerator.log({'accuracy': eval_metric, 'train_loss': total_loss.item() / len(train_dataloader), 'epoch': epoch, 'step': completed_steps}, step=completed_steps)
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
            all_results = {f'eval_{k}': v for (k, v) in eval_metric.items()}
            with open(os.path.join(args.output_dir, 'all_results.json'), 'w') as f:
                json.dump(all_results, f)
if __name__ == '__main__':
    main()