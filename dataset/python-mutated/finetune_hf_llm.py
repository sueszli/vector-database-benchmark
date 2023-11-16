import argparse
from filelock import FileLock
import functools
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import tree
from typing import Tuple
try:
    import deepspeed
except ImportError as e:
    raise RuntimeError('Please install deepspeed with `pip install --user deepspeed`.') from e
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DummyOptim, DummyScheduler, set_seed
import torch
import torch.nn as nn
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
import ray
from ray import train
import ray.util.scheduling_strategies
from ray.train.torch import TorchTrainer
from ray.train import Checkpoint
from utils import get_checkpoint_and_refs_dir, get_mirror_link, download_model, get_download_path
OPTIM_BETAS = (0.9, 0.999)
OPTIM_EPS = 1e-08
NUM_WARMUP_STEPS = 10
OPTIM_WEIGHT_DECAY = 0.0
ATTENTION_LAYER_NAME = 'self_attn'

def get_expected_lora_num_parameters(model, lora_config: LoraConfig, attn_layer_name: str=ATTENTION_LAYER_NAME):
    if False:
        while True:
            i = 10
    'Calculate the expected number of parameters for lora finetuning.'
    sum_params = 0
    num_attention_layers = 0
    modules = model.named_modules()
    loraified_modules = 0
    for (full_name, target) in modules:
        layer_name = full_name.split('.')[-1]
        if layer_name == attn_layer_name:
            num_attention_layers += 1
        elif layer_name in lora_config.modules_to_save:
            sum_params += 2 * target.weight.numel()
            print('Found non-lora-layer to checkpoint: ', layer_name, ' with num params ', target.weight.numel())
        else:
            for module_name in lora_config.target_modules:
                if layer_name == module_name:
                    loraified_modules += 1
                    if isinstance(target, nn.Linear):
                        sum_params += (target.in_features + target.out_features) * lora_config.r
                    elif isinstance(target, nn.Embedding):
                        sum_params += (target.embedding_dim + target.num_embeddings) * lora_config.r
    print(f"Detected {num_attention_layers} attention layers, containing {loraified_modules} modules to modify according to LoRA's `target_modules`. This should yield {sum_params} trainable parameters.")
    return sum_params

def get_number_of_params(model: nn.Module):
    if False:
        return 10
    sum = 0
    for (name, param) in model.named_parameters():
        if param.requires_grad:
            sum += param.numel()
    return sum

def collate_fn(batch, tokenizer, block_size, device):
    if False:
        print('Hello World!')
    out_batch = tokenizer(list(batch['input']), padding='max_length', max_length=block_size, truncation=True, return_tensors='pt')
    out_batch['labels'] = out_batch['input_ids'].clone()
    out_batch = tree.map_structure(lambda x: x.to(device), out_batch)
    return out_batch

def get_pretrained_path(model_id: str):
    if False:
        for i in range(10):
            print('nop')
    mirror_uri = get_mirror_link(model_id)
    (ckpt_path, _) = get_checkpoint_and_refs_dir(model_id=model_id, bucket_uri=mirror_uri, s3_sync_args=['--no-sign-request'])
    return ckpt_path

def get_tokenizer(model_name, special_tokens):
    if False:
        i = 10
        return i + 15
    pretrained_path = get_pretrained_path(model_name)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, legacy=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(special_tokens, special_tokens=True)
    return tokenizer

def evaluate(*, model, eval_ds, accelerator, bsize, ds_kwargs, as_test: bool=False) -> Tuple[float, float]:
    if False:
        while True:
            i = 10
    model.eval()
    losses = []
    eval_dataloader = eval_ds.iter_torch_batches(batch_size=bsize, **ds_kwargs)
    eval_ds_len = len(list(eval_ds.iter_batches(batch_size=1)))
    for (step, batch) in tqdm.tqdm(enumerate(eval_dataloader), total=eval_ds_len // (bsize + 1)):
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        losses.append(accelerator.gather(loss[None]))
        if as_test:
            break
    losses = torch.stack(losses)
    try:
        eval_loss = torch.mean(losses).item()
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float('inf')
    return (perplexity, eval_loss)

def _test_tokenizer(model_name):
    if False:
        while True:
            i = 10
    tokenizer = get_tokenizer(model_name=model_name, special_tokens=['<REPR_END>'])
    testoutput = tokenizer('<REPR_END>inform')['input_ids']
    expected = tokenizer('inform')['input_ids']
    assert testoutput[-1] == expected[-1], f'The tokenizer is not working as expected with special tokens, testoutput={testoutput}, expected={expected}'

def checkpoint_model(checkpoint_folder, ckpt_id, model, epoch, last_global_step, **kwargs):
    if False:
        i = 10
        return i + 15
    'Utility function for checkpointing model + optimizer dictionaries\n    The main purpose for this is to be able to resume training from that instant again.\n    '
    checkpoint_state_dict = {'epoch': epoch, 'last_global_step': last_global_step}
    checkpoint_state_dict.update(kwargs)
    model.save_checkpoint(checkpoint_folder, ckpt_id, checkpoint_state_dict)
    status_msg = f'checkpointing: checkpoint_folder={checkpoint_folder}, ckpt_id={ckpt_id}'
    print(status_msg)

def training_function(kwargs: dict):
    if False:
        return 10
    print('training_function called')
    cuda_visible_device = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    local_rank = int(os.environ['LOCAL_RANK'])
    device_id = cuda_visible_device[local_rank]
    os.environ['ACCELERATE_TORCH_DEVICE'] = f'cuda:{device_id}'
    config = kwargs['config']
    args = argparse.Namespace(**kwargs['args'])
    special_tokens = kwargs.get('special_tokens', [])
    model_id = config['model_name']
    bucket_uri = get_mirror_link(model_id)
    download_path = get_download_path(model_id)
    base_path = Path(download_path).parent
    base_path.mkdir(parents=True, exist_ok=True)
    lock_file = str(base_path / f"{model_id.replace('/', '--')}.lock")
    with FileLock(lock_file):
        download_model(model_id=model_id, bucket_uri=bucket_uri, s3_sync_args=['--no-sign-request'])
    lr = config['lr']
    num_epochs = int(config['num_epochs'])
    seed = int(config['seed'])
    batch_size = int(config['batch_size'])
    gradient_accumulation_steps = int(config['gradient_accumulation_steps'])
    ds_plugin = config['ds_plugin']
    ds_plugin.hf_ds_config.config['train_micro_batch_size_per_gpu'] = batch_size
    accelerator = Accelerator(deepspeed_plugin=ds_plugin, gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision=args.mx)
    set_seed(seed)
    train_ds = train.get_dataset_shard('train')
    valid_ds = train.get_dataset_shard('valid')
    train_ds_len = len(list(train_ds.iter_batches(batch_size=1)))
    _test_tokenizer(args.model_name)
    tokenizer = get_tokenizer(model_name=args.model_name, special_tokens=special_tokens)
    collate_partial = functools.partial(collate_fn, tokenizer=tokenizer, block_size=config['block_size'], device=accelerator.device)
    pretrained_path = get_pretrained_path(model_id)
    print(f'Loading model from {pretrained_path} ...')
    s = time.time()
    model = AutoModelForCausalLM.from_pretrained(pretrained_path, trust_remote_code=True, torch_dtype=torch.bfloat16, use_cache=False)
    print(f'Done loading model in {time.time() - s} seconds.')
    model.resize_token_embeddings(len(tokenizer))
    if config['lora']:
        s = time.time()
        lora_config = LoraConfig(**config['lora_config'])
        expected_num_parameters = get_expected_lora_num_parameters(lora_config=lora_config, model=model)
        print(f'Attempting to apply LoRA config: {lora_config}')
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        num_parameters = get_number_of_params(model)
        if num_parameters != expected_num_parameters:
            raise ValueError(f'Expected {expected_num_parameters} parameters, got {num_parameters} parameters. LoRA-ification failed.')
        print(f'LoRA-ification done in {time.time() - s} seconds. Estimated checkpoint size (fp16): {num_parameters * 2 / 1000000.0} MB')
    print(f'Number of checkpointed parameters: {get_number_of_params(model)}')
    print('Model initialized with pretrained weights. Training starting...')
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()
    optimizer_cls = torch.optim.AdamW if accelerator.state.deepspeed_plugin is None or 'optimizer' not in accelerator.state.deepspeed_plugin.deepspeed_config else DummyOptim
    optimizer = optimizer_cls(model.parameters(), lr=lr, betas=OPTIM_BETAS, weight_decay=OPTIM_WEIGHT_DECAY, eps=OPTIM_EPS)
    num_steps_per_epoch = math.ceil(train_ds_len / args.batch_size_per_device)
    total_training_steps = num_steps_per_epoch * num_epochs // gradient_accumulation_steps
    if accelerator.state.deepspeed_plugin is None or 'scheduler' not in accelerator.state.deepspeed_plugin.deepspeed_config:
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=NUM_WARMUP_STEPS * args.num_devices, num_training_steps=total_training_steps * args.num_devices)
    else:
        lr_scheduler = DummyScheduler(optimizer, warmup_num_steps=NUM_WARMUP_STEPS * args.num_devices, total_num_steps=total_training_steps * args.num_devices)
    s = time.time()
    (model, optimizer, lr_scheduler) = accelerator.prepare(model, optimizer, lr_scheduler)
    print(f'Prepare done in {time.time() - s} seconds.')
    if accelerator.is_main_process:
        print('Starting training ...')
        print('Number of batches on main process', train_ds_len // batch_size)
    for epoch in range(num_epochs):
        (fwd_time_sum, bwd_time_sum, optim_step_time_sum) = (0, 0, 0)
        s_epoch = time.time()
        model.train()
        loss_sum = torch.tensor(0.0).to(accelerator.device)
        train_dataloader = train_ds.iter_torch_batches(batch_size=batch_size, collate_fn=collate_partial)
        for (step, batch) in tqdm.tqdm(enumerate(train_dataloader), total=train_ds_len // batch_size + 1):
            with accelerator.accumulate(model):
                s_fwd = time.time()
                outputs = model(**batch)
                loss = outputs.loss
                loss_sum += loss.item()
                e_fwd = time.time()
                fwd_time = e_fwd - s_fwd
                fwd_time_sum += fwd_time
                s_bwd = time.time()
                accelerator.backward(loss)
                e_bwd = time.time()
                bwd_time = e_bwd - s_bwd
                bwd_time_sum += bwd_time
                s_opt_step = time.time()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                e_opt_step = time.time()
                optim_step_time_sum += e_opt_step - s_opt_step
            if accelerator.is_main_process:
                accelerator.print(f'[epoch {epoch} step {step}] loss: {loss.item()} step-time: {e_opt_step - s_fwd}')
            aggregated_loss = torch.mean(accelerator.gather(loss[None])).item()
            if config['as_test']:
                break
            if step != train_ds_len // batch_size - 1:
                train.report({'epoch': epoch, 'iteration': step, 'train_loss_batch': aggregated_loss, 'avg_train_loss_epoch': None, 'eval_loss': None, 'perplexity': None, 'num_iterations': step + 1, 'train_time_per_epoch': None, 'eval_time_per_epoch': None, 'fwd_time': fwd_time, 'bwd_time': bwd_time, 'avg_fwd_time_per_epoch': None, 'avg_bwd_time_per_epoch': None, 'learning_rate': lr_scheduler.get_lr()[0]})
        e_epoch = time.time()
        accelerator.print('Train time per epoch: ', e_epoch - s_epoch)
        eval_s_epoch = time.time()
        print('Running evaluation ...')
        (perplex, eloss) = evaluate(model=model, eval_ds=valid_ds, accelerator=accelerator, bsize=config['eval_batch_size'], ds_kwargs={'collate_fn': collate_partial}, as_test=config['as_test'])
        accelerator.print('Eval result loss', eloss)
        accelerator.print('Eval perplex', perplex)
        eval_e_epoch = time.time()
        accelerator.print('Eval time per epoch: ', eval_e_epoch - eval_s_epoch)
        accelerator.print('avg fwd time: ', fwd_time_sum / (step + 1))
        accelerator.print('avg bwd time: ', bwd_time_sum / (step + 1))
        accelerator.print('avg opt step time: ', optim_step_time_sum / (step + 1))
        metrics = {'epoch': epoch, 'iteration': step, 'train_loss_batch': aggregated_loss, 'avg_train_loss_epoch': loss_sum.item() / (step + 1), 'eval_loss': eloss, 'perplexity': perplex, 'num_iterations': step + 1, 'train_time_per_epoch': e_epoch - s_epoch, 'eval_time_per_epoch': eval_e_epoch - eval_s_epoch, 'fwd_time': fwd_time, 'bwd_time': bwd_time, 'avg_fwd_time_per_epoch': fwd_time_sum / (step + 1), 'avg_bwd_time_per_epoch': bwd_time_sum / (step + 1), 'learning_rate': lr_scheduler.get_lr()[0]}
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            accelerator.print(f'Saving the model locally at {temp_checkpoint_dir}')
            accelerator.wait_for_everyone()
            checkpoint_save_start = time.perf_counter()
            if accelerator.is_main_process:
                print('Saving tokenizer and config.')
                tokenizer.save_pretrained(temp_checkpoint_dir)
            accelerator.wait_for_everyone()
            aggregate_on_rank_0 = True
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(temp_checkpoint_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, safe_serialization=True, state_dict=accelerator.get_state_dict(model))
            accelerator.wait_for_everyone()
            print('Checkpoint save time: ', time.perf_counter() - checkpoint_save_start)
            checkpoint_upload_start = time.perf_counter()
            if aggregate_on_rank_0:
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir) if accelerator.is_main_process else None
            else:
                checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(metrics, checkpoint=checkpoint)
            print('Checkpoint upload time: ', time.perf_counter() - checkpoint_upload_start)
            print('Total checkpointing time: ', time.perf_counter() - checkpoint_save_start)
        if perplex < args.stop_perplexity:
            print(f'Perplexity reached {perplex} < {args.stop_perplexity}. Stopping.')
            break
        if config['as_test']:
            break

def parse_args():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(description='Simple example of training script.')
    parser.add_argument('--mx', type=str, default='bf16', choices=['no', 'fp16', 'bf16', 'fp8'], help='Whether to use mixed precision. Choosebetween fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10.and an Nvidia Ampere GPU.')
    parser.add_argument('--batch-size-per-device', '-bs', type=int, default=16, help='Batch size to use per device.')
    parser.add_argument('--stop-perplexity', default=0, type=float, help='Target perplexity to reach after which to stop training. Default is 0. If 0, training will not stop on perplexity.')
    parser.add_argument('--eval-batch-size-per-device', type=int, default=64, help='Batch size to use per device (For evaluation).')
    parser.add_argument('--num-devices', '-nd', type=int, default=4, help='Number of devices to use.')
    parser.add_argument('--grad_accum', type=int, default=1, help='Gradient accumulation steps.')
    parser.add_argument('--train_path', type=str, help='Path to training jsonl file')
    parser.add_argument('--test_path', type=str, help='Path to testing jsonl file')
    parser.add_argument('--special_token_path', type=str, help='Path to token json file')
    parser.add_argument('--no-grad-ckpt', action='store_true', help='If passed, will not use gradient checkpointing.')
    parser.add_argument('--output_dir', type=str, help='Path to output directory.')
    parser.add_argument('--model_name', default='meta-llama/Llama-2-7b-chat-hf', type=str)
    parser.add_argument('--num-epochs', type=int, default=1, help='Number of epochs to train for.')
    parser.add_argument('--num-checkpoints-to-keep', type=int, help='Number of checkpoints to keep, if None, all checkpoints will be kept, if set to n>=1, the top n checkpoint with min. evaluation perplexity will be kept.', default=None)
    parser.add_argument('--lr', type=float, default=5e-06, help='Learning rate to use.')
    parser.add_argument('--ctx-len', type=int, default=512, help='Learning rate to use.')
    parser.add_argument('--as-test', action='store_true', help='If passed, will run the script in test mode.')
    parser.add_argument('--ds-config', type=str, default='./deepspeed_configs/zero_3_llama_2_7b.json', help='Deepspeed config json to use.')
    parser.add_argument('--lora', action='store_true', default=False, help='If passed, will enable parameter efficient fine-tuning with LoRA (https://arxiv.org/pdf/2106.09685.pdf).')
    args = parser.parse_args()
    return args

def main():
    if False:
        return 10
    args = parse_args()
    if not args.output_dir:
        raise ValueError('--output_dir must be specified')
    config = vars(args)
    config.update(**{'lr': args.lr, 'num_epochs': args.num_epochs, 'seed': 42, 'batch_size': args.batch_size_per_device, 'gradient_accumulation_steps': args.grad_accum, 'model_name': args.model_name, 'block_size': args.ctx_len, 'eval_batch_size': args.eval_batch_size_per_device})
    if args.lora:
        with open('./lora_configs/lora.json', 'r') as json_file:
            lora_config = json.load(json_file)
        config['lora_config'] = lora_config
    ds_plugin = DeepSpeedPlugin(hf_ds_config=config.get('ds_config'))
    config.update(ds_plugin=ds_plugin)
    os.environ['RAY_AIR_LOCAL_CACHE_DIR'] = args.output_dir
    ray.init(runtime_env={'env_vars': {'HF_HOME': '/mnt/local_storage/.cache/huggingface', 'RAY_AIR_LOCAL_CACHE_DIR': os.environ['RAY_AIR_LOCAL_CACHE_DIR']}, 'working_dir': '.'})
    train_ds = ray.data.read_json(args.train_path)
    if args.test_path is not None:
        valid_ds = ray.data.read_json(args.test_path)
    else:
        valid_ds = None
    with open(args.special_token_path, 'r') as json_file:
        special_tokens = json.load(json_file)['tokens']
    assert 'ANYSCALE_ARTIFACT_STORAGE' in os.environ, 'ANYSCALE_ARTIFACT_STORAGE env var must be set!'
    artifact_storage = os.environ['ANYSCALE_ARTIFACT_STORAGE']
    user_name = re.sub('\\s+', '__', os.environ.get('ANYSCALE_USERNAME', 'user'))
    storage_path = f'{artifact_storage}/{user_name}/ft_llms_with_deepspeed/{args.model_name}'
    trial_name = f'{args.model_name}'.split('/')[-1]
    if args.lora:
        trial_name += '-lora'
    trainer = TorchTrainer(training_function, train_loop_config={'config': config, 'args': vars(args), 'special_tokens': special_tokens}, run_config=train.RunConfig(storage_path=storage_path, checkpoint_config=train.CheckpointConfig(num_to_keep=args.num_checkpoints_to_keep, checkpoint_score_attribute='perplexity', checkpoint_score_order='min')), scaling_config=train.ScalingConfig(num_workers=args.num_devices, use_gpu=True, resources_per_worker={'GPU': 1}), datasets={'train': train_ds, 'valid': valid_ds}, dataset_config=ray.train.DataConfig(datasets_to_split=['train', 'valid']))
    result: train.Result = trainer.fit()
    (best_checkpoint, best_checkpoint_metrics) = result.best_checkpoints[-1]
    print('Results are stored at:')
    print(result.path)
    print('Best checkpoint is stored at:')
    print(best_checkpoint)
    print(f"With perplexity: {best_checkpoint_metrics['perplexity']}")
if __name__ == '__main__':
    main()