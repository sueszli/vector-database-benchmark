import os
import sys
from typing import List
import time
import fire
import torch
from torch.utils.data import DataLoader
import transformers
from datasets import load_dataset
'\nUnused imports:\nimport torch.nn as nn\nimport bitsandbytes as bnb\n'
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_int8_training, set_peft_model_state_dict
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import BitsAndBytesConfig

def train(base_model: str='', data_path: str='./alpaca_data_cleaned.json', output_dir: str='./lora-alpaca', batch_size: int=128, micro_batch_size: int=4, num_epochs: int=3, learning_rate: float=0.0003, cutoff_len: int=256, val_set_size: int=2000, lora_r: int=8, lora_alpha: int=16, lora_dropout: float=0.05, lora_target_modules: List[str]=['q_proj', 'v_proj'], train_on_inputs: bool=True, group_by_length: bool=False, wandb_project: str='', wandb_run_name: str='', wandb_watch: str='', wandb_log_model: str='', resume_from_checkpoint: str=None, use_ipex: bool=False, bf16: bool=False, no_cuda: bool=True, xpu_backend: str='ccl'):
    if False:
        print('Hello World!')
    print(f'Training Alpaca-LoRA model with params:\nbase_model: {base_model}\ndata_path: {data_path}\noutput_dir: {output_dir}\nbatch_size: {batch_size}\nmicro_batch_size: {micro_batch_size}\nnum_epochs: {num_epochs}\nlearning_rate: {learning_rate}\ncutoff_len: {cutoff_len}\nval_set_size: {val_set_size}\nlora_r: {lora_r}\nlora_alpha: {lora_alpha}\nlora_dropout: {lora_dropout}\nlora_target_modules: {lora_target_modules}\ntrain_on_inputs: {train_on_inputs}\ngroup_by_length: {group_by_length}\nwandb_project: {wandb_project}\nwandb_run_name: {wandb_run_name}\nwandb_watch: {wandb_watch}\nwandb_log_model: {wandb_log_model}\nresume_from_checkpoint: {resume_from_checkpoint}\nuse_ipex: {use_ipex}\nbf16: {bf16}\n')
    assert base_model, "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    device_map = 'auto'
    pmi_world_size = int(os.environ.get('PMI_SIZE', -1))
    if pmi_world_size > 0:
        os.environ['WORLD_SIZE'] = str(pmi_world_size)
    else:
        os.environ['WORLD_SIZE'] = str(os.environ.get('WORLD_SIZE', 1))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    print(f'world_size: {world_size}!!')
    ddp = world_size != 1
    local_rank = 0
    if ddp:
        os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
        os.environ['LOCAL_RANK'] = str(os.environ.get('PMI_RANK', 0))
        local_rank = str(os.environ.get('PMI_RANK', 0))
        print('PMI_RANK(local_rank): ' + local_rank)
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    use_wandb = len(wandb_project) > 0 or ('WANDB_PROJECT' in os.environ and len(os.environ['WANDB_PROJECT']) > 0)
    if len(wandb_project) > 0:
        os.environ['WANDB_PROJECT'] = wandb_project
    if len(wandb_watch) > 0:
        os.environ['WANDB_WATCH'] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ['WANDB_LOG_MODEL'] = wandb_log_model
    model = LlamaForCausalLM.from_pretrained(base_model, low_cpu_mem_usage=True)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'

    def tokenize(prompt, add_eos_token=True):
        if False:
            print('Hello World!')
        result = tokenizer(prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None)
        if result['input_ids'][-1] != tokenizer.eos_token_id and len(result['input_ids']) < cutoff_len and add_eos_token:
            result['input_ids'].append(tokenizer.eos_token_id)
            result['attention_mask'].append(1)
        result['labels'] = result['input_ids'].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        if False:
            while True:
                i = 10
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, 'output': ''})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt['input_ids'])
            tokenized_full_prompt['labels'] = [-100] * user_prompt_len + tokenized_full_prompt['labels'][user_prompt_len:]
        return tokenized_full_prompt
    model = prepare_model_for_int8_training(model)
    config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=lora_target_modules, lora_dropout=lora_dropout, bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, config)
    if data_path.endswith('.json'):
        data = load_dataset('json', data_files=data_path)
    else:
        data = load_dataset(data_path)
    if resume_from_checkpoint:
        checkpoint_name = os.path.join(resume_from_checkpoint, 'pytorch_model.bin')
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(resume_from_checkpoint, 'adapter_model.bin')
            resume_from_checkpoint = False
        if os.path.exists(checkpoint_name):
            print(f'Restarting from {checkpoint_name}')
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f'Checkpoint {checkpoint_name} not found')
    model.print_trainable_parameters()
    if val_set_size > 0:
        print('[INFO] spliting and shuffling dataset...')
        train_val = data['train'].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        print('[INFO] shuffling and tokenizing train data...')
        train_data = train_val['train'].shuffle().map(generate_and_tokenize_prompt)
        print('[INFO] shuffling and tokenizing test data...')
        val_data = train_val['test'].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data['train'].shuffle().map(generate_and_tokenize_prompt)
        val_data = None
    print('[INFO] begining the training of transformers...')
    args = transformers.TrainingArguments(per_device_train_batch_size=micro_batch_size, gradient_accumulation_steps=gradient_accumulation_steps, warmup_steps=100, num_train_epochs=num_epochs, learning_rate=learning_rate, bf16=bf16, logging_steps=10, optim='adamw_torch', evaluation_strategy='epoch', save_strategy='steps', local_rank=local_rank, output_dir=output_dir, save_total_limit=3, ddp_find_unused_parameters=False, group_by_length=group_by_length, report_to='wandb' if use_wandb else None, run_name=wandb_run_name if use_wandb else None, xpu_backend=xpu_backend, no_cuda=no_cuda)
    print(f'[INFO] Process rank: {args.local_rank}, device: {args.device}' + f"distributed training: {args.parallel_mode.value == 'distributed'}")
    trainer = transformers.Trainer(model=model, train_dataset=train_data, eval_dataset=val_data, args=args, data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors='pt', padding=True))
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(model, type(model))
    start = time.time()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    end = time.time()
    print('training time is: ', end - start)
    if int(os.environ.get('PMI_RANK', -1)) == 0:
        model.save_pretrained(output_dir)
    elif int(os.environ.get('PMI_RANK', -1)) == -1:
        model.save_pretrained(output_dir)
    print("\n If there's a warning about missing keys above, please disregard :)")

def generate_prompt(data_point):
    if False:
        while True:
            i = 10
    if data_point['input']:
        return f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501\n\n### Instruction:\n{data_point['instruction']}\n\n### Input:\n{data_point['input']}\n\n### Response:\n{data_point['output']}"
    else:
        return f"Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501\n\n### Instruction:\n{data_point['instruction']}\n\n### Response:\n{data_point['output']}"
if __name__ == '__main__':
    fire.Fire(train)