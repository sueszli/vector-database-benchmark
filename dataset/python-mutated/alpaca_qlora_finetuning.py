import os
from typing import List
import fire
import torch
import transformers
from datasets import load_dataset
import accelerate
from transformers import LlamaTokenizer
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from utils.prompter import Prompter
import intel_extension_for_pytorch as ipex
from bigdl.llm.transformers import AutoModelForCausalLM
from bigdl.llm.transformers.qlora import get_peft_model, prepare_model_for_kbit_training

def train(base_model: str='meta-llama/Llama-2-7b-hf', saved_low_bit_model: str=None, data_path: str='yahma/alpaca-cleaned', output_dir: str='./bigdl-qlora-alpaca', batch_size: int=128, micro_batch_size: int=2, num_epochs: int=3, learning_rate: float=3e-05, cutoff_len: int=256, val_set_size: int=2000, lora_r: int=8, lora_alpha: int=16, lora_dropout: float=0.05, lora_target_modules: List[str]=['q_proj', 'v_proj', 'k_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj'], train_on_inputs: bool=True, add_eos_token: bool=False, group_by_length: bool=False, wandb_project: str='', wandb_run_name: str='', wandb_watch: str='', wandb_log_model: str='', resume_from_checkpoint: str=None, prompt_template_name: str='alpaca', gradient_checkpointing: bool=False, deepspeed: str=None):
    if False:
        while True:
            i = 10
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        print(f'Training Alpaca-LoRA model with params:\nbase_model: {base_model}\ndata_path: {data_path}\noutput_dir: {output_dir}\nbatch_size: {batch_size}\nmicro_batch_size: {micro_batch_size}\nnum_epochs: {num_epochs}\nlearning_rate: {learning_rate}\ncutoff_len: {cutoff_len}\nval_set_size: {val_set_size}\nlora_r: {lora_r}\nlora_alpha: {lora_alpha}\nlora_dropout: {lora_dropout}\nlora_target_modules: {lora_target_modules}\ntrain_on_inputs: {train_on_inputs}\nadd_eos_token: {add_eos_token}\ngroup_by_length: {group_by_length}\nwandb_project: {wandb_project}\nwandb_run_name: {wandb_run_name}\nwandb_watch: {wandb_watch}\nwandb_log_model: {wandb_log_model}\nresume_from_checkpoint: {resume_from_checkpoint or False}\nprompt template: {prompt_template_name}\n')
    assert base_model, "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size
    prompter = Prompter(prompt_template_name)
    device_map = 'auto'
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    ddp = world_size != 1
    if ddp:
        device_map = {'': int(os.environ.get('LOCAL_RANK') or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    use_wandb = len(wandb_project) > 0 or ('WANDB_PROJECT' in os.environ and len(os.environ['WANDB_PROJECT']) > 0)
    if len(wandb_project) > 0:
        os.environ['WANDB_PROJECT'] = wandb_project
    if len(wandb_watch) > 0:
        os.environ['WANDB_WATCH'] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ['WANDB_LOG_MODEL'] = wandb_log_model
    if saved_low_bit_model is not None:
        model = AutoModelForCausalLM.load_low_bit(saved_low_bit_model, optimize_model=False, torch_dtype=torch.bfloat16, modules_to_not_convert=['lm_head'])
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model, load_in_low_bit='nf4', optimize_model=False, torch_dtype=torch.bfloat16, modules_to_not_convert=['lm_head'])
    print(f"Model loaded on rank {os.environ.get('LOCAL_RANK')}")
    model = model.to(f"xpu:{os.environ.get('LOCAL_RANK', 0)}")
    print(f"Model moved to rank {os.environ.get('LOCAL_RANK')}")
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    print(f"Tokenizer loaded on rank {os.environ.get('LOCAL_RANK')}")
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'
    print(model)

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
            i = 10
            return i + 15
        full_prompt = prompter.generate_prompt(data_point['instruction'], data_point['input'], data_point['output'])
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(data_point['instruction'], data_point['input'])
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=add_eos_token)
            user_prompt_len = len(tokenized_user_prompt['input_ids'])
            if add_eos_token:
                user_prompt_len -= 1
            tokenized_full_prompt['labels'] = [-100] * user_prompt_len + tokenized_full_prompt['labels'][user_prompt_len:]
        return tokenized_full_prompt
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
    config = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=lora_target_modules, lora_dropout=lora_dropout, bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, config)
    if data_path.endswith('.json') or data_path.endswith('.jsonl'):
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
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f'Checkpoint {checkpoint_name} not found')
    model.print_trainable_parameters()
    if val_set_size > 0:
        train_val = data['train'].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        train_data = train_val['train'].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val['test'].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data['train'].shuffle().map(generate_and_tokenize_prompt)
        val_data = None
    trainer = transformers.Trainer(model=model, train_dataset=train_data, eval_dataset=val_data, args=transformers.TrainingArguments(per_device_train_batch_size=micro_batch_size, gradient_accumulation_steps=gradient_accumulation_steps, max_grad_norm=0.3, num_train_epochs=num_epochs, learning_rate=learning_rate, lr_scheduler_type='cosine', bf16=True, logging_steps=1, optim='adamw_torch', evaluation_strategy='steps' if val_set_size > 0 else 'no', save_strategy='steps', eval_steps=100 if val_set_size > 0 else None, save_steps=100, output_dir=output_dir, save_total_limit=100, load_best_model_at_end=True if val_set_size > 0 else False, ddp_find_unused_parameters=False if ddp else None, group_by_length=group_by_length, report_to='wandb' if use_wandb else None, run_name=wandb_run_name if use_wandb else None, gradient_checkpointing=gradient_checkpointing, ddp_backend='ccl', deepspeed=deepspeed), data_collator=transformers.DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors='pt', padding=True))
    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(output_dir)
    print("\n If there's a warning about missing keys above, please disregard :)")
if __name__ == '__main__':
    fire.Fire(train)