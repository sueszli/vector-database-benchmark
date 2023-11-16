import torch
import os
import transformers
from transformers import LlamaTokenizer
from peft import LoraConfig
from bigdl.llm.transformers.qlora import get_peft_model, prepare_model_for_kbit_training
from bigdl.llm.transformers import AutoModelForCausalLM
from datasets import load_dataset
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default='meta-llama/Llama-2-7b-hf', help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded, or the path to the huggingface checkpoint folder')
    parser.add_argument('--dataset', type=str, default='Abirate/english_quotes')
    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    dataset_path = args.dataset
    tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)
    data = load_dataset(dataset_path)

    def merge(row):
        if False:
            return 10
        row['prediction'] = row['quote'] + ' ->: ' + str(row['tags'])
        return row
    data = data.map(lambda samples: tokenizer(samples['prediction']), batched=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, load_in_low_bit='sym_int4', optimize_model=False, torch_dtype=torch.float16, modules_to_not_convert=['lm_head'])
    model = model.to('cpu')
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.enable_input_require_grads()
    config = LoraConfig(r=8, lora_alpha=32, target_modules=['q_proj', 'k_proj', 'v_proj'], lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
    model = get_peft_model(model, config)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'
    trainer = transformers.Trainer(model=model, train_dataset=data['train'], args=transformers.TrainingArguments(per_device_train_batch_size=4, gradient_accumulation_steps=1, warmup_steps=20, max_steps=200, learning_rate=0.0002, save_steps=100, bf16=True, logging_steps=20, output_dir='outputs', optim='adamw_hf'), data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False))
    model.config.use_cache = False
    result = trainer.train()
    print(result)