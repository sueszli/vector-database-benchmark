import argparse
import math
import os
import random
from argparse import Namespace
from typing import Sequence
import numpy as np
import torch
import transformers
import tritonclient.grpc as client_util
import trlx
from model_training.custom_datasets.formatting import QA_SPECIAL_TOKENS, format_pairs
from model_training.models import get_specific_model
from model_training.utils.utils import _strtobool, get_dataset, init_rng, read_yamls
from tritonclient.utils import np_to_triton_dtype
from trlx.data.configs import TRLConfig
from utils.ppo_utils import CustomPPOTrainer
from utils.utils import _strtobool, get_dataset, get_model, init_rng, read_yamls
from utils.utils_rl import prepare_tensor

def argument_parsing(notebook: bool=False, notebook_args: Sequence[str] | None=None, **kwargs):
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--wandb-entity', type=str, default='open-assistant')
    parser.add_argument('--rng_seed', type=int, help='rng seed')
    if notebook:
        (args, remaining) = parser.parse_known_args(notebook_args)
    else:
        (args, remaining) = parser.parse_known_args()
    conf = {}
    configs = read_yamls('./configs')
    for name in args.configs:
        if ',' in name:
            for n in name.split(','):
                conf.update(configs[n])
        else:
            conf.update(configs[name])
    conf['local_rank'] = args.local_rank
    if args.rng_seed is not None:
        conf['rng_seed'] = args.rng_seed
    parser = argparse.ArgumentParser()
    for (key, value) in kwargs.items():
        type_ = type(value) if value is not None else str
        parser.add_argument(f'--{key}', type=type_, default=value)
    for (key, value) in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = _strtobool
        parser.add_argument(f'--{key}', type=type_, default=value)
    return parser.parse_args(remaining)

def create_reward_fn(rank_config, sft_config):
    if False:
        i = 10
        return i + 15
    triton_host = os.environ.get('TRITON_HOST_RM')
    assert triton_host is not None, 'Specify reward model in the TRITON_HOST_RM environmental variable'
    (triton_url, triton_model) = triton_host.split('/')
    client = client_util.InferenceServerClient(url=triton_url, verbose=False)
    rank_tokenizer = transformers.AutoTokenizer.from_pretrained(rank_config.model_name, cache_dir=rank_config.cache_dir)
    sft_tokenizer = transformers.AutoTokenizer.from_pretrained(sft_config.model_name, cache_dir=sft_config.cache_dir)

    def reward_fn(samples, prompts, outputs):
        if False:
            while True:
                i = 10
        if len(samples) == 0:
            return []
        samples = [x.replace(sft_tokenizer.eos_token, rank_tokenizer.eos_token) for x in samples]
        samples = [x.replace(sft_tokenizer.pad_token, rank_tokenizer.pad_token) for x in samples]
        inputs = rank_tokenizer(samples, return_tensors='np', padding=True)
        mbs = rank_config.batch_size
        out = []
        for i in range(math.ceil(len(samples) / mbs)):
            batch_ixs = slice(i * mbs, (i + 1) * mbs)
            result = client.infer(triton_model, [prepare_tensor('input_ids', inputs.input_ids[batch_ixs].astype(np.int32)), prepare_tensor('attention_mask', inputs.attention_mask[batch_ixs].astype(np.int32))])
            rewards = result.as_numpy('rewards')
            out.extend(rewards)
        return out
    return reward_fn

def main():
    if False:
        for i in range(10):
            print('nop')
    training_conf = argument_parsing()
    rank_config = Namespace(**training_conf.rank_config)
    sft_config = Namespace(**training_conf.sft_config)
    triton_host_rm = os.getenv('TRITON_HOST_RM', training_conf.triton_host_rm)
    triton_host_sft = os.getenv('TRITON_HOST_REF', training_conf.triton_host_sft)
    os.environ['TRITON_HOST_RM'] = triton_host_rm
    os.environ['TRITON_HOST_REF'] = triton_host_sft
    init_rng(training_conf)
    eos_token = transformers.AutoTokenizer.from_pretrained(sft_config.model_name, cache_dir=sft_config.cache_dir).eos_token
    trlx_config = TRLConfig.load_yaml('configs/ppo_config.yaml')
    trlx_config.sft_config = sft_config
    (train, eval_dict) = get_dataset(training_conf, mode='rl')
    eval = eval_dict['oasst_export'] if 'oasst_export' in eval_dict else eval_dict[next(iter(eval_dict))]
    (prompts, eval_prompts) = tuple(map(lambda x: [''.join(format_pairs(x[i][0], eos_token, add_initial_reply_token=True)) for i in range(len(x))], (train, eval)))
    eval_prompts = [''.join(format_pairs(['Can you tell me about GLaDOS?'], eos_token, add_initial_reply_token=True)), ''.join(format_pairs(['What is the chemical symbol for gold?'], eos_token, add_initial_reply_token=True)), ''.join(format_pairs(['If you were the President of the United States, what would you do?'], eos_token, add_initial_reply_token=True))] + eval_prompts
    if training_conf.num_eval_prompts is not None and training_conf.num_eval_prompts > 0:
        eval_prompts = eval_prompts[:training_conf.num_eval_prompts]
    random.shuffle(prompts)
    with open('output.txt', 'w') as fp:
        for item in eval_prompts:
            fp.write('Prompt For RL: %s\n' % item)
    trlx_config.tokenizer.tokenizer_path = sft_config.model_name
    trlx_config.model.model_path = sft_config.model_name
    trlx_config.train.batch_size = int(training_conf.batch_size)
    trlx_config.method.chunk_size = int(training_conf.chunk_size)
    trlx_config.method.num_rollouts = int(training_conf.num_rollouts)
    trlx_config.train.total_steps = int(training_conf.total_steps)
    if training_conf.debug:
        print('Continuing in debug mode')
        prompts = prompts[:10]
        eval_prompts = eval_prompts[:10]
        trlx_config.method.num_rollouts = 1
    trainer = trlx.train(sft_config.model_name, reward_fn=create_reward_fn(rank_config, sft_config), prompts=prompts, eval_prompts=eval_prompts, config=trlx_config, stop_sequences=[eos_token])
    training_conf.output_dir = training_conf.output_dir if training_conf.output_dir else training_conf.model_name
    trainer.save_pretrained(training_conf.output_dir)
if __name__ == '__main__':
    main()