import json
import math
import os
import warnings
from time import time
from typing import List, Tuple
import numpy as np
import torch
import tritonclient.grpc as client_util
import trlx.utils.logging as logging
from huggingface_hub import hf_hub_download
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizer
from trlx.data.ppo_types import PPORLElement
from trlx.models.modeling_ppo import AutoModelForCausalLMWithHydraValueHead
from trlx.pipeline import BasePipeline, register_datapipeline
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_base_trainer import AccelerateRLTrainer
from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer
from trlx.utils import Clock
from trlx.utils.modeling import logprobs_of_labels
from utils.utils import get_model
from .utils_rl import prepare_tensor
logger = logging.get_logger(__name__)

class CustomCausalLMHydraWithValueHead(AutoModelForCausalLMWithHydraValueHead):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, config, tokenizer, kwargs=None, revision=None):
        if False:
            return 10
        '\n        Our custom loader that just modifies the loading of the base model so that patching and other stuff are supported.\n        '
        base_model = get_model(config, tokenizer, pad_vocab_size_to_multiple_of=1, check_freeze_layer=False)
        model = cls(base_model, num_layers_unfrozen=config.num_layers_unfrozen)
        pretrained_model_name_or_path = config.model_name
        if isinstance(pretrained_model_name_or_path, str):
            filename = os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin')
            sharded_index_filename = os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin.index.json')
            is_sharded = False
            if not os.path.exists(filename):
                try:
                    filename = hf_hub_download(pretrained_model_name_or_path, 'pytorch_model.bin', revision=revision)
                except Exception:
                    if os.path.exists(sharded_index_filename):
                        index_file_name = sharded_index_filename
                    else:
                        index_file_name = hf_hub_download(pretrained_model_name_or_path, 'pytorch_model.bin.index.json', revision=revision)
                    with open(index_file_name, 'r') as f:
                        index = json.load(f)
                    files_to_download = set()
                    for (k, v) in index['weight_map'].items():
                        if any([module in k for module in cls._supported_modules]):
                            files_to_download.add(v)
                    is_sharded = True
            if is_sharded:
                state_dict = {}
                for shard_file in files_to_download:
                    filename = os.path.join(pretrained_model_name_or_path, shard_file)
                    if not os.path.exists(filename):
                        filename = hf_hub_download(pretrained_model_name_or_path, shard_file, revision=revision)
                    state_dict.update(torch.load(filename, map_location='cpu'))
            else:
                state_dict = torch.load(filename, map_location='cpu')
        else:
            state_dict = pretrained_model_name_or_path.state_dict()
        model.post_init(state_dict=state_dict)
        return model

@register_trainer
class CustomPPOTrainer(AcceleratePPOTrainer, AccelerateRLTrainer):

    def __init__(self, config, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.tokenizer_path)
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            self.tokenizer.add_special_tokens({'pad_token': '<|padding|>'})
        self.tokenizer.padding_side = config.tokenizer.padding_side
        self.tokenizer.truncation_side = config.tokenizer.truncation_side
        print('len self.tokenizer', len(self.tokenizer))
        super().__init__(*args, config=config, **kwargs)
        self.ref_model = triton_server_ref_model()

    def decode(self, prompts: List[torch.LongTensor], samples: List[torch.LongTensor], prompt_sizes: torch.LongTensor=None, append_eos_token: bool=True) -> Tuple[List[str], List[str], List[str]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Decode tensor generations into lists of strings (`samples`: List[str], `prompts`: List[str], `outputs`: List[str])\n        '
        assert append_eos_token is True
        if prompt_sizes is None:
            prompt_sizes = [prompts.shape[1]] * len(prompts)
        (str_samples, str_prompts, str_outputs) = ([], [], [])
        for (prompt, sample, prompt_size) in zip(prompts, samples, prompt_sizes):
            if self.config.model.model_arch_type == 'seq2seq':
                raise NotImplementedError('Decoding for seq2seq models is not implemented yet')
                output_start_ix = 0
            else:
                output_start_ix = prompt_size
            PAD_TOKEN_ID = self.tokenizer.pad_token_id
            if not torch.is_tensor(sample):
                sample = torch.tensor(sample)
            if not torch.is_tensor(prompt):
                prompt = torch.tensor(prompt)
            str_prompt = self.tokenizer.decode(prompt[:prompt_size][prompt[:prompt_size] != PAD_TOKEN_ID], skip_special_tokens=False)
            str_output = self.tokenizer.decode(sample[output_start_ix:][sample[output_start_ix:] != PAD_TOKEN_ID], skip_special_tokens=False)
            trimmed = False
            if self.stop_sequences:
                for stop in self.stop_sequences:
                    stop_ix = str_output.find(stop)
                    if stop_ix >= 0:
                        str_output = str_output[:stop_ix].rstrip()
                        trimmed = True
            if append_eos_token and (trimmed or sample[-1] != self.tokenizer.eos_token_id):
                str_output += self.tokenizer.eos_token
            str_prompts.append(str_prompt)
            str_outputs.append(str_output)
            if self.config.model.model_arch_type == 'seq2seq':
                sample = str_prompt + self.tokenizer.sep_token + str_output
            else:
                sample = str_prompt + str_output
            str_samples.append(sample)
        return (str_samples, str_prompts, str_outputs)

    def get_arch(self, config):
        if False:
            print('Hello World!')
        if config.model.model_arch_type == 'seq2seq':
            raise NotImplementedError('Seq2Seq models are not implemented yet')
        else:
            model = CustomCausalLMHydraWithValueHead.from_pretrained(config.sft_config, self.tokenizer)
        return model

    def generate(self, input_ids, *args, **kwargs):
        if False:
            print('Hello World!')
        kwargs['forced_eos_token_id'] = self.tokenizer.eos_token_id
        kwargs['suppress_tokens'] = [self.tokenizer.pad_token_id]
        preds = super().generate(input_ids, *args, **kwargs)
        return preds

    def generate_eval(self, input_ids, *args, **kwargs):
        if False:
            return 10
        kwargs['forced_eos_token_id'] = self.tokenizer.eos_token_id
        kwargs['suppress_tokens'] = [self.tokenizer.pad_token_id]
        preds = super().generate(input_ids, *args, **kwargs)
        return preds

    def make_experience(self, num_rollouts: int=1024, iter_count: int=0):
        if False:
            return 10
        '\n        Replace padding with pad_token_id\n        '
        logger.info('Collecting rollouts')
        tbar = logging.tqdm(total=num_rollouts, disable=os.environ.get('RANK', 0) != '0', desc=f'[rollout 0 / {num_rollouts}]', position=logging.get_verbosity() >= logging.WARNING, leave=logging.get_verbosity() < logging.WARNING)
        ppo_rl_elements = []
        stats = {}
        clock = Clock()
        while len(ppo_rl_elements) < num_rollouts:
            batch = next(self.prompt_iterator)
            exp_generate_time = time()
            samples = self.generate(**batch)
            stats['time/exp_generate'] = time() - exp_generate_time
            prompt_tensors = batch.input_ids
            device = samples.device
            prompt_sizes = torch.tensor([prompt_tensors.shape[1]] * len(prompt_tensors), device=device)
            padded_samples = self.accelerator.pad_across_processes(samples, dim=1, pad_index=self.tokenizer.pad_token_id, pad_first=False)
            padded_prompts = self.accelerator.pad_across_processes(prompt_tensors, dim=1, pad_index=self.tokenizer.pad_token_id, pad_first=False)
            gathered_samples = self.accelerator.gather(padded_samples)
            gathered_prompts = self.accelerator.gather(padded_prompts)
            gathered_prompt_sizes = self.accelerator.gather(prompt_sizes)
            if self.accelerator.is_main_process:
                (all_str_samples, all_str_prompts, all_str_outputs) = self.decode(gathered_prompts, gathered_samples, gathered_prompt_sizes, append_eos_token=True)
                exp_score_time = time()
                all_scores = torch.tensor(self.reward_fn(samples=all_str_samples, prompts=all_str_prompts, outputs=all_str_outputs), dtype=torch.float, device=device)
                stats['time/exp_score'] = time() - exp_score_time
                all_scores = list(all_scores.reshape(self.accelerator.num_processes, -1).unbind())
            else:
                all_scores = None
            if torch.distributed.is_initialized():
                scores = torch.empty(len(samples), device=device)
                torch.distributed.scatter(scores, all_scores)
            else:
                scores = all_scores[0].clone().detach()
            (str_samples, str_prompts, str_outputs) = self.decode(prompt_tensors, samples, append_eos_token=True)
            outputs = self.tokenizer(str_outputs).input_ids
            if self.config.model.model_arch_type == 'seq2seq':
                for i in range(len(outputs)):
                    outputs[i] = [self.tokenizer.pad_token_id] + outputs[i]
            outputs = list(map(torch.LongTensor, outputs))
            maxsize = max(map(len, outputs))
            outputs = [F.pad(output, (0, maxsize - len(output)), value=self.tokenizer.pad_token_id) for output in outputs]
            sample_outputs = torch.vstack(outputs).to(device)
            if self.ref_mean is None:
                (self.ref_mean, self.ref_std) = (scores.mean(), scores.std())
            (all_scores_mean, all_scores_std) = self.running_moments.update(scores)
            stats['exp_scores/mean'] = all_scores_mean.item()
            stats['exp_scores/std'] = all_scores_std.item()
            stats['exp_scores/running_mean'] = self.running_moments.mean.item()
            stats['exp_scores/running_std'] = self.running_moments.std.item()
            if self.config.method.scale_reward == 'running':
                scores /= self.running_moments.std
            elif self.config.method.scale_reward == 'ref':
                scores /= self.ref_std
            clip_reward = self.config.method.cliprange_reward
            if clip_reward:
                scores = torch.clip(scores, -clip_reward, clip_reward)
            if self.config.model.model_arch_type == 'seq2seq':
                raise NotImplementedError
                attention_mask = batch.attention_mask.to(device)
                prompt_tensors = batch.input_ids.to(device)
                decoder_attention_mask = sample_outputs.not_equal(self.tokenizer.pad_token_id)
                decoder_attention_mask[:, 0] = 1
                with torch.no_grad():
                    outputs = self.model(input_ids=prompt_tensors, attention_mask=attention_mask, decoder_input_ids=sample_outputs, decoder_attention_mask=decoder_attention_mask)
                    logits = outputs.logits
                    values = outputs.value
                    if hasattr(self.model, 'frozen_head'):
                        ref_logits = self.model.forward_hydra(input_ids=prompt_tensors, attention_mask=attention_mask, decoder_input_ids=sample_outputs, decoder_attention_mask=decoder_attention_mask, return_dict=True).logits
                    else:
                        ref_logits = self.ref_model(input_ids=prompt_tensors, attention_mask=attention_mask, decoder_input_ids=sample_outputs, decoder_attention_mask=decoder_attention_mask, return_dict=True).logits
            else:
                all_tokens = torch.cat((prompt_tensors.to(device), sample_outputs), dim=1)
                attention_mask = all_tokens.not_equal(self.tokenizer.pad_token_id).long().to(device)
                with torch.no_grad():
                    (logits, *_, values) = self.model(all_tokens, attention_mask=attention_mask)
                    ref_logits = self.ref_model(all_tokens, attention_mask)
                    ref_logits = ref_logits.to(device)
            if self.config.model.model_arch_type == 'seq2seq':
                logprobs = logprobs_of_labels(logits[:, :-1, :], sample_outputs[:, 1:])
                ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], sample_outputs[:, 1:])
            else:
                logprobs = logprobs_of_labels(logits[:, :-1, :], all_tokens[:, 1:])
                ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], all_tokens[:, 1:])
            n_samples: int = samples.shape[0]
            if self.config.model.model_arch_type == 'seq2seq':
                attention_mask = sample_outputs != self.tokenizer.pad_token_id
                start = 0
            else:
                start = prompt_tensors.shape[1] - 1
            log_ratio = (logprobs - ref_logprobs) * attention_mask[:, :-1]
            self.mean_kl = (log_ratio.exp() - 1 - log_ratio).mean().to(device)
            logprobs = logprobs.cpu()
            ref_logprobs = ref_logprobs.cpu()
            prompt_tensors = prompt_tensors.cpu()
            sample_outputs = sample_outputs.cpu()
            values = values.cpu()[:, :-1]
            ends = start + attention_mask[:, start:].sum(1) + 1
            all_values = [values[ix, start:ends[ix]] for ix in range(n_samples)]
            all_logprobs = [logprobs[ix, start:ends[ix]] for ix in range(n_samples)]
            kl_penalty = self.kl_ctl.value * -log_ratio.cpu()
            kl_penalty = [xs[start:ends[ix]] for (ix, xs) in enumerate(kl_penalty)]
            rollout_count = 0
            for sample_idx in range(n_samples):
                rewards = kl_penalty[sample_idx]
                rewards[-1] += scores[sample_idx].cpu()
                ppo_rl_elements.append(PPORLElement(query_tensor=prompt_tensors[sample_idx], response_tensor=sample_outputs[sample_idx], logprobs=all_logprobs[sample_idx], values=all_values[sample_idx], rewards=rewards))
                rollout_count += 1
            exp_time = clock.tick()
            tbar.set_description(f'[rollout {len(ppo_rl_elements)} / {num_rollouts}]')
            tbar.update(min(rollout_count, num_rollouts))
        tbar.close()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.mean_kl, torch.distributed.ReduceOp.AVG)
        stats['policy/sqrt_kl'] = torch.sqrt(self.mean_kl).item()
        stats['kl_ctl_value'] = self.kl_ctl.value
        stats['time/exp'] = exp_time
        self.accelerator.log(stats, step=iter_count)
        self.push_to_store(ppo_rl_elements)

def triton_server_ref_model():
    if False:
        for i in range(10):
            print('nop')
    triton_host = os.environ.get('TRITON_HOST_REF')
    assert triton_host is not None, 'Specify reference model in the TRITON_HOST_REF environmental variable'
    (triton_url, triton_model) = triton_host.split('/')
    client = client_util.InferenceServerClient(url=triton_url, verbose=False)

    def ref_model(all_tokens, attention_masks):
        if False:
            while True:
                i = 10
        mbs = 8
        all_tokens = all_tokens.detach().cpu().numpy()
        attention_masks = attention_masks.detach().cpu().numpy()
        out = []
        for i in range(math.ceil(len(all_tokens) / mbs)):
            batch_ixs = slice(i * mbs, (i + 1) * mbs)
            result = client.infer(triton_model, [prepare_tensor('input_ids', all_tokens[batch_ixs].astype(np.int32)), prepare_tensor('attention_mask', attention_masks[batch_ixs].astype(np.int32))])
            logits = result.as_numpy('logits')
            out.append(torch.tensor(logits))
        return torch.cat(out, dim=0)
    return ref_model

@register_datapipeline
class CustomPromptPipeline(BasePipeline):
    """
    Tokenizes prompts, unless they are already tokenized, and truncates them to `max_prompt_length` from the right
    """

    def __init__(self, prompts: List[str], max_prompt_length: int, tokenizer: PreTrainedTokenizer):
        if False:
            return 10
        super().__init__()
        if max_prompt_length < 16:
            raise ValueError(f'`max_prompt_length` is {max_prompt_length}, this is too small (less than 16). Make sure all the config values are correct, when in doubt increase `seq_len` or decrease `max_new_tokens`.')
        model_inputs = tokenizer(prompts, truncation=True, padding=True, max_length=max_prompt_length, add_special_tokens=False)
        prompts_tokens_ = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']
        for prompt_tokens in prompts_tokens_:
            if tokenizer.eos_token_id not in prompt_tokens:
                warnings.warn('Found a prompt without an EOS token, which means it was truncated. Consider increasing the context size (`seq_len`)')
                break
        prompts_tokens = prompts_tokens_
        self.tokenizer = tokenizer
        self.prompts = [{'input_ids': tokens, 'attention_mask': mask} for (tokens, mask) in zip(prompts_tokens, attention_mask)]

    def __getitem__(self, ix: int):
        if False:
            i = 10
            return i + 15
        return self.prompts[ix]

    def __len__(self) -> int:
        if False:
            while True:
                i = 10
        return len(self.prompts)

    def create_loader(self, batch_size: int, shuffle=False) -> DataLoader:
        if False:
            i = 10
            return i + 15
        collate_fn = DataCollatorWithPadding(self.tokenizer) if self.tokenizer else torch.vstack
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)