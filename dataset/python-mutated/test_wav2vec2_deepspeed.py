import sys
from pathlib import Path
git_repo_path = Path(__file__).resolve().parents[3] / 'src'
sys.path.insert(1, str(git_repo_path))
import dataclasses
import io
import itertools
import json
import os
import unittest
from copy import deepcopy
from parameterized import parameterized
from transformers import TrainingArguments, is_torch_available
from transformers.integrations.deepspeed import is_deepspeed_available
from transformers.file_utils import WEIGHTS_NAME
from transformers.testing_utils import CaptureLogger, ExtendSysPath, TestCasePlus, execute_subprocess_async, get_gpu_count, mockenv_context, require_deepspeed, require_torch_gpu, require_torch_multi_gpu, slow
from transformers.trainer_utils import set_seed
set_seed(42)
models = {'base': 'patrickvonplaten/wav2vec2_tiny_random', 'robust': 'patrickvonplaten/wav2vec2_tiny_random_robust'}
ZERO2 = 'zero2'
ZERO3 = 'zero3'
stages = [ZERO2, ZERO3]

def custom_name_func(func, param_num, param):
    if False:
        return 10
    param_based_name = parameterized.to_safe_name('_'.join((str(x) for x in param.args)))
    return f'{func.__name__}_{param_based_name}'
params = list(itertools.product(stages, models.keys()))

@slow
@require_deepspeed
@require_torch_gpu
class TestDeepSpeedWav2Vec2(TestCasePlus):

    @parameterized.expand(params, name_func=custom_name_func)
    def test_fp32_non_distributed(self, stage, model):
        if False:
            return 10
        self.run_and_check(stage=stage, model=model, distributed=False, fp16=False)

    @require_torch_multi_gpu
    @parameterized.expand(params, name_func=custom_name_func)
    def test_fp32_distributed(self, stage, model):
        if False:
            return 10
        self.run_and_check(stage=stage, model=model, distributed=True, fp16=False)

    @parameterized.expand(params, name_func=custom_name_func)
    def test_fp16_non_distributed(self, stage, model):
        if False:
            print('Hello World!')
        self.run_and_check(stage=stage, model=model, distributed=False, fp16=True)

    @require_torch_multi_gpu
    @parameterized.expand(params, name_func=custom_name_func)
    def test_fp16_distributed(self, stage, model):
        if False:
            for i in range(10):
                print('nop')
        self.run_and_check(stage=stage, model=model, distributed=True, fp16=True)

    def do_checks(self, output_dir):
        if False:
            i = 10
            return i + 15
        pass

    def run_and_check(self, stage: str, model: str, eval_steps: int=10, distributed: bool=True, quality_checks: bool=True, fp16: bool=True):
        if False:
            while True:
                i = 10
        model_name = models[model]
        output_dir = self.run_trainer(stage=stage, model_name=model_name, eval_steps=eval_steps, num_train_epochs=1, distributed=distributed, fp16=fp16)
        self.do_checks(output_dir)
        return output_dir

    def run_trainer(self, stage: str, model_name: str, eval_steps: int=10, num_train_epochs: int=1, distributed: bool=True, fp16: bool=True):
        if False:
            print('Hello World!')
        output_dir = self.get_auto_remove_tmp_dir('./xxx', after=False)
        args = f'\n            --model_name_or_path {model_name}\n            --dataset_name hf-internal-testing/librispeech_asr_dummy\n            --dataset_config_name clean\n            --train_split_name validation\n            --validation_split_name validation\n            --output_dir {output_dir}\n            --num_train_epochs {str(num_train_epochs)}\n            --per_device_train_batch_size 2\n            --per_device_eval_batch_size 2\n            --evaluation_strategy steps\n            --learning_rate 5e-4\n            --warmup_steps 8\n            --orthography timit\n            --preprocessing_num_workers 1\n            --group_by_length\n            --freeze_feature_extractor\n            --report_to none\n            --save_steps 0\n            --eval_steps {eval_steps}\n            --report_to none\n        '.split()
        if fp16:
            args.extend(['--fp16'])
        ds_args = f'--deepspeed {self.test_file_dir_str}/ds_config_wav2vec2_{stage}.json'.split()
        script = [f'{self.examples_dir_str}/research_projects/wav2vec2/run_asr.py']
        launcher = self.get_launcher(distributed)
        cmd = launcher + script + args + ds_args
        execute_subprocess_async(cmd, env=self.get_env())
        return output_dir

    def get_launcher(self, distributed=False):
        if False:
            print('Hello World!')
        num_gpus = min(2, get_gpu_count()) if distributed else 1
        return f'deepspeed --num_nodes 1 --num_gpus {num_gpus}'.split()