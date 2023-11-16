import math
import os
import re
import sys
from pathlib import Path
from typing import Tuple
from unittest.mock import patch
from parameterized import parameterized
from transformers.testing_utils import CaptureStderr, ExtendSysPath, TestCasePlus, backend_device_count, execute_subprocess_async, get_torch_dist_unique_port, require_apex, require_bitsandbytes, require_torch, require_torch_gpu, require_torch_multi_accelerator, require_torch_non_multi_accelerator, slow, torch_device
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import set_seed
bindir = os.path.abspath(os.path.dirname(__file__))
with ExtendSysPath(f'{bindir}/../../examples/pytorch/translation'):
    from run_translation import main
set_seed(42)
MARIAN_MODEL = 'sshleifer/student_marian_en_ro_6_1'
MBART_TINY = 'sshleifer/tiny-mbart'

@require_torch
class TestTrainerExt(TestCasePlus):

    def run_seq2seq_quick(self, distributed=False, extra_args_str=None, predict_with_generate=True, do_train=True, do_eval=True, do_predict=True):
        if False:
            return 10
        output_dir = self.run_trainer(eval_steps=1, max_len=12, model_name=MBART_TINY, num_train_epochs=1, distributed=distributed, extra_args_str=extra_args_str, predict_with_generate=predict_with_generate, do_train=do_train, do_eval=do_eval, do_predict=do_predict)
        logs = TrainerState.load_from_json(os.path.join(output_dir, 'trainer_state.json')).log_history
        if not do_eval:
            return
        eval_metrics = [log for log in logs if 'eval_loss' in log.keys()]
        first_step_stats = eval_metrics[0]
        if predict_with_generate:
            assert 'eval_bleu' in first_step_stats
            last_step_stats = eval_metrics[-1]
            assert isinstance(last_step_stats['eval_bleu'], float)
            assert not math.isnan(float(last_step_stats['eval_loss'])), 'eval_loss must not be `nan`'

    @require_torch_non_multi_accelerator
    def test_run_seq2seq_no_dist(self):
        if False:
            while True:
                i = 10
        self.run_seq2seq_quick()

    @require_torch_multi_accelerator
    def test_run_seq2seq_dp(self):
        if False:
            i = 10
            return i + 15
        self.run_seq2seq_quick(distributed=False)

    @require_torch_multi_accelerator
    def test_run_seq2seq_ddp(self):
        if False:
            print('Hello World!')
        self.run_seq2seq_quick(distributed=True)

    @require_apex
    @require_torch_gpu
    def test_run_seq2seq_apex(self):
        if False:
            print('Hello World!')
        self.run_seq2seq_quick(distributed=True, extra_args_str='--fp16 --fp16_backend=apex')
        self.run_seq2seq_quick(distributed=True, extra_args_str='--fp16 --fp16_backend=apex')

    @parameterized.expand(['base', 'low', 'high', 'mixed'])
    @require_torch_multi_accelerator
    def test_trainer_log_level_replica(self, experiment_id):
        if False:
            while True:
                i = 10
        experiments = {'base': {'extra_args_str': '', 'n_matches': 1}, 'low': {'extra_args_str': '--log_level debug --log_level_replica debug', 'n_matches': 2}, 'high': {'extra_args_str': '--log_level error --log_level_replica debug', 'n_matches': 1}, 'mixed': {'extra_args_str': '--log_level error --log_level_replica error', 'n_matches': 0}}
        data = experiments[experiment_id]
        kwargs = {'distributed': True, 'predict_with_generate': False, 'do_eval': False, 'do_predict': False}
        log_info_string = 'Running training'
        with CaptureStderr() as cl:
            self.run_seq2seq_quick(**kwargs, extra_args_str=data['extra_args_str'])
        n_matches = len(re.findall(log_info_string, cl.err))
        self.assertEqual(n_matches, data['n_matches'])

    @slow
    def test_run_seq2seq(self):
        if False:
            i = 10
            return i + 15
        output_dir = self.run_trainer(eval_steps=2, max_len=128, model_name=MARIAN_MODEL, learning_rate=0.0003, num_train_epochs=10, distributed=False)
        logs = TrainerState.load_from_json(os.path.join(output_dir, 'trainer_state.json')).log_history
        eval_metrics = [log for log in logs if 'eval_loss' in log.keys()]
        first_step_stats = eval_metrics[0]
        last_step_stats = eval_metrics[-1]
        assert first_step_stats['eval_loss'] > last_step_stats['eval_loss'], 'model learned nothing'
        assert isinstance(last_step_stats['eval_bleu'], float)
        contents = os.listdir(output_dir)
        contents = {os.path.basename(p) for p in contents}
        assert 'generated_predictions.txt' in contents
        assert 'predict_results.json' in contents

    @slow
    @require_bitsandbytes
    def test_run_seq2seq_bnb(self):
        if False:
            i = 10
            return i + 15
        from transformers.training_args import OptimizerNames

        def train_and_return_metrics(optim: str) -> Tuple[int, float]:
            if False:
                print('Hello World!')
            extra_args = '--skip_memory_metrics 0'
            output_dir = self.run_trainer(max_len=128, model_name=MARIAN_MODEL, learning_rate=0.0003, num_train_epochs=1, optim=optim, distributed=True, extra_args_str=extra_args, do_eval=False, do_predict=False, n_gpus_to_use=1)
            logs = TrainerState.load_from_json(Path(output_dir, 'trainer_state.json')).log_history
            gpu_peak_mem_mb = int(logs[0]['train_mem_gpu_peaked_delta'] / 2 ** 20)
            gpu_alloc_mem_mb = int(logs[0]['train_mem_gpu_alloc_delta'] / 2 ** 20)
            loss = logs[0]['train_loss']
            return (gpu_peak_mem_mb, gpu_alloc_mem_mb, loss)
        (gpu_peak_mem_orig, gpu_alloc_mem_orig, loss_orig) = train_and_return_metrics(OptimizerNames.ADAMW_TORCH.value)
        (gpu_peak_mem_bnb, gpu_alloc_mem_bnb, loss_bnb) = train_and_return_metrics(OptimizerNames.ADAMW_BNB.value)
        gpu_alloc_mem_diff = gpu_alloc_mem_orig - gpu_alloc_mem_bnb
        gpu_total_mem_orig = gpu_peak_mem_orig + gpu_alloc_mem_orig
        gpu_total_mem_bnb = gpu_peak_mem_bnb + gpu_alloc_mem_bnb
        gpu_total_mem_diff = gpu_total_mem_orig - gpu_total_mem_bnb
        expected_savings = 120
        self.assertGreater(gpu_alloc_mem_diff, expected_savings, f'should use ~150MB less alloc gpu memory with BNB, compared to without it for this model but got a difference of {gpu_alloc_mem_diff}MB, with gpu_alloc_mem_orig={gpu_alloc_mem_orig}MB and gpu_alloc_mem_bnb={gpu_alloc_mem_bnb}MB')
        self.assertGreater(gpu_total_mem_diff, expected_savings, f'should use ~150MB less total gpu memory with BNB, compared to without it for this model but got a difference of {gpu_total_mem_diff}MB, with gpu_total_mem_orig={gpu_total_mem_orig}MB and gpu_total_mem_bnb={gpu_total_mem_bnb}MB')
        self.assertEqual(loss_orig, loss_bnb, f'loss should be the same, but got loss_orig={loss_orig}, loss_bnb={loss_bnb}')

    def run_trainer(self, max_len: int, model_name: str, num_train_epochs: int, learning_rate: float=0.003, optim: str='adafactor', distributed: bool=False, extra_args_str: str=None, eval_steps: int=0, predict_with_generate: bool=True, do_train: bool=True, do_eval: bool=True, do_predict: bool=True, n_gpus_to_use: int=None):
        if False:
            for i in range(10):
                print('nop')
        data_dir = self.test_file_dir / '../fixtures/tests_samples/wmt_en_ro'
        output_dir = self.get_auto_remove_tmp_dir()
        args_train = f'\n            --model_name_or_path {model_name}\n            --train_file {data_dir}/train.json\n            --validation_file {data_dir}/val.json\n            --test_file {data_dir}/test.json\n            --output_dir {output_dir}\n            --overwrite_output_dir\n            --max_train_samples 8\n            --max_source_length {max_len}\n            --max_target_length {max_len}\n            --do_train\n            --num_train_epochs {str(num_train_epochs)}\n            --per_device_train_batch_size 4\n            --learning_rate {learning_rate}\n            --warmup_steps 8\n            --logging_steps 0\n            --logging_strategy no\n            --save_steps {str(eval_steps)}\n            --group_by_length\n            --label_smoothing_factor 0.1\n            --target_lang ro_RO\n            --source_lang en_XX\n        '.split()
        args_eval = f'\n            --do_eval\n            --per_device_eval_batch_size 4\n            --max_eval_samples 8\n            --val_max_target_length {max_len}\n            --evaluation_strategy steps\n            --eval_steps {str(eval_steps)}\n        '.split()
        args_predict = '\n            --do_predict\n        '.split()
        args = []
        if do_train:
            args += args_train
        if do_eval:
            args += args_eval
        if do_predict:
            args += args_predict
        if predict_with_generate:
            args += '--predict_with_generate'.split()
        if do_train:
            if optim == 'adafactor':
                args += '--adafactor'.split()
            else:
                args += f'--optim {optim}'.split()
        if extra_args_str is not None:
            args += extra_args_str.split()
        if distributed:
            if n_gpus_to_use is None:
                n_gpus_to_use = backend_device_count(torch_device)
            master_port = get_torch_dist_unique_port()
            distributed_args = f'\n                -m torch.distributed.run\n                --nproc_per_node={n_gpus_to_use}\n                --master_port={master_port}\n                {self.examples_dir_str}/pytorch/translation/run_translation.py\n            '.split()
            cmd = [sys.executable] + distributed_args + args
            execute_subprocess_async(cmd, env=self.get_env())
        else:
            testargs = ['run_translation.py'] + args
            with patch.object(sys, 'argv', testargs):
                main()
        return output_dir