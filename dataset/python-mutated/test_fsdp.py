import itertools
import os
import unittest
from functools import partial
from parameterized import parameterized
import tests.trainer.test_trainer
from tests.trainer.test_trainer import TrainerIntegrationCommon
from transformers import is_torch_available
from transformers.testing_utils import TestCasePlus, backend_device_count, execute_subprocess_async, mockenv_context, require_accelerate, require_fsdp, require_torch_accelerator, require_torch_multi_accelerator, slow, torch_device
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import FSDPOption, set_seed
from transformers.utils import is_accelerate_available, is_torch_bf16_available_on_device
if is_torch_available():
    from transformers.pytorch_utils import is_torch_greater_or_equal_than_2_1
else:
    is_torch_greater_or_equal_than_2_1 = False
DEFAULT_MASTER_PORT = '10999'
dtypes = ['fp16']
if is_torch_bf16_available_on_device(torch_device):
    dtypes += ['bf16']
sharding_strategies = ['full_shard', 'shard_grad_op']
state_dict_types = ['FULL_STATE_DICT', 'SHARDED_STATE_DICT']
set_seed(42)
params = list(itertools.product(sharding_strategies, dtypes))

def get_master_port(real_launcher=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    When using a single gpu launcher emulation (i.e. not deepspeed or python -m torch.distributed)\n    the issue is that once the port is tied it can't be used anywhere else outside of this process,\n    since torch.dist doesn't free the port until the process exits. Therefore for the sake of being\n    able to run both emulated launcher and normal launcher tests we need 2 distinct ports.\n\n    This function will give the right port in the right context. For real launcher it'll give the\n    base port, for emulated launcher it'll give the base port + 1. In both cases a string is\n    returned.\n\n    Args:\n        `real_launcher`: whether a real launcher is going to be used, or the emulated one\n\n    "
    master_port_base = os.environ.get('DS_TEST_PORT', DEFAULT_MASTER_PORT)
    if not real_launcher:
        master_port_base = str(int(master_port_base) + 1)
    return master_port_base
if is_torch_available():
    from tests.trainer.test_trainer import RegressionModelConfig, RegressionPreTrainedModel
    get_regression_trainer = partial(tests.trainer.test_trainer.get_regression_trainer, log_level='info')
require_fsdp_version = require_fsdp
if is_accelerate_available():
    from accelerate.utils.constants import FSDP_PYTORCH_VERSION, FSDP_SHARDING_STRATEGY
    require_fsdp_version = partial(require_fsdp, min_version=FSDP_PYTORCH_VERSION)

def get_launcher(distributed=False, use_accelerate=False):
    if False:
        while True:
            i = 10
    num_gpus = min(2, backend_device_count(torch_device)) if distributed else 1
    master_port = get_master_port(real_launcher=True)
    if use_accelerate:
        return f'accelerate launch\n        --num_processes {num_gpus}\n        --main_process_port {master_port}\n        --use_fsdp\n        --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP\n        --fsdp_state_dict_type SHARDED_STATE_DICT\n        --fsdp_transformer_layer_cls_to_wrap BertLayer'.split()
    return f'torchrun --nnodes 1 --nproc-per-node {num_gpus} --master-port {master_port}'.split()

def _parameterized_custom_name_func(func, param_num, param):
    if False:
        print('Hello World!')
    param_based_name = parameterized.to_safe_name('_'.join((str(x) for x in param.args)))
    return f'{func.__name__}_{param_based_name}'

@require_accelerate
@require_torch_accelerator
@require_fsdp_version
class TrainerIntegrationFSDP(TestCasePlus, TrainerIntegrationCommon):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        master_port = get_master_port(real_launcher=False)
        self.dist_env_1_gpu = {'MASTER_ADDR': 'localhost', 'MASTER_PORT': master_port, 'RANK': '0', 'LOCAL_RANK': '0', 'WORLD_SIZE': '1'}
        self.fsdp_config = {'backward_prefetch': 'backward_pre', 'forward_prefetch': 'False', 'limit_all_gathers': 'False', 'use_orig_params': 'True', 'sync_module_states': 'True', 'activation_checkpointing': 'False', 'min_num_params': 1}

    def tearDown(self):
        if False:
            while True:
                i = 10
        super().tearDown()

    @parameterized.expand(params, name_func=_parameterized_custom_name_func)
    def test_fsdp_config(self, sharding_strategy, dtype):
        if False:
            print('Hello World!')
        output_dir = self.get_auto_remove_tmp_dir()
        kwargs = {'output_dir': output_dir, 'train_len': 128, 'save_steps': 5, 'learning_rate': 0.1, 'fsdp': f'{sharding_strategy} offload auto_wrap', 'fsdp_config': self.fsdp_config}
        kwargs[dtype] = True
        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(**kwargs)
            self.assertEqual(trainer.args.fsdp[0], sharding_strategy)
            self.assertEqual(trainer.args.fsdp[1], FSDPOption.OFFLOAD)
            self.assertEqual(trainer.args.fsdp[2], FSDPOption.AUTO_WRAP)
            for (k, v) in trainer.args.fsdp_config.items():
                self.assertEqual(v, self.fsdp_config[k])
            self.assertEqual(os.environ.get('ACCELERATE_USE_FSDP', 'false'), 'true')

    @parameterized.expand(params, name_func=_parameterized_custom_name_func)
    @require_torch_multi_accelerator
    @slow
    def test_basic_run(self, sharding_strategy, dtype):
        if False:
            return 10
        launcher = get_launcher(distributed=True, use_accelerate=False)
        output_dir = self.get_auto_remove_tmp_dir()
        args = self.get_base_args(output_dir, 1, 50).split() + [f'--{dtype}']
        fsdp_args = ['--fsdp', f'{sharding_strategy} auto_wrap', '--fsdp_transformer_layer_cls_to_wrap', 'BertLayer']
        script = [f'{self.examples_dir_str}/pytorch/text-classification/run_glue.py']
        cmd = launcher + script + args + fsdp_args
        execute_subprocess_async(cmd, env=self.get_env())

    @parameterized.expand(dtypes)
    @require_torch_multi_accelerator
    @slow
    @unittest.skipIf(not is_torch_greater_or_equal_than_2_1, reason='This test on pytorch 2.0 takes 4 hours.')
    def test_basic_run_with_cpu_offload(self, dtype):
        if False:
            return 10
        launcher = get_launcher(distributed=True, use_accelerate=False)
        output_dir = self.get_auto_remove_tmp_dir()
        args = self.get_base_args(output_dir, 1, 50).split() + [f'--{dtype}', '--max_steps', '10']
        fsdp_args = ['--fsdp', 'full_shard auto_wrap offload', '--fsdp_transformer_layer_cls_to_wrap', 'BertLayer']
        script = [f'{self.examples_dir_str}/pytorch/text-classification/run_glue.py']
        cmd = launcher + script + args + fsdp_args
        execute_subprocess_async(cmd, env=self.get_env())

    @parameterized.expand(state_dict_types, name_func=_parameterized_custom_name_func)
    @require_torch_multi_accelerator
    @slow
    def test_training_and_can_resume_normally(self, state_dict_type):
        if False:
            i = 10
            return i + 15
        output_dir = self.get_auto_remove_tmp_dir('./xxx', after=False)
        sharding_strategy = 'full_shard'
        use_accelerate = state_dict_type == 'SHARDED_STATE_DICT'
        launcher = get_launcher(True, use_accelerate=use_accelerate)
        args = self.get_base_args(output_dir, 2, 25).split()
        script = [f'{self.examples_dir_str}/pytorch/text-classification/run_glue.py']
        logs = self.run_cmd_and_get_logs(use_accelerate, sharding_strategy, launcher, script, args, output_dir)
        checkpoint = os.path.join(output_dir, 'checkpoint-115')
        resume_args = args + f'--resume_from_checkpoint {checkpoint}'.split()
        logs_resume = self.run_cmd_and_get_logs(use_accelerate, sharding_strategy, launcher, script, resume_args, output_dir)
        for (log, log1) in zip(logs, logs_resume):
            if 'learning_rate' in log:
                self.assertAlmostEqual(log['learning_rate'], log1['learning_rate'], delta=1e-05)

    def run_cmd_and_get_logs(self, use_accelerate, sharding_strategy, launcher, script, args, output_dir):
        if False:
            i = 10
            return i + 15
        if not use_accelerate:
            fsdp_args = ['--fsdp', f'{sharding_strategy} auto_wrap', '--fsdp_transformer_layer_cls_to_wrap', 'BertLayer']
            cmd = launcher + script + args + fsdp_args
        else:
            fsdp_config = f'\n                --fsdp_sharding_strategy {FSDP_SHARDING_STRATEGY.index(sharding_strategy.upper()) + 1}\n            '.split()
            cmd = launcher + fsdp_config + script + args
        execute_subprocess_async(cmd, env=self.get_env())
        logs = TrainerState.load_from_json(os.path.join(output_dir, 'trainer_state.json')).log_history
        return logs

    def get_base_args(self, output_dir, num_epochs, logging_steps):
        if False:
            return 10
        return f'\n            --model_name_or_path bert-base-cased\n            --task_name mrpc\n            --output_dir {output_dir}\n            --overwrite_output_dir\n            --do_train\n            --max_seq_length 128\n            --per_device_train_batch_size 16\n            --learning_rate 5e-5\n            --num_train_epochs {num_epochs}\n            --lr_scheduler_type cosine\n            --logging_steps {logging_steps}\n            --save_strategy epoch\n            --do_eval\n            --evaluation_strategy epoch\n            --report_to none\n        '