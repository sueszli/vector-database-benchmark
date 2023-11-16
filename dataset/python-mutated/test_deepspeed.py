import dataclasses
import io
import itertools
import json
import os
import unittest
from copy import deepcopy
from functools import partial
import datasets
from parameterized import parameterized
import tests.trainer.test_trainer
from tests.trainer.test_trainer import TrainerIntegrationCommon
from transformers import AutoModel, TrainingArguments, is_torch_available, logging
from transformers.integrations.deepspeed import HfDeepSpeedConfig, is_deepspeed_available, unset_hf_deepspeed_config
from transformers.testing_utils import CaptureLogger, CaptureStd, CaptureStderr, LoggingLevel, TestCasePlus, backend_device_count, execute_subprocess_async, mockenv_context, require_deepspeed, require_optuna, require_torch_accelerator, require_torch_multi_accelerator, slow, torch_device
from transformers.trainer_utils import get_last_checkpoint, set_seed
from transformers.utils import SAFE_WEIGHTS_NAME, is_torch_bf16_available_on_device
if is_torch_available():
    from tests.trainer.test_trainer import RegressionModelConfig, RegressionPreTrainedModel
    get_regression_trainer = partial(tests.trainer.test_trainer.get_regression_trainer, log_level='info')
set_seed(42)
DEFAULT_MASTER_PORT = '10999'
T5_SMALL = 't5-small'
T5_TINY = 'patrickvonplaten/t5-tiny-random'
GPT2_TINY = 'sshleifer/tiny-gpt2'

def load_json(path):
    if False:
        return 10
    with open(path) as f:
        return json.load(f)

def get_master_port(real_launcher=False):
    if False:
        for i in range(10):
            print('nop')
    "\n    When using a single gpu launcher emulation (i.e. not deepspeed or python -m torch.distributed)\n    the issue is that once the port is tied it can't be used anywhere else outside of this process,\n    since torch.dist doesn't free the port until the process exits. Therefore for the sake of being\n    able to run both emulated launcher and normal launcher tests we need 2 distinct ports.\n\n    This function will give the right port in the right context. For real launcher it'll give the\n    base port, for emulated launcher it'll give the base port + 1. In both cases a string is\n    returned.\n\n    Args:\n        `real_launcher`: whether a real launcher is going to be used, or the emulated one\n\n    "
    master_port_base = os.environ.get('DS_TEST_PORT', DEFAULT_MASTER_PORT)
    if not real_launcher:
        master_port_base = str(int(master_port_base) + 1)
    return master_port_base

def require_deepspeed_aio(test_case):
    if False:
        for i in range(10):
            print('nop')
    '\n    Decorator marking a test that requires deepspeed aio (nvme)\n    '
    if not is_deepspeed_available():
        return unittest.skip('test requires deepspeed')(test_case)
    import deepspeed
    from deepspeed.ops.aio import AsyncIOBuilder
    if not deepspeed.ops.__compatible_ops__[AsyncIOBuilder.NAME]:
        return unittest.skip('test requires deepspeed async-io')(test_case)
    else:
        return test_case
if is_deepspeed_available():
    from deepspeed.utils import logger as deepspeed_logger
    from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
    from transformers.integrations.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled

def get_launcher(distributed=False):
    if False:
        return 10
    num_gpus = min(2, backend_device_count(torch_device)) if distributed else 1
    master_port = get_master_port(real_launcher=True)
    return f'deepspeed --num_nodes 1 --num_gpus {num_gpus} --master_port {master_port}'.split()
ZERO2 = 'zero2'
ZERO3 = 'zero3'
FP16 = 'fp16'
BF16 = 'bf16'
HF_OPTIM = 'hf_optim'
HF_SCHEDULER = 'hf_scheduler'
DS_OPTIM = 'ds_optim'
DS_SCHEDULER = 'ds_scheduler'
optims = [HF_OPTIM, DS_OPTIM]
schedulers = [HF_SCHEDULER, DS_SCHEDULER]
stages = [ZERO2, ZERO3]
if is_torch_bf16_available_on_device(torch_device):
    dtypes = [FP16, BF16]
else:
    dtypes = [FP16]

def parameterized_custom_name_func(func, param_num, param):
    if False:
        i = 10
        return i + 15
    param_based_name = parameterized.to_safe_name('_'.join((str(x) for x in param.args)))
    return f'{func.__name__}_{param_based_name}'
params = list(itertools.product(stages, dtypes))
params_with_optims_and_schedulers = list(itertools.product(stages, dtypes, optims, schedulers))

@require_deepspeed
@require_torch_accelerator
class CoreIntegrationDeepSpeed(TestCasePlus, TrainerIntegrationCommon):
    """
    Testing non-Trainer DeepSpeed integration
    """

    def setUp(self):
        if False:
            return 10
        super().setUp()
        master_port = get_master_port(real_launcher=False)
        self.dist_env_1_gpu = {'MASTER_ADDR': 'localhost', 'MASTER_PORT': master_port, 'RANK': '0', 'LOCAL_RANK': '0', 'WORLD_SIZE': '1'}

    def tearDown(self):
        if False:
            return 10
        super().tearDown()
        unset_hf_deepspeed_config()

    def test_init_zero3_fp16(self):
        if False:
            print('Hello World!')
        ds_config = {'train_batch_size': 1, 'zero_optimization': {'stage': 3}}
        dschf = HfDeepSpeedConfig(ds_config)
        self.assertTrue(dschf.is_zero3())
        self.assertTrue(is_deepspeed_zero3_enabled())
        with LoggingLevel(logging.INFO):
            with mockenv_context(**self.dist_env_1_gpu):
                logger = logging.get_logger('transformers.modeling_utils')
                with CaptureLogger(logger) as cl:
                    AutoModel.from_pretrained(T5_TINY)
        self.assertIn('Detected DeepSpeed ZeRO-3', cl.out)
        del ds_config['zero_optimization']
        dschf = HfDeepSpeedConfig(ds_config)
        self.assertFalse(dschf.is_zero3())
        self.assertFalse(is_deepspeed_zero3_enabled())
        with LoggingLevel(logging.INFO):
            with mockenv_context(**self.dist_env_1_gpu):
                logger = logging.get_logger('transformers.modeling_utils')
                with CaptureLogger(logger) as cl:
                    AutoModel.from_pretrained(T5_TINY)
        self.assertNotIn('Detected DeepSpeed ZeRO-3', cl.out)

class TrainerIntegrationDeepSpeedWithCustomConfig(TestCasePlus):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        args = TrainingArguments('.')
        self.n_epochs = args.num_train_epochs
        self.batch_size = args.train_batch_size
        master_port = get_master_port(real_launcher=False)
        self.dist_env_1_gpu = {'MASTER_ADDR': 'localhost', 'MASTER_PORT': master_port, 'RANK': '0', 'LOCAL_RANK': '0', 'WORLD_SIZE': '1'}
        self.ds_config_file = {'zero2': f'{self.test_file_dir_str}/ds_config_zero2.json', 'zero3': f'{self.test_file_dir_str}/ds_config_zero3.json'}
        with io.open(self.ds_config_file[ZERO2], 'r', encoding='utf-8') as f:
            config_zero2 = json.load(f)
        with io.open(self.ds_config_file[ZERO3], 'r', encoding='utf-8') as f:
            config_zero3 = json.load(f)
            config_zero3['zero_optimization']['stage3_gather_16bit_weights_on_model_save'] = False
        self.ds_config_dict = {'zero2': config_zero2, 'zero3': config_zero3}

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        super().tearDown()
        unset_hf_deepspeed_config()

    def get_config_dict(self, stage):
        if False:
            while True:
                i = 10
        return deepcopy(self.ds_config_dict[stage])

@require_deepspeed
@require_torch_accelerator
class TrainerIntegrationDeepSpeed(TrainerIntegrationDeepSpeedWithCustomConfig, TrainerIntegrationCommon):
    """

    This class is for testing directly via get_regression_trainer

    It mixes in `TrainerIntegrationCommon` which already has a lot of helper validation methods
    which we can re-use here.

    Important: this class' setup can only work with a single gpu because it runs within the current
    pytest worker. For multi-gpu tests use TestDeepSpeedWithLauncher.

    Note: if any of the tests of this class get run there will be at least one gpu occupied by them
    until this pytest worker exits. This is because the gpu memory allocated by the cuda-kernels
    won't be released until this pytest worker exits.

    This may appear as some run-away tests if you watch `nvidia-smi` while other tests that fork new
    processes are run. So there will be one or two "stale" processes reported in `nvidia-smi`. This
    is not a bug.
    """

    def test_hf_ds_config_mismatch(self):
        if False:
            while True:
                i = 10
        ds_config = self.get_config_dict(ZERO2)
        per_device_train_batch_size = 2
        ds_config['train_micro_batch_size_per_gpu'] = per_device_train_batch_size + 2
        ds_config['train_batch_size'] = 1000
        gradient_accumulation_steps = 2
        ds_config['gradient_accumulation_steps'] = gradient_accumulation_steps + 2
        max_grad_norm = 1.0
        ds_config['gradient_clipping'] = max_grad_norm + 0.1
        (adam_beta1, adam_beta2) = (0.9, 0.99)
        ds_config['optimizer']['params']['betas'] = [adam_beta1 - 0.1, adam_beta2 - 0.1]
        fp16 = True
        ds_config['fp16']['enabled'] = not fp16
        keys = ['per_device_train_batch_size', 'train_batch_size', 'gradient_accumulation_steps', 'max_grad_norm', 'betas', 'fp16']
        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(local_rank=0, fp16=fp16, deepspeed=ds_config, per_device_train_batch_size=per_device_train_batch_size, gradient_accumulation_steps=gradient_accumulation_steps, max_grad_norm=max_grad_norm, adam_beta1=adam_beta1, adam_beta2=adam_beta2)
            with self.assertRaises(Exception) as context:
                trainer.train()
        for key in keys:
            self.assertTrue(key in str(context.exception), f'{key} is not in the exception message:\n{context.exception}')

    def test_hf_scheduler_hf_optimizer(self):
        if False:
            for i in range(10):
                print('nop')
        a = 0
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_zero2_dict = self.get_config_dict(ZERO2)
            del ds_config_zero2_dict['optimizer']
            del ds_config_zero2_dict['scheduler']
            ds_config_zero2_dict['zero_optimization']['offload_optimizer']['device'] = 'none'
            ds_config_zero2_dict['fp16']['initial_scale_power'] = 1
            trainer = get_regression_trainer(a=a, local_rank=0, fp16=True, deepspeed=ds_config_zero2_dict)
            trainer.train()
        new_a = trainer.model.a.item()
        self.assertNotEqual(new_a, a)

    def test_ds_scheduler_hf_optimizer(self):
        if False:
            return 10
        a = 0
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_zero2_dict = self.get_config_dict(ZERO2)
            del ds_config_zero2_dict['optimizer']
            ds_config_zero2_dict['zero_optimization']['offload_optimizer']['device'] = 'none'
            ds_config_zero2_dict['fp16']['initial_scale_power'] = 1
            trainer = get_regression_trainer(a=a, local_rank=0, fp16=True, deepspeed=ds_config_zero2_dict)
            trainer.train()
        new_a = trainer.model.a.item()
        self.assertNotEqual(new_a, a)

    def test_hf_scheduler_ds_optimizer(self):
        if False:
            i = 10
            return i + 15
        a = 0
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_zero2_dict = self.get_config_dict(ZERO2)
            del ds_config_zero2_dict['scheduler']
            ds_config_zero2_dict['zero_optimization']['offload_optimizer']['device'] = 'none'
            ds_config_zero2_dict['fp16']['initial_scale_power'] = 1
            trainer = get_regression_trainer(a=a, local_rank=0, fp16=True, deepspeed=ds_config_zero2_dict)
            trainer.train()
        new_a = trainer.model.a.item()
        self.assertNotEqual(new_a, a)

    @require_deepspeed_aio
    def test_stage3_nvme_offload(self):
        if False:
            i = 10
            return i + 15
        with mockenv_context(**self.dist_env_1_gpu):
            nvme_path = self.get_auto_remove_tmp_dir()
            nvme_config = {'device': 'nvme', 'nvme_path': nvme_path}
            ds_config_zero3_dict = self.get_config_dict(ZERO3)
            ds_config_zero3_dict['zero_optimization']['offload_optimizer'] = nvme_config
            ds_config_zero3_dict['zero_optimization']['offload_param'] = nvme_config
            trainer = get_regression_trainer(local_rank=0, fp16=True, deepspeed=ds_config_zero3_dict)
            with CaptureLogger(deepspeed_logger) as cl:
                trainer.train()
            self.assertIn('DeepSpeed info', cl.out, 'expected DeepSpeed logger output but got none')

    @require_optuna
    def test_hyperparameter_search(self):
        if False:
            return 10
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_zero3_dict = self.get_config_dict(ZERO3)

            def model_init():
                if False:
                    i = 10
                    return i + 15
                config = RegressionModelConfig(a=0, b=0, double_output=False)
                model = RegressionPreTrainedModel(config)
                return model
            trainer = get_regression_trainer(local_rank=0, fp16=True, model_init=model_init, deepspeed=ds_config_zero3_dict)
            n_trials = 3
            with CaptureLogger(deepspeed_logger) as cl:
                with CaptureStd() as cs:
                    trainer.hyperparameter_search(direction='maximize', n_trials=n_trials)
            self.assertIn('DeepSpeed info', cl.out, 'expected DeepSpeed logger output but got none')
            self.assertIn(f'Trial {n_trials - 1} finished with value', cs.err, 'expected hyperparameter_search output')
            self.assertIn('Best is trial', cs.err, 'expected hyperparameter_search output')

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_hf_optimizer_with_offload(self, stage, dtype):
        if False:
            while True:
                i = 10
        ds_config_dict = self.get_config_dict(stage)
        del ds_config_dict['optimizer']
        ds_config_dict['zero_optimization']['offload_optimizer']['device'] = 'cpu'
        ds_config_dict['zero_force_ds_cpu_optimizer'] = False
        with mockenv_context(**self.dist_env_1_gpu):
            kwargs = {'local_rank': 0, 'deepspeed': ds_config_dict}
            kwargs[dtype] = True
            trainer = get_regression_trainer(**kwargs)
            with CaptureLogger(deepspeed_logger) as cl:
                trainer.train()
            self.assertIn('DeepSpeed info', cl.out, 'expected DeepSpeed logger output but got none')

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_fake_notebook_no_launcher(self, stage, dtype):
        if False:
            for i in range(10):
                print('nop')
        with mockenv_context(**self.dist_env_1_gpu):
            kwargs = {'local_rank': 0, 'deepspeed': self.get_config_dict(stage)}
            kwargs[dtype] = True
            trainer = get_regression_trainer(**kwargs)
            with CaptureLogger(deepspeed_logger) as cl:
                trainer.train()
            self.assertIn('DeepSpeed info', cl.out, 'expected DeepSpeed logger output but got none')

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_early_get_last_lr(self, stage, dtype):
        if False:
            while True:
                i = 10
        with mockenv_context(**self.dist_env_1_gpu):
            a = b = 0.0
            kwargs = {'a': a, 'b': b, 'local_rank': 0, 'train_len': 8, 'deepspeed': self.get_config_dict(stage), 'per_device_train_batch_size': 8, 'logging_steps': 1}
            kwargs[dtype] = True
            trainer = get_regression_trainer(**kwargs)
            trainer.train()
            post_train_a = trainer.model.a.item()
            if stage == ZERO3 and dtype == FP16 or dtype == BF16:
                return
            self.assertEqual(post_train_a, a)

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_gradient_accumulation(self, stage, dtype):
        if False:
            print('Hello World!')
        train_len = 64
        a = b = 0.0
        kwargs = {'a': a, 'b': b, 'local_rank': 0, 'train_len': train_len, 'deepspeed': self.get_config_dict(stage)}
        kwargs[dtype] = True
        with mockenv_context(**self.dist_env_1_gpu):
            no_grad_accum_trainer = get_regression_trainer(**kwargs, per_device_train_batch_size=16, gradient_accumulation_steps=1)
            no_grad_accum_result = no_grad_accum_trainer.train()
            no_grad_accum_loss = no_grad_accum_result.training_loss
            no_grad_accum_a = no_grad_accum_trainer.model.a.item()
            no_grad_accum_b = no_grad_accum_trainer.model.b.item()
            self.assertNotEqual(no_grad_accum_a, a)
        with mockenv_context(**self.dist_env_1_gpu):
            yes_grad_accum_trainer = get_regression_trainer(**kwargs, per_device_train_batch_size=4, gradient_accumulation_steps=4)
            yes_grad_accum_result = yes_grad_accum_trainer.train()
            yes_grad_accum_loss = yes_grad_accum_result.training_loss
            yes_grad_accum_a = yes_grad_accum_trainer.model.a.item()
            yes_grad_accum_b = yes_grad_accum_trainer.model.b.item()
            self.assertNotEqual(yes_grad_accum_a, a)
        self.assertAlmostEqual(no_grad_accum_a, yes_grad_accum_a, places=5)
        self.assertAlmostEqual(no_grad_accum_b, yes_grad_accum_b, places=5)
        self.assertAlmostEqual(no_grad_accum_loss, yes_grad_accum_loss, places=2)

    def check_saved_checkpoints_deepspeed(self, output_dir, freq, total, stage, dtype):
        if False:
            print('Hello World!')
        file_list = [SAFE_WEIGHTS_NAME, 'training_args.bin', 'trainer_state.json', 'config.json']
        if stage == ZERO2:
            ds_file_list = ['mp_rank_00_model_states.pt']
        elif stage == ZERO3:
            ds_file_list = ['zero_pp_rank_0_mp_rank_00_model_states.pt']
        else:
            raise ValueError(f'unknown stage {stage}')
        if dtype == 'bf16':
            ds_file_list.append('bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt')
        for step in range(freq, total, freq):
            checkpoint = os.path.join(output_dir, f'checkpoint-{step}')
            self.assertTrue(os.path.isdir(checkpoint), f'[{stage}] {checkpoint} dir is not found')
            for filename in file_list:
                path = os.path.join(checkpoint, filename)
                self.assertTrue(os.path.isfile(path), f'[{stage}] {path} is not found')
            ds_path = os.path.join(checkpoint, f'global_step{step}')
            for filename in ds_file_list:
                path = os.path.join(ds_path, filename)
                self.assertTrue(os.path.isfile(path), f'[{stage}] {path} is not found')

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_save_checkpoints(self, stage, dtype):
        if False:
            i = 10
            return i + 15
        freq = 5
        output_dir = self.get_auto_remove_tmp_dir()
        ds_config_dict = self.get_config_dict(stage)
        if dtype == FP16:
            ds_config_dict['fp16']['initial_scale_power'] = 1
        if stage == ZERO3:
            ds_config_dict['zero_optimization']['stage3_gather_16bit_weights_on_model_save'] = True
        with mockenv_context(**self.dist_env_1_gpu):
            kwargs = {'output_dir': output_dir, 'save_steps': freq, 'deepspeed': ds_config_dict}
            kwargs[dtype] = True
            trainer = get_regression_trainer(**kwargs)
            trainer.train()
        total = int(self.n_epochs * 64 / self.batch_size)
        self.check_saved_checkpoints_deepspeed(output_dir, freq, total, stage, dtype)

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_can_resume_training_errors(self, stage, dtype):
        if False:
            while True:
                i = 10
        with mockenv_context(**self.dist_env_1_gpu):
            ds_config_dict = self.get_config_dict(stage)
            output_dir = self.get_auto_remove_tmp_dir()
            kwargs = {'output_dir': output_dir, 'deepspeed': ds_config_dict}
            kwargs[dtype] = True
            trainer = get_regression_trainer(**kwargs)
            with self.assertRaises(Exception) as context:
                trainer.train(resume_from_checkpoint=True)
            self.assertTrue('No valid checkpoint found in output directory' in str(context.exception), f'got exception: {context.exception}')
            with self.assertRaises(Exception) as context:
                checkpoint = os.path.join(output_dir, 'checkpoint-5')
                trainer.train(resume_from_checkpoint=f'{checkpoint}-bogus')
            self.assertTrue("Can't find a valid checkpoint at" in str(context.exception), f'got exception: {context.exception}')

    @parameterized.expand(params_with_optims_and_schedulers, name_func=parameterized_custom_name_func)
    def test_can_resume_training_normal(self, stage, dtype, optim, scheduler):
        if False:
            while True:
                i = 10
        if optim == HF_OPTIM and scheduler == HF_SCHEDULER:
            return
        output_dir = self.get_auto_remove_tmp_dir('./xxx', after=False)
        ds_config_dict = self.get_config_dict(stage)
        if dtype == FP16:
            ds_config_dict['fp16']['initial_scale_power'] = 1
        if stage == ZERO3:
            ds_config_dict['zero_optimization']['stage3_gather_16bit_weights_on_model_save'] = True
        if optim == HF_OPTIM:
            del ds_config_dict['optimizer']
        if scheduler == HF_SCHEDULER:
            del ds_config_dict['scheduler']
        kwargs = {'output_dir': output_dir, 'train_len': 128, 'save_steps': 5, 'learning_rate': 0.1, 'deepspeed': ds_config_dict}
        kwargs[dtype] = True
        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(**kwargs)
            trainer.train()
            (a, b) = (trainer.model.a.item(), trainer.model.b.item())
            state = dataclasses.asdict(trainer.state)
            checkpoint = os.path.join(output_dir, 'checkpoint-5')
            trainer = get_regression_trainer(**kwargs)
            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = (trainer.model.a.item(), trainer.model.b.item())
            state1 = dataclasses.asdict(trainer.state)
            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)
            checkpoint = os.path.join(output_dir, 'checkpoint-15')
            trainer = get_regression_trainer(**kwargs)
            trainer.train(resume_from_checkpoint=checkpoint)
            (a1, b1) = (trainer.model.a.item(), trainer.model.b.item())
            state1 = dataclasses.asdict(trainer.state)
            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_load_state_dict_from_zero_checkpoint(self, stage, dtype):
        if False:
            return 10
        output_dir = self.get_auto_remove_tmp_dir()
        ds_config_dict = self.get_config_dict(stage)
        kwargs = {'output_dir': output_dir, 'train_len': 4, 'per_device_train_batch_size': 4, 'num_train_epochs': 1, 'save_strategy': 'steps', 'save_steps': 1, 'learning_rate': 0.1, 'deepspeed': ds_config_dict}
        kwargs[dtype] = True
        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(**kwargs)
            trainer.train()
            (a, b) = (trainer.model.a.item(), trainer.model.b.item())
            state = dataclasses.asdict(trainer.state)
            checkpoint_dir = get_last_checkpoint(output_dir)
            model = load_state_dict_from_zero_checkpoint(trainer.model, checkpoint_dir)
            (a1, b1) = (model.a.item(), model.b.item())
            state1 = dataclasses.asdict(trainer.state)
            self.assertEqual(a, a1)
            self.assertEqual(b, b1)
            self.check_trainer_state_are_the_same(state, state1)

    def test_config_object(self):
        if False:
            while True:
                i = 10
        output_dir = self.get_auto_remove_tmp_dir()
        kwargs = {'output_dir': output_dir, 'train_len': 8, 'fp16': True}
        ds_config_zero3_dict = self.get_config_dict(ZERO3)
        ds_config_zero2_dict = self.get_config_dict(ZERO2)
        with mockenv_context(**self.dist_env_1_gpu):
            trainer = get_regression_trainer(deepspeed=ds_config_zero3_dict, **kwargs)
            self.assertTrue(is_deepspeed_zero3_enabled())
            trainer = get_regression_trainer(deepspeed=ds_config_zero3_dict, **kwargs)
            trainer.train()
            self.assertTrue(is_deepspeed_zero3_enabled())
            trainer = get_regression_trainer(deepspeed=ds_config_zero2_dict, **kwargs)
            self.assertFalse(is_deepspeed_zero3_enabled())
            config = deepspeed_config()
            self.assertTrue(bool(config), 'Deepspeed config should be accessible')
            trainer.accelerator.state._reset_state()
            del trainer
            config = deepspeed_config()
            self.assertFalse(is_deepspeed_zero3_enabled())
            self.assertFalse(bool(config), 'Deepspeed config should not be accessible')

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_load_best_model(self, stage, dtype):
        if False:
            while True:
                i = 10
        from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer
        output_dir = self.get_auto_remove_tmp_dir()
        ds_config_dict = self.get_config_dict(stage)
        del ds_config_dict['optimizer']
        del ds_config_dict['scheduler']
        ds_config_dict['zero_force_ds_cpu_optimizer'] = False
        ds_config_dict['zero_optimization']['stage3_gather_16bit_weights_on_model_save'] = True
        with mockenv_context(**self.dist_env_1_gpu):
            args_dict = {'per_device_train_batch_size': 1, 'per_device_eval_batch_size': 1, 'gradient_accumulation_steps': 1, 'learning_rate': 0.0001, 'num_train_epochs': 1, 'do_train': True, 'do_eval': True, 'optim': 'adafactor', 'evaluation_strategy': 'steps', 'eval_steps': 1, 'save_strategy': 'steps', 'save_steps': 1, 'load_best_model_at_end': True, 'max_steps': 1, 'deepspeed': ds_config_dict, 'report_to': 'none'}
            training_args = TrainingArguments(output_dir, **args_dict)
            tokenizer = T5Tokenizer.from_pretrained(T5_TINY)
            model = T5ForConditionalGeneration.from_pretrained(T5_TINY)

            def _add_eos_to_examples(example):
                if False:
                    i = 10
                    return i + 15
                example['input_text'] = f"question: {example['question']}  context: {example['context']}"
                example['target_text'] = example['answers']['text'][0] if len(example['answers']['text']) > 0 else ''
                return example

            def _convert_to_features(example_batch):
                if False:
                    for i in range(10):
                        print('nop')
                input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'], pad_to_max_length=True, max_length=512, truncation=True)
                target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], pad_to_max_length=True, max_length=16, truncation=True)
                encodings = {'input_ids': input_encodings['input_ids'], 'attention_mask': input_encodings['attention_mask'], 'labels': target_encodings['input_ids']}
                return encodings

            def get_dataset():
                if False:
                    i = 10
                    return i + 15
                data_file = str(self.tests_dir / 'fixtures/tests_samples/SQUAD/sample.json')
                data_files = {'train': data_file, 'validation': data_file}
                raw_datasets = datasets.load_dataset('json', data_files=data_files, field='data')
                train_dataset = raw_datasets['train'].map(_add_eos_to_examples).map(_convert_to_features, batched=True)
                valid_dataset = deepcopy(train_dataset)
                return (train_dataset, valid_dataset)
            (train_dataset, eval_dataset) = get_dataset()
            trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
            trainer.train()
            trainer.evaluate()

@slow
@require_deepspeed
@require_torch_accelerator
class TestDeepSpeedWithLauncher(TestCasePlus):
    """This class is for testing via an external script - can do multiple gpus"""

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    @require_torch_multi_accelerator
    def test_basic_distributed(self, stage, dtype):
        if False:
            return 10
        self.run_and_check(stage=stage, dtype=dtype, distributed=True)

    def test_do_eval_no_train(self):
        if False:
            print('Hello World!')
        self.run_and_check(stage=ZERO3, dtype=FP16, eval_steps=1, distributed=False, do_train=False, do_eval=True)

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_fp32_non_distributed(self, stage, dtype):
        if False:
            while True:
                i = 10
        self.run_and_check(stage=stage, dtype=dtype, model_name=T5_TINY, distributed=False, do_train=True, do_eval=True, quality_checks=False, fp32=True)

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    @require_torch_multi_accelerator
    def test_fp32_distributed(self, stage, dtype):
        if False:
            while True:
                i = 10
        self.run_and_check(stage=stage, dtype=dtype, model_name=T5_TINY, distributed=True, do_train=True, do_eval=True, quality_checks=False, fp32=True)

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_resume_train_not_from_ds_checkpoint(self, stage, dtype):
        if False:
            for i in range(10):
                print('nop')
        do_train = True
        do_eval = False
        kwargs = {'stage': stage, 'dtype': dtype, 'eval_steps': 1, 'distributed': True, 'do_train': do_train, 'do_eval': do_eval}
        output_dir = self.run_and_check(**kwargs)
        output_dir = self.run_trainer(**kwargs, model_name=output_dir)
        self.do_checks(output_dir, do_train=do_train, do_eval=do_eval)

    @parameterized.expand(['bf16', 'fp16', 'fp32'])
    @require_torch_multi_accelerator
    def test_inference(self, dtype):
        if False:
            i = 10
            return i + 15
        if dtype == 'bf16' and (not is_torch_bf16_available_on_device(torch_device)):
            self.skipTest('test requires bfloat16 hardware support')
        fp32 = True if dtype == 'fp32' else False
        self.run_and_check(stage=ZERO3, dtype=FP16, model_name=T5_TINY, distributed=True, do_train=False, do_eval=True, quality_checks=False, fp32=fp32)

    def do_checks(self, output_dir, do_train=True, do_eval=True, quality_checks=True):
        if False:
            while True:
                i = 10
        if do_train:
            train_metrics = load_json(os.path.join(output_dir, 'train_results.json'))
            self.assertIn('train_samples_per_second', train_metrics)
            if quality_checks:
                self.assertGreater(train_metrics['train_samples_per_second'], 0.5)
        if do_eval:
            eval_metrics = load_json(os.path.join(output_dir, 'eval_results.json'))
            self.assertIn('eval_bleu', eval_metrics)
            if quality_checks:
                self.assertGreater(eval_metrics['eval_bleu'], 1)

    def run_and_check(self, stage, dtype, model_name: str=T5_SMALL, eval_steps: int=10, distributed: bool=True, do_train: bool=True, do_eval: bool=True, quality_checks: bool=True, fp32: bool=False, extra_args_str: str=None, remove_args_str: str=None):
        if False:
            for i in range(10):
                print('nop')
        output_dir = self.run_trainer(stage=stage, dtype=dtype, model_name=model_name, eval_steps=eval_steps, num_train_epochs=1, do_train=do_train, do_eval=do_eval, distributed=distributed, fp32=fp32, extra_args_str=extra_args_str, remove_args_str=remove_args_str)
        self.do_checks(output_dir, do_train=do_train, do_eval=do_eval, quality_checks=quality_checks)
        return output_dir

    def run_trainer(self, stage: str, dtype: str, model_name: str, eval_steps: int=10, num_train_epochs: int=1, do_train: bool=False, do_eval: bool=True, distributed: bool=True, fp32: bool=False, extra_args_str: str=None, remove_args_str: str=None):
        if False:
            for i in range(10):
                print('nop')
        max_len = 32
        data_dir = self.test_file_dir / '../fixtures/tests_samples/wmt_en_ro'
        output_dir = self.get_auto_remove_tmp_dir()
        args = f'\n            --model_name_or_path {model_name}\n            --train_file {data_dir}/train.json\n            --validation_file {data_dir}/val.json\n            --output_dir {output_dir}\n            --overwrite_output_dir\n            --max_source_length {max_len}\n            --max_target_length {max_len}\n            --val_max_target_length {max_len}\n            --warmup_steps 8\n            --predict_with_generate\n            --save_steps 0\n            --eval_steps {eval_steps}\n            --group_by_length\n            --label_smoothing_factor 0.1\n            --source_lang en\n            --target_lang ro\n            --report_to none\n        '.split()
        args.extend(['--source_prefix', '"translate English to Romanian: "'])
        if not fp32:
            args.extend([f'--{dtype}'])
        actions = 0
        if do_train:
            actions += 1
            args.extend(f'\n            --do_train\n            --num_train_epochs {str(num_train_epochs)}\n            --max_train_samples 16\n            --per_device_train_batch_size 2\n            --learning_rate 3e-3\n            '.split())
        if do_eval:
            actions += 1
            args.extend('\n            --do_eval\n            --max_eval_samples 16\n            --per_device_eval_batch_size 2\n            '.split())
        assert actions > 0, 'need at least do_train or do_eval for the test to run'
        if extra_args_str is not None:
            args.extend(extra_args_str.split())
        if remove_args_str is not None:
            remove_args = remove_args_str.split()
            args = [x for x in args if x not in remove_args]
        ds_args = f'--deepspeed {self.test_file_dir_str}/ds_config_{stage}.json'.split()
        script = [f'{self.examples_dir_str}/pytorch/translation/run_translation.py']
        launcher = get_launcher(distributed)
        cmd = launcher + script + args + ds_args
        execute_subprocess_async(cmd, env=self.get_env())
        return output_dir

    @parameterized.expand(params, name_func=parameterized_custom_name_func)
    def test_clm(self, stage, dtype):
        if False:
            return 10
        data_dir = self.tests_dir / 'fixtures'
        output_dir = self.get_auto_remove_tmp_dir()
        args = f'\n            --model_name_or_path {GPT2_TINY}\n            --train_file {data_dir}/sample_text.txt\n            --validation_file {data_dir}/sample_text.txt\n            --output_dir {output_dir}\n            --overwrite_output_dir\n            --do_train\n            --do_eval\n            --max_train_samples 16\n            --max_eval_samples 16\n            --per_device_train_batch_size 2\n            --per_device_eval_batch_size 2\n            --num_train_epochs 1\n            --warmup_steps 8\n            --block_size 64\n            --report_to none\n            '.split()
        args.extend([f'--{dtype}'])
        ds_args = f'--deepspeed {self.test_file_dir_str}/ds_config_{stage}.json'.split()
        script = [f'{self.examples_dir_str}/pytorch/language-modeling/run_clm.py']
        launcher = get_launcher(distributed=True)
        cmd = launcher + script + args + ds_args
        execute_subprocess_async(cmd, env=self.get_env())

    def test_clm_from_config_zero3_fp16(self):
        if False:
            while True:
                i = 10
        data_dir = self.tests_dir / 'fixtures'
        output_dir = self.get_auto_remove_tmp_dir()
        args = f'\n            --model_type gpt2\n            --tokenizer_name {GPT2_TINY}\n            --train_file {data_dir}/sample_text.txt\n            --validation_file {data_dir}/sample_text.txt\n            --output_dir {output_dir}\n            --overwrite_output_dir\n            --do_train\n            --max_train_samples 4\n            --per_device_train_batch_size 2\n            --num_train_epochs 1\n            --warmup_steps 8\n            --block_size 8\n            --fp16\n            --report_to none\n            '.split()
        ds_args = f'--deepspeed {self.test_file_dir_str}/ds_config_zero3.json'.split()
        script = [f'{self.examples_dir_str}/pytorch/language-modeling/run_clm.py']
        launcher = get_launcher(distributed=True)
        cmd = launcher + script + args + ds_args
        with CaptureStderr() as cs:
            execute_subprocess_async(cmd, env=self.get_env())
        self.assertIn('Detected DeepSpeed ZeRO-3', cs.err)