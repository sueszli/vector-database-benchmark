import os
import shutil
import tempfile
import unittest
from fairseq import options
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.data.data_utils import raise_if_valid_subsets_unintentionally_ignored
from .utils import create_dummy_data, preprocess_lm_data, train_language_model

def make_lm_config(data_dir=None, extra_flags=None, task='language_modeling', arch='transformer_lm_gpt2_tiny'):
    if False:
        print('Hello World!')
    task_args = [task]
    if data_dir is not None:
        task_args += [data_dir]
    train_parser = options.get_training_parser()
    train_args = options.parse_args_and_arch(train_parser, ['--task', *task_args, '--arch', arch, '--optimizer', 'adam', '--lr', '0.0001', '--max-tokens', '500', '--tokens-per-sample', '500', '--save-dir', data_dir, '--max-epoch', '1'] + (extra_flags or []))
    cfg = convert_namespace_to_omegaconf(train_args)
    return cfg

def write_empty_file(path):
    if False:
        i = 10
        return i + 15
    with open(path, 'w'):
        pass
    assert os.path.exists(path)

class TestValidSubsetsErrors(unittest.TestCase):
    """Test various filesystem, clarg combinations and ensure that error raising happens as expected"""

    def _test_case(self, paths, extra_flags):
        if False:
            return 10
        with tempfile.TemporaryDirectory() as data_dir:
            [write_empty_file(os.path.join(data_dir, f'{p}.bin')) for p in paths + ['train']]
            cfg = make_lm_config(data_dir, extra_flags=extra_flags)
            raise_if_valid_subsets_unintentionally_ignored(cfg)

    def test_default_raises(self):
        if False:
            print('Hello World!')
        with self.assertRaises(ValueError):
            self._test_case(['valid', 'valid1'], [])
        with self.assertRaises(ValueError):
            self._test_case(['valid', 'valid1', 'valid2'], ['--valid-subset', 'valid,valid1'])

    def partially_specified_valid_subsets(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            self._test_case(['valid', 'valid1', 'valid2'], ['--valid-subset', 'valid,valid1'])
        self._test_case(['valid', 'valid1', 'valid2'], ['--valid-subset', 'valid,valid1', '--ignore-unused-valid-subsets'])

    def test_legal_configs(self):
        if False:
            print('Hello World!')
        self._test_case(['valid'], [])
        self._test_case(['valid', 'valid1'], ['--ignore-unused-valid-subsets'])
        self._test_case(['valid', 'valid1'], ['--combine-val'])
        self._test_case(['valid', 'valid1'], ['--valid-subset', 'valid,valid1'])
        self._test_case(['valid', 'valid1'], ['--valid-subset', 'valid1'])
        self._test_case(['valid', 'valid1'], ['--combine-val', '--ignore-unused-valid-subsets'])
        self._test_case(['valid1'], ['--valid-subset', 'valid1'])

    def test_disable_validation(self):
        if False:
            print('Hello World!')
        self._test_case([], ['--disable-validation'])
        self._test_case(['valid', 'valid1'], ['--disable-validation'])

    def test_dummy_task(self):
        if False:
            for i in range(10):
                print('nop')
        cfg = make_lm_config(task='dummy_lm')
        raise_if_valid_subsets_unintentionally_ignored(cfg)

    def test_masked_dummy_task(self):
        if False:
            while True:
                i = 10
        cfg = make_lm_config(task='dummy_masked_lm')
        raise_if_valid_subsets_unintentionally_ignored(cfg)

class TestCombineValidSubsets(unittest.TestCase):

    def _train(self, extra_flags):
        if False:
            while True:
                i = 10
        with self.assertLogs() as logs:
            with tempfile.TemporaryDirectory('test_transformer_lm') as data_dir:
                create_dummy_data(data_dir, num_examples=20)
                preprocess_lm_data(data_dir)
                shutil.copyfile(f'{data_dir}/valid.bin', f'{data_dir}/valid1.bin')
                shutil.copyfile(f'{data_dir}/valid.idx', f'{data_dir}/valid1.idx')
                train_language_model(data_dir, 'transformer_lm', ['--max-update', '0', '--log-format', 'json'] + extra_flags, run_validation=False)
        return [x.message for x in logs.records]

    def test_combined(self):
        if False:
            print('Hello World!')
        flags = ['--combine-valid-subsets', '--required-batch-size-multiple', '1']
        logs = self._train(flags)
        assert any(['valid1' in x for x in logs])
        assert not any(['valid1_ppl' in x for x in logs])

    def test_subsets(self):
        if False:
            i = 10
            return i + 15
        flags = ['--valid-subset', 'valid,valid1', '--required-batch-size-multiple', '1']
        logs = self._train(flags)
        assert any(['valid_ppl' in x for x in logs])
        assert any(['valid1_ppl' in x for x in logs])