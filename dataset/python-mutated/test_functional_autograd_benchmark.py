from torch.testing._internal.common_utils import TestCase, run_tests, slowTest, IS_WINDOWS
import subprocess
import tempfile
import os
import unittest
PYTORCH_COLLECT_COVERAGE = bool(os.environ.get('PYTORCH_COLLECT_COVERAGE'))

class TestFunctionalAutogradBenchmark(TestCase):

    def _test_runner(self, model, disable_gpu=False):
        if False:
            i = 10
            return i + 15
        with tempfile.NamedTemporaryFile() as out_file:
            cmd = ['python3', '../benchmarks/functional_autograd_benchmark/functional_autograd_benchmark.py']
            cmd += ['--num-iters', '0']
            cmd += ['--task-filter', 'vjp']
            cmd += ['--model-filter', model]
            cmd += ['--output', out_file.name]
            if disable_gpu:
                cmd += ['--gpu', '-1']
            res = subprocess.run(cmd)
            self.assertTrue(res.returncode == 0)
            out_file.seek(0, os.SEEK_END)
            self.assertTrue(out_file.tell() > 0)

    @unittest.skipIf(IS_WINDOWS, 'NamedTemporaryFile on windows does not have all the features we need.')
    @unittest.skipIf(PYTORCH_COLLECT_COVERAGE, 'Can deadlocks with gcov, see https://github.com/pytorch/pytorch/issues/49656')
    def test_fast_tasks(self):
        if False:
            for i in range(10):
                print('nop')
        fast_tasks = ['resnet18', 'ppl_simple_reg', 'ppl_robust_reg', 'wav2letter', 'transformer', 'multiheadattn']
        for task in fast_tasks:
            self._test_runner(task)

    @slowTest
    @unittest.skipIf(IS_WINDOWS, 'NamedTemporaryFile on windows does not have all the features we need.')
    def test_slow_tasks(self):
        if False:
            print('Hello World!')
        slow_tasks = ['fcn_resnet', 'detr']
        for task in slow_tasks:
            self._test_runner(task, disable_gpu=True)
if __name__ == '__main__':
    run_tests()