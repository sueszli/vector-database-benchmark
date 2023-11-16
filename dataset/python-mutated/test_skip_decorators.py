import os
import unittest
import pytest
from parameterized import parameterized
from transformers.testing_utils import require_torch, require_torch_gpu, slow, torch_device
params = [(1,)]

def check_slow():
    if False:
        i = 10
        return i + 15
    run_slow = bool(os.getenv('RUN_SLOW', 0))
    if run_slow:
        assert True
    else:
        assert False, 'should have been skipped'

def check_slow_torch_cuda():
    if False:
        for i in range(10):
            print('nop')
    run_slow = bool(os.getenv('RUN_SLOW', 0))
    if run_slow and torch_device == 'cuda':
        assert True
    else:
        assert False, 'should have been skipped'

@require_torch
class SkipTester(unittest.TestCase):

    @slow
    @require_torch_gpu
    def test_2_skips_slow_first(self):
        if False:
            i = 10
            return i + 15
        check_slow_torch_cuda()

    @require_torch_gpu
    @slow
    def test_2_skips_slow_last(self):
        if False:
            while True:
                i = 10
        check_slow_torch_cuda()

    @parameterized.expand(params)
    @slow
    def test_param_slow_last(self, param=None):
        if False:
            for i in range(10):
                print('nop')
        check_slow()

@slow
@require_torch_gpu
def test_pytest_2_skips_slow_first():
    if False:
        for i in range(10):
            print('nop')
    check_slow_torch_cuda()

@require_torch_gpu
@slow
def test_pytest_2_skips_slow_last():
    if False:
        print('Hello World!')
    check_slow_torch_cuda()

@slow
@pytest.mark.parametrize('param', [1])
def test_pytest_param_slow_first(param):
    if False:
        print('Hello World!')
    check_slow()

@pytest.mark.parametrize('param', [1])
@slow
def test_pytest_param_slow_last(param):
    if False:
        for i in range(10):
            print('nop')
    check_slow()