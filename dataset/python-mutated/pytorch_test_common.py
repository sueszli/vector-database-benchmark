from __future__ import annotations
import functools
import os
import random
import sys
import unittest
from typing import Optional
import numpy as np
import packaging.version
import torch
from torch.autograd import function
from torch.onnx._internal import diagnostics
from torch.testing._internal import common_utils
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(-1, pytorch_test_dir)
torch.set_default_tensor_type('torch.FloatTensor')
BATCH_SIZE = 2
RNN_BATCH_SIZE = 7
RNN_SEQUENCE_LENGTH = 11
RNN_INPUT_SIZE = 5
RNN_HIDDEN_SIZE = 3

def _skipper(condition, reason):
    if False:
        print('Hello World!')

    def decorator(f):
        if False:
            for i in range(10):
                print('nop')

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if False:
                print('Hello World!')
            if condition():
                raise unittest.SkipTest(reason)
            return f(*args, **kwargs)
        return wrapper
    return decorator
skipIfNoCuda = _skipper(lambda : not torch.cuda.is_available(), 'CUDA is not available')
skipIfTravis = _skipper(lambda : os.getenv('TRAVIS'), 'Skip In Travis')
skipIfNoBFloat16Cuda = _skipper(lambda : not torch.cuda.is_bf16_supported(), 'BFloat16 CUDA is not available')
skipIfQuantizationBackendQNNPack = _skipper(lambda : torch.backends.quantized.engine == 'qnnpack', 'Not compatible with QNNPack quantization backend')

def skipIfUnsupportedMinOpsetVersion(min_opset_version):
    if False:
        while True:
            i = 10

    def skip_dec(func):
        if False:
            while True:
                i = 10

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            if self.opset_version < min_opset_version:
                raise unittest.SkipTest(f'Unsupported opset_version: {self.opset_version} < {min_opset_version}')
            return func(self, *args, **kwargs)
        return wrapper
    return skip_dec

def skipIfUnsupportedMaxOpsetVersion(max_opset_version):
    if False:
        while True:
            i = 10

    def skip_dec(func):
        if False:
            print('Hello World!')

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if False:
                print('Hello World!')
            if self.opset_version > max_opset_version:
                raise unittest.SkipTest(f'Unsupported opset_version: {self.opset_version} > {max_opset_version}')
            return func(self, *args, **kwargs)
        return wrapper
    return skip_dec

def skipForAllOpsetVersions():
    if False:
        print('Hello World!')

    def skip_dec(func):
        if False:
            print('Hello World!')

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if False:
                return 10
            if self.opset_version:
                raise unittest.SkipTest('Skip verify test for unsupported opset_version')
            return func(self, *args, **kwargs)
        return wrapper
    return skip_dec

def skipTraceTest(skip_before_opset_version: Optional[int]=None, reason: str=''):
    if False:
        i = 10
        return i + 15
    'Skip tracing test for opset version less than skip_before_opset_version.\n\n    Args:\n        skip_before_opset_version: The opset version before which to skip tracing test.\n            If None, tracing test is always skipped.\n        reason: The reason for skipping tracing test.\n\n    Returns:\n        A decorator for skipping tracing test.\n    '

    def skip_dec(func):
        if False:
            for i in range(10):
                print('nop')

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            if skip_before_opset_version is not None:
                self.skip_this_opset = self.opset_version < skip_before_opset_version
            else:
                self.skip_this_opset = True
            if self.skip_this_opset and (not self.is_script):
                raise unittest.SkipTest(f'Skip verify test for torch trace. {reason}')
            return func(self, *args, **kwargs)
        return wrapper
    return skip_dec

def skipScriptTest(skip_before_opset_version: Optional[int]=None, reason: str=''):
    if False:
        while True:
            i = 10
    'Skip scripting test for opset version less than skip_before_opset_version.\n\n    Args:\n        skip_before_opset_version: The opset version before which to skip scripting test.\n            If None, scripting test is always skipped.\n        reason: The reason for skipping scripting test.\n\n    Returns:\n        A decorator for skipping scripting test.\n    '

    def skip_dec(func):
        if False:
            while True:
                i = 10

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if False:
                return 10
            if skip_before_opset_version is not None:
                self.skip_this_opset = self.opset_version < skip_before_opset_version
            else:
                self.skip_this_opset = True
            if self.skip_this_opset and self.is_script:
                raise unittest.SkipTest(f'Skip verify test for TorchScript. {reason}')
            return func(self, *args, **kwargs)
        return wrapper
    return skip_dec

def skip_min_ort_version(reason: str, version: str, dynamic_only: bool=False):
    if False:
        print('Hello World!')

    def skip_dec(func):
        if False:
            for i in range(10):
                print('nop')

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            if packaging.version.parse(self.ort_version).release < packaging.version.parse(version).release:
                if dynamic_only and (not self.dynamic_shapes):
                    return func(self, *args, **kwargs)
                raise unittest.SkipTest(f'ONNX Runtime version: {version} is older than required version {version}. Reason: {reason}.')
            return func(self, *args, **kwargs)
        return wrapper
    return skip_dec

def skip_dynamic_fx_test(reason: str):
    if False:
        return 10
    'Skip dynamic exporting test.\n\n    Args:\n        reason: The reason for skipping dynamic exporting test.\n\n    Returns:\n        A decorator for skipping dynamic exporting test.\n    '

    def skip_dec(func):
        if False:
            i = 10
            return i + 15

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if False:
                return 10
            if self.dynamic_shapes:
                raise unittest.SkipTest(f'Skip verify dynamic shapes test for FX. {reason}')
            return func(self, *args, **kwargs)
        return wrapper
    return skip_dec

def skip_load_checkpoint_after_model_creation(reason: str):
    if False:
        print('Hello World!')
    'Skip loading checkpoint right after model initialization.\n\n    Args:\n        reason: The reason for skipping dynamic exporting test.\n\n    Returns:\n        A decorator for skipping dynamic exporting test.\n    '

    def skip_dec(func):
        if False:
            for i in range(10):
                print('nop')

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if False:
                print('Hello World!')
            if self.load_checkpoint_during_init:
                raise unittest.SkipTest(f'Skip loading checkpoint during model initialization for FX tests. {reason}')
            return func(self, *args, **kwargs)
        return wrapper
    return skip_dec

def skip_op_level_debug_test(reason: str):
    if False:
        print('Hello World!')
    'Skip tests with op_level_debug enabled.\n\n    Args:\n        reason: The reason for skipping tests with op_level_debug enabled.\n\n    Returns:\n        A decorator for skipping tests with op_level_debug enabled.\n    '

    def skip_dec(func):
        if False:
            print('Hello World!')

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if False:
                print('Hello World!')
            if self.op_level_debug:
                raise unittest.SkipTest(f'Skip test with op_level_debug enabled. {reason}')
            return func(self, *args, **kwargs)
        return wrapper
    return skip_dec

def skip_in_ci(reason: str):
    if False:
        print('Hello World!')
    'Skip test in CI.\n\n    Args:\n        reason: The reason for skipping test in CI.\n\n    Returns:\n        A decorator for skipping test in CI.\n    '

    def skip_dec(func):
        if False:
            while True:
                i = 10

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            if os.getenv('CI'):
                raise unittest.SkipTest(f'Skip test in CI. {reason}')
            return func(self, *args, **kwargs)
        return wrapper
    return skip_dec

def xfail(reason: str):
    if False:
        i = 10
        return i + 15
    'Expect failure.\n\n    Args:\n        reason: The reason for expected failure.\n\n    Returns:\n        A decorator for expecting test failure.\n    '
    return unittest.expectedFailure

def skipIfUnsupportedOpsetVersion(unsupported_opset_versions):
    if False:
        i = 10
        return i + 15

    def skip_dec(func):
        if False:
            return 10

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            if self.opset_version in unsupported_opset_versions:
                raise unittest.SkipTest('Skip verify test for unsupported opset_version')
            return func(self, *args, **kwargs)
        return wrapper
    return skip_dec

def skipShapeChecking(func):
    if False:
        print('Hello World!')

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if False:
            return 10
        self.check_shape = False
        return func(self, *args, **kwargs)
    return wrapper

def skipDtypeChecking(func):
    if False:
        i = 10
        return i + 15

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.check_dtype = False
        return func(self, *args, **kwargs)
    return wrapper

def flatten(x):
    if False:
        while True:
            i = 10
    return tuple(function._iter_filter(lambda o: isinstance(o, torch.Tensor))(x))

def set_rng_seed(seed):
    if False:
        print('Hello World!')
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

class ExportTestCase(common_utils.TestCase):
    """Test case for ONNX export.

    Any test case that tests functionalities under torch.onnx should inherit from this class.
    """

    def setUp(self):
        if False:
            return 10
        super().setUp()
        set_rng_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        diagnostics.engine.clear()