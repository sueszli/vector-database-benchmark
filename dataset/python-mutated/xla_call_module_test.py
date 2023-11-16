"""Tests for XLA call module op wrapper."""
import os
import re
from typing import Optional, Sequence
import unittest
from absl.testing import parameterized
import numpy as np
from tensorflow.compiler.mlir.stablehlo import stablehlo
from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test

def serialize(module_str: str) -> tuple[str, int]:
    if False:
        while True:
            i = 10
    target = stablehlo.get_minimum_version()
    byte_str = stablehlo.serialize_portable_artifact(module_str, target)
    return (byte_str, xla.call_module_maximum_supported_version())

class XlaCallModuleOpTest(xla_test.XLATestCase, parameterized.TestCase):

    def _assertOpOutputMatchesExpected(self, op, args, expected, equality_fn=None):
        if False:
            return 10
        'Asserts op(*args) == expected.'
        with self.test_scope():
            tf_func = def_function.function(op, autograph=False, jit_compile=True)
            result = tf_func(*args)
            if not equality_fn:
                equality_fn = self.assertAllClose
            equality_fn(result, expected, rtol=0.001)

    def testing_platform(self):
        if False:
            i = 10
            return i + 15
        'Current testing platform, one of CPU, GPU, TPU.'
        if self.device in ['CPU', 'XLA_CPU']:
            return 'CPU'
        elif self.device in ['GPU', 'XLA_GPU']:
            if test.is_built_with_rocm():
                return 'ROCM'
            else:
                return 'CUDA'
        elif self.device in ['TPU', 'XLA_TPU']:
            return 'TPU'
        else:
            assert False, f'Unexpected self.device={self.device!r}'

    def test_basic(self):
        if False:
            print('Hello World!')
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            (module, version) = serialize('\nmodule @jit_f.0 {\n  func.func public @main(%arg0: tensor<3xf32>) -> tensor<3xf32> {\n    %0 = stablehlo.cosine %arg0 : tensor<3xf32>\n    %1 = stablehlo.sine %0 : tensor<3xf32>\n    return %1 : tensor<3xf32>\n  }\n}\n')
            return xla.call_module([x], version=version, module=module, Tout=[x.dtype], Sout=[x.shape], platforms=[self.testing_platform()])
        self._assertOpOutputMatchesExpected(f, (x,), (np.sin(np.cos(x)),))

    def test_basic_with_token_v8(self):
        if False:
            while True:
                i = 10
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        def f(x):
            if False:
                i = 10
                return i + 15
            (module, _) = serialize('\nmodule @jit_f.0 {\n  func.func public @main(%arg0: !stablehlo.token, %arg1: tensor<3xf32>) -> (!stablehlo.token, tensor<3xf32>) {\n    %0 = stablehlo.cosine %arg1 : tensor<3xf32>\n    %1 = stablehlo.sine %0 : tensor<3xf32>\n    return %arg0, %1 : !stablehlo.token, tensor<3xf32>\n  }\n}\n')
            return xla.call_module([x], version=8, module=module, Tout=[x.dtype], Sout=[x.shape], has_token_input_output=True, platforms=[self.testing_platform()])
        self._assertOpOutputMatchesExpected(f, (x,), (np.sin(np.cos(x)),))

    def test_basic_with_multiple_tokens(self):
        if False:
            print('Hello World!')
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        def f(x):
            if False:
                i = 10
                return i + 15
            (module, version) = serialize('\nmodule @jit_f.0 {\n  func.func public @main(%arg0: !stablehlo.token {jax.token = true}, %arg1: !stablehlo.token {jax.token = true}, %arg2: tensor<3xf32>) -> (!stablehlo.token, !stablehlo.token, tensor<3xf32>) {\n    %0 = stablehlo.cosine %arg2 : tensor<3xf32>\n    %1 = stablehlo.sine %0 : tensor<3xf32>\n    return %arg0, %arg1, %1 : !stablehlo.token, !stablehlo.token, tensor<3xf32>\n  }\n}\n')
            return xla.call_module([x], version=version, module=module, Tout=[x.dtype], Sout=[x.shape], platforms=[self.testing_platform()])
        self._assertOpOutputMatchesExpected(f, (x,), (np.sin(np.cos(x)),))

    def test_basic_with_tokens_preceeded_by_other_args(self):
        if False:
            return 10
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        def f(x):
            if False:
                i = 10
                return i + 15
            (module, version) = serialize('\nmodule @jit_f.0 {\n  func.func public @main(%arg0: tensor<i32>, %arg1: !stablehlo.token {jax.token = true}, %arg2: !stablehlo.token {jax.token = true}, %arg3: tensor<3xf32>) -> (!stablehlo.token, !stablehlo.token, tensor<3xf32>) {\n    %0 = stablehlo.cosine %arg3 : tensor<3xf32>\n    %1 = stablehlo.sine %0 : tensor<3xf32>\n    return %arg1, %arg2, %1 : !stablehlo.token, !stablehlo.token, tensor<3xf32>\n  }\n}\n')
            return xla.call_module([np.int32(0), x], version=version, module=module, Tout=[x.dtype], Sout=[x.shape], platforms=[self.testing_platform()])
        self._assertOpOutputMatchesExpected(f, (x,), (np.sin(np.cos(x)),))

    def test_compare(self):
        if False:
            i = 10
            return i + 15
        x = np.uint32(2)
        res = np.bool_(True)

        def f(x):
            if False:
                return 10
            (module, version) = serialize('\nmodule @jit_f_jax.0 {\n  func.func public @main(%arg0: tensor<ui32>) -> tensor<i1> {\n    %0 = stablehlo.constant dense<1> : tensor<ui32>\n    %1 = "stablehlo.compare"(%arg0, %0) {compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction GE>} : (tensor<ui32>, tensor<ui32>) -> tensor<i1>\n    return %1 : tensor<i1>\n  }\n}\n')
            return xla.call_module([x], version=version, module=module, Tout=[res.dtype], Sout=[res.shape], platforms=[self.testing_platform()])
        self._assertOpOutputMatchesExpected(f, (x,), (res,))

    def test_multiple_args_results(self):
        if False:
            return 10
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([11.0, 12.0, 13.0, 14.0], dtype=np.float64)

        def f(x, y):
            if False:
                for i in range(10):
                    print('nop')
            (module, version) = serialize('\nmodule @jit_f.0 {\n  func.func public @main(%arg0: tensor<3xf32>, %arg1: tensor<4xf64>) -> (tensor<3xf32>, tensor<4xf64>) {\n    %0 = stablehlo.sine %arg0 : tensor<3xf32>\n    %1 = stablehlo.cosine %arg1 : tensor<4xf64>\n    return %0, %1 : tensor<3xf32>, tensor<4xf64>\n  }\n}\n')
            return xla.call_module([x, y], version=version, module=module, Tout=[x.dtype, y.dtype], Sout=[x.shape, y.shape], platforms=[self.testing_platform()])
        self._assertOpOutputMatchesExpected(f, (x, y), (np.sin(x), np.cos(y)))

    @parameterized.named_parameters((dict(testcase_name='_' + dim_var_type, dim_var_type=dim_var_type) for dim_var_type in ('i32',)))
    def test_poly_basic(self, *, dim_var_type: str):
        if False:
            return 10
        x = np.arange(6, dtype=np.float32).reshape((2, 3))

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            (module, version) = serialize(f'\nmodule @jit_f.0 attributes {{jax.uses_shape_polymorphism = true}} {{\n  func.func public @main(%arg1: tensor<2x?xf32>) -> (tensor<2x?xf32>, tensor<{dim_var_type}>) {{\n    %arg0_new_i32 = "stablehlo.get_dimension_size"(%arg1) {{dimension = 1 : i64}} : (tensor<2x?xf32>) -> tensor<i32>\n    %arg0_new = stablehlo.convert %arg0_new_i32 : (tensor<i32>) -> tensor<{dim_var_type}>\n    %0, %1 = call @dyn_main(%arg0_new, %arg1) : (tensor<{dim_var_type}>, tensor<2x?xf32>) -> (tensor<2x?xf32>, tensor<{dim_var_type}>)\n    return %0, %1 : tensor<2x?xf32>, tensor<{dim_var_type}>\n  }}\n  func.func private @dyn_main(%arg0: tensor<{dim_var_type}> {{jax.global_constant = "b"}}, %arg1: tensor<2x?xf32>) -> (tensor<2x?xf32>, tensor<{dim_var_type}>) {{\n    %0 = stablehlo.sine %arg1 : tensor<2x?xf32>\n    return %0, %arg0 : tensor<2x?xf32>, tensor<{dim_var_type}>\n  }}\n}}\n')
            return xla.call_module([x], module=module, version=version, Tout=[x.dtype, np.int32], Sout=[(None, 3), ()], platforms=[self.testing_platform()])
        self._assertOpOutputMatchesExpected(f, (x,), (np.sin(x), x.shape[1]))

    def test_poly_unranked(self):
        if False:
            print('Hello World!')
        x = np.arange(6, dtype=np.float32).reshape((2, 3))

        def f(x):
            if False:
                print('Hello World!')
            (module, version) = serialize('\nmodule @jit_f.0 attributes {jax.uses_shape_polymorphism = true} {\n  func.func public @main(%arg1: tensor<*xf32>) -> tensor<*xf32> {\n    %0 = stablehlo.sine %arg1 : tensor<*xf32>\n    return %0 : tensor<*xf32>\n  }\n}\n')
            return xla.call_module([x], module=module, version=version, Tout=[x.dtype], Sout=[(None, None)], platforms=[self.testing_platform()])
        self._assertOpOutputMatchesExpected(f, (x,), (np.sin(x),))

    def test_wrong_actual_args_errors(self):
        if False:
            while True:
                i = 10
        x = np.arange(6, dtype=np.float32).reshape((3, 2))
        y = np.arange(6, dtype=np.int32).reshape((2, 3))
        (module, version) = serialize('\nmodule @jit_f.0 attributes {jax.uses_shape_polymorphism = true} {\n  func.func public @main(%arg0: tensor<?x2xf32>, %arg1: tensor<*xi32>) -> tensor<?x2xf32> {\n    return %arg0 : tensor<?x2xf32>\n  }\n}\n')

        def f(x, y):
            if False:
                while True:
                    i = 10
            return xla.call_module([x, y], module=module, version=version, Tout=[x.dtype], Sout=[(None, 2)], platforms=[self.testing_platform()])
        self._assertOpOutputMatchesExpected(f, (x, y), (x,))
        x_bad_etype = x.astype(np.int32)
        with self.assertRaisesRegex(errors.InvalidArgumentError, 'Element type mismatch for argument 0 passed to XlaCallModule: expecting tensor<\\?x2xf32>, got tensor<3x2xi32>'):
            self._assertOpOutputMatchesExpected(f, (x_bad_etype, y), (x_bad_etype,))
        y_bad_etype = y.astype(np.float32)
        with self.assertRaisesRegex(errors.InvalidArgumentError, 'Element type mismatch for argument 1 passed to XlaCallModule: expecting tensor<\\*xi32>, got tensor<2x3xf32>'):
            self._assertOpOutputMatchesExpected(f, (x, y_bad_etype), (x,))
        x_bad_shape = np.arange(15, dtype=np.float32).reshape(5, 3)
        with self.assertRaisesRegex(errors.InvalidArgumentError, 'Shape mismatch for argument 0 passed to XlaCallModule: expecting tensor<\\?x2xf32>, got tensor<5x3xf32>'):
            self._assertOpOutputMatchesExpected(f, (x_bad_shape, y), (x_bad_shape,))

    @parameterized.named_parameters((dict(testcase_name='_' + platform_idx_type, platform_idx_type=platform_idx_type) for platform_idx_type in ('i32', 'i64')))
    def test_platforms_basic(self, *, platform_idx_type: str):
        if False:
            return 10
        x = np.float32(0.0)
        (module, version) = serialize(f'\nmodule @jit_f.0 {{\n  func.func public @main(%arg_platform_idx: tensor<{platform_idx_type}> {{jax.global_constant = "_platform_index"}}, %arg0: tensor<f32>) -> tensor<f32> {{\n    %0 = stablehlo.convert %arg_platform_idx : (tensor<{platform_idx_type}>) -> tensor<i32>\n    %to_add = "stablehlo.case"(%0) ({{\n      %cpu_val = stablehlo.constant dense<2.> : tensor<f32>\n      stablehlo.return %cpu_val : tensor<f32>\n    }}, {{\n      %gpu_val = stablehlo.constant dense<3.> : tensor<f32>\n      stablehlo.return %gpu_val : tensor<f32>\n    }}, {{\n      %tpu_val = stablehlo.constant dense<4.> : tensor<f32>\n      stablehlo.return %tpu_val : tensor<f32>\n    }}) : (tensor<i32>) -> tensor<f32>\n    %1 = stablehlo.add %arg0, %to_add : tensor<f32>\n    return %1 : tensor<f32>\n  }}\n}}\n')
        platforms = ['CPU', 'CUDA', 'ROCM', 'TPU']

        def f(x):
            if False:
                print('Hello World!')
            return xla.call_module([x], version=version, module=module, Tout=[np.float32], Sout=[()], platforms=platforms)
        expected_value = x + dict(CPU=2.0, CUDA=3.0, ROCM=3.0, TPU=4.0)[self.testing_platform()]
        self._assertOpOutputMatchesExpected(f, (x,), (expected_value,))

    def test_platforms_unknown_custom_call(self):
        if False:
            return 10
        if self.testing_platform() == 'ROCM':
            raise unittest.SkipTest('Not intended for ROCM')
        x = np.float32(0.0)
        (module, version) = serialize('\nmodule @jit_f.0 {\n  func.func public @main(%arg_platform_idx: tensor<i32> {jax.global_constant = "_platform_index"}, %arg0: tensor<f32>) -> tensor<f32> {\n    %to_add = "stablehlo.case"(%arg_platform_idx) ({\n      %cpu_val = stablehlo.constant dense<2.> : tensor<f32>\n      stablehlo.return %cpu_val : tensor<f32>\n    }, {\n      %gpu_val = stablehlo.constant dense<3.> : tensor<f32>\n      stablehlo.return %gpu_val : tensor<f32>\n    }, {\n      %tpu_val = stablehlo.constant dense<4.> : tensor<f32>\n      stablehlo.return %tpu_val : tensor<f32>\n    }, {\n      %rocm_val = stablehlo.custom_call @non_existent_target(%arg0) : (tensor<f32>) -> tensor<f32>\n      stablehlo.return %rocm_val : tensor<f32>\n    }) : (tensor<i32>) -> tensor<f32>\n    %0 = stablehlo.add %arg0, %to_add : tensor<f32>\n    return %0 : tensor<f32>\n  }\n}\n')
        platforms = ['CPU', 'CUDA', 'TPU', 'ROCM']

        def f(x):
            if False:
                print('Hello World!')
            return xla.call_module([x], version=version, module=module, Tout=[np.float32], Sout=[()], platforms=platforms)
        expected_value = x + dict(CPU=2.0, CUDA=3.0, TPU=4.0)[self.testing_platform()]
        self._assertOpOutputMatchesExpected(f, (x,), (expected_value,))

    def test_platforms_and_poly(self):
        if False:
            i = 10
            return i + 15
        x = np.arange(6, dtype=np.float32)
        (module, version) = serialize('\nmodule @jit_f_jax attributes {jax.uses_shape_polymorphism = true} {\n  func.func public @main(%arg_platform_idx: tensor<i32> {jax.global_constant = "_platform_index"}, %arg0: tensor<?xf32>) -> (tensor<?xf32>) {\n    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf32>) -> tensor<i32>\n    %5 = call @_wrapped_jax_export_main(%arg_platform_idx, %0, %arg0) : (tensor<i32>, tensor<i32>, tensor<?xf32>) -> tensor<?xf32>\n    return %5 : tensor<?xf32>\n  }\n\n  func.func private @_wrapped_jax_export_main(%arg_platform_idx: tensor<i32> {jax.global_constant = "_platform_index"}, %arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?xf32>) -> (tensor<?xf32>) {\n    %to_add = "stablehlo.case"(%arg_platform_idx) ({\n      %cpu_val = stablehlo.constant dense<2.> : tensor<f32>\n      stablehlo.return %cpu_val : tensor<f32>\n    }, {\n      %gpu_val = stablehlo.constant dense<3.> : tensor<f32>\n      stablehlo.return %gpu_val : tensor<f32>\n    }, {\n      %tpu_val = stablehlo.constant dense<4.> : tensor<f32>\n      stablehlo.return %tpu_val : tensor<f32>\n    }) : (tensor<i32>) -> tensor<f32>\n    %1 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>\n    %3 = stablehlo.dynamic_broadcast_in_dim %to_add, %1, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>\n    %4 = stablehlo.add %3, %arg1 : tensor<?xf32>\n    return %4 : tensor<?xf32>\n  }\n}\n')
        platforms = ['CPU', 'CUDA', 'ROCM', 'TPU']

        def f(x):
            if False:
                i = 10
                return i + 15
            return xla.call_module([x], version=version, module=module, Tout=[np.float32], Sout=[()], platforms=platforms)
        expected_value = x + dict(CPU=2.0, CUDA=3.0, ROCM=3.0, TPU=4.0)[self.testing_platform()]
        self._assertOpOutputMatchesExpected(f, (x,), (expected_value,))

    def test_platforms_and_poly_and_tokens(self):
        if False:
            print('Hello World!')
        x = np.arange(6, dtype=np.float32)
        (module, version) = serialize('\nmodule @jit_f_jax attributes {jax.uses_shape_polymorphism = true} {\n  func.func public @main(%arg_platform_idx: tensor<i32> {jax.global_constant = "_platform_index"}, %arg_tok: !stablehlo.token {jax.token = true}, %arg0: tensor<?xf32>) -> (!stablehlo.token, tensor<?xf32>) {\n    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xf32>) -> tensor<i32>\n    %5:2 = call @_wrapped_jax_export_main(%arg_platform_idx, %0, %arg_tok, %arg0) : (tensor<i32>, tensor<i32>, !stablehlo.token, tensor<?xf32>) -> (!stablehlo.token, tensor<?xf32>)\n    return %5#0, %5#1 : !stablehlo.token, tensor<?xf32>\n  }\n\n  func.func private @_wrapped_jax_export_main(%arg_platform_idx: tensor<i32> {jax.global_constant = "_platform_index"}, %arg0: tensor<i32> {jax.global_constant = "b"}, %arg_tok: !stablehlo.token {jax.token = true}, %arg1: tensor<?xf32>) -> (!stablehlo.token, tensor<?xf32>) {\n    %to_add = "stablehlo.case"(%arg_platform_idx) ({\n      %cpu_val = stablehlo.constant dense<2.> : tensor<f32>\n      stablehlo.return %cpu_val : tensor<f32>\n    }, {\n      %gpu_val = stablehlo.constant dense<3.> : tensor<f32>\n      stablehlo.return %gpu_val : tensor<f32>\n    }, {\n      %tpu_val = stablehlo.constant dense<4.> : tensor<f32>\n      stablehlo.return %tpu_val : tensor<f32>\n    }) : (tensor<i32>) -> tensor<f32>\n    %1 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>\n    %3 = stablehlo.dynamic_broadcast_in_dim %to_add, %1, dims = [] : (tensor<f32>, tensor<1xi32>) -> tensor<?xf32>\n    %4 = stablehlo.add %3, %arg1 : tensor<?xf32>\n    return %arg_tok, %4 : !stablehlo.token, tensor<?xf32>\n  }\n}\n')
        platforms = ['CPU', 'CUDA', 'ROCM', 'TPU']

        def f(x):
            if False:
                return 10
            return xla.call_module([x], version=version, module=module, Tout=[np.float32], Sout=[()], platforms=platforms)
        expected_value = x + dict(CPU=2.0, CUDA=3.0, ROCM=3.0, TPU=4.0)[self.testing_platform()]
        self._assertOpOutputMatchesExpected(f, (x,), (expected_value,))
    platforms_errors_module_str = '\n  module @jit_f.0 {\n    func.func public @main(%arg_platform_idx: tensor<i32>, %arg0: tensor<f32>) -> tensor<f32> {\n      return %arg0 : tensor<f32>\n    }\n  }\n'

    def platforms_errors_helper(self, *, module_str: str, platforms: Sequence[str]=('CPU', 'CUDA', 'ROCM', 'TPU'), disabled_checks: Sequence[str]=(), expected_error: Optional[Exception]=None, expected_error_message: str=''):
        if False:
            print('Hello World!')
        (module, version) = serialize(module_str)
        x = np.float32(0.0)

        def f(x):
            if False:
                return 10
            return xla.call_module([x], version=version, module=module, Tout=[np.float32], Sout=[()], platforms=platforms, disabled_checks=disabled_checks)
        if expected_error is None:
            self._assertOpOutputMatchesExpected(f, (x,), (x,))
        else:
            with self.assertRaisesRegex(expected_error, expected_error_message):
                self._assertOpOutputMatchesExpected(f, (x,), (x,))

    def platforms_errors_singleton_platform(self):
        if False:
            i = 10
            return i + 15
        self.platforms_errors_helper(module_str=self.platforms_errors_module_str, platforms=(self.testing_platform(),), expected_error=errors.InvalidArgumentError, expected_error_message='Incorrect number of arguments passed to XlaCallModule = 1. The module main function takes 2 arguments of which 0 platform index arguments, 0 dimension arguments and 0 token arguments.')

    def platforms_errors_no_platform_index_arg(self):
        if False:
            print('Hello World!')
        module_str = self.platforms_errors_module_str.replace('%arg_platform_idx: tensor<i32>, %arg0: tensor<f32>', '')
        self.platforms_errors_helper(module_str=module_str, expected_error=errors.InvalidArgumentError, expected_error_message='The module should have a platform index argument but it has no arguments')

    def platforms_errors_platform_index_i16(self):
        if False:
            print('Hello World!')
        module_str = self.platforms_errors_module_str.replace('i32', 'i16')
        self.platforms_errors_helper(module_str=module_str, expected_error=errors.InvalidArgumentError, expected_error_message='Module argument at index 0 should be a 0-dimensional 32-bit or 64-bit integer-tensor platform index argument .* has type tensor<i16>')

    def platforms_errors_platform_index_non_scalar(self):
        if False:
            print('Hello World!')
        module_str = self.platforms_errors_module_str.replace('tensor<i32>', 'tensor<1xi32>')
        self.platforms_errors_helper(module_str=module_str, expected_error=errors.InvalidArgumentError, expected_error_message='Module argument at index 0 should be a 0-dimensional 32-bit integer-tensor platform index argument .* has type tensor<1xi32>')

    def platforms_errors_platform_index_unranked(self):
        if False:
            return 10
        module_str = self.platforms_errors_module_str.replace('tensor<i32>', 'tensor<*xi32>')
        self.platforms_errors_helper(module_str=module_str, expected_error=errors.InvalidArgumentError, expected_error_message='Module argument at index 0 should be a 0-dimensional 32-bit integer-tensor platform index argument')

    def platforms_errors_different_from_current(self):
        if False:
            while True:
                i = 10
        platform_check_disabled_by_flags = '--tf_xla_call_module_disabled_checks=platform' in os.getenv('TF_XLA_FLAGS', '')
        self.platforms_errors_helper(module_str=self.platforms_errors_module_str, platforms=['RANDOM_PLATFORM_1', 'RANDOM_PLATFORM_2'], expected_error=None if platform_check_disabled_by_flags else errors.NotFoundError, expected_error_message='current platform .* is not among the platforms')

    def platforms_errors_dissabled_check(self):
        if False:
            for i in range(10):
                print('nop')
        self.platforms_errors_helper(module_str=self.platforms_errors_module_str, platforms=('RANDOM_PLATFORM_1', 'RANDOM_PLATFORM_2'), disabled_checks=(xla.call_module_disable_check_platform(),), expected_error=None, expected_error_message='current platform .* is not among the platforms')

    def platforms_errors_empty(self):
        if False:
            print('Hello World!')
        self.platforms_errors_helper(module_str=self.platforms_errors_module_str, platforms=[], disabled_checks=[xla.call_module_disable_check_platform()], expected_error=None, expected_error_message='current platform .* is not among the platforms')

    def test_shape_assertion_success(self):
        if False:
            while True:
                i = 10
        x = np.ones((3, 5), dtype=np.int32)
        res = np.int32(x.shape[0])

        def f(x):
            if False:
                return 10
            (module, version) = serialize('\nmodule @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {\n  func.func public @main(%arg1: tensor<?x5xi32>) -> tensor<i32> {\n    %b = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x5xi32>) -> tensor<i32>\n    %3 = stablehlo.constant dense<3> : tensor<i32>\n    %ok = stablehlo.compare  EQ, %b, %3,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>\n    stablehlo.custom_call @shape_assertion(%ok) {\n      error_message = "The error message",\n      has_side_effect = true\n    } : (tensor<i1>) -> ()\n    return %b : tensor<i32>\n  }\n\n}\n')
            return xla.call_module([x], version=version, module=module, Tout=[res.dtype], Sout=[res.shape], platforms=[self.testing_platform()])
        self._assertOpOutputMatchesExpected(f, (x,), (res,))

    def test_shape_assertion_failure(self):
        if False:
            return 10
        x = np.ones((3, 5), dtype=np.int32)
        res = np.int32(x.shape[0])

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            (module, version) = serialize('\nmodule @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {\n  func.func public @main(%arg1: tensor<?x5xi32>) -> tensor<i32> {\n    %b = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x5xi32>) -> tensor<i32>\n    %4 = stablehlo.constant dense<4> : tensor<i32>\n    %5 = stablehlo.constant dense<5> : tensor<i32>\n    %11 = stablehlo.constant dense<11> : tensor<i32>\n    %ok = stablehlo.compare  EQ, %b, %4,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>\n    stablehlo.custom_call @shape_assertion(%ok, %b, %4, %5, %4, %5, %4, %5, %4, %5, %4, %5, %11) {\n      error_message = "Expecting {0} == {1}. Extra {2,=5}, {3}, {{0}, {4}, {5}, {6}, {7}, {11}.",\n      has_side_effect = true\n    } : (tensor<i1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> ()\n    return %b : tensor<i32>\n  }\n}\n')
            return xla.call_module([x], version=version, module=module, Tout=[res.dtype], Sout=[res.shape], platforms=[self.testing_platform()])
        disabled_shape_assertions_check = '--tf_xla_call_module_disabled_checks=shape_assertions' in os.getenv('TF_XLA_FLAGS', '')
        if disabled_shape_assertions_check:
            self._assertOpOutputMatchesExpected(f, (x,), (res,))
        else:
            with self.assertRaisesRegex(errors.InvalidArgumentError, re.escape('Expecting 3 == 4. Extra   5  , 4, {0}, 5, 4, 5, 4, 11.')):
                self._assertOpOutputMatchesExpected(f, (x,), (res,))

    def test_invalid_shape_assertion(self):
        if False:
            while True:
                i = 10
        arg_i1 = np.bool_(True)
        arg_i32 = np.int32(2)
        res = arg_i32
        disabled_shape_assertions_check = '--tf_xla_call_module_disabled_checks=shape_assertions' in os.getenv('TF_XLA_FLAGS', '')
        if disabled_shape_assertions_check:
            self.skipTest('Test is N/A when shape_assertions are disabled')
        subtest_count = 1

        def one_subtest(error_msg: str, module_str: str):
            if False:
                return 10

            def f(*args):
                if False:
                    return 10
                (module, version) = serialize(module_str)
                return xla.call_module(list(args), version=version, module=module, Tout=[res.dtype], Sout=[res.shape], platforms=[self.testing_platform()])
            nonlocal subtest_count
            subtest_count += 1
            with self.subTest(count=subtest_count, error_msg=error_msg):
                with self.assertRaisesRegex(errors.InvalidArgumentError, error_msg):
                    self._assertOpOutputMatchesExpected(f, (arg_i1, arg_i32), (res,))
        one_subtest('expects assert_what .* to be a constant of type tensor<i1>', '\nmodule @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {\n  func.func public @main(%arg_i1: tensor<i1>, %arg_i32: tensor<i32>) -> tensor<i32> {\n    %ok = stablehlo.constant dense<0> : tensor<i32>\n    stablehlo.custom_call @shape_assertion(%ok) {\n      error_message = "Some error",\n      has_side_effect = true\n    } : (tensor<i32>) -> ()\n    return %arg_i32 : tensor<i32>\n  }\n}\n')
        one_subtest('expects static assert_what', '\nmodule @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {\n  func.func public @main(%arg_i1: tensor<i1>, %arg_i32: tensor<i32>) -> tensor<i32> {\n    stablehlo.custom_call @shape_assertion(%arg_i1) {\n      error_message = "Some error",\n      has_side_effect = true\n    } : (tensor<i1>) -> ()\n    return %arg_i32 : tensor<i32>\n  }\n}\n')
        one_subtest('expects has_side_effect=true', '\nmodule @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {\n  func.func public @main(%arg_i1: tensor<i1>, %arg_i32: tensor<i32>) -> tensor<i32> {\n    %ok = stablehlo.constant dense<false> : tensor<i1>\n    stablehlo.custom_call @shape_assertion(%ok) {\n      error_message = "Some error",\n      has_side_effect = false\n    } : (tensor<i1>) -> ()\n    return %arg_i32 : tensor<i32>\n  }\n}\n')
        one_subtest('expects error_message .* Found specifier {0}', '\nmodule @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {\n  func.func public @main(%arg_i1: tensor<i1>, %arg_i32: tensor<i32>) -> tensor<i32> {\n    %ok = stablehlo.constant dense<false> : tensor<i1>\n    stablehlo.custom_call @shape_assertion(%ok) {\n      error_message = "Some error {0}",\n      has_side_effect = true\n    } : (tensor<i1>) -> ()\n    return %arg_i32 : tensor<i32>\n  }\n}\n')
        one_subtest('expects static error_message_input', '\nmodule @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {\n  func.func public @main(%arg_i1: tensor<i1>, %arg_i32: tensor<i32>) -> tensor<i32> {\n    %ok = stablehlo.constant dense<false> : tensor<i1>\n    stablehlo.custom_call @shape_assertion(%ok, %arg_i32) {\n      error_message = "Some error {0}",\n      has_side_effect = true\n    } : (tensor<i1>, tensor<i32>) -> ()\n    return %arg_i32 : tensor<i32>\n  }\n}\n')
        one_subtest('expects error_message_input .* to be a constant of type tensor<i32>', '\nmodule @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {\n  func.func public @main(%arg_i1: tensor<i1>, %arg_i32: tensor<i32>) -> tensor<i32> {\n    %ok = stablehlo.constant dense<false> : tensor<i1>\n    %c = stablehlo.constant dense<2.0> : tensor<f32>\n    stablehlo.custom_call @shape_assertion(%ok, %c) {\n      error_message = "Some error {0}",\n      has_side_effect = true\n    } : (tensor<i1>, tensor<f32>) -> ()\n    return %arg_i32 : tensor<i32>\n  }\n}\n')

    def test_dynamic_iota(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.ones((3, 5), dtype=np.int32)
        res = np.arange(x.shape[0], dtype=np.int32)

        def f(x):
            if False:
                return 10
            (module, version) = serialize('\nmodule @jit_fun.1 attributes {jax.uses_shape_polymorphism = true} {\n  func.func public @main(%arg1: tensor<?x5xi32>) -> tensor<?xi32> {\n    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x5xi32>) -> tensor<i32>\n    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?x5xi32>) -> tensor<?xi32>\n    return %0 : tensor<?xi32>\n  }\n  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?x5xi32>) -> tensor<?xi32> {\n    %0 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>\n    %1 = "stablehlo.dynamic_iota"(%0) {iota_dimension = 0 : i64} : (tensor<1xi32>) -> tensor<?xi32>\n    return %1 : tensor<?xi32>\n  }\n}\n')
            return xla.call_module([x], version=version, module=module, Tout=[res.dtype], Sout=[(None,)], platforms=[self.testing_platform()])
        self._assertOpOutputMatchesExpected(f, (x,), (res,))

    def test_build_graph_with_any_platform(self):
        if False:
            for i in range(10):
                print('nop')
        'We can construct the tf.Graph on all platforms.'
        x = np.float32(0.0)
        (module, version) = serialize('\nmodule @jit_f.0 {\n  func.func public @main(%arg_platform_idx: tensor<i32>, %arg0: tensor<f32>) -> tensor<f32> {\n    return %arg0 : tensor<f32>\n  }\n}\n')
        platforms = ['TPU']

        def f(x):
            if False:
                i = 10
                return i + 15
            return xla.call_module([x], version=version, module=module, Tout=[np.float32], Sout=[()], platforms=platforms)
        tf_graph = def_function.function(f).get_concrete_function(x).graph
        self.assertIn('XlaCallModule', str(tf_graph.as_graph_def()))

    def test_dynamic_reshape(self):
        if False:
            while True:
                i = 10
        x = np.ones((4, 3), dtype=np.float32)
        res = x.reshape((-1,))

        def f(x):
            if False:
                print('Hello World!')
            (module, version) = serialize('\nmodule @jit_fun_flat_jax attributes {jax.uses_shape_polymorphism = true} {\n  func.func public @main(%arg1: tensor<?x3xf32>) -> tensor<?xf32> {\n    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x3xf32>) -> tensor<i32>\n    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?x3xf32>) -> tensor<?xf32>\n    return %0 : tensor<?xf32>\n  }\n  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?x3xf32>) -> tensor<?xf32> {\n    %0 = stablehlo.constant dense<3> : tensor<i32>\n    %1 = stablehlo.multiply %arg0, %0 : tensor<i32>\n    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>\n    %3 = stablehlo.dynamic_reshape %arg1, %2 : (tensor<?x3xf32>, tensor<1xi32>) -> tensor<?xf32>\n    return %3 : tensor<?xf32>\n  }\n}\n')
            return xla.call_module([x], version=version, module=module, Tout=[res.dtype], Sout=[(None,)], platforms=[self.testing_platform()])
        self._assertOpOutputMatchesExpected(f, (x,), (res,))

    def test_dynamic_gather(self):
        if False:
            print('Hello World!')
        x = np.ones((3, 4), dtype=np.float32)
        res = np.ones((3, 2), dtype=np.float32)

        def f(x):
            if False:
                return 10
            (module, version) = serialize('\nmodule @jit_fun_flat_jax attributes {jax.uses_shape_polymorphism = true} {\n  func.func public @main(%arg1: tensor<?x4xf32>) -> tensor<?x2xf32> {\n    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x4xf32>) -> tensor<i32>\n    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?x4xf32>) -> tensor<?x2xf32>\n    return %0 : tensor<?x2xf32>\n  }\n  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?x4xf32>) -> tensor<?x2xf32> {\n    %0 = stablehlo.constant dense<0> : tensor<i64>\n    %1 = stablehlo.constant dense<0> : tensor<1xi64>\n    %2 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>\n    %3 = stablehlo.constant dense<2> : tensor<1xi32>\n    %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>\n    %5 = "stablehlo.dynamic_gather"(%arg1, %1, %4) {dimension_numbers = #stablehlo.gather<offset_dims = [0, 1], start_index_map = [1]>, indices_are_sorted = true} : (tensor<?x4xf32>, tensor<1xi64>, tensor<2xi32>) -> tensor<?x2xf32>\n    return %5 : tensor<?x2xf32>\n  }\n}\n')
            return xla.call_module([x], version=version, module=module, Tout=[res.dtype], Sout=[(None, 2)], platforms=[self.testing_platform()])
        self._assertOpOutputMatchesExpected(f, (x,), (res,))

    def test_real_dynamic_slice(self):
        if False:
            return 10
        x = np.ones((3, 4), dtype=np.float32)
        res = x[-1, :]

        def f(x):
            if False:
                return 10
            (module, version) = serialize('\nmodule @jit_fun_flat_jax attributes {jax.uses_shape_polymorphism = true} {\n  func.func public @main(%arg1: tensor<?x4xf32>) -> tensor<4xf32> {\n    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x4xf32>) -> tensor<i32>\n    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?x4xf32>) -> tensor<4xf32>\n    return %0 : tensor<4xf32>\n  }\n  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?x4xf32>) -> tensor<4xf32> {\n    %0 = stablehlo.constant dense<-1> : tensor<i32>\n    %1 = stablehlo.add %arg0, %0 : tensor<i32>\n    %2 = stablehlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>\n    %3 = stablehlo.constant dense<0> : tensor<1xi32>\n    %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>\n    %5 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>\n    %6 = stablehlo.constant dense<4> : tensor<1xi32>\n    %7 = stablehlo.concatenate %5, %6, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>\n    %10 = stablehlo.constant dense<1> : tensor<2xi32>\n    %11 = stablehlo.real_dynamic_slice %arg1, %4, %7, %10 : (tensor<?x4xf32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<1x4xf32>\n    %12 = stablehlo.reshape %11 : (tensor<1x4xf32>) -> tensor<4xf32>\n    return %12 : tensor<4xf32>\n  }\n}\n')
            return xla.call_module([x], version=version, module=module, Tout=[x.dtype], Sout=[(4,)], platforms=[self.testing_platform()])
        self._assertOpOutputMatchesExpected(f, (x,), (res,))

    def test_dynamic_update_slice(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.ones((3, 4), dtype=np.float32)
        idx = np.int32(-2)
        res = x

        def f(x, idx):
            if False:
                return 10
            (module, version) = serialize('\nmodule @jit_fun_flat_jax attributes {jax.uses_shape_polymorphism = true} {\n  func.func public @main(%arg1: tensor<?x4xf32>, %arg2: tensor<i32>) -> tensor<?x4xf32> {\n    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x4xf32>) -> tensor<i32>\n    %0 = call @dyn_main(%arg0_new, %arg1, %arg2) : (tensor<i32>, tensor<?x4xf32>, tensor<i32>) -> tensor<?x4xf32>\n    return %0 : tensor<?x4xf32>\n  }\n  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?x4xf32>, %arg2: tensor<i32>) -> tensor<?x4xf32> {\n    %0 = stablehlo.constant dense<0> : tensor<i32>\n    %1 = stablehlo.compare  LT, %arg2, %0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>\n    %2 = stablehlo.add %arg2, %arg0 : tensor<i32>\n    %3 = stablehlo.select %1, %2, %arg2 : tensor<i1>, tensor<i32>\n    %4 = stablehlo.constant dense<0> : tensor<i32>\n    %5 = stablehlo.dynamic_update_slice %arg1, %arg1, %3, %4 : (tensor<?x4xf32>, tensor<?x4xf32>, tensor<i32>, tensor<i32>) -> tensor<?x4xf32>\n    return %5 : tensor<?x4xf32>\n  }\n}\n')
            return xla.call_module([x, idx], version=version, module=module, Tout=[res.dtype], Sout=[(None, 4)], platforms=[self.testing_platform()])
        self._assertOpOutputMatchesExpected(f, (x, idx), (res,))

    def test_dynamic_broadcast_in_dim(self):
        if False:
            i = 10
            return i + 15
        x = np.ones((3, 4), dtype=np.float32)
        y = np.ones((2, 3, 4), dtype=np.float32)
        res = (np.broadcast_to(x, y.shape), x + y)

        def f(x, y):
            if False:
                i = 10
                return i + 15
            (module, version) = serialize('\nmodule @jit_fun.0 attributes {jax.uses_shape_polymorphism = true} {\n  func.func public @main(%arg1: tensor<?x4xf32>, %arg2: tensor<2x?x4xf32>) -> (tensor<2x?x4xf32>, tensor<2x?x4xf32>) {\n    %arg0_new = "stablehlo.get_dimension_size"(%arg2) {dimension = 1 : i64} : (tensor<2x?x4xf32>) -> tensor<i32>\n    %0, %1 = call @dyn_main(%arg0_new, %arg1, %arg2) : (tensor<i32>, tensor<?x4xf32>, tensor<2x?x4xf32>) -> (tensor<2x?x4xf32>, tensor<2x?x4xf32>)\n    return %0, %1 : tensor<2x?x4xf32>, tensor<2x?x4xf32>\n  }\n  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?x4xf32>, %arg2: tensor<2x?x4xf32>) -> (tensor<2x?x4xf32>, tensor<2x?x4xf32>) {\n    %0 = stablehlo.constant dense<2> : tensor<1xi32>\n    %2 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>\n    %3 = stablehlo.constant dense<4> : tensor<1xi32>\n    %4 = "stablehlo.concatenate"(%0, %2, %3) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>\n    %5 = "stablehlo.dynamic_broadcast_in_dim"(%arg1, %4) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<?x4xf32>, tensor<3xi32>) -> tensor<2x?x4xf32>\n    %6 = stablehlo.add %5, %arg2 : (tensor<2x?x4xf32>, tensor<2x?x4xf32>) -> tensor<2x?x4xf32>\n    return %5, %6 : tensor<2x?x4xf32>, tensor<2x?x4xf32>\n  }\n}\n')
            return xla.call_module([x, y], version=version, module=module, Tout=[res[0].dtype, res[1].dtype], Sout=[(2, None, 4), (2, None, 4)], platforms=[self.testing_platform()])
        self._assertOpOutputMatchesExpected(f, (x, y), res)

    @unittest.skip('TODO(necula): test is flaky')
    def test_reduce(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.arange(5, dtype=np.int32)
        res = np.sum(x) * x.shape[0]

        def f(x):
            if False:
                print('Hello World!')
            (module, version) = serialize('\nmodule @jit_fun attributes {jax.uses_shape_polymorphism = true} {\n  func.func public @main(%arg1: tensor<?xi32>) -> tensor<i32> {\n    %arg0_new = "stablehlo.get_dimension_size"(%arg2) {dimension = 0 : i64} : (tensor<?xi32>) -> tensor<i32>\n    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?xi32>) -> tensor<i32>\n    return %0 : tensor<i32>\n  }\n  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?xi32>) -> tensor<i32> {\n    %0 = stablehlo.constant dense<0> : tensor<i32>\n    %1 = stablehlo.reduce(%arg1 init: %0) across dimensions = [0] : (tensor<?xi32>, tensor<i32>) -> tensor<i32>\n     reducer(%arg2: tensor<i32>, %arg3: tensor<i32>)  {\n      %4 = stablehlo.add %arg2, %arg3 : tensor<i32>\n      "stablehlo.return"(%4) : (tensor<i32>) -> ()\n    }\n    %2 = stablehlo.multiply %1, %arg0 : tensor<i32>\n    return %2 : tensor<i32>\n  }\n}\n')
            return xla.call_module([x], version=version, module=module, Tout=[res.dtype], Sout=[res.shape], platforms=[self.testing_platform()])
        self._assertOpOutputMatchesExpected(f, (x,), (res,))

    def test_reduce_broadcast(self):
        if False:
            i = 10
            return i + 15
        x = np.broadcast_to(np.arange(3, dtype=np.float32).reshape(3, 1), (3, 5))
        res = np.arange(3, dtype=np.float32).reshape(3, 1) * 5

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            (module, version) = serialize('\nmodule @jit_fun_flat_jax attributes {jax.uses_shape_polymorphism = true} {\n  func.func public @main(%arg1: tensor<?x5xf32>) -> tensor<?x1xf32> {\n    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?x5xf32>) -> tensor<i32>\n    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?x5xf32>) -> tensor<?x1xf32>\n    return %0 : tensor<?x1xf32>\n  }\n  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?x5xf32>) -> tensor<?x1xf32> {\n    %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>\n    %1 = stablehlo.reduce(%arg1 init: %0) across dimensions = [1] : (tensor<?x5xf32>, tensor<f32>) -> tensor<?xf32>\n     reducer(%arg2: tensor<f32>, %arg3: tensor<f32>)  {\n      %6 = stablehlo.add %arg2, %arg3 : tensor<f32>\n      stablehlo.return %6 : tensor<f32>\n    }\n    %2 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>\n    %3 = stablehlo.constant dense<1> : tensor<1xi32>\n    %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>\n    %5 = stablehlo.dynamic_broadcast_in_dim %1, %4, dims = [0] : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x1xf32>\n    return %5 : tensor<?x1xf32>\n  }\n}\n')
            return xla.call_module([x], version=version, module=module, Tout=[res.dtype], Sout=[(None, 1)], platforms=[self.testing_platform()])
        self._assertOpOutputMatchesExpected(f, (x,), (res,))

    def test_call(self):
        if False:
            for i in range(10):
                print('nop')
        'A chain of calls.'
        x = np.ones((5,), dtype=np.float32)
        res = np.arange(x.shape[0], dtype=np.int32)

        def f(x):
            if False:
                return 10
            (module, version) = serialize('\nmodule @jit_fun_3 attributes {jax.uses_shape_polymorphism = true} {\n  func.func public @main(%arg1: tensor<?xf32>) -> tensor<?xi32> {\n    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?xf32>) -> tensor<i32>\n    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?xf32>) -> tensor<?xi32>\n    return %0 : tensor<?xi32>\n  }\n  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?xf32>) -> tensor<?xi32> {\n    %0 = call @f(%arg0, %arg1) : (tensor<i32>, tensor<?xf32>) -> tensor<?xi32>\n    return %0 : tensor<?xi32>\n  }\n  func.func private @f(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?xf32>) -> tensor<?xi32> {\n    %0 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>\n    %1 = "stablehlo.dynamic_iota"(%0) {iota_dimension = 0 : i64} : (tensor<1xi32>) -> tensor<?xi32>\n    return %1 : tensor<?xi32>\n  }\n}\n')
            return xla.call_module([x], version=version, module=module, Tout=[res.dtype], Sout=[()], platforms=[self.testing_platform()])
        self._assertOpOutputMatchesExpected(f, (x,), (res,))

    def test_identity(self):
        if False:
            while True:
                i = 10
        x = np.ones((5,), dtype=np.float32)
        res = x

        def f(x):
            if False:
                i = 10
                return i + 15
            (module, version) = serialize('\nmodule @jit_fun_3 attributes {jax.uses_shape_polymorphism = true} {\n  func.func public @main(%arg1: tensor<?xf32>) -> tensor<?xf32> {\n    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?xf32>) -> tensor<i32>\n    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?xf32>) -> tensor<?xf32>\n    return %0 : tensor<?xf32>\n  }\n  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?xf32>) -> tensor<?xf32> {\n    return %arg1 : tensor<?xf32>\n  }\n}\n')
            return xla.call_module([x], version=version, module=module, Tout=[res.dtype], Sout=[()], platforms=[self.testing_platform()])
        self._assertOpOutputMatchesExpected(f, (x,), (res,))

    def test_while(self):
        if False:
            print('Hello World!')
        'A while loop with carryied dynamic shapes.'
        x = np.ones((5,), dtype=np.float32)
        res0 = np.copy(x)
        for _ in range(5):
            res0 += np.arange(x.shape[0], dtype=np.float32)
        res1 = np.int64(5)

        def f(x):
            if False:
                i = 10
                return i + 15
            (module, version) = serialize('\nmodule @jit_fun_flat_jax attributes {jax.uses_shape_polymorphism = true} {\n  func.func public @main(%arg1: tensor<?xf32>) -> (tensor<?xf32>, tensor<i64>) {\n    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?xf32>) -> tensor<i32>\n    %0, %1 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?xf32>) -> (tensor<?xf32>, tensor<i64>)\n    return %0, %1 : tensor<?xf32>, tensor<i64>\n  }\n  func.func private @dyn_main(%arg0: tensor<i32> {jax.global_constant = "b"}, %arg1: tensor<?xf32>) -> (tensor<?xf32>, tensor<i64>) {\n    %0 = stablehlo.constant dense<0> : tensor<i64>\n    %1:2 = "stablehlo.while"(%arg1, %0) ({\n    ^bb0(%arg2: tensor<?xf32>, %arg3: tensor<i64>):\n      %2 = stablehlo.constant dense<5> : tensor<i64>\n      %3 = stablehlo.compare  LT, %arg3, %2,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>\n      stablehlo.return %3 : tensor<i1>\n    }, {\n    ^bb0(%arg2: tensor<?xf32>, %arg3: tensor<i64>):\n      %2 = stablehlo.reshape %arg0 : (tensor<i32>) -> tensor<1xi32>\n      %3 = stablehlo.dynamic_iota %2, dim = 0 : (tensor<1xi32>) -> tensor<?xf32>\n      %4 = stablehlo.add %arg2, %3 : tensor<?xf32>\n      %5 = stablehlo.constant dense<1> : tensor<i64>\n      %6 = stablehlo.add %arg3, %5 : tensor<i64>\n      stablehlo.return %4, %6 : tensor<?xf32>, tensor<i64>\n    }) : (tensor<?xf32>, tensor<i64>) -> (tensor<?xf32>, tensor<i64>)\n    return %1#0, %1#1 : tensor<?xf32>, tensor<i64>\n  }\n}\n')
            return xla.call_module([x], version=version, module=module, Tout=[res0.dtype, res1.dtype], Sout=[(None,), res1.shape], platforms=[self.testing_platform()])
        self._assertOpOutputMatchesExpected(f, (x,), (res0, res1))

    def test_skip_shape_refinement(self):
        if False:
            return 10
        x = np.ones((5,), dtype=np.float32)
        res = x
        module_attrs = ''

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            (module, version) = serialize(f'\nmodule @jit_fun_3 {module_attrs} {{\n  func.func public @main(%arg1: tensor<?xf32>) -> tensor<?xf32> {{\n    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {{dimension = 0 : i64}} : (tensor<?xf32>) -> tensor<i32>\n    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?xf32>) -> tensor<?xf32>\n    return %0 : tensor<?xf32>\n  }}\n  func.func private @dyn_main(%arg0: tensor<i32> {{jax.global_constant = "b"}}, %arg1: tensor<?xf32>) -> tensor<?xf32> {{\n    return %arg1 : tensor<?xf32>\n  }}\n}}\n')
            return xla.call_module([x], version=version, module=module, Tout=[res.dtype], Sout=[()], platforms=[self.testing_platform()])
        module_attrs = ''
        with self.assertRaisesRegex(errors.InvalidArgumentError, 'Module has dynamic shapes'):
            self._assertOpOutputMatchesExpected(f, (x,), (res,))
        module_attrs = 'attributes {jax.uses_shape_polymorphism = false}'
        with self.assertRaisesRegex(errors.InvalidArgumentError, 'Module has dynamic shapes'):
            self._assertOpOutputMatchesExpected(f, (x,), (res,))

    def test_uses_shape_polymorphism_before_version_8(self):
        if False:
            print('Hello World!')
        x = np.ones((5,), dtype=np.float32)
        res = x

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            version = 7
            (module, _) = serialize('\nmodule @jit_fun_3 {\n  func.func public @main(%arg1: tensor<?xf32>) -> tensor<?xf32> {\n    %arg0_new = "stablehlo.get_dimension_size"(%arg1) {dimension = 0 : i64} : (tensor<?xf32>) -> tensor<i32>\n    %0 = call @dyn_main(%arg0_new, %arg1) : (tensor<i32>, tensor<?xf32>) -> tensor<?xf32>\n    return %0 : tensor<?xf32>\n  }\n  func.func private @dyn_main(%arg0: tensor<i32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {\n    return %arg1 : tensor<?xf32>\n  }\n}\n')
            return xla.call_module([x], version=version, module=module, Tout=[res.dtype], Sout=[()], platforms=[self.testing_platform()])
        self._assertOpOutputMatchesExpected(f, (x,), (res,))

    def test_tf_call_function(self):
        if False:
            i = 10
            return i + 15
        'A TensorFlow function call inside StableHLO.'
        x = np.int32(2)
        y = np.int32(3)
        res = x + y

        @function.Defun(dtypes.int32, dtypes.int32)
        def foo(x, y):
            if False:
                return 10
            return x + y

        def f(x, y):
            if False:
                while True:
                    i = 10
            (module, version) = serialize('\nmodule @jit_fun_flat_jax {\n  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {\n    %0 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {\n      tf.backend_config = {called_index = 0}\n    } : (tensor<i32>, tensor<i32>) -> tensor<i32>\n    return %0 : tensor<i32>\n  }\n}\n')
            return xla.call_module([x, y], version=version, module=module, Tout=[res.dtype], Sout=[res.shape], platforms=[self.testing_platform()], function_list=(foo,))
        self._assertOpOutputMatchesExpected(f, (x, y), (res,))

    def test_tf_call_function_multiple_funcs(self):
        if False:
            return 10
        'Multiple TensorFlow function calls inside StableHLO.'
        x = np.int32(2)
        y = np.int32(3)
        res = x + y + (x + y)

        @function.Defun(dtypes.int32, dtypes.int32)
        def foo(x, y):
            if False:
                i = 10
                return i + 15
            return x + y

        @function.Defun(dtypes.int32, dtypes.int32)
        def bar(x, y):
            if False:
                return 10
            return foo(x, y)

        def f(x, y):
            if False:
                print('Hello World!')
            (module, version) = serialize('\nmodule @jit_fun_flat_jax {\n  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {\n    %0 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {\n      tf.backend_config = {called_index = 0}\n    } : (tensor<i32>, tensor<i32>) -> tensor<i32>\n    %1 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {\n      tf.backend_config = {called_index = 1}\n    } : (tensor<i32>, tensor<i32>) -> tensor<i32>\n    %2 = stablehlo.custom_call @tf.call_tf_function(%0, %1) {\n      tf.backend_config = {called_index = 1}\n    } : (tensor<i32>, tensor<i32>) -> tensor<i32>\n    return %2 : tensor<i32>\n  }\n}\n')
            return xla.call_module([x, y], version=version, module=module, Tout=[res.dtype], Sout=[res.shape], platforms=[self.testing_platform()], function_list=(foo, bar))
        self._assertOpOutputMatchesExpected(f, (x, y), (res,))

    def test_shape_polymorphic_tf_call_function(self):
        if False:
            return 10
        'A TensorFlow function call inside StableHLO.'
        x = np.full((2,), 2, dtype=np.int32)
        y = np.full((2,), 3, dtype=np.int32)
        res = x + y

        @function.Defun(dtypes.int32, dtypes.int32)
        def foo(x, y):
            if False:
                i = 10
                return i + 15
            return x + y

        def f(x, y):
            if False:
                print('Hello World!')
            (module, version) = serialize('\nmodule @jit_fun_flat_jax attributes {jax.uses_shape_polymorphism = true} {\n  func.func public @main(%arg0: tensor<?xi32>, %arg1: tensor<?xi32>) -> tensor<?xi32> {\n    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?xi32>) -> tensor<i32>\n    %1 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1, %0) {\n      tf.backend_config = {called_index = 0},\n      indices_of_shape_operands = dense<[2]> : tensor<1xi64>\n    } : (tensor<?xi32>, tensor<?xi32>, tensor<i32>) -> tensor<?xi32>\n    return %1 : tensor<?xi32>\n  }\n}\n')
            return xla.call_module([x, y], version=version, module=module, Tout=[res.dtype], Sout=[res.shape], platforms=[self.testing_platform()], function_list=(foo,))
        self._assertOpOutputMatchesExpected(f, (x, y), (res,))

    def test_tf_call_function_with_token(self):
        if False:
            return 10
        'A TensorFlow function call inside StableHLO.'
        x = np.int32(2)
        y = np.int32(3)
        res = x + y

        @function.Defun(dtypes.int32, dtypes.int32)
        def foo(x, y):
            if False:
                i = 10
                return i + 15
            return x + y

        def f(x, y):
            if False:
                print('Hello World!')
            (module, version) = serialize('\nmodule @jit_fun_flat_jax {\n  func.func public @main(%arg0: !stablehlo.token, %arg1: tensor<i32>, %arg2: tensor<i32>) -> (!stablehlo.token, tensor<i32>) {\n    %0:2 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1, %arg2) {\n      tf.backend_config = {called_index = 0, has_token_input_output = true}\n    } : (!stablehlo.token, tensor<i32>, tensor<i32>) -> (!stablehlo.token, tensor<i32>)\n    return %0#0, %0#1 : !stablehlo.token, tensor<i32>\n  }\n}\n')
            return xla.call_module([x, y], version=version, module=module, Tout=[res.dtype], Sout=[res.shape], platforms=[self.testing_platform()], function_list=(foo,))
        self._assertOpOutputMatchesExpected(f, (x, y), (res,))

    def test_tf_call_function_nested(self):
        if False:
            i = 10
            return i + 15
        'Nested XlaCallModule inside TensorFlow function calls.'
        x = np.int32(2)
        y = np.int32(3)
        res = x + y

        @function.Defun(dtypes.int32, dtypes.int32)
        def add(x, y):
            if False:
                print('Hello World!')
            return x + y

        @function.Defun(dtypes.int32, dtypes.int32)
        def nested_xla_call(x, y):
            if False:
                for i in range(10):
                    print('nop')
            (module, version) = serialize('\nmodule @jit_fun_flat_jax {\n  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {\n    %0 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {\n      tf.backend_config = {called_index = 0}\n    } : (tensor<i32>, tensor<i32>) -> tensor<i32>\n    return %0 : tensor<i32>\n  }\n}\n')
            return xla.call_module([x, y], version=version, module=module, Tout=[res.dtype], Sout=[res.shape], platforms=[self.testing_platform()], function_list=(add,))

        @function.Defun(dtypes.int32, dtypes.int32)
        def call(x, y):
            if False:
                return 10
            return nested_xla_call(x, y)

        def f(x, y):
            if False:
                return 10
            (module, version) = serialize('\nmodule @jit_fun_flat_jax {\n  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {\n    %0 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {\n      tf.backend_config = {called_index = 0}\n    } : (tensor<i32>, tensor<i32>) -> tensor<i32>\n    return %0 : tensor<i32>\n  }\n}\n')
            return xla.call_module([x, y], version=version, module=module, Tout=[res.dtype], Sout=[res.shape], platforms=[self.testing_platform()], function_list=(call,))
        self._assertOpOutputMatchesExpected(f, (x, y), (res,))

    def test_tf_call_function_nested_func_renaming(self):
        if False:
            while True:
                i = 10
        'Multiple custom calls with identically named private functions.'
        x = np.int32(2)
        y = np.int32(3)
        res0 = x + y
        res1 = x - y

        @function.Defun(dtypes.int32, dtypes.int32)
        def add(x, y):
            if False:
                i = 10
                return i + 15
            (module, version) = serialize('\nmodule @jit_fun_flat_jax {\n  func.func private @call(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {\n    %0 = stablehlo.add %arg0, %arg1 : tensor<i32>\n    return %0 : tensor<i32>\n  }\n\n  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {\n    %0 = func.call @call(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>\n    return %0 : tensor<i32>\n  }\n}\n')
            return xla.call_module([x, y], version=version, module=module, Tout=[res0.dtype], Sout=[res0.shape], platforms=[self.testing_platform()])

        @function.Defun(dtypes.int32, dtypes.int32)
        def subtract(x, y):
            if False:
                for i in range(10):
                    print('nop')
            (module, version) = serialize('\nmodule @jit_fun_flat_jax {\n  func.func private @call(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {\n    %0 = stablehlo.subtract %arg0, %arg1 : tensor<i32>\n    return %0 : tensor<i32>\n  }\n\n  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<i32> {\n    %0 = func.call @call(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>\n    return %0 : tensor<i32>\n  }\n}\n')
            return xla.call_module([x, y], version=version, module=module, Tout=[res1.dtype], Sout=[res1.shape], platforms=[self.testing_platform()])

        def f(x, y):
            if False:
                return 10
            (module, version) = serialize('\nmodule @jit_fun_flat_jax {\n  func.func public @main(%arg0: tensor<i32>, %arg1: tensor<i32>) -> (tensor<i32>, tensor<i32>) {\n    %0 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {\n      tf.backend_config = {called_index = 0}\n    } : (tensor<i32>, tensor<i32>) -> tensor<i32>\n    %1 = stablehlo.custom_call @tf.call_tf_function(%arg0, %arg1) {\n      tf.backend_config = {called_index = 1}\n    } : (tensor<i32>, tensor<i32>) -> tensor<i32>\n    return %0, %1 : tensor<i32>, tensor<i32>\n  }\n}\n')
            return xla.call_module([x, y], version=version, module=module, Tout=[res0.dtype, res1.dtype], Sout=[res0.shape, res1.shape], platforms=[self.testing_platform()], function_list=(add, subtract))
        self._assertOpOutputMatchesExpected(f, (x, y), (res0, res1))

    def test_op_backward_compatibility(self):
        if False:
            for i in range(10):
                print('nop')
        'Test for ensuring XlaCallModuleOp backward compatiblity.'
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        def f(x):
            if False:
                return 10
            (module, version) = serialize('\nmodule @jit_f.0 {\n  func.func public @main(%arg0: tensor<3xf32>) -> tensor<3xf32> {\n    %0 = stablehlo.cosine %arg0 : tensor<3xf32>\n    %1 = stablehlo.sine %0 : tensor<3xf32>\n    return %1 : tensor<3xf32>\n  }\n}\n')
            return gen_xla_ops.xla_call_module([x], version=version, module=module, Tout=[x.dtype], Sout=[x.shape], platforms=[self.testing_platform()])
        self._assertOpOutputMatchesExpected(f, (x,), (np.sin(np.cos(x)),))
if __name__ == '__main__':
    ops.enable_eager_execution(config=config_pb2.ConfigProto(log_device_placement=True))
    googletest.main()