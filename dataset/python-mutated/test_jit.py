import torch
from jit.test_tracer import TestTracer, TestMixTracingScripting
from jit.test_recursive_script import TestRecursiveScript
from jit.test_type_sharing import TestTypeSharing
from jit.test_logging import TestLogging
from jit.test_backends import TestBackends, TestBackendsWithCompiler
from jit.test_backend_nnapi import TestNnapiBackend
from jit.test_list_dict import TestList, TestDict, TestNamedTuple, TestScriptDict, TestScriptList
from jit.test_async import TestAsync
from jit.test_await import TestAwait
from jit.test_data_parallel import TestDataParallel
from jit.test_models import TestModels
from jit.test_modules import TestModules
from jit.test_autodiff import TestAutodiffJit
from jit.test_autodiff_subgraph_slicing import TestAutodiffSubgraphSlicing
from jit.test_custom_operators import TestCustomOperators
from jit.test_graph_rewrite_passes import TestGraphRewritePasses
from jit.test_class_type import TestClassType
from jit.test_builtins import TestBuiltins, TestTensorBuiltins
from jit.test_ignore_context_manager import TestIgnoreContextManager
from jit.test_symbolic_shape_analysis import TestSymbolicShapeAnalysis
from jit.test_op_decompositions import TestOpDecompositions
from jit.test_unsupported_ops import TestUnsupportedOps
from jit.test_freezing import TestFreezing, TestFrozenOptimizations, TestMKLDNNReinplacing
from jit.test_peephole import TestPeephole
from jit.test_alias_analysis import TestAliasAnalysis
from jit.test_save_load import TestSaveLoad, TestSaveLoadFlatbuffer
from jit.test_save_load_for_op_version import TestSaveLoadForOpVersion
from jit.test_module_containers import TestModuleContainers
from jit.test_python_bindings import TestPythonBindings
from jit.test_python_ir import TestPythonIr
from jit.test_functional_blocks import TestFunctionalBlocks
from jit.test_remove_mutation import TestRemoveMutation
from jit.test_torchbind import TestTorchbind
from jit.test_module_interface import TestModuleInterface
from jit.test_with import TestWith
from jit.test_enum import TestEnum
from jit.test_string_formatting import TestStringFormatting
from jit.test_profiler import TestProfiler
from jit.test_slice import TestSlice
from jit.test_ignorable_args import TestIgnorableArgs
from jit.test_hooks import TestHooks
from jit.test_warn import TestWarn
from jit.test_isinstance import TestIsinstance
from jit.test_cuda import TestCUDA
from jit.test_python_builtins import TestPythonBuiltinOP
from jit.test_typing import TestTyping
from jit.test_hash import TestHash
from jit.test_complex import TestComplex
from jit.test_jit_utils import TestJitUtils
from jit.test_scriptmod_ann import TestScriptModuleInstanceAttributeTypeAnnotation
from jit.test_types import TestTypesAndAnnotation
from jit.test_misc import TestMisc
from jit.test_upgraders import TestUpgraders
from jit.test_pdt import TestPDT
from jit.test_tensor_creation_ops import TestTensorCreationOps
from jit.test_module_apis import TestModuleAPIs
from jit.test_script_profile import TestScriptProfile
from jit.test_convert_activation import TestFunctionalToInplaceActivation, TestInplaceToFunctionalActivation
from jit.test_parametrization import TestParametrization
from jit.test_attr import TestGetDefaultAttr
from jit.test_aten_pow import TestAtenPow
from jit.test_optimize_for_mobile_preserve_debug_info import TestOptimizeForMobilePreserveDebugInfo
from jit.test_union import TestUnion
from jit.test_batch_mm import TestBatchMM
from jit.test_dtype_analysis import TestDtypeAnalysis, TestDtypeCustomRulesCPU
from jit.test_device_analysis import TestDeviceAnalysis
from jit.test_dce import TestDCE
from jit.test_sparse import TestSparse
from jit.test_tensor_methods import TestTensorMethods
from jit.test_dataclasses import TestDataclasses
from torch import Tensor
from torch._C import TensorType, BoolType, parse_ir, _propagate_shapes
from torch.autograd import Variable
from torch.jit.annotations import BroadcastingList2, BroadcastingList3, Any
from torch.nn.utils.rnn import PackedSequence
from torch.testing import FileCheck, make_tensor
import torch.autograd.profiler
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal import jit_utils
from torch.testing._internal.common_jit import check_against_reference
from torch.testing._internal.common_utils import run_tests, IS_WINDOWS, TEST_WITH_UBSAN, suppress_warnings, BUILD_WITH_CAFFE2, IS_SANDCASTLE, GRAPH_EXECUTOR, ProfilingMode, TestCase, freeze_rng_state, slowTest, TemporaryFileName, enable_profiling_mode_for_profiling_tests, TEST_MKL, set_default_dtype, num_profiled_runs, skipIfCrossRef, IS_MACOS, skipIfTorchDynamo
from torch.testing._internal.jit_utils import JitTestCase, enable_cpu_fuser, disable_autodiff_subgraph_inlining, _trace, do_input_map, get_execution_plan, make_global, execWrapper, _inline_everything, _tmp_donotuse_dont_inline_everything, RUN_CUDA
from torch.testing._internal.jit_metaprogramming_utils import get_script_args, create_input, unpack_variables, additional_module_tests, EXCLUDE_SCRIPT_MODULES, get_nn_module_name_from_kwargs, get_nn_mod_test_name, script_method_template
from torch.testing._internal.common_nn import module_tests, new_module_tests, criterion_tests
from torch.testing._internal.test_module.future_div import div_int_future, div_float_future
from torch.testing._internal.test_module.no_future_div import div_int_nofuture, div_float_nofuture
from collections import defaultdict, namedtuple, OrderedDict
from copy import deepcopy
from itertools import product
from textwrap import dedent
from typing import List, Dict, NamedTuple, Optional, Tuple, Union
import copy
import functools
import inspect
import io
import itertools
import math
import numpy as np
import os
import pickle
import pickletools
import random
import re
import shutil
import string
import sys
import tempfile
import types
import typing
import unittest
import warnings
import zipfile
import tracemalloc

def canonical(graph):
    if False:
        while True:
            i = 10
    return torch._C._jit_pass_canonicalize(graph).str(False)

def LSTMCellF(input, hx, cx, *params):
    if False:
        for i in range(10):
            print('nop')
    return LSTMCell(input, (hx, cx), *params)

def doAutodiffCheck(testname):
    if False:
        print('Hello World!')
    if 'test_t_' in testname or testname == 'test_t':
        return False
    if GRAPH_EXECUTOR == ProfilingMode.SIMPLE:
        return False
    if GRAPH_EXECUTOR == ProfilingMode.LEGACY:
        return True
    test_exceptions = ['test_nn_dropout', 'test_nn_log_softmax', 'test_nn_relu', 'test_nn_softmax', 'test_nn_threshold', 'test_nn_lp_pool2d', 'test_nn_lp_pool1d', 'test_nn_gumbel_softmax_hard', 'test_nn_gumbel_softmax', 'test_nn_multilabel_soft_margin_loss', 'test_nn_batch_norm', 'test_nn_max_pool2d_with_indices', 'test___rdiv___constant', 'test___rdiv___scalar_constant', 'test_split', 'test_split_dim', 'test_split_dim_neg0', 'test_split_size_list', 'test_split_size_list_dim', 'test_split_size_list_dim_neg0', 'test_split_with_sizes', 'test_split_with_sizes_dim', 'test_split_with_sizes_dim_neg0', 'test_split_with_sizes_size_0', 'test_nn_max_pool2d_with_indices']
    if testname in test_exceptions:
        return False
    return True
torch._C._jit_set_texpr_fuser_enabled(GRAPH_EXECUTOR == ProfilingMode.PROFILING)
torch._C._jit_set_profiling_executor(GRAPH_EXECUTOR != ProfilingMode.LEGACY)

def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    if False:
        return 10
    (hx, cx) = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
    (ingate, forgetgate, cellgate, outgate) = gates.chunk(4, 1)
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)
    cy = forgetgate * cx + ingate * cellgate
    hy = outgate * torch.tanh(cy)
    return (hy, cy)

def LSTMCellC(*args, **kwargs):
    if False:
        return 10
    (hy, cy) = LSTMCellF(*args, **kwargs)
    return torch.cat((hy, cy))

def LSTMCellS(x, hx, cx, w_ih, w_hh, b_ih, b_hh):
    if False:
        while True:
            i = 10
    gates = x.mm(w_ih.t()) + hx.mm(w_hh.t()) + b_ih + b_hh
    (ingate, forgetgate, cellgate, outgate) = gates.chunk(4, 1)
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)
    cy = forgetgate * cx + ingate * cellgate
    hy = outgate * torch.tanh(cy)
    return (hy, cy)

def MiLSTMCell(x, hx, cx, w_ih, w_hh, alpha, beta_i, beta_h, bias):
    if False:
        for i in range(10):
            print('nop')
    Wx = x.mm(w_ih.t())
    Uz = hx.mm(w_hh.t())
    gates = alpha * Wx * Uz + beta_i * Wx + beta_h * Uz + bias
    (ingate, forgetgate, cellgate, outgate) = gates.chunk(4, 1)
    ingate = ingate.sigmoid()
    forgetgate = forgetgate.sigmoid()
    cellgate = cellgate.tanh()
    outgate = outgate.sigmoid()
    cy = forgetgate * cx + ingate * cellgate
    hy = outgate * cy.tanh()
    return (hy, cy)

def get_lstm_inputs(device, training=False, seq_length=None):
    if False:
        i = 10
        return i + 15
    input_shape = (3, 10) if seq_length is None else (seq_length, 3, 10)
    input = torch.randn(*input_shape, dtype=torch.float, device=device, requires_grad=training)
    hx = torch.randn(3, 20, dtype=torch.float, device=device, requires_grad=training)
    cx = torch.randn(3, 20, dtype=torch.float, device=device, requires_grad=training)
    module = nn.LSTMCell(10, 20).to(device, torch.float)
    if training:
        params = tuple(module.parameters())
    else:
        params = tuple((p.requires_grad_(False) for p in module.parameters()))
    return (input, hx, cx) + params

def get_milstm_inputs(device, training=False):
    if False:
        i = 10
        return i + 15
    minibatch = 3
    input_size = 10
    hidden_size = 20
    x = torch.randn(minibatch, input_size, device=device, dtype=torch.float)
    hx = torch.randn(minibatch, hidden_size, device=device, dtype=torch.float)
    cx = torch.randn(minibatch, hidden_size, device=device, dtype=torch.float)
    ih = torch.randn(4 * hidden_size, input_size, device=device, dtype=torch.float, requires_grad=training)
    hh = torch.randn(4 * hidden_size, hidden_size, device=device, dtype=torch.float, requires_grad=training)
    alpha = torch.randn(4 * hidden_size, dtype=torch.float, device=device, requires_grad=training)
    ibeta = torch.randn(4 * hidden_size, dtype=torch.float, device=device, requires_grad=training)
    hbeta = torch.randn(4 * hidden_size, dtype=torch.float, device=device, requires_grad=training)
    bias = torch.randn(4 * hidden_size, dtype=torch.float, device=device, requires_grad=training)
    return (x, hx, cx, ih, hh, alpha, ibeta, hbeta, bias)

def get_fn(file_name, script_path):
    if False:
        print('Hello World!')
    import importlib.util
    spec = importlib.util.spec_from_file_location(file_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = module.fn
    return fn

def get_grad_executor(plan_state, diff_graph_idx=None, skip_check=False):
    if False:
        return 10
    if diff_graph_idx is None:
        nodes = list(plan_state.graph.nodes())
        if not skip_check:
            nodes = list(filter(lambda n: n.kind() != 'prim::BailOut' and n.kind() != 'prim::BailoutTemplate', nodes))
            if len(nodes) == 1 or (len(nodes) == 2 and nodes[1].kind() == 'prim::TupleConstruct'):
                pass
            elif len(nodes) == 2 and nodes[0].kind() == 'prim::RequiresGradCheck' and (nodes[1].kind() == 'prim::If'):
                pass
            else:
                raise RuntimeError("Can't get a grad_executor for a non-differentiable graph")
    grad_executors = list(plan_state.code.grad_executor_states())
    return grad_executors[diff_graph_idx or 0]

def all_backward_graphs(script_module, diff_graph_idx=None):
    if False:
        print('Hello World!')
    ge_state = script_module.get_debug_state()
    fwd_plan = get_execution_plan(ge_state)
    grad_executor_state = get_grad_executor(fwd_plan, diff_graph_idx=diff_graph_idx)
    bwd_plans = list(grad_executor_state.execution_plans.values())
    return [p.graph.copy() for p in bwd_plans]

def backward_graph(script_module, diff_graph_idx=None, skip_check=False):
    if False:
        for i in range(10):
            print('nop')
    ge_state = script_module.get_debug_state()
    fwd_plan = get_execution_plan(ge_state)
    grad_executor_state = get_grad_executor(fwd_plan, diff_graph_idx=diff_graph_idx, skip_check=skip_check)
    bwd_plan = get_execution_plan(grad_executor_state)
    return bwd_plan.graph.copy()

def _sum_of_list(tensorlist):
    if False:
        print('Hello World!')
    s = 0
    for t in tensorlist:
        s += t.sum()
    return s

class FooToPickle(torch.nn.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.bar = torch.jit.ScriptModule()

@skipIfTorchDynamo()
class TestJitProfiler(JitTestCase):
    """
    This runs tests that requires setting some global states like torch._C._set_graph_executor_optimize
    and restore the values afterward, i.e. test_profiler. This is to address the flaky issue in
    https://github.com/pytorch/pytorch/issues/91483 in which test_profiler was flaky and failed in the
    middle without the chance to restore torch._C._set_graph_executor_optimize to its original value.
    This causes issues for all future tests running after.

    Using a separate test class here, so that there is no need to run setup and teardown for all tests
    in TestJit.
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.graph_executor_optimize_opt = torch._C._get_graph_executor_optimize()

    def tearDown(self):
        if False:
            return 10
        super().tearDown()
        torch._C._set_graph_executor_optimize(self.graph_executor_optimize_opt)

    def test_profiler(self):
        if False:
            while True:
                i = 10
        torch._C._set_graph_executor_optimize(False)

        def other_fn(x):
            if False:
                i = 10
                return i + 15
            return x * 2
        x = torch.rand(3, 4)
        traced_other_fn = torch.jit.trace(other_fn, x)

        def fn(x):
            if False:
                i = 10
                return i + 15
            y = traced_other_fn(x)
            fut = torch.jit._fork(traced_other_fn, x)
            y = torch.jit._wait(fut)
            return y
        traced_fn = torch.jit.trace(fn, x)
        with torch.autograd.profiler.profile() as prof:
            traced_fn(x)
        mul_events = defaultdict(int)
        other_fn_events = defaultdict(int)
        for e in prof.function_events:
            if e.name == 'aten::mul':
                self.assertTrue(e.thread not in mul_events)
                mul_events[e.thread] = e.time_range.elapsed_us()
            elif e.name == 'other_fn':
                self.assertTrue(e.thread not in other_fn_events)
                other_fn_events[e.thread] = e.time_range.elapsed_us()
        self.assertTrue(len(mul_events) == 2)
        self.assertTrue(len(other_fn_events) == 2)
        for (thread, mul_time) in mul_events.items():
            self.assertTrue(thread in other_fn_events)
            self.assertTrue(other_fn_events[thread] >= mul_time)

@skipIfTorchDynamo()
class TestJit(JitTestCase):

    @unittest.skip('Requires a lot of RAM')
    def test_big(self):
        if False:
            i = 10
            return i + 15
        m = torch.jit.ScriptModule()
        gig = int(1024 * 1024 * 1024 / 4)
        m.v0 = nn.Parameter(torch.full((2,), 1, dtype=torch.float))
        m.v1 = nn.Parameter(torch.full((5, gig), 2, dtype=torch.float))
        m.v2 = nn.Parameter(torch.full((2,), 3, dtype=torch.float))
        m.v3 = nn.Parameter(torch.full((5, gig), 4, dtype=torch.float))
        m2 = self.getExportImportCopy(m)
        self.assertEqual(tuple(m.parameters()), tuple(m2.parameters()))

    def test_inferred_as_tensor(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(RuntimeError, "Inferred the value for argument 'dim' to be of type 'Tensor' because it was not annotated with an explicit type"):

            @torch.jit.script
            def dot(points, query, dim):
                if False:
                    i = 10
                    return i + 15
                return (points * query).sum(dim)

    def test_constants_pkl(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def fn(x):
            if False:
                return 10
            return x
        buf = io.BytesIO()
        torch.jit.save(fn, buf)
        buf.seek(0)
        files = zipfile.ZipFile(buf).filelist
        self.assertTrue(any(('archive/constants.pkl' == f.filename for f in files)))

    def test_script_fn_pkl(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(pickle.PickleError, 'ScriptFunction cannot be pickled'):

            @torch.jit.script
            def fn(x: torch.Tensor) -> torch.Tensor:
                if False:
                    print('Hello World!')
                return x
            pkl_fn = pickle.dumps(fn, protocol=0)

    def test_restore_device(self):
        if False:
            while True:
                i = 10

        class M(torch.jit.ScriptModule):

            def __init__(self, cpu_device_str):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.p0 = nn.Parameter(torch.tensor([0.3], dtype=torch.float, device=cpu_device_str))
                self.b0 = torch.tensor([0.9], dtype=torch.float, device=cpu_device_str)
        m = M('cpu')
        m2 = self.getExportImportCopy(m)
        self.assertEqual(tuple(m.parameters()), tuple(m2.parameters()))
        self.assertEqual(tuple(m.buffers()), tuple(m2.buffers()))
        self.assertFalse(m2.p0.is_cuda)
        self.assertFalse(m2.b0.is_cuda)

    @unittest.skipIf(not RUN_CUDA, 'restore device requires CUDA')
    def test_restore_device_cuda(self):
        if False:
            print('Hello World!')

        class MyModule(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.register_buffer('b0', torch.randn(1, 3))
                self.p0 = nn.Parameter(torch.randn(2, 3))

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x + self.b0 + self.p0
        m = MyModule()
        m.cuda(torch.cuda.device_count() - 1)
        cuda_device_str = 'cuda:' + str(torch.cuda.device_count() - 1)
        self.assertTrue(m.p0.is_cuda)
        self.assertTrue(m.b0.is_cuda)
        m2 = self.getExportImportCopy(m)
        self.assertEqual(tuple(m.parameters()), tuple(m2.parameters()))
        self.assertEqual(tuple(m.buffers()), tuple(m2.buffers()))
        self.assertEqual(str(m2.p0.device), cuda_device_str)
        self.assertEqual(str(m2.b0.device), cuda_device_str)
        cpu_device_str = 'cpu'
        m3 = self.getExportImportCopy(m, map_location=cpu_device_str)
        self.assertEqual(str(m3.p0.device), cpu_device_str)
        self.assertEqual(str(m3.b0.device), cpu_device_str)
        m4 = self.getExportImportCopy(m3, map_location=torch.device('cuda:0'))
        self.assertEqual(str(m4.p0.device), 'cuda:0')
        self.assertEqual(str(m4.b0.device), 'cuda:0')
        input = torch.rand(2, 3).cuda(torch.cuda.device_count() - 1)
        origin_result = m(input)
        self.assertEqual(origin_result, m2(input))
        self.assertEqual(origin_result, m3(input.cpu()))
        self.assertEqual(origin_result, m4(input.cuda(0)))

    def test_trace_retains_train(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x
        m = M()
        m.eval()
        tm = torch.jit.trace(m, torch.rand(3))
        self.assertEqual(tm.training, m.training)

    @unittest.skipIf(not RUN_CUDA, 'restore device requires CUDA')
    def test_restore_shared_storage_on_cuda(self):
        if False:
            for i in range(10):
                print('nop')

        class Foo(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                whole_tensor = torch.randn(4, 5, dtype=torch.float, device='cpu')
                self.p0 = nn.Parameter(whole_tensor.narrow(0, 0, 1))
                self.register_buffer('b0', whole_tensor.narrow(0, 3, 1))
        m = Foo()
        m2 = self.getExportImportCopy(m, map_location=torch.device('cuda:0'))
        self.assertEqual(tuple(m.parameters()), tuple(m2.parameters()))
        self.assertEqual(tuple(m.buffers()), tuple(m2.buffers()))
        self.assertTrue(m2.p0.is_cuda)
        self.assertTrue(m2.b0.is_cuda)
        self.assertTrue(m2.p0.is_shared())
        self.assertTrue(m2.b0.is_shared())
        self.assertEqual(m2.b0.storage().data_ptr(), m2.p0.storage().data_ptr())

    def test_add_relu_fusion(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def __init__(self, relu_op):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.relu_op = relu_op

            def forward(self, a, b, c):
                if False:
                    for i in range(10):
                        print('nop')
                tmp = torch.add(a, b)
                x = self.relu_op(tmp)
                d = torch.add(a, c)
                return x + d
        a = torch.rand((7, 11))
        a = a * -10
        a = a + 5
        b = torch.rand((7, 11))
        c = torch.rand((7, 11))
        m = torch.jit.script(M(torch.relu))
        orig_res = m(a, b, c)
        torch._C._jit_pass_fuse_add_relu(m.graph)
        buffer = io.BytesIO()
        torch.jit.save(m, buffer)
        buffer.seek(0)
        m = torch.jit.load(buffer)
        new_res = m(a, b, c)
        FileCheck().check_not('aten::relu(').check('aten::_add_relu(').run(m.graph)
        torch.testing.assert_close(orig_res, new_res)
        a = torch.rand((7, 11))
        a = a * -10
        a = a + 5
        b = torch.rand((7, 11))
        c = torch.rand((7, 11))
        m = torch.jit.script(M(torch.relu_))
        orig_res = m(a, b, c)
        torch._C._jit_pass_fuse_add_relu(m.graph)
        buffer = io.BytesIO()
        torch.jit.save(m, buffer)
        buffer.seek(0)
        m = torch.jit.load(buffer)
        new_res = m(a, b, c)
        FileCheck().check_not('aten::relu_(').check('aten::_add_relu(').run(m.graph)
        torch.testing.assert_close(orig_res, new_res)

        class Madd_(torch.nn.Module):

            def __init__(self, relu_op):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.relu_op = relu_op

            def forward(self, a, b):
                if False:
                    return 10
                x = a.add_(b)
                x = self.relu_op(x)
                return x
        a = torch.rand((7, 11))
        a = a * -10
        a = a + 5
        b = torch.rand((7, 11))
        a_copy = a.clone()
        m = torch.jit.script(Madd_(torch.relu_))
        orig_res = m(a, b)
        torch._C._jit_pass_fuse_add_relu(m.graph)
        buffer = io.BytesIO()
        torch.jit.save(m, buffer)
        buffer.seek(0)
        m = torch.jit.load(buffer)
        new_res = m(a_copy, b)
        FileCheck().check_not('aten::add_(').check_not('aten::relu_(').check('aten::_add_relu_(').run(m.graph)
        torch.testing.assert_close(orig_res, new_res)
        torch.testing.assert_close(orig_res, a_copy)

        class Madd_out(torch.nn.Module):

            def __init__(self, relu_op):
                if False:
                    return 10
                super().__init__()
                self.relu_op = relu_op

            def forward(self, a, b):
                if False:
                    return 10
                x = torch.add(a, b, out=a)
                x = self.relu_op(x)
                return x
        a = torch.rand((7, 11))
        a = a * -10
        a = a + 5
        b = torch.rand((7, 11))
        a = torch.rand((7, 11))
        a = a * -10
        a = a + 5
        b = torch.rand((7, 11))
        a_copy = a.clone()
        m = torch.jit.script(Madd_out(torch.relu_))
        orig_res = m(a, b)
        torch._C._jit_pass_fuse_add_relu(m.graph)
        buffer = io.BytesIO()
        torch.jit.save(m, buffer)
        buffer.seek(0)
        m = torch.jit.load(buffer)
        new_res = m(a_copy, b)
        FileCheck().check_not('aten::add(').check_not('aten::relu_(').check('aten::_add_relu(').run(m.graph)
        torch.testing.assert_close(orig_res, new_res)
        torch.testing.assert_close(orig_res, a_copy)

    def test_repeat_interleave_script(self):
        if False:
            while True:
                i = 10

        def fn(input: torch.Tensor, repeats: torch.Tensor) -> torch.Tensor:
            if False:
                return 10
            output = input.repeat_interleave(repeats)
            return output
        fn_scripted = torch.jit.script(fn)
        input = torch.tensor([5, 7], dtype=torch.int64)
        repeats = torch.tensor([3, 6], dtype=torch.int64)
        output = fn(input, repeats)
        output_scripted = fn_scripted(input, repeats)
        self.assertEqual(output_scripted, output)

    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "Simple executor doesn't have shape information")
    def test_peephole_optimize_shape_ops(self):
        if False:
            print('Hello World!')

        def test_input(func, input, result):
            if False:
                while True:
                    i = 10
            self.assertEqual(func(input, profile_and_replay=True), result)
            gre = func.graph_for(input)
            FileCheck().check_not('prim::If').run(gre)

        def test_dim():
            if False:
                i = 10
                return i + 15

            @torch.jit.script
            def func(x):
                if False:
                    for i in range(10):
                        print('nop')
                if x.dim() == 1:
                    return 1
                else:
                    return 2
            test_input(func, torch.tensor([0.5]), 1)
            test_input(func, torch.tensor([[0.5]]), 2)
        test_dim()

        def test_size_index():
            if False:
                return 10

            @torch.jit.script
            def func(x):
                if False:
                    while True:
                        i = 10
                if x.size(0) == 1:
                    return 1
                else:
                    return 2
            test_input(func, torch.rand([1, 2]), 1)
            test_input(func, torch.rand([1, 3]), 1)

            @torch.jit.script
            def neg_index(x):
                if False:
                    for i in range(10):
                        print('nop')
                if x.size(-2) == 1:
                    return 1
                else:
                    return 2
            test_input(neg_index, torch.rand([1, 2]), 1)
            test_input(neg_index, torch.rand([1, 3]), 1)
        if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
            test_size_index()

        def test_dtype():
            if False:
                i = 10
                return i + 15

            @torch.jit.script
            def func(x):
                if False:
                    i = 10
                    return i + 15
                if x.dtype == torch.float32:
                    return 1
                else:
                    return 2
            test_input(func, torch.tensor(0.5, dtype=torch.float32), 1)
            test_input(func, torch.tensor(0.5, dtype=torch.int64), 2)
        test_dtype()

        def test_is_floating_poiint():
            if False:
                while True:
                    i = 10

            @torch.jit.script
            def func(x):
                if False:
                    i = 10
                    return i + 15
                if x.is_floating_point():
                    return 1
                else:
                    return 2
            test_input(func, torch.tensor(0.5, dtype=torch.float32), 1)
            test_input(func, torch.tensor(0.5, dtype=torch.int64), 2)
        test_is_floating_poiint()

        def test_device():
            if False:
                i = 10
                return i + 15

            @torch.jit.script
            def func_1(x):
                if False:
                    while True:
                        i = 10
                if x.device == torch.device('cuda:0'):
                    a = 0
                else:
                    a = 1
                return a

            @torch.jit.script
            def func_2(x):
                if False:
                    for i in range(10):
                        print('nop')
                if x.is_cuda:
                    a = 0
                else:
                    a = 1
                return a
            test_input(func_1, torch.tensor(0.5), 1)
            test_input(func_2, torch.tensor(0.5), 1)
            if RUN_CUDA:
                test_input(func_1, torch.tensor(0.5, device='cuda:0'), 0)
                test_input(func_2, torch.tensor(0.5, device='cuda:0'), 0)
        test_device()

    def test_attrs(self):
        if False:
            print('Hello World!')

        def foo(x):
            if False:
                print('Hello World!')
            return (x.device, x.shape, x.is_cuda, x.is_mkldnn, x.is_quantized, x.requires_grad, x.T, x.mT, x.H, x.mH)
        scripted = torch.jit.script(foo)
        x = torch.rand(3, 4)
        self.assertEqual(scripted(x), foo(x))

    def test_layout(self):
        if False:
            return 10

        @torch.jit.script
        def check(x, y):
            if False:
                return 10
            return x.layout == y.layout
        x = torch.rand(3, 4)
        y = torch.rand(3, 4)
        self.assertTrue(check(x, y))

    def test_matrix_transpose(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def check(x):
            if False:
                i = 10
                return i + 15
            return torch.equal(x.mT, x.transpose(-2, -1))
        x = torch.rand(3, 4)
        self.assertTrue(check(x))

    def test_transpose(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def check(x):
            if False:
                while True:
                    i = 10
            return torch.equal(x.T, x.t())
        x = torch.rand(3, 4)
        self.assertTrue(check(x))

    def test_matrix_conj_transpose(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def check(x):
            if False:
                return 10
            return torch.equal(x.mH, x.transpose(-2, -1).conj())
        x = torch.rand(3, 4)
        self.assertTrue(check(x))
        x = make_tensor((3, 4), device='cpu', dtype=torch.complex64)
        self.assertTrue(check(x))

    def test_conj_transpose(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def check(x):
            if False:
                i = 10
                return i + 15
            return torch.equal(x.H, x.t().conj())
        x = torch.rand(3, 4)
        self.assertTrue(check(x))
        x = make_tensor((3, 4), device='cpu', dtype=torch.complex64)
        self.assertTrue(check(x))

    def test_T_mT_H_mH(self):
        if False:
            while True:
                i = 10

        def T(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.mT

        def mT(x):
            if False:
                while True:
                    i = 10
            return x.mT

        def H(x):
            if False:
                i = 10
                return i + 15
            return x.H

        def mH(x):
            if False:
                while True:
                    i = 10
            return x.mH
        x = torch.rand(3, 4)
        y = make_tensor((3, 4), device='cpu', dtype=torch.complex64)
        self.checkScript(T, (x,))
        self.checkScript(mT, (x,))
        self.checkScript(H, (x,))
        self.checkScript(mH, (x,))
        self.checkScript(T, (y,))
        self.checkScript(mT, (y,))
        self.checkScript(H, (y,))
        self.checkScript(mH, (y,))

    def test_nn_conv(self):
        if False:
            for i in range(10):
                print('nop')

        class Mod(nn.Module):

            def __init__(self, conv):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = conv

            def forward(self, input):
                if False:
                    return 10
                return self.conv(input)
        inputs = [(Mod(nn.Conv1d(16, 33, 3, stride=2)), torch.randn(20, 16, 5)), (Mod(nn.Conv2d(16, 33, 3, stride=2)), torch.randn(20, 16, 5, 10)), (Mod(nn.Conv3d(16, 33, 3, stride=2)), torch.randn(20, 16, 3, 5, 4)), (Mod(nn.ConvTranspose1d(16, 33, 3, stride=2)), torch.randn(20, 16, 5)), (Mod(nn.ConvTranspose2d(16, 33, 3, stride=2)), torch.randn(20, 16, 5, 10)), (Mod(nn.ConvTranspose3d(16, 33, 3, stride=2)), torch.randn(20, 16, 3, 5, 4))]
        for (m, inp) in inputs:
            self.checkModule(m, (inp,))

    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, 'Not implemented for Simple or Legacy')
    def test_debug_flush_compilation_cache(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            return x + 2

        class Mod(nn.Module):

            def forward(self, t):
                if False:
                    print('Hello World!')
                return t + 2
        m = torch.jit.script(Mod())
        x = torch.rand(1, 10)
        with enable_profiling_mode_for_profiling_tests():
            jitted = self.checkScript(foo, (x,))
            states = jitted.get_debug_state()
            jitted._debug_flush_compilation_cache()
            with self.assertRaisesRegex(RuntimeError, 'INTERNAL ASSERT FAILED'):
                states = jitted.get_debug_state()
            NUM_RUNS = 1
            with num_profiled_runs(NUM_RUNS):
                m(x)
                m(x)
                fwd = m._c._get_method('forward')
                states = m.get_debug_state()
                fwd._debug_flush_compilation_cache()
                with self.assertRaisesRegex(RuntimeError, 'INTERNAL ASSERT FAILED'):
                    states = m.get_debug_state()

    def test_numel(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def get_numel_script(x):
            if False:
                while True:
                    i = 10
            return x.numel()
        x = torch.rand(3, 4)
        numel = get_numel_script(x)
        self.assertEqual(numel, x.numel())

    def test_element_size(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def get_element_size_script(x):
            if False:
                print('Hello World!')
            return x.element_size()
        x = torch.rand(3, 4)
        element_size = get_element_size_script(x)
        self.assertEqual(element_size, x.element_size())

    def test_Sequential(self):
        if False:
            print('Hello World!')

        class Seq(nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.seq = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 30))

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                for l in self.seq:
                    x = l(x)
                return x
        m = torch.jit.script(Seq())
        assert m.graph

    def test_ModuleList(self):
        if False:
            while True:
                i = 10

        class Mod(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.model = nn.ModuleList([nn.Linear(10, 10) for _ in range(10)])
                self.model += (nn.Linear(10, 20),)
                self.model.append(nn.Linear(20, 30))
                self.model.extend([nn.Linear(30, 40), nn.Linear(40, 50)])

            def forward(self, v):
                if False:
                    return 10
                for m in self.model:
                    v = m(v)
                return v
        m = torch.jit.script(Mod())
        assert m.graph

    def test_disabled(self):
        if False:
            while True:
                i = 10
        torch.jit._state.disable()
        try:

            def f(x, y):
                if False:
                    i = 10
                    return i + 15
                return x + y
            self.assertIs(torch.jit.trace(f, (torch.randn(2, 2), torch.randn(2, 2))), f)
            self.assertIs(torch.jit.script(f), f)

            class MyModule(torch.jit.ScriptModule):

                @torch.jit.script_method
                def method(self, x):
                    if False:
                        return 10
                    return x
            self.assertTrue(inspect.ismethod(MyModule.method) or inspect.isfunction(MyModule.method))
        finally:
            torch.jit._state.enable()

    def test_train_eval(self):
        if False:
            return 10

        class Sub(nn.Module):

            def forward(self, input):
                if False:
                    print('Hello World!')
                if self.training:
                    return input
                else:
                    return -input

        class MyModule(torch.jit.ScriptModule):

            def __init__(self, module):
                if False:
                    print('Hello World!')
                super().__init__()
                self.module = module

            @torch.jit.script_method
            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return self.module(input) + 1
        m = MyModule(Sub())
        input = torch.rand(3, 4)
        self.assertEqual(input + 1, m(input))
        m.eval()
        self.assertEqual(-input + 1, m(input))
        input = torch.randn(6, 10)
        batchnorm = nn.BatchNorm1d(10)
        dropout = nn.Dropout(p=0.2)
        m_batchnorm = MyModule(batchnorm)
        self.assertEqual(batchnorm(input) + 1, m_batchnorm(input))
        batchnorm.eval()
        m_batchnorm.eval()
        self.assertEqual(batchnorm(input) + 1, m_batchnorm(input))
        m_dropout = MyModule(dropout)
        dropout.eval()
        m_dropout.eval()
        self.assertEqual(dropout(input) + 1, m_dropout(input))

    def test_nn_lp_pool2d(self):
        if False:
            while True:
                i = 10

        class Mod(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.l = torch.nn.LPPool2d(2, 3)
                self.n = torch.nn.LPPool2d(2, (7, 1))

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return (self.l(x), self.n(x), torch.nn.functional.lp_pool2d(x, float(2), 3), torch.nn.functional.lp_pool2d(x, 2, 3), torch.nn.functional.lp_pool2d(x, float(2), (7, 1)))
        self.checkModule(Mod(), (torch.rand(1, 3, 7, 7),))

    def test_nn_lp_pool1d(self):
        if False:
            for i in range(10):
                print('nop')

        class Mod(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.l = torch.nn.LPPool1d(2, 3)
                self.n = torch.nn.LPPool1d(2, 7)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return (self.l(x), self.n(x), torch.nn.functional.lp_pool1d(x, float(2), 3), torch.nn.functional.lp_pool1d(x, 2, 3), torch.nn.functional.lp_pool1d(x, float(2), 7))
        self.checkModule(Mod(), (torch.rand(1, 3, 7),))

    def test_nn_padding_functional(self):
        if False:
            for i in range(10):
                print('nop')

        class Mod(nn.Module):

            def __init__(self, *pad):
                if False:
                    print('Hello World!')
                super().__init__()
                self.pad = pad

            def forward(self, x):
                if False:
                    print('Hello World!')
                return F.pad(x, self.pad, mode='constant', value=3.5)
        inputs = [(Mod(1, 2), torch.randn(1, 3, 4)), (Mod(1, 2, 3, 4), torch.randn(1, 3, 4)), (Mod(1, 2, 3, 4, 5, 6), torch.randn(1, 3, 4))]
        for (m, inp) in inputs:
            self.checkModule(m, (inp,))

    def test_nn_padding(self):
        if False:
            for i in range(10):
                print('nop')

        class Mod(nn.Module):

            def __init__(self, padding):
                if False:
                    print('Hello World!')
                super().__init__()
                self.padding = padding

            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return self.padding(input)
        inputs = [(Mod(nn.ConstantPad1d(2, 3.5)), torch.randn(1, 2, 4)), (Mod(nn.ConstantPad2d(2, 3.5)), torch.randn(1, 2, 2)), (Mod(nn.ConstantPad3d(3, 3.5)), torch.randn(16, 3, 10, 20, 30)), (Mod(nn.ReflectionPad1d(2)), torch.arange(8, dtype=torch.float).reshape(1, 2, 4)), (Mod(nn.ReflectionPad2d(2)), torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)), (Mod(nn.ReflectionPad3d(3)), torch.randn(16, 3, 8, 32, 48)), (Mod(nn.ReplicationPad1d(2)), torch.arange(8, dtype=torch.float).reshape(1, 2, 4)), (Mod(nn.ReplicationPad2d(2)), torch.arange(9, dtype=torch.float).reshape(1, 1, 3, 3)), (Mod(nn.ReplicationPad3d(3)), torch.randn(16, 3, 8, 32, 48)), (Mod(nn.ZeroPad2d(2)), torch.randn(1, 1, 3, 3))]
        for (m, inp) in inputs:
            self.checkModule(m, (inp,))

    def test_script_autograd_grad(self):
        if False:
            i = 10
            return i + 15

        def test_simple_grad(x, y):
            if False:
                return 10
            z = x + 2 * y + x * y
            return torch.autograd.grad((z.sum(),), (x, y))

        def test_simple_grad_with_grad_outputs(x, y):
            if False:
                return 10
            z = x + 2 * y + x * y
            grad_outputs = torch.jit.annotate(List[Optional[torch.Tensor]], [torch.ones((2, 2))])
            return torch.autograd.grad((z,), (x, y), grad_outputs)

        def test_one_output_not_requires_grad(x, y):
            if False:
                while True:
                    i = 10
            z = 2 * y + y
            return torch.autograd.grad((z.sum(),), (x, y), allow_unused=True)

        def test_retain_graph(x, y):
            if False:
                print('Hello World!')
            z = x + 2 * y + x * y
            torch.autograd.grad((z.sum(),), (x, y), retain_graph=True)
            torch.autograd.grad((z.sum(),), (x, y))
        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2, requires_grad=True)
        self.checkScript(test_simple_grad, (x, y), inputs_requires_grad=True)
        self.checkScript(test_simple_grad_with_grad_outputs, (x, y), inputs_requires_grad=True)
        self.checkScript(test_one_output_not_requires_grad, (x, y), inputs_requires_grad=True)
        self.checkScript(test_retain_graph, (x, y), inputs_requires_grad=True)

    def test_script_backward(self):
        if False:
            while True:
                i = 10

        def checkBackwardScript(fn, inputs):
            if False:
                return 10
            scripted_fn = torch.jit.script(fn)
            FileCheck().check('torch.autograd.backward').run(scripted_fn.code)
            recording_inputs = do_input_map(lambda t: t.detach().requires_grad_(), inputs)
            fn(*inputs)
            scripted_fn(*recording_inputs)
            for (inp1, inp2) in zip(inputs, recording_inputs):
                self.assertEqual(inp1.grad, inp2.grad)

        def test_tensor_backward(input):
            if False:
                for i in range(10):
                    print('nop')
            output = torch.relu(input)
            output = output.softmax(0)
            sum_out = output.sum()
            sum_out.backward()

        def test_torch_autograd_backward(input):
            if False:
                for i in range(10):
                    print('nop')
            output = torch.relu(input)
            output = output.softmax(0)
            torch.autograd.backward(output.sum())

        def test_torch_autograd_backward_with_grad_tensors(input):
            if False:
                return 10
            output = torch.relu(input)
            output = output.softmax(0)
            grad_outputs = torch.jit.annotate(List[Optional[torch.Tensor]], [torch.ones((2, 2))])
            torch.autograd.backward((output,), grad_outputs)
        inp = torch.randn(2, 2, requires_grad=True)
        checkBackwardScript(test_tensor_backward, (inp,))
        checkBackwardScript(test_torch_autograd_backward, (inp,))
        checkBackwardScript(test_torch_autograd_backward_with_grad_tensors, (inp,))

    def test_script_backward_twice(self):
        if False:
            for i in range(10):
                print('nop')

        def checkBackwardTwiceScript(fn, inputs, retain_graph_=False):
            if False:
                return 10

            class jit_profiling_executor_false:

                def __enter__(self):
                    if False:
                        return 10
                    torch._C._jit_set_profiling_executor(False)

                def __exit__(self, *args):
                    if False:
                        print('Hello World!')
                    torch._C._jit_set_profiling_executor(GRAPH_EXECUTOR != ProfilingMode.LEGACY)
            with jit_profiling_executor_false(), torch.jit.optimized_execution(True):
                scripted_fn = torch.jit.script(fn, inputs)
                FileCheck().check('prim::DifferentiableGraph').run(scripted_fn.graph_for(*inputs))
                result = scripted_fn(*inputs)
                result.sum().backward(retain_graph=retain_graph_)
                if not retain_graph_:
                    self.assertRaisesRegex(RuntimeError, 'Specify retain_graph=True', lambda : result.sum().backward())
                else:
                    result.sum().backward()

        def test_script_backward_twice_with_saved_values(input1, input2):
            if False:
                print('Hello World!')
            tmp1 = torch.mul(input1, input2)
            tmp2 = torch.abs(tmp1)
            if torch.equal(input1, input2):
                tmp2 = torch.acos(tmp2)
            else:
                tmp2 = torch.atan(tmp2)
            result = torch.add(tmp2, input2)
            return result
        inp1 = torch.randn(2, 2, requires_grad=True)
        inp2 = torch.randn(2, 2, requires_grad=True)
        checkBackwardTwiceScript(test_script_backward_twice_with_saved_values, (inp1, inp2), False)
        checkBackwardTwiceScript(test_script_backward_twice_with_saved_values, (inp1, inp2), True)

    def test_diff_subgraph_clones_constants(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def f(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return x + x + y + x + y + x + y + x + y + x

        def count_constants(graph):
            if False:
                print('Hello World!')
            return sum((node.kind() == 'prim::Constant' for node in graph.nodes()))
        graph = f.graph.copy()
        self.run_pass('cse', graph)
        self.run_pass('create_autodiff_subgraphs', graph)
        nodes = list(graph.nodes())
        self.assertEqual(count_constants(graph), 1)
        self.assertEqual(count_constants(nodes[1].g('Subgraph')), 1)

    @unittest.skip('Need to be adjusted to Graph Executor')
    def test_arg_configurations(self):
        if False:
            print('Hello World!')
        'Different arg configurations should trigger different traces'
        x = Variable(torch.FloatTensor(4, 4).uniform_())
        x_double = Variable(x.data.double())
        x_grad = Variable(x.data.clone(), requires_grad=True)
        y = Variable(torch.randn(4))
        configurations = [(x,), (x_double,), (x_grad,), (y,), ([x, x],), ([x, y],)]
        if torch.cuda.is_available():
            x_cuda = Variable(x.data.cuda())
            configurations += [(x_cuda,), ([x, x_cuda],), ([x_cuda, x],), ([[x_cuda, x]],)]
            if torch.cuda.device_count() > 1:
                x_cuda_1 = Variable(x.data.cuda(1))
                configurations += [(x_cuda_1,), ([x_cuda, x_cuda_1],)]

        @torch.jit.compile(nderivs=0)
        def fn(*args):
            if False:
                for i in range(10):
                    print('nop')
            (in_vars, _) = torch._C._jit_flatten(args)
            return in_vars[0] + 1
        for (i, config) in enumerate(configurations):
            self.assertFalse(fn.has_trace_for(*config))
            fn(*config)
            self.assertTrue(fn.has_trace_for(*config))
            for unk_config in configurations[i + 1:]:
                self.assertFalse(fn.has_trace_for(*unk_config))
        self.assertEqual(fn.hits, 0)

    def test_torch_sum(self):
        if False:
            return 10

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.sum(x)

        def fn1(x, dim: int):
            if False:
                for i in range(10):
                    print('nop')
            return torch.sum(x, dim)
        x = torch.randn(3, 4)
        self.checkScript(fn, (x,))
        self.checkScript(fn1, (x, 1))
        self.checkScript(fn1, (x, 0))

    def test_cse(self):
        if False:
            print('Hello World!')
        x = torch.tensor([0.4, 0.3], requires_grad=True)
        y = torch.tensor([0.7, 0.5], requires_grad=True)

        def fn(x, y):
            if False:
                while True:
                    i = 10
            w = (x + y) * (x + y) * (x + y)
            t = torch.tanh(w) + torch.tanh(w)
            z = (x + y) * (x + y) * (x + y) + t
            return z
        (g, _) = torch.jit._get_trace_graph(fn, (x, y))
        self.run_pass('cse', g)
        do_exactly = True
        FileCheck().check_count('add', 1).check_count('mul', 2, do_exactly).check_count('tanh', 1, do_exactly).check_count('add', 2, do_exactly).check_next('return').run(str(g))
        self.assertExportImport(g, (x, y))

    def test_cse_not_introduce_aliasing(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def tensor_alias_outputs(x):
            if False:
                while True:
                    i = 10
            return (x + x, x + x)
        self.run_pass('cse', tensor_alias_outputs.graph)
        FileCheck().check_count('aten::add', 2).run(tensor_alias_outputs.graph)

        @torch.jit.script
        def ints_alias_outputs(x):
            if False:
                print('Hello World!')
            return (x + x, x + x)
        self.run_pass('cse', ints_alias_outputs.graph)
        FileCheck().check_count('aten::add', 1, exactly=True).run(ints_alias_outputs.graph)

    def test_recursive_cse(self):
        if False:
            while True:
                i = 10
        input_str = '\ngraph(%x : Tensor,\n      %y : Tensor,\n      %20 : int):\n  %2 : int = prim::Constant[value=1]()\n  %3 : Tensor = aten::add(%x, %y, %2)\n  %4 : int = aten::add(%2, %20)\n  %5 : bool = aten::Bool(%4)\n  %z : int = prim::If(%5)\n    # CHECK: block\n    block0():\n      # CHECK-NOT: aten::add\n      %z.1 : int = aten::add(%2, %20)\n      -> (%z.1)\n    block1():\n      -> (%2)\n  return (%z)\n'
        graph = parse_ir(input_str)
        self.run_pass('cse', graph)
        FileCheck().run(input_str, graph)

    def test_pattern_based_rewrite(self):
        if False:
            while True:
                i = 10
        input_str = '\ngraph(%x, %y, %z):\n    # CHECK-NOT: aten::mul\n    # CHECK: my::fused_mulmul\n    %t = aten::mul(%x, %y)\n    %p = aten::mul(%t, %z)\n    # CHECK: my::fused_mulmul\n    %u = aten::mul(%p, %x)\n    %o = aten::mul(%u, %y)\n    return (%o)'
        graph = parse_ir(input_str)
        torch._C._jit_pass_custom_pattern_based_rewrite_graph('\ngraph(%a, %b, %c):\n  %q = aten::mul(%a, %b)\n  %r = aten::mul(%q, %c)\n  return (%r)', '\ngraph(%a, %b, %c):\n  %r = my::fused_mulmul(%a, %b, %c)\n  return (%r)', graph)
        FileCheck().run(input_str, graph)
        input_str = '\ngraph(%x, %y, %z):\n    # CHECK-NOT: aten::mul\n    # CHECK: my::fused_mulmul\n    %t = aten::mul(%x, %y)\n    %p = aten::mul(%t, %z)\n    # CHECK-NEXT: aten::mul\n    %u = aten::mul(%p, %x)\n    return (%u)'
        graph = parse_ir(input_str)
        torch._C._jit_pass_custom_pattern_based_rewrite_graph('\ngraph(%a, %b, %c):\n  %q = aten::mul(%a, %b)\n  %r = aten::mul(%q, %c)\n  return (%r)', '\ngraph(%a, %b, %c):\n  %r = my::fused_mulmul(%a, %b, %c)\n  return (%r)', graph)
        FileCheck().run(input_str, graph)
        input_str = '\ngraph(%x, %y, %z):\n    # CHECK-NOT: aten::mul\n    # CHECK-NOT: aten::add\n    %c = prim::Const[value=1]()\n    %t = aten::mul(%x, %y)\n    %p = aten::add(%t, %z, %c)\n    # CHECK: my::muladd\n    # CHECK-NEXT: return\n    return (%p)'
        graph = parse_ir(input_str)
        torch._C._jit_pass_custom_pattern_based_rewrite_graph('\ngraph(%a, %b, %c, %d):\n  %q = aten::mul(%a, %b)\n  %r = aten::add(%q, %c, %d)\n  return (%r)', '\ngraph(%a, %b, %c, %d):\n  %r = my::muladd(%a, %b, %c, %d)\n  return (%r)', graph)
        FileCheck().run(input_str, graph)
        input_str = '\ngraph(%x, %y, %z):\n    # CHECK-NOT: aten::mul\n    %c = prim::Const[value=1]()\n    # CHECK: aten::add\n    %t = aten::mul(%x, %y)\n    # CHECK-NEXT: aten::sub\n    %p = aten::add(%t, %z, %c)\n    # CHECK-NOT: aten::add\n    # CHECK-NEXT: return\n    return (%p)'
        graph = parse_ir(input_str)
        torch._C._jit_pass_custom_pattern_based_rewrite_graph('\ngraph(%a, %b, %c, %d):\n  %q = aten::mul(%a, %b)\n  %r = aten::add(%q, %c, %d)\n  return (%r)', '\ngraph(%a, %b, %c, %d):\n  %q = aten::add(%a, %b, %d)\n  %r = aten::sub(%q, %c, %d)\n  return (%r)', graph)
        FileCheck().run(input_str, graph)
        input_str = '\ngraph(%x, %y, %z):\n    %c = prim::Const[value=1]()\n    # CHECK-NOT: aten::mul\n    %t = aten::mul(%x, %y)\n    # CHECK: aten::add(%x, %z\n    %p = aten::add(%t, %z, %c)\n    # CHECK-NEXT: return\n    return (%p)'
        graph = parse_ir(input_str)
        torch._C._jit_pass_custom_pattern_based_rewrite_graph('\ngraph(%Pa, %Pb):\n  %Pq = aten::mul(%Pa, %Pb)\n  return (%Pq)', '\ngraph(%Ra, %Rb):\n  return (%Ra)', graph)
        FileCheck().run(input_str, graph)

    @_tmp_donotuse_dont_inline_everything
    def test_pattern_based_module_rewrite(self):
        if False:
            while True:
                i = 10

        class Test(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 20, 5, 1)
                self.bn = torch.nn.BatchNorm2d(num_features=20)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x = self.conv(x)
                x = self.bn(x)
                return x
        m = torch.jit.script(Test())
        torch._C._jit_pass_custom_pattern_based_rewrite_graph('\n        graph(%self, %x):\n                %conv = match::module[name="Conv2d"](%self)\n                %y = prim::CallMethod[name="forward"](%conv, %x)\n                %bn = match::module[name="BatchNorm2d"](%self)\n                %z = prim::CallMethod[name="forward"](%bn, %y)\n                return (%z)', '\n        graph(%self, %x):\n          %z = my::matched_conv_bn(%self, %x)\n          return (%z)', m._c._get_method('forward').graph)
        FileCheck().check('my::matched_conv_bn').run(m._c._get_method('forward').graph)

    def test_pattern_based_rewrite_with_source_range_preserved(self):
        if False:
            for i in range(10):
                print('nop')

        class TestModule1(torch.nn.Module):

            def forward(self, x, y, z, w):
                if False:
                    for i in range(10):
                        print('nop')
                x = x + y
                x = x * z
                return w - x
        input_pattern = '\n        graph(%x, %y, %z, %const):\n            %t = aten::add(%x, %y, %const)\n            %o = aten::mul(%t, %z)\n            return (%o)'
        replacement_pattern = '\n        graph(%x, %y, %z, %const):\n            %o = my::add_mul(%x, %y, %z, %const)\n            return (%o)'
        scripted_model = torch.jit.script(TestModule1())
        graph = scripted_model.graph
        value_mappings = [('o', 't')]
        for node in graph.nodes():
            if node.kind() == 'aten::add':
                source_range_1 = node.sourceRange()
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(input_pattern, replacement_pattern, scripted_model.graph, value_name_pairs=value_mappings)
        graph = scripted_model.graph
        for node in graph.nodes():
            if node.kind() == 'my::add_mul':
                source_range_2 = node.sourceRange()
        self.assertTrue(source_range_1 == source_range_2)

        class TestModule2(torch.nn.Module):

            def forward(self, x, y, z, w):
                if False:
                    for i in range(10):
                        print('nop')
                x = x + y
                x = x + z
                x = x * z
                x = x * w
                return x - 2
        input_pattern = '\n        graph(%x, %y, %const):\n            %o = aten::add(%x, %y, %const)\n            return (%o)'
        replacement_pattern = '\n        graph(%x, %y, %const):\n            %o = my::add(%x, %y, %const)\n            return (%o)'
        scripted_model = copy.deepcopy(torch.jit.script(TestModule2()))
        graph_copy = scripted_model.graph.copy()
        value_mappings = [('o', 'o')]
        source_range_add_1 = None
        for node in graph_copy.nodes():
            if source_range_add_1 is None and node.kind() == 'aten::add':
                source_range_add_1 = node.sourceRange()
            if source_range_add_1 is not None and node.kind() == 'aten::add':
                source_range_add_2 = node.sourceRange()
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(input_pattern, replacement_pattern, graph_copy, value_name_pairs=value_mappings)
        source_range_my_add_1 = None
        for node in graph_copy.nodes():
            if source_range_my_add_1 is None and node.kind() == 'my::add':
                source_range_my_add_1 = node.sourceRange()
            if source_range_my_add_1 is not None and node.kind() == 'my::add':
                source_range_my_add_2 = node.sourceRange()
        self.assertTrue(source_range_add_1 == source_range_my_add_1)
        self.assertTrue(source_range_add_2 == source_range_my_add_2)
        input_pattern = '\n        graph(%x, %y, %z, %const):\n            %t = aten::add(%x, %y, %const)\n            %o = aten::add(%t, %z, %const)\n            return (%o)'
        replacement_pattern = '\n        graph(%x, %y, %z, %const):\n            %o = my::double_add(%x, %y, %z, %const)\n            return (%o)'
        scripted_model = torch.jit.script(TestModule2())
        graph_copy = scripted_model.graph.copy()
        value_mappings = [('o', 't')]
        source_range_1 = None
        source_range_2 = None
        for node in graph_copy.nodes():
            if node.kind() == 'aten::add':
                source_range_1 = node.sourceRange()
                break
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(input_pattern, replacement_pattern, graph_copy, value_name_pairs=value_mappings)
        for node in graph_copy.nodes():
            if node.kind() == 'my::double_add':
                source_range_2 = node.sourceRange()
        self.assertTrue(source_range_1 == source_range_2)
        input_pattern = '\n        graph(%x, %y):\n            %t = aten::mul(%x, %y)\n            return (%t)'
        replacement_pattern = '\n        graph(%x, %y):\n            %t = my::add(%x, %y)\n            %o = my::add(%t, %y)\n            return (%o)'
        scripted_model = torch.jit.script(TestModule2())
        graph_copy = scripted_model.graph.copy()
        value_mappings = [('t', 't'), ('o', 't')]
        source_range_mul_1 = None
        for node in graph_copy.nodes():
            if source_range_mul_1 is None and node.kind() == 'aten::mul':
                source_range_mul_1 = node.sourceRange()
            if source_range_mul_1 is not None and node.kind() == 'aten::mul':
                source_range_mul_2 = node.sourceRange()
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(input_pattern, replacement_pattern, graph_copy, value_name_pairs=value_mappings)
        source_range_add_1 = None
        for node in graph_copy.nodes():
            if source_range_add_1 is None and node.kind() == 'my::add':
                source_range_add_1 = node.sourceRange()
            if source_range_add_1 is not None and node.kind() == 'my::add':
                source_range_add_2 = node.sourceRange()
        self.assertTrue(source_range_mul_1 == source_range_add_1)
        self.assertTrue(source_range_mul_2 == source_range_add_2)
        input_pattern = '\n        graph(%x, %y, %z):\n            %t = aten::mul(%x, %y)\n            %o = aten::mul(%t, %z)\n            return (%o)'
        replacement_pattern = '\n        graph(%x, %y, %z):\n            %o = my::double_mul(%x, %y, %z)\n            return (%o)'
        scripted_model = torch.jit.script(TestModule2())
        graph_copy = scripted_model.graph.copy()
        for node in graph_copy.nodes():
            if node.kind() == 'aten::mul':
                source_range_1 = node.sourceRange()
        torch._C._jit_pass_custom_pattern_based_rewrite_graph(input_pattern, replacement_pattern, graph_copy)
        for node in graph_copy.nodes():
            if node.kind() == 'my::double_mul':
                source_range_2 = node.sourceRange()
        self.assertFalse(source_range_1 == source_range_2)

    def test_expand_quantlint(self):
        if False:
            while True:
                i = 10
        pass

    def test_expand_fold_quant_inputs(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_shape_analysis_broadcast(self):
        if False:
            i = 10
            return i + 15

        def broadcast(a, b):
            if False:
                while True:
                    i = 10
            return a + b
        x = torch.randn(3, 1, 5, requires_grad=True)
        y = torch.randn(4, 1, 8, 5, requires_grad=True)
        graph = torch.jit.script(broadcast).graph
        torch._C._jit_pass_complete_shape_analysis(graph, (x, y), False)
        FileCheck().check('Float(4, 3, 8, 5, strides=[120, 40, 5, 1], device=cpu)').run(str(graph))

    def test_shape_analysis_unsqueeze_in_loop(self):
        if False:
            while True:
                i = 10
        input_str = 'graph(%x.1 : Tensor):\n          %4 : bool = prim::Constant[value=1]()\n          %1 : int = prim::Constant[value=2]()\n          %7 : int = prim::Constant[value=0]()\n          # CHECK: FloatTensor(requires_grad=0, device=cpu) = prim::Loop\n          %x : Tensor = prim::Loop(%1, %4, %x.1)\n            # CHECK: : FloatTensor(requires_grad=0, device=cpu)):\n            block0(%i : int, %x.6 : Tensor):\n              # CHECK: FloatTensor(requires_grad=0, device=cpu) = aten::unsqueeze\n              %x.3 : Tensor = aten::unsqueeze(%x.6, %7)\n              -> (%4, %x.3)\n          return (%x)'
        graph = parse_ir(input_str)
        torch._C._jit_pass_complete_shape_analysis(graph, (torch.zeros(2, 2, dtype=torch.float32),), False)
        FileCheck().run(input_str, graph)

    def test_script_tensor_type(self):
        if False:
            while True:
                i = 10

        def foo(x, t: torch.dtype):
            if False:
                i = 10
                return i + 15
            return x.type(t)
        scr = torch.jit.script(foo)
        x = torch.rand(3, 4)
        for t in [torch.int8, torch.float64, torch.float32, torch.bfloat16, torch.complex64, torch.complex128, torch.bool]:
            self.assertEqual(scr(x, t), foo(x, t))

    def test_shape_analysis_masked_select(self):
        if False:
            print('Hello World!')
        input_str = 'graph(%0 : Float(),\n          %1 : Bool()):\n          # CHECK: Float(*, requires_grad=0, device=cpu) = aten::masked_select\n          %2 : Tensor = aten::masked_select(%0, %1) # test/test_jit.py:15261:0\n          return (%2)'
        graph = parse_ir(input_str)
        x = torch.ones(1, dtype=torch.float32)[0]
        mask = x.ge(0.5)
        torch._C._jit_pass_complete_shape_analysis(graph, (x, mask), False)
        FileCheck().run(input_str, graph)

    @unittest.skip('verify needs to be updated to work with GraphExecutors')
    def test_verify(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.tensor([0.4], requires_grad=True)
        y = torch.tensor([0.7], requires_grad=True)

        @torch.jit.compile
        def f(x, y):
            if False:
                i = 10
                return i + 15
            z = torch.sigmoid(x * (x + y))
            w = torch.abs(x * x * x + y) + Variable(torch.ones(1))
            return (z, w)
        torch.jit.verify(f, (x, y), loss_fn=lambda z, w: z * w, devices=[])

    @unittest.skip('Need to instrument GraphExecutors a bit more')
    def test_flags(self):
        if False:
            for i in range(10):
                print('nop')
        (x, y) = torch.randn(2, 2)
        y = Variable(torch.randn(2, 2))

        @torch.jit.compile
        def fn(x, y):
            if False:
                return 10
            return (x * x + y * y + x * y).sum()
        grads = {}
        for (rx, ry) in product((True, False), repeat=2):
            x.requires_grad = rx
            y.requires_grad = ry
            self.assertFalse(fn.has_trace_for(x, y))
            out = fn(x, y)
            self.assertFalse(fn.has_trace_for(x, y))
            for (v, name, compute) in [(x, 'x', rx), (y, 'y', ry)]:
                if not compute:
                    continue
                (grad_v,) = torch.autograd.grad(out, v, retain_graph=True)
                expected_grad = grads.setdefault(name, grad_v)
                self.assertEqual(grad_v, expected_grad)
            self.assertEqual(fn.has_trace_for(x, y), rx or ry)

    def test_python_ir(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.tensor([0.4], requires_grad=True)
        y = torch.tensor([0.7], requires_grad=True)

        def doit(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return torch.sigmoid(torch.tanh(x * (x + y)))
        (g, _) = torch.jit._get_trace_graph(doit, (x, y))
        self.run_pass('dce', g)
        self.run_pass('canonicalize', g)
        g2 = torch._C.Graph()
        g_to_g2 = {}
        for node in g.inputs():
            g_to_g2[node] = g2.addInput()
        for node in g.nodes():
            n_ = g2.createClone(node, lambda x: g_to_g2[x])
            g2.appendNode(n_)
            for (o, no) in zip(node.outputs(), n_.outputs()):
                g_to_g2[o] = no
        for node in g.outputs():
            g2.registerOutput(g_to_g2[node])
        t_node = g2.create('prim::TensorTest').t_('a', torch.ones([2, 2]))
        self.assertEqual(t_node.attributeNames(), ['a'])
        g2.appendNode(t_node)
        self.assertTrue(torch.equal(torch.ones(2, 2), t_node.t('a')))
        for node in g.nodes():
            self.assertTrue(g2.findNode(node.kind()) is not None)

    def test_permute_inputs_binding(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def foo(i, j, k):
            if False:
                i = 10
                return i + 15
            pass
        g = foo.graph
        idxs = []
        for (i, inp) in enumerate(g.inputs()):
            inp.setDebugName(f'inp{i}')
            idxs.append(i)
        permuted_idxs = list(np.random.permutation(idxs))
        g.permuteInputs(permuted_idxs)
        for (i, inp) in enumerate(g.inputs()):
            self.assertEqual(f'inp{permuted_idxs[i]}', inp.debugName())

    @unittest.skipIf(IS_MACOS, 'Failing on MacOS only')
    def test_python_ir_utils(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def foo(inp):
            if False:
                for i in range(10):
                    print('nop')
            x = inp + 1
            y = x / 2
            z = y * y
            return z
        add_node = foo.graph.findNode('aten::add')
        div_node = foo.graph.findNode('aten::div')
        with foo.graph.insert_point_guard(add_node):
            with foo.graph.insert_point_guard(div_node):
                foo.graph.insertConstant('goodbye')
            foo.graph.insertConstant('hello')
        with foo.graph.insert_point_guard(foo.graph.findNode('aten::mul')):
            foo.graph.insertConstant('hello')
        FileCheck().check('hello').check('goodbye').check('hello').run(foo.graph)
        self.assertTrue(add_node.matches(add_node.schema()))
        self.assertFalse(add_node.matches(div_node.schema()))

    def test_python_ir_utils_graph(self):
        if False:
            return 10

        @torch.jit.script
        def unrolled_mul(x: torch.Tensor, y: int):
            if False:
                print('Hello World!')
            out = x
            for _ in range(y - 1):
                out = out + x
            return out

        @torch.jit.script
        def foo(x):
            if False:
                return 10
            return x * 4
        g = foo.graph
        muls = g.findAllNodes('aten::mul')
        scalar_muls = filter(lambda x: x.matches('aten::mul(Tensor self, Scalar other) -> Tensor'), muls)
        mul_constant_int = filter(lambda x: isinstance(list(x.inputs())[1].toIValue(), int), scalar_muls)
        for mul in mul_constant_int:
            with g.insert_point_guard(mul):
                outputs = g.insertGraph(unrolled_mul.graph, list(mul.inputs()))
                assert len(outputs) == len(list(mul.outputs()))
                for (new_out, old_out) in zip(outputs, g.outputs()):
                    old_out.replaceAllUsesWith(new_out)
                mul.destroy()
        FileCheck().check_not('aten::mul').check('aten::add').run(foo.graph)
        self.assertEqual(foo(torch.ones([2, 2])), torch.ones([2, 2]) * 4)

    @unittest.skipIf(IS_SANDCASTLE, 'gtest runs these in sandcastle')
    @unittest.skipIf(RUN_CUDA, 'covered by test_cpp_cuda')
    @unittest.skipIf(not torch._C._jit_has_cpp_tests(), 'Tests were not built, use BUILD_TEST=1')
    def test_cpp(self):
        if False:
            i = 10
            return i + 15
        from cpp.jit import tests_setup
        tests_setup.setup()
        torch._C._jit_run_cpp_tests()
        tests_setup.shutdown()

    def test_batchnorm(self):
        if False:
            return 10
        x = torch.ones(2, 2, 2, 2)
        (g, outputs, inputs) = torch.jit._get_trace_graph(nn.BatchNorm2d(2), x, _force_outplace=True, return_inputs=True)
        m = self.createFunctionFromGraph(g)
        self.assertEqual(outputs, m(*inputs))

    def test_dropout(self):
        if False:
            return 10
        x = torch.ones(2, 2)
        with torch.random.fork_rng(devices=[]):
            (g, outputs, inputs) = torch.jit._get_trace_graph(nn.Dropout(0.6), x, return_inputs=True)
        with torch.random.fork_rng(devices=[]):
            m = self.createFunctionFromGraph(g)
            self.assertEqual(outputs, m(*inputs))

    @unittest.skipIf(not RUN_CUDA, 'test requires CUDA')
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, "skip if profiling isn't enabled")
    def test_native_dropout_corner_case(self):
        if False:
            while True:
                i = 10
        with disable_autodiff_subgraph_inlining():

            def t(x, p: float, t: bool):
                if False:
                    return 10
                o = torch.dropout(x, p, t)
                return o
            jit_t = torch.jit.script(t)
            x = torch.randn(5).requires_grad_()
            FileCheck().check('prim::DifferentiableGraph').run(jit_t.graph_for(x, 1.0, True, profile_and_replay=True))
            for train in [True, False]:
                for p in [0.0, 1.0]:
                    for device in ['cuda', 'cpu']:
                        x = torch.randn(5).to(device=device).requires_grad_()
                        x_ref = x.detach().requires_grad_()
                        o = jit_t(x, p, train)
                        o_ref = t(x_ref, p, train)
                        o.sum().backward()
                        o_ref.sum().backward()
                        assert o.equal(o_ref)
                        assert x.grad.equal(x_ref.grad)

    @slowTest
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, 'Testing differentiable graph')
    def test_dropout_module_requires_grad(self):
        if False:
            print('Hello World!')
        with enable_profiling_mode_for_profiling_tests():

            class MyModule(torch.nn.Module):

                def __init__(self, M):
                    if False:
                        while True:
                            i = 10
                    super().__init__()
                    self.dropout = torch.nn.Dropout(0.5)
                    self.linear = torch.nn.Linear(M, M)

                def forward(self, input):
                    if False:
                        return 10
                    input = self.dropout(input)
                    output = self.linear(input)
                    return output

            def profile(func, X):
                if False:
                    i = 10
                    return i + 15
                with torch.autograd.profiler.profile() as prof:
                    func(X)
                return [e.name for e in prof.function_events]
            M = 1000
            scripted = torch.jit.script(MyModule(M))
            for training in (True, False):
                if training:
                    scripted.train()
                else:
                    scripted.eval()
                for requires_grad in (True, False):
                    X = torch.randn(M, M, requires_grad=requires_grad)
                    if requires_grad:
                        FileCheck().check('aten::native_dropout').run(scripted.graph_for(X, profile_and_replay=True))
                    self.assertEqual(training, 'aten::bernoulli_' in profile(scripted, X))

    @unittest.skipIf(GRAPH_EXECUTOR == ProfilingMode.SIMPLE, 'Testing differentiable graph')
    @skipIfTorchDynamo('Torchdynamo cannot correctly handle profiler.profile calls')
    def test_dropout_func_requires_grad(self):
        if False:
            print('Hello World!')

        def dropout_training(input):
            if False:
                for i in range(10):
                    print('nop')
            return F.dropout(input, 0.5, training=True)

        def dropout_eval(input):
            if False:
                print('Hello World!')
            return F.dropout(input, 0.5, training=False)

        def profile(func, X):
            if False:
                i = 10
                return i + 15
            with torch.autograd.profiler.profile() as prof:
                func(X)
            return [e.name for e in prof.function_events]
        M = 1000
        scripted_training = torch.jit.script(dropout_training)
        scripted_eval = torch.jit.script(dropout_eval)
        with disable_autodiff_subgraph_inlining():
            for requires_grad in (True, False):
                X = torch.randn(M, M, requires_grad=requires_grad)
                if requires_grad:
                    FileCheck().check('aten::native_dropout').run(scripted_training.graph_for(X, profile_and_replay=True))
                self.assertIn('aten::bernoulli_', profile(scripted_training, X))
                self.assertNotIn('aten::bernoulli_', profile(scripted_eval, X))

    @unittest.skipIf(not RUN_CUDA, 'test_dropout_cuda require CUDA')
    def test_dropout_cuda(self):
        if False:
            while True:
                i = 10

        def _zero_rate(t):
            if False:
                while True:
                    i = 10
            return torch.true_divide((t == 0).sum(), t.numel())
        x = torch.ones(1000, 1000).cuda().requires_grad_()
        with enable_profiling_mode_for_profiling_tests():

            @torch.jit.script
            def func(x):
                if False:
                    return 10
                return torch.nn.functional.dropout(x)
            with freeze_rng_state():
                out_ref = torch.nn.functional.dropout(x)
                grad_ref = torch.autograd.grad(out_ref.sum(), x)
            with freeze_rng_state():
                out = func(x)
                grad = torch.autograd.grad(out.sum(), x)
            self.assertEqual(_zero_rate(out), _zero_rate(out_ref), rtol=0.001, atol=0.0001)
            self.assertEqual(_zero_rate(grad[0]), _zero_rate(grad_ref[0]), rtol=0.001, atol=0.0001)

    def test_torch_ops_overloaded(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(RuntimeError, 'failed to many any schema'):
            torch.ops.aten.add('a', 1)
        self.assertEqual('ab', torch.ops.aten.add('a', 'b'))
        (a, b) = (torch.rand(3, 4), torch.rand(3, 4))
        self.assertEqual(a + b, torch.ops.aten.add(a, b))
        self.assertEqual(a + 1, torch.ops.aten.add(a, 1))

    def test_torch_ops_kwonly(self):
        if False:
            i = 10
            return i + 15
        (a, b) = (torch.rand(3, 4), torch.rand(3, 4))
        with self.assertRaisesRegex(RuntimeError, 'positional argument'):
            torch.ops.aten.add(a, b, 2)
        self.assertEqual(a.prod(1), torch.ops.aten.prod(a, 1))

    def test_torch_complex(self):
        if False:
            print('Hello World!')

        def fn(real, img):
            if False:
                for i in range(10):
                    print('nop')
            return torch.complex(real, img)

        def fn_out(real, img, out):
            if False:
                return 10
            return torch.complex(real, img, out=out)
        self.checkScript(fn, (torch.rand(3, 4), torch.rand(3, 4)))
        self.checkScript(fn, (torch.ones(5, 1, 4), torch.ones(5, 1, 4)))
        self.checkScript(fn, (torch.zeros(1, 6), torch.ones(6, 1)))
        self.checkScript(fn, (torch.zeros(1, 6), torch.zeros(6, 1)))
        self.checkScript(fn, (torch.empty(3, 4), torch.empty(3, 4)))
        real = torch.tensor([1, 2], dtype=torch.float32)
        img = torch.tensor([3, 4], dtype=torch.float32)
        out = torch.empty([3, 4], dtype=torch.complex64)
        self.checkScript(fn_out, (real, img, out))
        real = torch.tensor([5, 2], dtype=torch.float64)
        img = torch.tensor([3, 4], dtype=torch.float64)
        out = torch.empty([5, 2], dtype=torch.complex128)
        self.checkScript(fn_out, (real, img, out))
        real = torch.ones([1, 2])
        img = torch.ones([1, 2])
        out = torch.empty([1, 2], dtype=torch.complex64)
        self.checkScript(fn_out, (real, img, out))
        real = torch.ones([3, 8, 7])
        img = torch.ones([3, 8, 7])
        out = torch.empty([3, 8, 7], dtype=torch.complex64)
        self.checkScript(fn_out, (real, img, out))
        real = torch.empty([3, 2, 6])
        img = torch.empty([3, 2, 6])
        out = torch.empty([3, 2, 6], dtype=torch.complex64)
        self.checkScript(fn_out, (real, img, out))
        real = torch.zeros([1, 3])
        img = torch.empty([3, 1])
        out = torch.empty([3, 3], dtype=torch.complex64)
        self.checkScript(fn_out, (real, img, out))
        real = torch.ones([2, 5])
        img = torch.empty([2, 1])
        out = torch.empty([2, 5], dtype=torch.complex64)
        self.checkScript(fn_out, (real, img, out))
        real = torch.ones([2, 5])
        img = torch.zeros([2, 1])
        out = torch.empty([2, 5], dtype=torch.complex64)
        self.checkScript(fn_out, (real, img, out))

    def test_einsum(self):
        if False:
            print('Hello World!')

        def check(fn, jitted, *args):
            if False:
                while True:
                    i = 10
            self.assertGraphContains(jitted.graph, kind='aten::einsum')
            self.assertEqual(fn(*args), jitted(*args))

        def equation_format(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return torch.einsum('i,j->ij', (x, y))

        def equation_format_varargs(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return torch.einsum('i,j->ij', x, y)

        def sublist_format(x, y):
            if False:
                print('Hello World!')
            return torch.einsum(x, [0], y, [1], [0, 1])
        x = make_tensor((5,), dtype=torch.float32, device='cpu')
        y = make_tensor((10,), dtype=torch.float32, device='cpu')
        for fn in [equation_format, equation_format_varargs, sublist_format]:
            check(fn, torch.jit.script(fn), x, y)
            check(fn, torch.jit.trace(fn, (x, y)), x, y)

    @skipIfTorchDynamo('TorchDynamo fails with unknown reason')
    def test_python_ivalue(self):
        if False:
            for i in range(10):
                print('nop')
        py_array = np.arange(15)
        ret_py_obj = torch._C._ivalue_debug_python_object(py_array)
        self.assertEqual(py_array, ret_py_obj)
        ret_py_obj = torch._C._ivalue_debug_python_object(F.relu)
        self.assertEqual(F.relu, ret_py_obj)

        def test_func_scope_helper(inp):
            if False:
                for i in range(10):
                    print('nop')
            inp_refcount = sys.getrefcount(inp)
            ivalue_holder = torch._C._ivalue_debug_python_object(inp)
            self.assertEqual(inp_refcount + 1, sys.getrefcount(ivalue_holder))
            return ivalue_holder + 1
        test_input = 2200
        before_count = sys.getrefcount(test_input)
        test_func_scope_helper(test_input)
        after_count = sys.getrefcount(test_input)
        self.assertEqual(before_count, after_count)

    def test_decompose_addmm(self):
        if False:
            return 10

        def does_decompose():
            if False:
                for i in range(10):
                    print('nop')

            @torch.jit.script
            def addmm(mat, mat1, mat2):
                if False:
                    while True:
                        i = 10
                a = mat.addmm(mat1, mat2)
                b = mat.addmm(mat1, mat2, alpha=1.0, beta=1.0)
                return a + b
            mat = torch.randn(2, 2)
            mat1 = torch.randn(2, 4)
            mat2 = torch.randn(4, 2)
            out_ref = addmm(mat, mat1, mat2)
            self.run_pass('decompose_ops', addmm.graph)
            out_test = addmm(mat, mat1, mat2)
            self.assertEqual(out_ref, out_test)
            FileCheck().check_not('addmm').run(str(addmm.graph))

        def doesnt_decompose():
            if False:
                for i in range(10):
                    print('nop')

            @torch.jit.script
            def addmm(mat, mat1, mat2, alpha, beta):
                if False:
                    while True:
                        i = 10
                a = mat.addmm(mat1, mat2, alpha=4.2, beta=2.0)
                b = mat.addmm(mat1, mat2, alpha=int(alpha), beta=int(beta))
                return a + b
            orig = str(addmm.graph)
            self.run_pass('decompose_ops', addmm.graph)
            self.assertTrue(orig == str(addmm.graph))
        does_decompose()
        doesnt_decompose()

    @suppress_warnings
    def test_sparse_tensors(self):
        if False:
            while True:
                i = 10

        @torch.jit.ignore
        def get_sparse():
            if False:
                while True:
                    i = 10
            return torch.sparse_coo_tensor((2, 3), dtype=torch.float32)

        @torch.jit.script
        def test_is_sparse(input):
            if False:
                return 10
            return input.is_sparse
        script_out_is_sparse = test_is_sparse(get_sparse())
        script_out_is_dense = test_is_sparse(torch.randn(2, 3))
        self.assertEqual(script_out_is_sparse, True)
        self.assertEqual(script_out_is_dense, False)

        def test_basic_sparse(input):
            if False:
                while True:
                    i = 10
            output = get_sparse()
            return (output, input)
        self.checkScript(test_basic_sparse, (get_sparse(),))
        self.checkScript(test_basic_sparse, (torch.tensor([1]),))

        def test_sparse_sum(input):
            if False:
                for i in range(10):
                    print('nop')
            return torch.sparse.sum(input)
        self.checkScript(test_sparse_sum, (get_sparse(),))

        def test_sparse_mm(input1, input2):
            if False:
                for i in range(10):
                    print('nop')
            return torch.sparse.mm(input1, input2)
        self.checkScript(test_sparse_mm, (get_sparse(), torch.randn(3, 4)))

        def test_sparse_addmm(input, input1, input2):
            if False:
                print('Hello World!')
            return torch.sparse.addmm(input, input1, input2)

        def test_sparse_addmm_alpha_beta(input, input1, input2):
            if False:
                return 10
            return torch.sparse.addmm(input, input1, input2, alpha=1.3, beta=1.5)
        self.checkScript(test_sparse_addmm, (torch.randn(2, 4), get_sparse(), torch.randn(3, 4)))
        self.checkScript(test_sparse_addmm_alpha_beta, (torch.randn(2, 4), get_sparse(), torch.randn(3, 4)))

    @suppress_warnings
    def test_sparse_csr_tensors(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.ignore
        def get_sparse_csr():
            if False:
                print('Hello World!')
            return torch.randn(3, 3).to_sparse_csr()

        @torch.jit.script
        def test_is_sparse_csr(input):
            if False:
                print('Hello World!')
            return input.is_sparse_csr
        script_out_is_sparse_csr = test_is_sparse_csr(get_sparse_csr())
        script_out_is_dense_csr = test_is_sparse_csr(torch.randn(3, 3))
        self.assertEqual(script_out_is_sparse_csr, True)
        self.assertEqual(script_out_is_dense_csr, False)

    @unittest.skipIf(not RUN_CUDA, 'requires CUDA')
    def test_device_not_equal(self):
        if False:
            print('Hello World!')

        def compare_device(x: torch.device):
            if False:
                print('Hello World!')
            return x != torch.device('cuda:0')

        def compare_two_device(x: torch.device, y: torch.device):
            if False:
                print('Hello World!')
            return x != y
        self.checkScript(compare_device, (torch.device('cuda:0'),))
        self.checkScript(compare_two_device, (torch.device('cuda:0'), torch.device('cuda:1')))

    def test_constant_prop_simple(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def constant_prop(input_int):
            if False:
                i = 10
                return i + 15
            a = 2 * 3
            b = a + 2
            return b - input_int
        out_ref = constant_prop(2)
        self.run_pass('constant_propagation', constant_prop.graph)
        out_test = constant_prop(2)
        self.assertEqual(out_ref, out_test)
        graph_str = str(constant_prop.graph)
        self.assertTrue('aten::add' not in graph_str and 'aten::mul' not in graph_str)
        const = constant_prop.graph.findNode('prim::Constant').output().toIValue()
        self.assertEqual(const, 8)

    def test_constant_prop_nested(self):
        if False:
            return 10

        @torch.jit.script
        def constant_prop(a):
            if False:
                print('Hello World!')
            b = 2 + 1
            if bool(a < 2):
                c = b + 2
            else:
                c = b - 2
            return c
        out_ref = constant_prop(torch.tensor(2))
        self.run_pass('constant_propagation', constant_prop.graph)
        out_test = constant_prop(torch.tensor(2))
        self.assertEqual(out_ref, out_test)
        if_node = constant_prop.graph.findNode('prim::If')
        for block in if_node.blocks():
            for node in block.nodes():
                self.assertTrue(node.kind() == 'prim::Constant')

    def test_constant_prop_print(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def constant_prop(input_tensor):
            if False:
                return 10
            a = 2 * 3
            print(a)
            b = a + 2
            return b + input_tensor
        self.run_pass('constant_propagation', constant_prop.graph)
        graph = constant_prop.graph
        print_node = graph.findNode('prim::Print')
        self.assertTrue(print_node.input().toIValue() == 6)

    def test_constant_prop_rand(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def constant_prop():
            if False:
                return 10
            a = torch.randn([3])
            b = a + 2
            return b
        self.run_pass('constant_propagation', constant_prop.graph)
        self.assertTrue('aten::randn' in str(constant_prop.graph))

    def test_constant_prop_none(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def typed_none():
            if False:
                i = 10
                return i + 15
            return None

        @torch.jit.script
        def constant_prop():
            if False:
                print('Hello World!')
            a = typed_none()
            b = typed_none()
            if a is None and b is None:
                a = 2
            else:
                a = 1
            return a
        self.run_pass('constant_propagation', constant_prop.graph)
        FileCheck().check('prim::Constant').run(constant_prop.graph)

    def test_constant_prop_if_inline(self):
        if False:
            return 10

        @torch.jit.script
        def constant_prop():
            if False:
                while True:
                    i = 10
            cond = True
            a = 1
            if cond:
                a = 1 * 2
            else:
                a = 1 // 0
            return a
        self.run_pass('constant_propagation', constant_prop.graph)

    def test_constant_prop_exception(self):
        if False:
            print('Hello World!')

        def bad_index(x):
            if False:
                while True:
                    i = 10
            y = 0
            if x:
                a = [1, 2, 3]
                y = a[4]
            return y
        self.checkScript(bad_index, (False,))

    def test_constant_prop_aliasing_type(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            return (len([1]), len(torch.tensor([2])))
        FileCheck().check_dag('aten::tensor').check_dag('aten::len').run(foo.graph)

        @torch.jit.script
        def fn():
            if False:
                i = 10
                return i + 15
            if 1 == 1:
                return 1
            else:
                return 2
        FileCheck().check_not('prim::If').run(fn.graph)

    def test_unchecked_cast(self):
        if False:
            print('Hello World!')

        def test(cond):
            if False:
                for i in range(10):
                    print('nop')
            a = torch.tensor([10])
            if cond:
                b = None
            else:
                b = a
            if b is not None:
                b[0] = 5
            return a.int()
        self.checkScript(test, (True,))
        self.checkScript(test, (False,))

    def test_constant_prop_if_constant(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def constant_prop(a, b):
            if False:
                i = 10
                return i + 15
            c0 = 1
            c1 = 1
            c2 = 1
            if bool(a):
                if bool(b):
                    if 1 == 1:
                        c0 = c0 + 1
                        if 1 == 2:
                            c1 = c1 + 1
                            c2 = c2 + 1
            else:
                c1 = c1 + 1
            if 1 == 1:
                c0 = c0 + 1
                c2 = c2 + 4
            return a + c0 + c1 + c2
        graph = constant_prop.graph
        self.run_pass('constant_propagation', graph)
        ifs = graph.findAllNodes('prim::If', recurse=False)
        snd_if_inlined = len(ifs) == 1
        self.assertTrue(snd_if_inlined)
        first_if = ifs[0]
        self.assertTrue(first_if.outputsSize() == 2)
        second_if = first_if.findNode('prim::If', recurse=False)
        self.assertTrue(second_if.outputsSize() == 1)
        self.assertTrue(second_if.findNode('prim::If') is None)

    def test_constant_prop_loop_constant(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def constant_prop(cond, iter):
            if False:
                while True:
                    i = 10
            b = 0
            while True:
                print('stays')
            for _ in range(2):
                print('stays')
            for _ in range(iter):
                print('stays')
            while cond:
                print('stays')
            while False:
                print('removed')
            for _i in range(0):
                print('removed')
            for _i in range(-4):
                print('removed')
            return b
        self.run_pass('constant_propagation', constant_prop.graph)
        graph = canonical(constant_prop.graph)
        self.assertTrue(graph.count('removed') == 0)
        self.assertTrue(graph.count('stays') == 1)
        self.assertTrue(graph.count('prim::Print') == 4)

    def test_constant_prop_remove_output(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def constant_prop(iter):
            if False:
                i = 10
                return i + 15
            a = 1
            b = 1
            c = 1
            for i in range(iter):
                if 1 == 2:
                    a = 10
                if i == 5:
                    b = 2
                    c = 3
            print(a, b, c)
        graph = constant_prop.graph
        self.run_pass('constant_propagation', graph)
        self.assertTrue(graph.findNode('prim::Loop').outputsSize() == 2)

    def test_constant_insertion(self):
        if False:
            while True:
                i = 10
        funcs_template = dedent('\n        def func():\n            return {constant_constructor}\n        ')

        def check_constant(constant_constructor):
            if False:
                while True:
                    i = 10
            scope = {}
            funcs_str = funcs_template.format(constant_constructor=constant_constructor)
            execWrapper(funcs_str, globals(), scope)
            cu = torch.jit.CompilationUnit(funcs_str)
            f_script = cu.func
            self.run_pass('constant_propagation', f_script.graph)
            FileCheck().check_count('prim::Constant', 1, exactly=True).run(f_script.graph)
            self.assertEqual(scope['func'](), f_script())
            imported = self.getExportImportCopy(f_script)
            self.assertEqual(imported(), f_script())
        constants = ['None', '-.5', '0', '1', 'True', 'False', "''", "'a'", "'b'", 'torch.tensor(1)', '[True, False]', '[0., .5]', '[torch.tensor(4), torch.tensor(2)]', '[0, 1]', "['0', '1']", '[True, None]', '[.5, None, .2]']
        for type in ['Tensor', 'str', 'int', 'float', 'bool']:
            constants.append('torch.jit.annotate(List[ ' + type + '], [])')
        for constant in constants:
            check_constant(constant)
        for key_type in ['str', 'int', 'float']:
            for value_type in ['Tensor', 'bool', 'str', 'int', 'float']:
                check_constant('torch.jit.annotate(Dict[ ' + key_type + ', ' + value_type + '], {})')
                check_constant('torch.jit.annotate(Dict[ ' + key_type + ', Optional[' + value_type + ']], {})')
        for i in range(len(constants)):
            for j in range(i + 1, len(constants)):
                tup_constant = constants[i] + ', ' + constants[j]
                check_constant(tup_constant)
        dict_constants = []
        for i in range(len(constants)):
            if not isinstance(eval(constants[i]), (str, int, float)):
                continue
            for j in range(len(constants)):
                dict_constant = '{ ' + constants[i] + ': ' + constants[j] + '}'
                check_constant(dict_constant)
                dict_constants.append(dict_constant)
        constants = constants + dict_constants
        funcs_template = dedent('\n        def func():\n            print({constant_constructor})\n        ')
        single_elem_tuples = ('(' + x + ',)' for x in constants)
        input_arg = ', '.join(single_elem_tuples)
        scope = {}
        funcs_str = funcs_template.format(constant_constructor=input_arg)
        execWrapper(funcs_str, globals(), scope)
        cu = torch.jit.CompilationUnit(funcs_str)
        f_script = cu.func
        self.run_pass('constant_propagation', f_script.graph)
        self.assertEqual(len(constants) + 1, str(f_script.graph).count('prim::Constant'))
        self.run_pass('cse', f_script.graph)
        self.assertEqual(len(constants) + 1, str(f_script.graph).count('prim::Constant'))
        funcs_template = dedent('\n        def func():\n            a = {constant_constructor}\n            print(a)\n            b = {constant_constructor}\n            print(b)\n        ')
        xprod = itertools.product(constants, constants)
        for tup in ('(' + x + ',)' for x in constants):
            funcs_str = funcs_template.format(constant_constructor=tup)
            scope = {}
            execWrapper(funcs_str, globals(), scope)
            cu = torch.jit.CompilationUnit(funcs_str)
            f_script = cu.func
            self.run_pass('constant_propagation_immutable_types', f_script.graph)
            num_constants = str(f_script.graph).count('prim::Constant')
            self.run_pass('cse', f_script.graph)
            FileCheck().check_count('prim::Constant', num_constants, exactly=True).run(f_script.graph)

    @unittest.skipIf(not RUN_CUDA, 'requires CUDA')
    def test_cuda_export_restore(self):
        if False:
            while True:
                i = 10

        class Sub(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.weight = nn.Parameter(torch.randn(3, 4))

            @torch.jit.script_method
            def forward(self, thing):
                if False:
                    return 10
                return self.weight + thing

        class M(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.mod = Sub()

            @torch.jit.script_method
            def forward(self, v):
                if False:
                    while True:
                        i = 10
                return self.mod(v)
        m = M()
        m.cuda()
        m2 = self.getExportImportCopy(m)
        m2.cuda()
        input = torch.rand(3, 4).cuda()
        self.assertEqual(m(input), m2(input))

    @slowTest
    def test_export_batchnorm(self):
        if False:
            for i in range(10):
                print('nop')
        for mode in ['eval', 'train']:
            for clazz in [torch.nn.BatchNorm1d(100), torch.nn.BatchNorm1d(100, affine=False), torch.nn.BatchNorm2d(100), torch.nn.BatchNorm2d(100, affine=False)]:
                getattr(clazz, mode)()
                input = torch.randn(20, 100) if isinstance(clazz, torch.nn.BatchNorm1d) else torch.randn(20, 100, 35, 45)
                traced = torch.jit.trace(clazz, (input,))
                imported = self.getExportImportCopy(traced)
                x = torch.randn(20, 100) if isinstance(clazz, torch.nn.BatchNorm1d) else torch.randn(20, 100, 35, 45)
                self.assertEqual(traced(x), imported(x))

    def test_export_rnn(self):
        if False:
            i = 10
            return i + 15
        for clazz in [nn.RNN(10, 20, 2), nn.GRU(10, 20, 2)]:

            class RNNTest(torch.nn.Module):

                def __init__(self):
                    if False:
                        while True:
                            i = 10
                    super().__init__()
                    self.rnn = clazz

                def forward(self, x, lengths, h0):
                    if False:
                        print('Hello World!')
                    packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)
                    (out, h) = self.rnn(packed, h0)
                    (padded_outs, _) = torch.nn.utils.rnn.pad_packed_sequence(out)
                    return padded_outs
            test = RNNTest()
            traced = torch.jit.trace(test, (torch.randn(5, 3, 10), torch.LongTensor([3, 2, 1]), torch.randn(2, 3, 20)))
            imported = self.getExportImportCopy(traced)
            (x, lengths, h0) = (torch.randn(7, 4, 10), torch.LongTensor([7, 3, 2, 1]), torch.randn(2, 4, 20))
            self.assertEqual(traced(x, lengths, h0), imported(x, lengths, h0))

    def test_export_lstm(self):
        if False:
            for i in range(10):
                print('nop')

        class LSTMTest(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.rnn = nn.LSTM(10, 20, 2)

            def forward(self, x, lengths, hiddens):
                if False:
                    for i in range(10):
                        print('nop')
                (h0, c0) = hiddens
                packed = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)
                (out, (h, c)) = self.rnn(packed, (h0, c0))
                (padded_outs, _) = torch.nn.utils.rnn.pad_packed_sequence(out)
                return padded_outs
        test = LSTMTest()
        traced = torch.jit.trace(test, (torch.randn(5, 3, 10), torch.LongTensor([3, 2, 1]), (torch.randn(2, 3, 20), torch.randn(2, 3, 20))))
        imported = self.getExportImportCopy(traced)
        (x, lengths, h0, c0) = (torch.randn(7, 3, 10), torch.LongTensor([7, 5, 2]), torch.randn(2, 3, 20), torch.randn(2, 3, 20))
        self.assertEqual(traced(x, lengths, (h0, c0)), imported(x, lengths, (h0, c0)))

    def test_unique_state_dict(self):
        if False:
            return 10

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                shared_param = torch.nn.Parameter(torch.ones(1))
                self.register_parameter('w1', shared_param)
                self.register_parameter('w2', shared_param)

            def forward(self, input):
                if False:
                    print('Hello World!')
                return input + self.w1 + self.w2
        model = MyModule()
        unittest.TestCase.assertEqual(self, len(torch.jit._unique_state_dict(model, keep_vars=False)), 1)
        unittest.TestCase.assertEqual(self, len(torch.jit._unique_state_dict(model, keep_vars=True)), 1)

    def test_export_dropout(self):
        if False:
            return 10
        test = torch.nn.Dropout()
        test.eval()
        traced = torch.jit.trace(test, (torch.rand(3, 4),), check_trace=False)
        imported = self.getExportImportCopy(traced)
        x = torch.randn(3, 4)
        self.assertEqual(traced(x), imported(x))

    def test_pretty_printer(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def if_test(a, b):
            if False:
                for i in range(10):
                    print('nop')
            c = a
            if bool(a < b):
                c = b
            else:
                c = a
            return c

        @torch.jit.script
        def if_one(a, b):
            if False:
                return 10
            c = b
            if bool(a < b):
                c = a
            return c

        @torch.jit.script
        def while_test(a, i):
            if False:
                i = 10
                return i + 15
            while bool(i < 3):
                a *= a
                i += 1
            return a

        @torch.jit.script
        def while_if_test(a, b):
            if False:
                while True:
                    i = 10
            c = 0
            while bool(a < 10):
                a = a + 1
                b = b + 1
                if bool(a > b):
                    c = 2
                else:
                    c = 3
            return a + 1 + c

        @torch.jit.script
        def loop_use_test(y):
            if False:
                return 10
            x = y + 1
            z = x + 5
            while bool(y < 8):
                y += 1
                z = x
            return (x, z)

        @torch.jit.ignore
        def python_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return x + 10

        @torch.jit.script
        def python_op_name_test(y):
            if False:
                while True:
                    i = 10
            return python_fn(y)

        @torch.jit.script
        def empty_int_list_test(y):
            if False:
                return 10
            x = torch.jit.annotate(List[int], [])
            return x[0]

        @torch.jit.script
        def empty_float_list_test(y):
            if False:
                i = 10
                return i + 15
            return [1.0, 2.0, 3.0]

        @torch.jit.script
        def print_weird_test(y):
            if False:
                i = 10
                return i + 15
            print('hi\x0e')
        self.assertExpected(if_test.code, 'if_test')
        self.assertExpected(if_one.code, 'if_one')
        self.assertExpected(while_test.code, 'while_test')
        self.assertExpected(while_if_test.code, 'while_if_test')
        self.assertExpected(loop_use_test.code, 'loop_use_test')
        self.assertExpected(python_op_name_test.code, 'python_op_name_test')
        self.assertExpected(empty_int_list_test.code, 'empty_int_list_test')
        self.assertExpected(empty_float_list_test.code, 'empty_float_list_test')
        self.assertExpected(print_weird_test.code, 'print_weird_test')

    def test_cu_escaped_number(self):
        if False:
            i = 10
            return i + 15
        cu = torch.jit.CompilationUnit('\n            def foo(a):\n                print("hi\x0e")\n        ')
        self.assertExpected(cu.foo.code)

    def test_import_method(self):
        if False:
            i = 10
            return i + 15
        with torch._jit_internal._disable_emit_hooks():

            class Foo(torch.jit.ScriptModule):

                @torch.jit.script_method
                def forward(self, x, y):
                    if False:
                        while True:
                            i = 10
                    return 2 * x + y
            foo = Foo()
            buffer = io.BytesIO()
            torch.jit.save(foo, buffer)
            buffer.seek(0)
            foo_loaded = torch.jit.load(buffer)
            self.assertExpected(foo_loaded.forward.code)

    @unittest.skip('temporarily disable the test for fwd compatibility')
    def test_non_ascii_string(self):
        if False:
            return 10

        class Foo(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.a = 'Over ๕๗ 57'

            @torch.jit.script_method
            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                return self.a + 'hi¡'
        foo = Foo()
        buffer = io.BytesIO()
        torch.jit.save(foo, buffer)
        buffer.seek(0)
        foo_loaded = torch.jit.load(buffer)
        self.assertExpected(foo_loaded.forward.code)

    def test_function_default_values(self):
        if False:
            print('Hello World!')
        outer_var = torch.tensor(20)
        outer_var2 = torch.tensor(30)
        a = torch.tensor(0.5)
        b = torch.tensor(10)

        @torch.jit.script
        def simple_fn(x, a=a, b=b, c=outer_var + outer_var2):
            if False:
                return 10
            return x + a + b + c
        self.assertEqual(simple_fn(torch.ones(1)), torch.ones(1) + 0.5 + 10 + (20 + 30))
        self.assertEqual(simple_fn(torch.ones(1), torch.tensor(1), torch.tensor(3), torch.tensor(4)), torch.ones(1) + 1 + 3 + 4)
        outer_c = torch.tensor(9)
        outer_flag = torch.tensor(False)

        @torch.jit.script
        def bool_fn(x, a=outer_c, flag=outer_flag):
            if False:
                i = 10
                return i + 15
            if bool(flag):
                result = x
            else:
                result = x + a
            return result
        self.assertEqual(bool_fn(torch.ones(1)), torch.ones(1) + 9)
        self.assertEqual(bool_fn(torch.ones(1), torch.tensor(1), torch.tensor(True)), torch.ones(1))

        @torch.jit.script
        def none_fn(x=None):
            if False:
                i = 10
                return i + 15
            return x
        self.assertEqual(none_fn(), None)
        self.assertEqual(none_fn(1), 1)

        @torch.jit.script
        def hints(x, a=0.5, b=10):
            if False:
                for i in range(10):
                    print('nop')
            return x + a + b
        self.assertEqual(hints(torch.ones(1)), torch.ones(1) + 0.5 + 10)
        with self.assertRaisesRegex(RuntimeError, 'Expected a default value'):

            @torch.jit.script
            def hints_bad_types(x, a=10, b=0.5):
                if False:
                    while True:
                        i = 10
                return x + a + b
        with self.assertRaisesRegex(RuntimeError, 'Expected a default value'):

            @torch.jit.script
            def bad_no_optional(x=None):
                if False:
                    for i in range(10):
                        print('nop')
                return x

    def test_module_default_values(self):
        if False:
            i = 10
            return i + 15
        four = torch.tensor(4)

        class Test(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, input, other=four):
                if False:
                    while True:
                        i = 10
                return input + other
        t = Test()
        self.assertEqual(t(torch.ones(1)), torch.ones(1) + 4)

    def test_mutable_default_values(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(Exception, 'Mutable default parameters'):

            @torch.jit.script
            def foo(x=(1, [])):
                if False:
                    i = 10
                    return i + 15
                return x

        class Test(torch.nn.Module):

            def forward(self, input=[]):
                if False:
                    i = 10
                    return i + 15
                return input
        with self.assertRaisesRegex(Exception, 'Mutable default parameters'):
            torch.jit.script(Test())

    @skipIfTorchDynamo('TorchDynamo fails with unknown reason')
    def test_warnings(self):
        if False:
            return 10
        import warnings

        def fn(x):
            if False:
                return 10
            if bool(x < 2):
                warnings.warn('x is less than 2')
            return x

        class M(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                if bool(x < 2):
                    warnings.warn('x is less than 2')
                return x
        scripted_mod = torch.jit.script(M())
        scripted_fn = torch.jit.script(fn)
        with warnings.catch_warnings(record=True) as warns:
            fn(torch.ones(1))
        with warnings.catch_warnings(record=True) as script_warns:
            scripted_fn(torch.ones(1))
        with warnings.catch_warnings(record=True) as script_mod_warns:
            scripted_mod(torch.ones(1))
        self.assertEqual(str(warns[0]), str(script_warns[0]))
        self.assertEqual(len(script_mod_warns), 1)
        self.assertEqual(str(warns[0].message), str(script_mod_warns[0].message))

    def test_no_erroneous_warnings(self):
        if False:
            for i in range(10):
                print('nop')
        import warnings

        def fn(x):
            if False:
                i = 10
                return i + 15
            if bool(x > 0):
                warnings.warn('This should NOT be printed')
                x += 1
            return x
        with warnings.catch_warnings(record=True) as warns:
            fn_script = torch.jit.script(fn)
            fn_script(torch.tensor(0))
        warns = [str(w.message) for w in warns]
        self.assertEqual(len(warns), 0)

    @unittest.skipIf(True, 'TODO: re-enable with https://github.com/pytorch/pytorch/pull/29339')
    def test_torch_load_error(self):
        if False:
            while True:
                i = 10

        class J(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, input):
                if False:
                    print('Hello World!')
                return input + 100
        j = J()
        with TemporaryFileName() as fname:
            j.save(fname)
            with self.assertRaisesRegex(RuntimeError, 'is a zip'):
                torch.load(fname)

    def test_torch_load_zipfile_check(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def fn(x):
            if False:
                print('Hello World!')
            return x + 10
        with TemporaryFileName() as fname:
            fn.save(fname)
            with open(fname, 'rb') as f:
                self.assertTrue(torch.serialization._is_zipfile(f))

    def test_python_bindings(self):
        if False:
            i = 10
            return i + 15
        lstm_cell = torch.jit.script(LSTMCellS)

        def lstm(x, hx, cx, w_ih, w_hh, b_ih, b_hh):
            if False:
                return 10
            for i in range(x.size(0)):
                (hx, cx) = lstm_cell(x[i], hx, cx, w_ih, w_hh, b_ih, b_hh)
            return hx
        slstm = torch.jit.script(lstm)
        inputs = get_lstm_inputs('cpu', training=True, seq_length=10)
        slstm(*inputs).sum().backward()
        global fw_graph
        fw_graph = slstm.graph_for(*inputs)
        nodes = list(fw_graph.nodes())
        tested_blocks = False
        for node in nodes:
            for output in node.outputs():
                self.assertTrue(hasattr(output, 'type'))
                self.assertTrue(output.type() is not None)
            for input in node.inputs():
                self.assertTrue(hasattr(input, 'type'))
                self.assertTrue(input.type() is not None)
            for block in node.blocks():
                tested_blocks = True
                self.assertTrue(hasattr(block, 'inputs'))
                self.assertTrue(hasattr(block, 'outputs'))
                for output in block.outputs():
                    self.assertTrue(hasattr(output, 'type'))
                    self.assertTrue(output.type() is not None)
                for input in block.inputs():
                    self.assertTrue(hasattr(input, 'type'))
                    self.assertTrue(input.type() is not None)
                self.assertTrue(hasattr(block, 'returnNode'))
                self.assertTrue(type(block.returnNode()) == torch._C.Node)
                self.assertTrue(hasattr(block, 'paramNode'))
                self.assertTrue(type(block.paramNode()) == torch._C.Node)
        self.assertTrue(tested_blocks)

    def test_export_opnames(self):
        if False:
            i = 10
            return i + 15

        class Foo(torch.jit.ScriptModule):

            def one(self, x, y):
                if False:
                    return 10
                return x + y

            def two(self, x):
                if False:
                    return 10
                return 2 * x

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.one(self.two(x), x)

        class Bar(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.sub = Foo()

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.sub.forward(x)
        bar = Bar()
        ops = torch.jit.export_opnames(bar)
        expected = ['aten::add.Tensor', 'aten::mul.Scalar']
        self.assertTrue(set(expected).issubset(set(ops)))

    def test_pytorch_jit_env_off(self):
        if False:
            i = 10
            return i + 15
        import subprocess
        env = os.environ.copy()
        env['PYTORCH_JIT'] = '0'
        try:
            subprocess.check_output([sys.executable, '-c', 'import torch'], env=env)
        except subprocess.CalledProcessError as e:
            raise RuntimeError("Could not 'import torch' with PYTORCH_JIT=0") from e

    def test_print_op_module(self):
        if False:
            print('Hello World!')
        s = str(torch.ops)
        self.assertRegex(s, 'ops')

    def test_print_classes_module(self):
        if False:
            print('Hello World!')
        s = str(torch.classes)
        self.assertRegex(s, 'classes')

    def test_print_torch_ops_modules(self):
        if False:
            print('Hello World!')
        s = str(torch._ops.ops.quantized)
        self.assertRegex(s, 'torch.ops')
        s = str(torch._ops.ops.atan)
        self.assertRegex(s, 'torch.ops')

    def test_hide_source_ranges_context_manager(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def foo(x):
            if False:
                return 10
            return torch.add(x, x)
        graph = foo.graph
        source_range_regex = '# .*\\.py'
        self.assertRegex(graph.__repr__(), source_range_regex)
        with torch.jit._hide_source_ranges():
            self.assertNotRegex(graph.__repr__(), source_range_regex)
            self.assertRegex(graph.str(print_source_ranges=True), source_range_regex)
        self.assertRegex(graph.__repr__(), source_range_regex)

@skipIfTorchDynamo()
class TestFrontend(JitTestCase):

    def test_instancing_error(self):
        if False:
            return 10

        @torch.jit.ignore
        class MyScriptClass:

            def unscriptable(self):
                if False:
                    print('Hello World!')
                return 'a' + 200

        class TestModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return MyScriptClass()
        with self.assertRaises(torch.jit.frontend.FrontendError) as cm:
            torch.jit.script(TestModule())
        checker = FileCheck()
        checker.check('Cannot instantiate class')
        checker.check('def forward')
        checker.run(str(cm.exception))

    def test_dictionary_as_example_inputs_for_jit_trace(self):
        if False:
            for i in range(10):
                print('nop')

        class TestModule_v1(torch.nn.Module):

            def forward(self, key2=None, key3=None, key4=None, key5=None, key1=None, key6=None):
                if False:
                    print('Hello World!')
                return key1 + key2 + key3

        class TestModule_v2(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                return x + y

        def test_func(x, y):
            if False:
                i = 10
                return i + 15
            return x + y
        model_1 = TestModule_v1()
        model_2 = TestModule_v2()
        value1 = torch.ones(1)
        value2 = torch.ones(1)
        value3 = torch.ones(1)
        example_input_dict = {'key1': value1, 'key2': value2, 'key3': value3}
        example_input_dict_func = {'x': value1, 'y': value2}
        traced_model_1 = torch.jit.trace(model_1, example_kwarg_inputs=example_input_dict, strict=False)
        traced_model_1_m = torch.jit.trace_module(model_1, {'forward': example_input_dict}, example_inputs_is_kwarg=True, strict=False)
        traced_model_2 = torch.jit.trace(model_2, example_kwarg_inputs={'x': torch.rand([2]), 'y': torch.rand([2])})
        traced_func = torch.jit.trace(test_func, example_kwarg_inputs=example_input_dict_func, strict=False)
        res_1 = traced_model_1(**example_input_dict)
        res_1_m = traced_model_1_m(**example_input_dict)
        self.assertEqual(res_1, 3 * torch.ones(1))
        self.assertEqual(res_1_m, 3 * torch.ones(1))
        res_func = traced_func(**example_input_dict_func)
        self.assertEqual(res_func, 2 * torch.ones(1))
        with self.assertRaisesRegex(RuntimeError, "forward\\(\\) is missing value for argument 'x'."):
            res_2 = traced_model_2(**{'z': torch.rand([2]), 'y': torch.rand([2])})
        with self.assertRaisesRegex(RuntimeError, "forward\\(\\) is missing value for argument 'y'."):
            res_2 = traced_model_2(**{'x': torch.rand([2]), 'z': torch.rand([2])})

@skipIfTorchDynamo()
class TestScript(JitTestCase):

    def test_repeated_script_on_function(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        @torch.jit.script
        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return x
        torch.jit.script(torch.jit.script(fn))

    def test_pretty_print_function(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def foo(x):
            if False:
                return 10
            return torch.nn.functional.interpolate(x)
        FileCheck().check('interpolate').run(foo.code)

    def test_inlined_graph(self):
        if False:
            print('Hello World!')
        '\n        Check that the `inlined_graph` property correctly returns an inlined\n        graph, both through function calls and method calls.\n        '

        @torch.jit.script
        def foo(x):
            if False:
                print('Hello World!')
            return torch.add(x, x)

        class MyNestedMod(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.sub(x, x)

        class MyMod(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.nested = MyNestedMod()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.nested(x)
                x = foo(x)
                return torch.mul(x, x)
        m = torch.jit.script(MyMod())
        FileCheck().check('aten::sub').check('aten::add').check('aten::mul').run(m.inlined_graph)

    def test_static_method_on_module(self):
        if False:
            while True:
                i = 10
        '\n        Check that the `@staticmethod` annotation on a function on a module works.\n        '

        class MyCell(torch.nn.Module):

            @staticmethod
            def do_it(x, h):
                if False:
                    i = 10
                    return i + 15
                new_h = torch.tanh(x + h)
                return (new_h, new_h)

            def forward(self, x, h):
                if False:
                    while True:
                        i = 10
                return self.do_it(x, h)
        my_cell = torch.jit.script(MyCell())
        x = torch.rand(3, 4)
        h = torch.rand(3, 4)
        jitted_cell = my_cell(x, h)
        non_jitted_cell = MyCell().do_it(x, h)
        self.assertEqual(jitted_cell, non_jitted_cell)

    def test_code_with_constants(self):
        if False:
            return 10
        '\n        Check that the `code_with_constants` property correctly returns graph CONSTANTS in the\n        CONSTANTS.cN format used in the output of the `code` property.\n        '

        @torch.jit.script
        def foo(x=torch.ones(1)):
            if False:
                print('Hello World!')
            return x

        class Moddy(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return foo()
        m = torch.jit.script(Moddy())
        (src, CONSTANTS) = m.code_with_constants
        self.assertEqual(CONSTANTS.c0, torch.ones(1))
        self.assertEqual(src, m.code)

    def test_code_with_constants_restore(self):
        if False:
            while True:
                i = 10
        '\n        Check that the `code_with_constants` property correctly works on restoration after save() + load()\n        '

        @torch.jit.script
        def foo(x=torch.ones(1)):
            if False:
                i = 10
                return i + 15
            return x

        class Moddy(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return foo()
        m = torch.jit.script(Moddy())
        (src, CONSTANTS) = m.code_with_constants
        eic = self.getExportImportCopy(m)
        (src_eic, CONSTANTS_eic) = eic.code_with_constants
        self.assertEqual(src, src_eic)
        self.assertEqual(CONSTANTS.c0, CONSTANTS_eic.c0)

    def test_oneline_func(self):
        if False:
            i = 10
            return i + 15

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return x
        self.checkScript(fn, (torch.ones(2, 2),))

    def test_request_bailout(self):
        if False:
            i = 10
            return i + 15
        with enable_profiling_mode_for_profiling_tests():

            def fct_loop(x):
                if False:
                    for i in range(10):
                        print('nop')
                for i in range(3):
                    x = torch.cat((x, x), 0)
                return x
            x = torch.ones(2, 3, 4, dtype=torch.float32)
            expected = fct_loop(x)
            jitted = torch.jit.script(fct_loop)
            jitted(x)
            jitted(x)
            dstate = jitted.get_debug_state()
            eplan = get_execution_plan(dstate)
            num_bailouts = eplan.code.num_bailouts()
            for i in range(0, num_bailouts):
                eplan.code.request_bailout(i)
                self.assertEqual(jitted(x), expected)

    @unittest.skip('bailouts are being deprecated')
    def test_dominated_bailout(self):
        if False:
            for i in range(10):
                print('nop')
        with enable_profiling_mode_for_profiling_tests():

            @torch.jit.script
            def foo(x):
                if False:
                    i = 10
                    return i + 15
                dim = x.dim()
                if dim == 0:
                    y = int(x)
                else:
                    y = x.size()[dim - 1]
                return y
            x = torch.zeros(2)
            self.assertEqual(foo(x), 2)
            self.assertEqual(foo(x), 2)
            g = torch.jit.last_executed_optimized_graph()
            g_s = str(g)
            g_s = g_s[0:g_s.find('return')]
            FileCheck().check_count('prim::BailOut[', 1, exactly=True).run(g_s)

            @torch.jit.script
            def foo(x):
                if False:
                    i = 10
                    return i + 15
                dim = x.dim()
                x.add_(3)
                if dim == 0:
                    return 0
                else:
                    return x.size()[dim - 1]
            x = torch.zeros(2)
            self.assertEqual(foo(x), 2)
            self.assertEqual(foo(x), 2)
            g = torch.jit.last_executed_optimized_graph()
            FileCheck().check('prim::BailOut[').check('aten::add_').check_next('prim::BailOut[').check('return').run(g)
            with torch.enable_grad():

                @torch.jit.ignore
                def disable_grad():
                    if False:
                        print('Hello World!')
                    torch.set_grad_enabled(False)

                @torch.jit.ignore
                def enable_grad():
                    if False:
                        for i in range(10):
                            print('nop')
                    torch.set_grad_enabled(True)

                @torch.jit.script
                def foo(x):
                    if False:
                        while True:
                            i = 10
                    x = x + 1
                    dim = x.dim()
                    disable_grad()
                    if dim == 0:
                        y = int(x)
                    else:
                        y = x.size()[dim - 1]
                    enable_grad()
                    return y
                x = torch.zeros(2, requires_grad=True)
                self.assertEqual(foo(x), 2)
                self.assertEqual(foo(x), 2)
                g = torch.jit.last_executed_optimized_graph()
                FileCheck().check('disable_grad').check('BailOut[').check('BailoutTemplate').run(g)

    @skipIfTorchDynamo('Torchdynamo cannot correctly handle profiler.profile calls')
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, "skip if profiling isn't enabled")
    def test_profiling_merge(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def test_not_const(x):
            if False:
                for i in range(10):
                    print('nop')
            if x.size(0) == 1:
                return 1
            else:
                return 2
        with enable_profiling_mode_for_profiling_tests():
            with num_profiled_runs(2):
                test_not_const(torch.rand([1, 2]))
                test_not_const(torch.rand([2, 2]))
                graph_str = torch.jit.last_executed_optimized_graph()
                FileCheck().check('profiled_type=Float(*, 2, strides=[2, 1], requires_grad=0, device=cpu').run(graph_str)
                FileCheck().check_not('profiled_type=Float(1, 2, strides=[2, 1], requires_grad=0, device=cpu').run(graph_str)

    def test_nested_bailouts(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def fct_loop(x):
            if False:
                while True:
                    i = 10
            for i in range(3):
                x = torch.cat((x, x), 0)
            return x
        x = torch.ones(2, 3, 4, dtype=torch.float32)
        out = fct_loop(x)
        jit_trace = torch.jit.trace(fct_loop, x)
        out_trace = jit_trace(x)

    def test_no_self_arg_ignore_function(self):
        if False:
            while True:
                i = 10

        class MyModule(nn.Module):

            @torch.jit.ignore
            def call_np():
                if False:
                    return 10
                return np.random.choice(2, p=[0.95, 0.05])

            def forward(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.call_np()
        with self.assertRaisesRegex(Exception, 'does not have a self argument'):
            torch.jit.script(MyModule())

    def test_loop_liveness(self):
        if False:
            while True:
                i = 10
        with enable_profiling_mode_for_profiling_tests():

            @torch.jit.script
            def f(i):
                if False:
                    print('Hello World!')
                l = []
                for n in [2, 1]:
                    l.append(torch.zeros(n, i))
                return l[0]
            f(2)
            f(1)

    def test_bailout_loop_carried_deps_name_clash(self):
        if False:
            print('Hello World!')
        with enable_profiling_mode_for_profiling_tests():
            NUM_ITERATIONS = 10

            @torch.jit.script
            def fct_loop(z, size):
                if False:
                    while True:
                        i = 10
                counters = torch.jit.annotate(List[int], [])
                j = 0
                y = torch.ones(2)
                for i in range(size):
                    counters.append(i + j)
                    y = torch.cat((y, torch.ones(z)), 0)
                    j = j + 1
                return (y, counters)
            inputs = [1, 2, 3, 4]
            expected = [x * 2 for x in range(NUM_ITERATIONS)]
            for inp in inputs:
                results = fct_loop(inp, NUM_ITERATIONS)
                self.assertEqual(results[1], expected)

    def test_bailout_loop_counter_transition(self):
        if False:
            print('Hello World!')
        with enable_profiling_mode_for_profiling_tests():
            NUM_ITERATIONS = 10

            @torch.jit.script
            def fct_loop(z, size):
                if False:
                    return 10
                counters = torch.jit.annotate(List[int], [])
                y = torch.ones(2)
                for i in range(size):
                    counters.append(i)
                    y = torch.cat((y, torch.ones(z)), 0)
                return (y, counters)
            inputs = [1, 2, 3, 4]
            expected = list(range(NUM_ITERATIONS))
            for inp in inputs:
                results = fct_loop(inp, NUM_ITERATIONS)
                self.assertEqual(results[1], expected)

    def test_ignored_method_binding(self):
        if False:
            while True:
                i = 10

        class Bar(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.x: int = 0

            @torch.jit.export
            def setx(self, x: int):
                if False:
                    i = 10
                    return i + 15
                self.x = x

            @torch.jit.export
            def getx(self):
                if False:
                    i = 10
                    return i + 15
                return self.x

            @torch.jit.ignore
            def ignored_getx(self):
                if False:
                    print('Hello World!')
                return self.x
        b = Bar()
        b.setx(123)
        sb = torch.jit.script(b)
        self.assertEqual(sb.getx(), 123)
        self.assertEqual(sb.ignored_getx(), 123)
        sb.setx(456)
        self.assertEqual(sb.getx(), 456)
        self.assertEqual(sb.ignored_getx(), 456)

    def test_set_attribute_through_optional(self):
        if False:
            print('Hello World!')

        class A(torch.nn.Module):
            __annotations__ = {'x': Optional[torch.Tensor]}

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.x = None

            @torch.jit.ignore
            def foo(self):
                if False:
                    print('Hello World!')
                if self.x is None:
                    self.x = torch.tensor([3])
                return self.x

            def forward(self, x):
                if False:
                    return 10
                a = self.foo()
                return x + 1
        m = torch.jit.script(A())
        self.assertEqual(m.x, None)
        m(torch.rand(1))
        self.assertEqual(m.x, torch.tensor([3]))

    def test_mutate_constant(self):
        if False:
            while True:
                i = 10

        class M(torch.jit.ScriptModule):
            __constants__ = ['foo']

            def __init__(self, foo):
                if False:
                    print('Hello World!')
                super().__init__()
                self.foo = foo
        m = M(5)
        with self.assertRaises(RuntimeError):
            m.foo = 6

    def test_class_attribute(self):
        if False:
            while True:
                i = 10

        class M(torch.jit.ScriptModule):
            FOO = 0

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.foo = self.FOO
        m = M()
        self.assertEqual(m.foo, M.FOO)

    def test_class_attribute_in_script(self):
        if False:
            return 10

        class M(torch.jit.ScriptModule):
            FOO = 0

            @torch.jit.script_method
            def forward(self):
                if False:
                    i = 10
                    return i + 15
                return self.FOO
        with self.assertRaises(RuntimeError):
            M()

    def test_not_initialized_err(self):
        if False:
            print('Hello World!')

        class M(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.foo = torch.rand(2, 3)
        with self.assertRaises(RuntimeError):
            M()

    def test_attribute_in_init(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.foo = torch.jit.Attribute(0.1, float)
                assert 0.0 < self.foo
        M()

    def test_scriptable_fn_as_attr(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def __init__(self, fn):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.fn = fn

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.fn(x)
        m = M(torch.sigmoid)
        inp = torch.rand(2, 3)
        self.checkModule(m, (inp,))

    def test_sequence_parsing(self):
        if False:
            return 10
        tests = [('return [x, x,]', True), ('return [x x]', 'expected ]'), ('return x, x,', True), ('return bar(x, x,)', True), ('return bar()', 'Argument x not provided'), ('for a, b, in x, x,:\n        pass', 'List of iterables'), ('a, b, = x, x,\n    return a + b', True)]
        for (exp, result) in tests:
            cu = torch.jit.CompilationUnit()
            full = f'\ndef bar(x, y):\n    return x + y\ndef foo(x):\n    {exp}\n            '
            if isinstance(result, str):
                with self.assertRaisesRegex(RuntimeError, result):
                    cu.define(full)
            else:
                cu.define(full)

    def test_namedtuple_python(self):
        if False:
            while True:
                i = 10
        global MyTuple, MyMod
        MyTuple = namedtuple('MyTuple', ['a'])

        @torch.jit.unused
        def fn():
            if False:
                while True:
                    i = 10
            return MyTuple(1)

        @torch.jit.script
        def fn2():
            if False:
                i = 10
                return i + 15
            return fn()
        FileCheck().check('NamedTuple').run(fn2.graph)

        class MyMod(torch.nn.Module):

            @torch.jit.unused
            def fn(self):
                if False:
                    return 10
                return MyTuple(1)

            def forward(self, x):
                if False:
                    return 10
                if 1 == 1:
                    return MyTuple(torch.rand(2, 3))
                else:
                    return self.fn()
        torch.jit.script(MyMod())

    def test_unused_decorator(self):
        if False:
            i = 10
            return i + 15

        class MyMod(torch.nn.Module):

            @torch.jit.unused
            @torch.no_grad()
            def fn(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return next(x)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.fn(x)
        torch.jit.script(MyMod())

    @_inline_everything
    def test_lazy_script(self):
        if False:
            print('Hello World!')

        def untraceable(x):
            if False:
                i = 10
                return i + 15
            if x.ndim > 2:
                print('hello')
            else:
                print('goodbye')
            return x + 2

        def fn(x):
            if False:
                while True:
                    i = 10
            return untraceable(x)
        with self.capture_stdout():
            traced_bad = torch.jit.trace(fn, [torch.ones(2, 2)])
        FileCheck().check_not('goodbye').check_not('hello').run(traced_bad.graph)
        untraceable = torch.jit.script_if_tracing(untraceable)

        def fn2(x):
            if False:
                print('Hello World!')
            return untraceable(x)
        with self.capture_stdout():
            traced = torch.jit.trace(fn, [torch.ones(2, 2)])
        FileCheck().check('goodbye').run(traced.graph)

        def foo(x: int):
            if False:
                print('Hello World!')
            return x + 1

        @torch.jit.script_if_tracing
        def fee(x: int=2):
            if False:
                i = 10
                return i + 15
            return foo(1) + x
        fee_compiled = torch.jit.script(fee)
        self.assertEqual(fee_compiled(), fee())

        @torch.jit.script
        def hum():
            if False:
                print('Hello World!')
            return fee(x=3)
        self.assertEqual(hum(), 5)

    def test_big_int_literals(self):
        if False:
            print('Hello World!')

        def ok():
            if False:
                return 10
            a = 9223372036854775807
            return a

        def toobig():
            if False:
                print('Hello World!')
            a = 9223372036854775808
            return a

        def waytoobig():
            if False:
                i = 10
                return i + 15
            a = 99999999999999999999
            return a
        self.checkScript(ok, [])
        with self.assertRaisesRegex(RuntimeError, 'out of range'):
            torch.jit.script(toobig)
        with self.assertRaisesRegex(RuntimeError, 'out of range'):
            torch.jit.script(waytoobig)

    def test_hex_literals(self):
        if False:
            while True:
                i = 10

        def test1():
            if False:
                i = 10
                return i + 15
            return 11184810

        def test2():
            if False:
                i = 10
                return i + 15
            return 11184810

        def test3():
            if False:
                print('Hello World!')
            return -11184810
        self.checkScript(test1, [])
        self.checkScript(test2, [])
        self.checkScript(test3, [])

        def ok():
            if False:
                print('Hello World!')
            a = 9223372036854775807
            return a

        def toobig():
            if False:
                i = 10
                return i + 15
            a = 18446744073709551615
            return a

        def waytoobig():
            if False:
                return 10
            a = 340282366920938463463374607431768211455
            return a
        self.checkScript(ok, [])
        with self.assertRaisesRegex(RuntimeError, 'out of range'):
            torch.jit.script(toobig)
        with self.assertRaisesRegex(RuntimeError, 'out of range'):
            torch.jit.script(waytoobig)

    def test_big_float_literals(self):
        if False:
            i = 10
            return i + 15

        def ok():
            if False:
                while True:
                    i = 10
            a = 1e309
            return a

        def check(fn):
            if False:
                for i in range(10):
                    print('nop')
            self.assertTrue(fn() == ok())
        check(torch.jit.script(ok))
        cu = torch.jit.CompilationUnit()
        cu.define(dedent(inspect.getsource(ok)))
        check(cu.ok)

    def _test_device_type(self, dest):
        if False:
            print('Hello World!')

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return (x.type, x.index)
        device = torch.ones(2).to(dest).device
        self.checkScript(fn, [device])

    def test_device_type(self):
        if False:
            while True:
                i = 10
        self._test_device_type('cpu')

    @unittest.skipIf(not RUN_CUDA, 'Requires CUDA')
    def test_device_type_cuda(self):
        if False:
            while True:
                i = 10
        self._test_device_type('cuda')

    def test_string_device_implicit_conversion(self):
        if False:
            return 10

        @torch.jit.script
        def fn(x: torch.device):
            if False:
                for i in range(10):
                    print('nop')
            return x
        self.assertEqual(fn('cpu'), torch.device('cpu'))
        with self.assertRaisesRegex(RuntimeError, 'Expected one of'):
            fn('invalid_device')

    def test_eval_python(self):
        if False:
            return 10

        def _test(m):
            if False:
                for i in range(10):
                    print('nop')
            self.assertTrue(m(torch.ones(2, 2)))
            self.assertTrue(m.training)
            self.assertTrue(m._c.getattr('training'))
            m.eval()
            self.assertFalse(m.training)
            self.assertFalse(m._c.getattr('training'))
            self.assertFalse(m(torch.ones(2, 2)))
            buffer = io.BytesIO()
            torch.jit.save(m, buffer)
            buffer.seek(0)
            loaded = torch.jit.load(buffer)
            self.assertFalse(loaded.training)
            self.assertFalse(loaded._c.getattr('training'))

        class M(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.training

        class OldM(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    return 10
                return self.training
        _test(torch.jit.script(M()))
        _test(OldM())

    def test_inherit_method(self):
        if False:
            i = 10
            return i + 15

        class A(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return x + self.bar(x)

        class B(A):

            @torch.jit.script_method
            def bar(self, x):
                if False:
                    i = 10
                    return i + 15
                return x * x
        with self.assertRaisesRegex(RuntimeError, 'attribute'):
            A()
        v = torch.rand(3, 4)
        b = B()
        self.assertEqual(b(v), v + v * v)

        class C(torch.jit.ScriptModule):

            @torch.jit.script_method
            def bar(self, x):
                if False:
                    i = 10
                    return i + 15
                return x

        class D(C, B):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
        self.assertEqual(D()(v), v + v)

    def test_tensor_subclasses(self):
        if False:
            while True:
                i = 10

        def check_subclass(x, tensor):
            if False:
                i = 10
                return i + 15
            template = dedent('\n                def func(input: {}) -> {}:\n                    return torch.zeros((input.shape[0], 1), dtype=input.dtype)\n                ')
            self._check_code(template.format(x, x), 'func', [tensor])
        check_subclass('torch.LongTensor', torch.LongTensor([[1, 2], [3, 4]]))
        check_subclass('torch.DoubleTensor', torch.DoubleTensor([[1.2, 2.3], [3.4, 4.5]]))
        check_subclass('torch.IntTensor', torch.IntTensor([[1, 2], [3, 4]]))
        check_subclass('torch.BoolTensor', torch.BoolTensor([[False, True], [True, False]]))

        def check_subclass_warn(input: torch.LongTensor) -> torch.LongTensor:
            if False:
                for i in range(10):
                    print('nop')
            return torch.zeros((input.shape[0], 1), dtype=input.dtype)
        with warnings.catch_warnings(record=True) as warns:
            scripted = torch.jit.script(check_subclass_warn)
        FileCheck().check('TorchScript will treat type annotations of Tensor').run(str(warns[0]))

    def test_first_class_module(self):
        if False:
            return 10

        class Foo(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.foo = nn.Parameter(torch.rand(3, 4))

            @torch.jit.script_method
            def forward(self, input):
                if False:
                    return 10
                self.foo = input
                return self.foo
        foo = Foo()
        input = torch.rand(3, 4)
        foo.forward(input)
        self.assertEqual(input, foo.foo)

    @_tmp_donotuse_dont_inline_everything
    def test_first_class_calls(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        class Foo:

            def __init__(self, x):
                if False:
                    return 10
                self.bar = x

            def stuff(self, x):
                if False:
                    return 10
                return self.bar + x

        @torch.jit.script
        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            return x * x + Foo(x).stuff(2 * x)

        @torch.jit.script
        def bar(x):
            if False:
                while True:
                    i = 10
            return foo(x) * foo(x)
        x = torch.rand(3, 4)
        self.assertEqual(bar(x), (x * x + 3 * x) * (x * x + 3 * x))

    def test_static_methods(self):
        if False:
            print('Hello World!')

        class M(nn.Module):

            @staticmethod
            def my_method(x):
                if False:
                    for i in range(10):
                        print('nop')
                return x + 100

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x + M.my_method(x)

        class N(nn.Module):

            @staticmethod
            def my_method(x):
                if False:
                    print('Hello World!')
                return x * 100

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x - M.my_method(x) + N.my_method(x)
        self.checkModule(M(), (torch.ones(2, 2),))
        self.checkModule(N(), (torch.ones(2, 2),))

    def test_invalid_prefix_annotation(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(RuntimeError, 'annotation prefix in line'):
            with self.capture_stdout() as captured:

                @torch.jit.script
                def invalid_prefix_annotation1(a):
                    if False:
                        print('Hello World!')
                    return a + 2
        with self.assertRaisesRegex(RuntimeError, 'annotation prefix in line'):
            with self.capture_stdout() as captured:

                @torch.jit.script
                def invalid_prefix_annotation2(a):
                    if False:
                        i = 10
                        return i + 15
                    return a + 2
        with self.assertRaisesRegex(RuntimeError, 'annotation prefix in line'):
            with self.capture_stdout() as captured:

                @torch.jit.script
                def invalid_prefix_annotation3(a):
                    if False:
                        i = 10
                        return i + 15
                    return a + 2

    def test_builtin_function_attributes(self):
        if False:
            print('Hello World!')

        class Add(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.add = torch.add

            def forward(self, input):
                if False:
                    return 10
                return self.add(input, input)
        self.checkModule(Add(), [torch.randn(2, 2)])

    def test_pybind_type_comparisons(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def f():
            if False:
                while True:
                    i = 10
            return None
        node = list(f.graph.nodes())[0]
        t = node.outputsAt(0).type()
        self.assertIsNotNone(t)

    @unittest.skipIf(IS_WINDOWS, 'TODO: need to fix the test case')
    def test_unmatched_type_annotation(self):
        if False:
            i = 10
            return i + 15
        message1 = re.escape('Number of type annotations (2) did not match the number of function parameters (1):')
        message2 = 'def invalid2\\(a\\):\n\\s*~+\\.*\\s+<--- HERE\n\\s+# type: \\(Int, Int\\) -> Int\n\\s+return a \\+ 2'
        message3 = 'def invalid4\\(a\\):\n\\s*~+\\.*\\s+<--- HERE\n\\s+# type: \\(Int, Int\\) -> Int\n\\s+return a \\+ 2'
        with self.assertRaisesRegex(RuntimeError, message1):

            @torch.jit.script
            def invalid1(a):
                if False:
                    i = 10
                    return i + 15
                return a + 2
        with self.assertRaisesRegex(RuntimeError, message2):

            @torch.jit.script
            def invalid2(a):
                if False:
                    i = 10
                    return i + 15
                return a + 2
        with self.assertRaisesRegex(RuntimeError, message1):

            def invalid3(a):
                if False:
                    while True:
                        i = 10
                return a + 2
            torch.jit.script(invalid3)
        with self.assertRaisesRegex(RuntimeError, message3):

            def invalid4(a):
                if False:
                    print('Hello World!')
                return a + 2
            torch.jit.script(invalid4)

    def test_calls_in_type_annotations(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(RuntimeError, 'Type annotation should not contain calls'):

            def spooky(a):
                if False:
                    while True:
                        i = 10
                return a + 2
            print(torch.__file__)
            torch.jit.annotations.get_signature(spooky, None, 1, True)

    def test_is_optional(self):
        if False:
            while True:
                i = 10
        ann = Union[List[int], List[float]]
        torch._jit_internal.is_optional(ann)

    def test_interpreter_fuzz(self):
        if False:
            while True:
                i = 10
        import builtins
        templates = ['torch.rand(3, 4)', '({} + {})', '-{}', '({} * {})', 'torch.tanh({})', 'VAR {}']

        def gen_code():
            if False:
                while True:
                    i = 10
            src_lines = ['def f():']
            exprs = []
            n_variables = 0

            def get_expr(idx):
                if False:
                    while True:
                        i = 10
                elem = exprs[idx]
                exprs[idx] = exprs[-1]
                exprs.pop()
                return elem

            def select_expr_or_var():
                if False:
                    for i in range(10):
                        print('nop')
                idx = random.randrange(0, len(exprs) + n_variables)
                if idx < len(exprs):
                    return get_expr(idx)
                else:
                    return f'v{idx - len(exprs)}'
            for i in range(50):
                n = None
                while n is None or n > len(exprs) + n_variables:
                    template = random.choice(templates)
                    n = template.count('{}')
                if 'VAR' in template:
                    src_lines.append(f'  v{n_variables} = {select_expr_or_var()}')
                    n_variables += 1
                else:
                    exprs.append(template.format(*(select_expr_or_var() for _ in range(n))))
            src_lines.append('  return ({})\n'.format(''.join((f'v{i},' for i in range(n_variables)))))
            return '\n'.join(src_lines)
        for i in range(100):
            g = {'torch': torch}
            code = gen_code()
            builtins.exec(code, g, None)
            cu = torch.jit.CompilationUnit(code)
            with freeze_rng_state():
                o1 = g['f']()
            with freeze_rng_state():
                o2 = cu.f()
            self.assertEqual(o1, o2)

    @skipIfTorchDynamo('TorchDynamo fails with unknown reason')
    def test_cpp_module_iterator(self):
        if False:
            return 10
        a = nn.Module()
        a.name = 'a'
        a.p = nn.Parameter(torch.rand(3, 4))
        a.foo = nn.Module()
        a.foo.name = 'foo'
        a.foo.register_buffer('b', torch.rand(1, 1))
        a.foo.bar = nn.Module()
        a.foo.bar.name = 'bar'
        a.foo.bar.an_int = 4
        a.another = nn.Module()
        a.another.name = 'another'
        sa = torch.jit.script(a)
        result = torch._C._jit_debug_module_iterators(sa._c)

        def replace(e):
            if False:
                for i in range(10):
                    print('nop')
            if e is a.p:
                return 'P'
            elif e is a.foo.b:
                return 'B'
            elif isinstance(e, torch._C.ScriptModule):
                return e.getattr('name')
            return e
        for v in result.values():
            for i in range(len(v)):
                if isinstance(v[i], tuple):
                    (n, v2) = v[i]
                    v[i] = (n, replace(v2))
                else:
                    v[i] = replace(v[i])
            v.sort()
        expected = {'buffers': [], 'buffers_r': ['B'], 'children': ['another', 'foo'], 'modules': ['a', 'another', 'bar', 'foo'], 'named_attributes': [('_is_full_backward_hook', None), ('another', 'another'), ('foo', 'foo'), ('name', 'a'), ('p', 'P'), ('training', True)], 'named_attributes_r': [('_is_full_backward_hook', None), ('another', 'another'), ('another._is_full_backward_hook', None), ('another.name', 'another'), ('another.training', True), ('foo', 'foo'), ('foo._is_full_backward_hook', None), ('foo.b', 'B'), ('foo.bar', 'bar'), ('foo.bar._is_full_backward_hook', None), ('foo.bar.an_int', 4), ('foo.bar.name', 'bar'), ('foo.bar.training', True), ('foo.name', 'foo'), ('foo.training', True), ('name', 'a'), ('p', 'P'), ('training', True)], 'named_buffers': [], 'named_buffers_r': [('foo.b', 'B')], 'named_children': [('another', 'another'), ('foo', 'foo')], 'named_modules': [('', 'a'), ('another', 'another'), ('foo', 'foo'), ('foo.bar', 'bar')], 'named_parameters': [('p', 'P')], 'named_parameters_r': [('p', 'P')], 'parameters': ['P'], 'parameters_r': ['P']}
        self.assertEqual(expected, result)

    def test_parameter_order(self):
        if False:
            print('Hello World!')
        m = nn.Module()
        for (i, name) in enumerate(string.ascii_letters):
            setattr(m, name, nn.Parameter(torch.tensor([float(i)])))
        ms = torch.jit.script(m)
        print(torch.cat(list(m.parameters())))
        print(torch.cat(list(ms.parameters())))
        self.assertEqual(list(m.parameters()), list(ms.parameters()))

    def test_python_op_builtins(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.unused
        def fn(x):
            if False:
                i = 10
                return i + 15
            return sum(x)

        @torch.jit.script
        def script_fn(x):
            if False:
                print('Hello World!')
            return fn(x)

    def test_submodule_twice(self):
        if False:
            return 10

        @torch.jit.script
        def foo(x):
            if False:
                while True:
                    i = 10
            return x * x

        class What(torch.jit.ScriptModule):

            def __init__(self, x):
                if False:
                    print('Hello World!')
                super().__init__()
                self.foo = x
        a = What(foo)
        c = What(foo)

    def test_training_param(self):
        if False:
            print('Hello World!')

        class What(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    while True:
                        i = 10
                if self.training:
                    r = x
                else:
                    r = x + 4
                if self.training:
                    r = r + 1
                return r
        w = What()
        self.assertEqual(4, w(3))
        w.train(False)
        self.assertEqual(7, w(3))
        self.assertFalse('training' in w.state_dict())

    def test_class_as_attribute(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        class Foo321:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.x = 3

        class FooBar1234(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.f = Foo321()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return x + self.f.x
        scripted = torch.jit.script(FooBar1234())
        eic = self.getExportImportCopy(scripted)
        x = torch.rand(3, 4)
        self.assertEqual(scripted(x), eic(x))

    def test_module_str(self):
        if False:
            i = 10
            return i + 15

        class Foo(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.relu(x)
        f = torch.jit.script(Foo())
        str_f = str(f._c)
        self.assertTrue(str_f.startswith('ScriptObject'))
        self.assertTrue('__torch__.' in str_f)
        self.assertTrue('.Foo' in str_f)

    def test_jitter_bug(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def fn2(input, kernel_size):
            if False:
                while True:
                    i = 10
            if kernel_size[0] > 1:
                _stride = [2]
            else:
                _stride = kernel_size
            print(_stride, kernel_size)
            return input

        @torch.jit.script
        def fn(input):
            if False:
                return 10
            return fn2(input, [1])

    def test_parser_kwargonly(self):
        if False:
            i = 10
            return i + 15
        cu = torch.jit.CompilationUnit('\n            def foo(x, *, y) -> Tuple[Tensor, Tensor]:\n                return x, x\n            def bar(x):\n                return foo(x, y=x)\n        ')
        self.assertTrue('*' in str(cu.foo.schema))
        with self.assertRaisesRegex(RuntimeError, 'not provided'):
            torch.jit.CompilationUnit('\n                def foo(x, *, y) -> Tuple[Tensor, Tensor]:\n                    return x, x\n                def bar(x):\n                    return foo(x, x)\n            ')

    def test_annoying_doubles(self):
        if False:
            while True:
                i = 10
        mod = types.ModuleType('temp')
        mod.inf = float('inf')
        mod.ninf = float('-inf')
        mod.nan = float('nan')
        with torch._jit_internal._disable_emit_hooks():

            class Foo(torch.jit.ScriptModule):

                @torch.jit.script_method
                def forward(self):
                    if False:
                        while True:
                            i = 10
                    return (math.pi, 0.1, mod.inf, mod.ninf, 2.225073858507201e-308, mod.nan)
            foo = Foo()
            buffer = io.BytesIO()
            torch.jit.save(foo, buffer)
            buffer.seek(0)
            foo_loaded = torch.jit.load(buffer)
            r = foo()
            r2 = foo_loaded()
            self.assertTrue(r[:-1] == r2[:-1])
            self.assertTrue(math.isnan(r[-1]) and math.isnan(r2[-1]))

    def test_type_annotate(self):
        if False:
            print('Hello World!')

        def foo(a):
            if False:
                while True:
                    i = 10
            return torch.jit.annotate(torch.Tensor, a)
        self.checkScript(foo, (torch.rand(3),))

        def bar():
            if False:
                return 10
            a = torch.jit.annotate(List[int], [])
            for _ in range(10):
                a.append(4)
            return a
        self.checkScript(bar, ())

        def baz(a):
            if False:
                return 10
            return torch.jit.annotate(float, a)
        self.checkScript(baz, (torch.rand(()),))

        def annotate_none():
            if False:
                return 10
            return torch.jit.annotate(Optional[torch.Tensor], None)
        self.checkScript(annotate_none, ())

    def test_robust_op_resolution(self):
        if False:
            print('Hello World!')
        neg = torch.add

        def stuff(x):
            if False:
                for i in range(10):
                    print('nop')
            return neg(x, x)
        a = (torch.rand(3),)
        self.checkScript(stuff, a)

    def test_nested_aug_assign(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        class SomeClass:

            def __init__(self):
                if False:
                    return 10
                self.num = 99

            def __iadd__(self, x):
                if False:
                    i = 10
                    return i + 15
                self.num += x
                return self

            def __eq__(self, other):
                if False:
                    for i in range(10):
                        print('nop')
                return self.num == other.num

        @torch.jit.script
        class SomeOutOfPlaceClass:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.num = 99

            def __add__(self, x):
                if False:
                    while True:
                        i = 10
                self.num = x
                return self

            def __eq__(self, other):
                if False:
                    for i in range(10):
                        print('nop')
                return self.num == other.num

        class Child(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.x = 2
                self.o = SomeClass()
                self.oop = SomeOutOfPlaceClass()
                self.list = [1, 2, 3]

        class A(nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.child = Child()

            def forward(self):
                if False:
                    i = 10
                    return i + 15
                self.child.x += 1
                self.child.o += 5
                self.child.oop += 5
                some_list = [1, 2]
                self.child.list += some_list
                self.child.list *= 2
                return (self.child.x, self.child.o, self.child.list, self.child.oop)
        a = A()
        sa = torch.jit.script(A())
        eager_result = a()
        script_result = sa()
        self.assertEqual(eager_result, script_result)
        self.assertEqual(a.child.x, sa.child.x)
        self.assertEqual(a.child.o, sa.child.o)
        self.assertEqual(a.child.list, sa.child.list)

        @torch.jit.script
        class SomeNonAddableClass:

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.num = 99

            def __eq__(self, other):
                if False:
                    return 10
                return self.num == other.num

        class A(nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.x = SomeNonAddableClass()

            def forward(self):
                if False:
                    print('Hello World!')
                self.x += SomeNonAddableClass()
                return self.x
        with self.assertRaisesRegex(RuntimeError, 'Cannot emit inplace op'):
            torch.jit.script(A())

    def test_var_aug_assign(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        class SomeNonAddableClass:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.num = 99

            def __eq__(self, other):
                if False:
                    while True:
                        i = 10
                return self.num == other.num
        with self.assertRaisesRegex(RuntimeError, 'Cannot emit inplace op'):

            @torch.jit.script
            def fn():
                if False:
                    while True:
                        i = 10
                a = SomeNonAddableClass()
                a += SomeNonAddableClass()
                return a

        @torch.jit.script
        class SomeClass:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.num = 99

            def __iadd__(self, x):
                if False:
                    print('Hello World!')
                self.num += x
                return self

            def __eq__(self, other):
                if False:
                    return 10
                return self.num == other.num

        @torch.jit.script
        class SomeOutOfPlaceClass:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.num = 99

            def __add__(self, x):
                if False:
                    print('Hello World!')
                self.num = x
                return self

            def __eq__(self, other):
                if False:
                    print('Hello World!')
                return self.num == other.num

        def fn2():
            if False:
                return 10
            a = SomeClass()
            a_copy = a
            a += 20
            assert a is a_copy
            b = SomeOutOfPlaceClass()
            b_copy = b
            b += 99
            assert b is b_copy
            c = [1, 2, 3]
            c_copy = c
            c *= 2
            assert c is c_copy
            c += [4, 5, 6]
            d = torch.ones(2, 2)
            d_copy = d
            d += torch.ones(2, 2)
            assert d is d_copy
            return (a, b, c, d)
        self.checkScript(fn2, [])

    def test_nested_list_construct(self):
        if False:
            i = 10
            return i + 15

        def foo():
            if False:
                for i in range(10):
                    print('nop')
            return [[4]] + [[4, 5]]
        self.checkScript(foo, ())

    def test_file_line_error(self):
        if False:
            return 10

        def foobar(xyz):
            if False:
                while True:
                    i = 10
            return torch.blargh(xyz)
        (_, lineno) = inspect.getsourcelines(foobar)
        with self.assertRaisesRegex(RuntimeError, f'test_jit.py", line {lineno + 1}'):
            scripted = torch.jit.script(foobar)

    def test_file_line_error_class_defn(self):
        if False:
            print('Hello World!')

        class FooBar:

            def baz(self, xyz):
                if False:
                    i = 10
                    return i + 15
                return torch.blargh(xyz)
        (_, lineno) = inspect.getsourcelines(FooBar)
        with self.assertRaisesRegex(RuntimeError, f'test_jit.py", line {lineno + 2}'):
            torch.jit.script(FooBar)

    def test_file_line_graph(self):
        if False:
            while True:
                i = 10

        def foobar(xyz):
            if False:
                return 10
            return torch.neg(xyz)
        scripted = torch.jit.script(foobar)
        (_, lineno) = inspect.getsourcelines(foobar)
        fc = FileCheck().check(f'test_jit.py:{lineno + 1}:19')
        fc.run(scripted.graph)
        fc.run(str(scripted.graph))

    def test_file_line_save_load(self):
        if False:
            for i in range(10):
                print('nop')

        class Scripted(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, xyz):
                if False:
                    while True:
                        i = 10
                return torch.neg(xyz)
        scripted = Scripted()
        buffer = scripted.save_to_buffer()
        bytesio = io.BytesIO(buffer)
        scripted = torch.jit.load(bytesio)
        (_, lineno) = inspect.getsourcelines(Scripted)
        fc = FileCheck().check(f':{lineno + 3}')
        fc.run(scripted.graph)
        fc.run(str(scripted.graph))

    def test_file_line_string(self):
        if False:
            print('Hello World!')
        scripted = torch.jit.CompilationUnit('\ndef foo(xyz):\n    return torch.neg(xyz)\n        ')
        fc = FileCheck().check('<string>:3:11')
        fc.run(scripted.foo.graph)
        fc.run(str(scripted.foo.graph))

    @skipIfCrossRef
    def test_file_line_trace(self):
        if False:
            i = 10
            return i + 15

        def foobar(xyz):
            if False:
                for i in range(10):
                    print('nop')
            return torch.neg(xyz)
        scripted = torch.jit.trace(foobar, torch.rand(3, 4))
        (_, lineno) = inspect.getsourcelines(foobar)
        fc = FileCheck().check(f'test_jit.py:{lineno + 1}:0')
        fc.run(scripted.graph)
        fc.run(str(scripted.graph))

    def test_serialized_source_ranges(self):
        if False:
            i = 10
            return i + 15

        class FooTest(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x, w):
                if False:
                    while True:
                        i = 10
                return torch.mm(x, w.t())
        ft = FooTest()
        loaded = self.getExportImportCopy(ft)
        (_, lineno) = inspect.getsourcelines(FooTest)
        with self.assertRaisesRegex(RuntimeError, f'test_jit.py", line {lineno + 3}'):
            loaded(torch.rand(3, 4), torch.rand(30, 40))

    def test_serialized_source_ranges_graph(self):
        if False:
            return 10

        class FooTest3(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x, w):
                if False:
                    while True:
                        i = 10
                return torch.mm(x, w.t())
        ft = FooTest3()
        loaded = self.getExportImportCopy(ft)
        (_, lineno) = inspect.getsourcelines(FooTest3)
        fc = FileCheck().check(f'test_jit.py:{lineno + 3}')
        fc.run(loaded.graph)

    def test_serialized_source_ranges2(self):
        if False:
            for i in range(10):
                print('nop')

        class FooTest2(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self):
                if False:
                    for i in range(10):
                        print('nop')
                raise RuntimeError('foo')
        (_, lineno) = inspect.getsourcelines(FooTest2)
        with self.assertRaisesRegex(torch.jit.Error, f'test_jit.py", line {lineno + 3}'):
            ft = FooTest2()
            loaded = self.getExportImportCopy(ft)
            loaded()

    def test_serialized_source_ranges_dont_jitter(self):
        if False:
            while True:
                i = 10

        class FooTest3(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, lim):
                if False:
                    print('Hello World!')
                first = 1
                second = 1
                i = 1
                somenum = 5
                dontmutateme = 3
                third = 0
                while bool(i < lim):
                    third = first + second
                    first = second
                    second = third
                    j = 0
                    while j < 10:
                        somenum = somenum * 2
                        j = j + 1
                    i = i + j
                    i = i + dontmutateme
                st = second + third
                fs = first + second
                return (third, st, fs)
        ft3 = FooTest3()

        def debug_records_from_mod(self, mod):
            if False:
                for i in range(10):
                    print('nop')
            buffer = io.BytesIO()
            torch.jit.save(ft3, buffer)
            buffer.seek(0)
            archive = zipfile.ZipFile(buffer)
            files = filter(lambda x: x.startswith('archive/code/'), archive.namelist())
            debug_files = list(filter(lambda f: f.endswith('.debug_pkl'), files))
            self.assertEqual(len(debug_files), 1)
            debug_file = archive.open(debug_files[0])
            return (pickle.load(debug_file), buffer)
        (records1, buffer) = debug_records_from_mod(self, ft3)
        buffer.seek(0)
        loaded = torch.jit.load(buffer)
        (records2, buffer) = debug_records_from_mod(self, loaded)
        buffer.seek(0)
        loaded2 = torch.jit.load(buffer)
        (records3, _) = debug_records_from_mod(self, loaded2)
        self.assertEqual(records1, records2)
        self.assertEqual(records2, records3)

    def test_serialized_source_ranges_no_dups(self):
        if False:
            print('Hello World!')

        class FooTest3(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, lim):
                if False:
                    for i in range(10):
                        print('nop')
                first = 1
                second = 1
                i = 1
                somenum = 5
                dontmutateme = 3
                third = 0
                while bool(i < lim):
                    third = first + second
                    first = second
                    second = third
                    j = 0
                    while j < 10:
                        somenum = somenum * 2
                        j = j + 1
                    i = i + j
                    i = i + dontmutateme
                st = second + third
                fs = first + second
                return (third, st, fs)
        ft3 = FooTest3()

        def debug_records_from_mod(mod):
            if False:
                print('Hello World!')
            buffer = io.BytesIO()
            torch.jit.save(ft3, buffer)
            buffer.seek(0)
            archive = zipfile.ZipFile(buffer)
            files = list(filter(lambda x: x.startswith('archive/code/'), archive.namelist()))
            debug_files = filter(lambda f: f.endswith('.debug_pkl'), files)
            debug_files = (archive.open(f) for f in debug_files)
            debug_files = (pickle.load(f) for f in debug_files)
            debug_files = (f[2] for f in debug_files)
            return list(debug_files)
        debug_files = debug_records_from_mod(ft3)
        for debug_file in debug_files:
            for i in range(len(debug_file) - 1):
                (offset, source_range_tag, source_range) = debug_file[i]
                (offset2, source_range_tag2, source_range2) = debug_file[i + 1]
                self.assertNotEqual(source_range, source_range2)

    def test_circular_dependency(self):
        if False:
            return 10
        '\n        https://github.com/pytorch/pytorch/issues/25871\n        '

        class A(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x

        class B(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.foo = torch.nn.ModuleList([A()])

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                for f in self.foo:
                    x = f(x)
                return x

        class C(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.foo = torch.nn.Sequential(B())

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    while True:
                        i = 10
                for f in self.foo:
                    x = f(x)
                return x
        self.getExportImportCopy(C())

    def test_serialize_long_lines(self):
        if False:
            i = 10
            return i + 15

        class OrderModuleLong(torch.nn.Module):

            def forward(self, long_arg_name: List[torch.Tensor]):
                if False:
                    while True:
                        i = 10
                return [(long_arg_name[1],), (long_arg_name[0].argmax(),)]
        src = str(torch.jit.script(OrderModuleLong()).code)
        FileCheck().check('long_arg_name[1]').check('argmax').run(src)

    def test_tensor_shape(self):
        if False:
            print('Hello World!')
        x = torch.empty(34, 56, 78)

        def f(x):
            if False:
                i = 10
                return i + 15
            return x.shape
        self.checkScript(f, (x,))

    def test_block_input_grad_in_loop(self):
        if False:
            i = 10
            return i + 15
        x = torch.randn(3, 3, requires_grad=False)
        y = torch.randn(3, 3, requires_grad=True)

        def grad_in_loop(x, y):
            if False:
                i = 10
                return i + 15
            for i in range(100):
                x = y @ x
            return x
        scripted = torch.jit.script(grad_in_loop)
        outer = scripted.graph_for(x, y)
        loop = outer.findNode('prim::Loop')
        loop_block = next(loop.blocks())
        param_node = loop_block.paramNode()
        x_value = list(param_node.outputs())[1]
        self.assertTrue(x_value.requires_grad())

    def test_tensor_grad(self):
        if False:
            while True:
                i = 10
        x = torch.randn(3, 4, requires_grad=True)
        y = torch.randn(3, 4, requires_grad=False)

        def f_requires_grad(x):
            if False:
                return 10
            return x.requires_grad
        self.checkScript(f_requires_grad, (x,))
        self.checkScript(f_requires_grad, (y,))

        def f_grad(x):
            if False:
                return 10
            return x.grad
        x.sum().backward()
        self.checkScript(f_grad, (x,))
        self.checkScript(f_grad, (y,))

    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, 'shape analysis is only enabled in Legacy')
    def test_prim_grad_undefined(self):
        if False:
            while True:
                i = 10
        x = torch.ones(2)

        def f_grad(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.grad
        scripted = self.checkScript(f_grad, (x,))
        g = scripted.graph_for(x)
        prim_grad_node = g.findNode('prim::grad')
        self.assertTrue(next(prim_grad_node.outputs()).type().undefined() is None)

    def test_tensor_data(self):
        if False:
            return 10
        x = torch.randn(3, 4, requires_grad=True)
        y = torch.randn(4, 5)

        def f_data(x):
            if False:
                return 10
            return x.data
        scripted_f_data = torch.jit.script(f_data)
        scripted_x = scripted_f_data(x)
        self.assertEqual(scripted_x, f_data(x))
        self.assertEqual(scripted_x.requires_grad, False)
        scripted_y = scripted_f_data(y)
        self.assertEqual(scripted_y, f_data(y))
        self.assertEqual(scripted_x.requires_grad, False)

    def test_tensor_dtype(self):
        if False:
            i = 10
            return i + 15
        x_byte = torch.empty(34, 56, 78, dtype=torch.uint8)
        x_long = torch.empty(34, 56, 78, dtype=torch.long)
        x_float32 = torch.empty(34, 56, 78, dtype=torch.float32)

        @torch.jit.script
        def byte(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.dtype == torch.uint8

        @torch.jit.script
        def long(x):
            if False:
                return 10
            return x.dtype == torch.long

        @torch.jit.script
        def float32(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.dtype == torch.float32
        self.assertTrue(byte(x_byte))
        self.assertFalse(byte(x_long))
        self.assertFalse(byte(x_float32))
        self.assertFalse(long(x_byte))
        self.assertTrue(long(x_long))
        self.assertFalse(long(x_float32))
        self.assertFalse(float32(x_byte))
        self.assertFalse(float32(x_long))
        self.assertTrue(float32(x_float32))

    @unittest.skipIf(not RUN_CUDA, 'device tests require CUDA')
    def test_tensor_device(self):
        if False:
            print('Hello World!')
        cpu = torch.empty(34, 56, 78, device='cpu')
        gpu = torch.empty(34, 56, 78, device='cuda')

        @torch.jit.script
        def same_device(x, y):
            if False:
                while True:
                    i = 10
            return x.device == y.device
        self.assertTrue(same_device(cpu, cpu))
        self.assertTrue(same_device(gpu, gpu))
        self.assertFalse(same_device(cpu, gpu))

    @unittest.skipIf(not RUN_CUDA, 'device tests require CUDA')
    def test_tensor_to_device(self):
        if False:
            print('Hello World!')

        def to_device(x):
            if False:
                print('Hello World!')
            return x.to(device='cuda').to(device=torch.device('cpu'))
        self.checkScript(to_device, (torch.ones(3, 4),))

    def test_tensor_to_cpu(self):
        if False:
            return 10

        def to_cpu(x):
            if False:
                print('Hello World!')
            return x.cpu()
        x = torch.ones(3, 4)
        script_fn = torch.jit.script(to_cpu)
        self.assertEqual(to_cpu(x).device, script_fn(x).device)
        self.checkScript(to_cpu, (x,))

    @unittest.skipIf(not RUN_CUDA, 'device tests require CUDA')
    def test_tensor_to_cuda(self):
        if False:
            i = 10
            return i + 15

        def to_cuda(x):
            if False:
                print('Hello World!')
            return x.cuda()
        x = torch.ones(3, 4)
        script_fn = torch.jit.script(to_cuda)
        self.assertEqual(to_cuda(x).device, script_fn(x).device)
        self.checkScript(to_cuda, (x,))

    def test_generic_list_errors(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(RuntimeError, 'previously matched to type'):

            @torch.jit.script
            def foo(x):
                if False:
                    return 10
                return [[x]] + [[1]]

    def test_script_cu(self):
        if False:
            return 10
        cu = torch.jit.CompilationUnit('\n            def foo(a):\n                b = a\n                return b\n        ')
        a = Variable(torch.rand(1))
        self.assertEqual(a, cu.foo(a))

    def test_string_cu(self):
        if False:
            while True:
                i = 10
        cu = torch.jit.CompilationUnit('\n            def foo(a):\n                print(a, """a\\n\tb\\n""", 2, "aa")\n                return a\n        ')
        FileCheck().check('aa').check('a\\n\\tb\\n').run(str(cu.foo.graph))

    def test_function_compilation_caching(self):
        if False:
            print('Hello World!')

        def fun():
            if False:
                for i in range(10):
                    print('nop')
            return 1 + 2
        fun_compiled = torch.jit.script(fun)
        self.assertIs(fun_compiled.graph, torch.jit.script(fun).graph)

        def fun():
            if False:
                for i in range(10):
                    print('nop')
            return 3 + 4
        num_ref_counts = sys.getrefcount(fun)
        fun_compiled_2 = torch.jit.script(fun)
        self.assertIsNot(fun_compiled, fun_compiled_2)
        self.assertEqual(fun_compiled_2(), 7)
        self.assertTrue(sys.getrefcount(fun), num_ref_counts)

    def test_string_ops(self):
        if False:
            for i in range(10):
                print('nop')

        def foo():
            if False:
                print('Hello World!')
            a = 'a' + 'b'
            return (a + a, 'ab' == 'b', 'ab' != 'b', 'ab' == 'ab', 'ab' != 'ab')
        self.checkScript(foo, ())

    def test_string_sorted(self):
        if False:
            while True:
                i = 10

        def foo(strs: List[str]):
            if False:
                return 10
            return sorted(strs)
        FileCheck().check('graph').check_next('str[] = aten::sorted').check_next('return').run(str(torch.jit.script(foo).graph))
        inputs = ['str3', 'str2', 'str1']
        self.checkScript(foo, (inputs,))

    def test_string_sort(self):
        if False:
            i = 10
            return i + 15

        def foo(strs: List[str]):
            if False:
                while True:
                    i = 10
            strs.sort()
            return strs
        inputs = ['str3', 'str2', 'str1']
        self.checkScript(foo, (inputs,))

    def test_tuple_sorted(self):
        if False:
            while True:
                i = 10

        def foo(tups: List[Tuple[int, int]]):
            if False:
                while True:
                    i = 10
            return sorted(tups)
        inputs = [(1, 2), (0, 2), (1, 3)]
        self.checkScript(foo, (inputs,))

    def test_tuple_sort(self):
        if False:
            while True:
                i = 10

        def foo(tups: List[Tuple[int, int]]):
            if False:
                for i in range(10):
                    print('nop')
            tups.sort()
            return tups
        inputs = [(1, 2), (0, 2), (1, 3)]
        self.checkScript(foo, (inputs,))

    def test_tuple_sort_reverse(self):
        if False:
            print('Hello World!')

        def foo(tups: List[Tuple[int, int]]):
            if False:
                return 10
            tups.sort(reverse=True)
            return tups
        inputs = [(1, 2), (0, 2), (1, 3)]
        self.checkScript(foo, (inputs,))

    def test_tuple_unsortable_element_type(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def foo():
            if False:
                return 10
            tups = [({1: 2}, {2: 3})]
            tups.sort()
            return tups
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'are not sortable', 'tups.sort'):
            foo()

    def test_tuple_unsortable_diff_type(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def foo(inputs: List[Any]):
            if False:
                return 10
            inputs.sort()
            return inputs
        inputs = [(1, 2), ('foo', 'bar')]
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'Only values of same type can be compared', 'inputs.sort'):
            foo(inputs)

    def test_tuple_nested_sort(self):
        if False:
            i = 10
            return i + 15

        def foo(inputs: List[Tuple[int, Tuple[int, str]]]):
            if False:
                while True:
                    i = 10
            inputs.sort()
            return inputs
        inputs = [(1, (2, 'foo')), (1, (2, 'bar')), (1, (0, 'bar'))]
        self.checkScript(foo, (inputs,))

    def test_tuple_unsortable_nested_diff_type(self):
        if False:
            return 10

        @torch.jit.script
        def foo(inputs: List[Any]):
            if False:
                return 10
            inputs.sort()
            return inputs
        inputs = [(1, (2, 3)), (2, ('foo', 'bar'))]
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'Only values of same type can be compared', 'inputs.sort'):
            foo(inputs)

    def test_string_new_line(self):
        if False:
            return 10
        with self.assertRaisesRegex(RuntimeError, 'expected a valid token*'):
            torch.jit.CompilationUnit('\n            def test_while(a):\n                print("\n                    a")\n                return a\n            ')

    def test_string_single_escape(self):
        if False:
            return 10
        with self.assertRaisesRegex(RuntimeError, 'expected a valid token*'):
            torch.jit.CompilationUnit('\n            def test_while(a):\n                print("\\")\n                return a\n            ')

    def test_script_annotation(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def foo(a):
            if False:
                while True:
                    i = 10
            return a + a + a
        s = Variable(torch.rand(2))
        self.assertEqual(s + s + s, foo(s))

    def test_torch_pow(self):
        if False:
            while True:
                i = 10

        def func(a, b):
            if False:
                return 10
            return pow(a, b)

        def func2(a, b, c, d):
            if False:
                for i in range(10):
                    print('nop')
            return pow(pow(c + a, b), d)

        def func3(a: int, b: float):
            if False:
                return 10
            return pow(a, b)

        def func4():
            if False:
                return 10
            return pow(2, -2)

        def func5(x, y):
            if False:
                i = 10
                return i + 15
            return pow(x.item(), y.item())

        def func6(a: int, b: int):
            if False:
                print('Hello World!')
            return pow(a, b)
        a = torch.rand(1)
        b = torch.rand(1)
        c = torch.rand(1)
        d = torch.rand(1)
        self.checkScript(func, (a, b))
        self.checkScript(func2, (a, b, c, d))
        self.checkScript(func3, (4, -0.5))
        self.checkScript(func4, ())
        self.checkScript(func6, (2, 4))
        inputs = [torch.tensor(2), torch.tensor(-2), torch.tensor(0.5), torch.tensor(0.2)]
        for x in inputs:
            for y in inputs:
                if x < 0:
                    continue
                else:
                    self.checkScript(func5, (x, y))

    @unittest.skipIf(not RUN_CUDA, 'device tests require CUDA')
    def test_pow_scalar_backward_cuda(self):
        if False:
            i = 10
            return i + 15
        with enable_profiling_mode_for_profiling_tests():
            for dtype in [torch.float, torch.double]:

                @torch.jit.script
                def func(a, b):
                    if False:
                        i = 10
                        return i + 15
                    return (a * 2) ** b
                a = torch.rand(1, requires_grad=True, device='cuda', dtype=dtype)
                func(a, 1, profile_and_replay=True).backward()

                @torch.jit.script
                def func(a, b):
                    if False:
                        i = 10
                        return i + 15
                    return a ** (b * 2 + 1)
                a = torch.rand(1, requires_grad=True, device='cuda', dtype=dtype)
                func(2, a, profile_and_replay=True).backward()

    def _check_code(self, code_str, fn_name, inputs):
        if False:
            print('Hello World!')
        scope = {}
        exec(code_str, globals(), scope)
        cu = torch.jit.CompilationUnit(code_str)
        self.assertEqual(cu.func(*inputs), scope[fn_name](*inputs))

    @unittest.skipIf(not RUN_CUDA, 'no CUDA')
    def test_scriptmodule_releases_tensors_cuda(self):
        if False:
            while True:
                i = 10
        with enable_profiling_mode_for_profiling_tests():

            @torch.jit.script
            def fn(x, y):
                if False:
                    while True:
                        i = 10
                return x.sigmoid() * y.tanh()

            def test(backward=False):
                if False:
                    print('Hello World!')
                x = torch.randn(3, 3, dtype=torch.double, device='cuda', requires_grad=True)
                y = torch.randn(3, 3, dtype=torch.double, device='cuda', requires_grad=True)
                out = fn(x, y, profile_and_replay=True)
                if backward:
                    out.sum().backward()
            with self.assertLeaksNoCudaTensors():
                test()
                test()
                test()
            if GRAPH_EXECUTOR != ProfilingMode.SIMPLE:
                with self.assertLeaksNoCudaTensors():
                    test(backward=True)
                    test(backward=True)
                    test(backward=True)

    def test_index(self):
        if False:
            return 10

        def consec(size, start=0):
            if False:
                for i in range(10):
                    print('nop')
            numel = torch.tensor(size).prod().item()
            return torch.arange(numel).view(size)

        def consec_list(size):
            if False:
                for i in range(10):
                    print('nop')
            return list(range(size))

        def random_string(size):
            if False:
                while True:
                    i = 10
            letters = string.ascii_lowercase
            return ''.join((random.choice(letters) for i in range(size)))

        def check_indexing(indexing, tensor):
            if False:
                return 10
            template = dedent('\n            def func(x):\n                return x{}\n            ')
            self._check_code(template.format(indexing), 'func', [tensor])

        def check_dynamic_indexing(indexing, tensor, value1, value2):
            if False:
                print('Hello World!')
            value1 = torch.tensor(value1)
            value2 = torch.tensor(value2)
            template = dedent('\n            def func(x, value1, value2):\n                i = int(value1)\n                j = int(value2)\n                return x{}\n            ')
            self._check_code(template.format(indexing), 'func', [tensor, value1, value2])

        def check_indexing_list_int(indexing, list):
            if False:
                print('Hello World!')
            template = dedent('\n            def func(x):\n                # type: (List[int]) -> Any\n                return x{}\n            ')
            self._check_code(template.format(indexing), 'func', [list])

        def check_indexing_str(indexing, str):
            if False:
                for i in range(10):
                    print('nop')
            template = dedent('\n            def func(x):\n                # type: (str) -> Any\n                return x{}\n            ')
            self._check_code(template.format(indexing), 'func', [str])
        check_indexing('[0]', consec((3, 3)))
        check_indexing('[1]', consec((3, 3), 10))
        check_indexing('[2]', consec((3, 3), 19))
        check_indexing('[2]', consec((3,)))
        check_indexing('[-1]', consec((3, 3), 19))
        check_indexing('[0:2]', consec((3, 3, 3)))
        check_indexing('[1:-1]', consec((3, 3, 3)))
        check_indexing('[-3:-1]', consec((6, 3)))
        check_indexing('[1:]', consec((3, 3)))
        check_indexing('[:1]', consec((3, 3)))
        check_indexing('[:]', consec((3, 2)))
        check_indexing('[0, 1]', consec((3, 3)))
        check_indexing('[0, 1]', consec((3, 3, 2)))
        check_indexing('[1, 0, 2]', consec((3, 3, 3)))
        check_indexing('[2, -1]', consec((3, 3)))
        check_indexing('[0, 1:2]', consec((3, 3)))
        check_indexing('[0, :1]', consec((3, 3, 2)))
        check_indexing('[1, 2:]', consec((3, 3, 3)))
        check_indexing('[-1, 1:, 0]', consec((3, 3, 3, 3)))
        check_indexing('[1:, -1, 0]', consec((3, 3, 3, 3)))
        check_indexing('[-1, 2:, 1:2]', consec((3, 3, 3, 3)))
        check_indexing('[-1, 1:, 0]', consec((3, 3, 3, 3)))
        check_indexing('[-1, :, 0, 2]', consec((3, 3, 3, 3)))
        check_indexing('[0:0]', consec((2, 2)))
        check_indexing('[0:0, 1]', consec((3, 3)))
        check_indexing('[1+1]', consec((3, 3)))
        check_indexing('[1:(0 + 2)]', consec((3, 3, 3)))
        check_indexing('[None, 0]', consec((3, 3)))
        check_indexing('[1, None]', consec((3, 3), 10))
        check_indexing('[None, None, 2]', consec((3, 3), 19))
        check_indexing('[None, 2, None]', consec((3,)))
        check_indexing('[0:2, None]', consec((3, 3, 3)))
        check_indexing('[None, 1:-1]', consec((3, 3, 3)))
        check_indexing('[None, -3:-1, None]', consec((6, 3)))
        check_indexing('[-1, None, 2:, None, 1:2]', consec((3, 3, 3, 3)))
        check_indexing('[None, -1, None, 2:, None, 1:2, None]', consec((3, 3, 3, 3)))
        check_dynamic_indexing('[i + j]', consec((3, 3)), 0, 1)
        check_dynamic_indexing('[i:j, i]', consec((3, 3, 2)), 0, 2)
        check_indexing_list_int('[0]', consec_list(6))
        check_indexing_list_int('[1]', consec_list(7))
        check_indexing_list_int('[2]', consec_list(8))
        check_indexing_list_int('[2]', consec_list(9))
        check_indexing_list_int('[-1]', consec_list(10))
        check_indexing_list_int('[0:2]', consec_list(11))
        check_indexing_list_int('[1:-1]', consec_list(12))
        check_indexing_list_int('[-3:-1]', consec_list(13))
        check_indexing_list_int('[1:]', consec_list(15))
        check_indexing_list_int('[:1]', consec_list(16))
        check_indexing_list_int('[:]', consec_list(17))
        check_indexing_list_int('[::]', consec_list(0))
        check_indexing_list_int('[1000::]', consec_list(0))
        check_indexing_list_int('[:1000:]', consec_list(0))
        check_indexing_list_int('[::-1]', consec_list(7))
        check_indexing_list_int('[:3:-1]', consec_list(7))
        check_indexing_list_int('[3::-1]', consec_list(7))
        check_indexing_list_int('[1000::-1]', consec_list(7))
        check_indexing_list_int('[3:0:-1]', consec_list(7))
        check_indexing_list_int('[3:-1000:-1]', consec_list(7))
        check_indexing_list_int('[0:0:-1]', consec_list(7))
        check_indexing_list_int('[0:-1000:-1]', consec_list(7))
        check_indexing_list_int('[::-1]', consec_list(0))
        check_indexing_list_int('[::-1]', consec_list(7))
        check_indexing_list_int('[::-2]', consec_list(7))
        check_indexing_list_int('[::2]', consec_list(7))
        check_indexing_list_int('[::42]', consec_list(7))
        check_indexing_list_int('[::-42]', consec_list(7))
        check_indexing_list_int('[::42]', consec_list(0))
        check_indexing_list_int('[::-42]', consec_list(0))
        check_indexing_list_int('[::9223372036854775807]', consec_list(42))
        check_indexing_list_int('[::-9223372036854775807]', consec_list(42))
        with self.assertRaisesRegex(RuntimeError, 'out of bounds'):
            check_indexing_list_int('[::-9223372036854775808]', consec_list(42))
        with self.assertRaisesRegex(RuntimeError, 'should have non-zero step'):
            check_indexing_list_int('[::0]', consec_list(42))
        check_indexing_str('[0]', random_string(6))
        check_indexing_str('[1]', random_string(7))
        check_indexing_str('[2]', random_string(8))
        check_indexing_str('[2]', random_string(9))
        check_indexing_str('[-1]', random_string(10))
        check_indexing_str('[0:2]', random_string(11))
        check_indexing_str('[1:-1]', random_string(12))
        check_indexing_str('[-3:-1]', random_string(13))
        check_indexing_str('[1:]', random_string(15))
        check_indexing_str('[:1]', random_string(16))
        check_indexing_str('[:]', random_string(17))
        check_indexing_str('[::]', random_string(0))
        check_indexing_str('[1000::]', random_string(0))
        check_indexing_str('[:1000:]', random_string(0))
        check_indexing_str('[::-1]', random_string(7))
        check_indexing_str('[:3:-1]', random_string(7))
        check_indexing_str('[3::-1]', random_string(7))
        check_indexing_str('[1000::-1]', random_string(7))
        check_indexing_str('[3:0:-1]', random_string(7))
        check_indexing_str('[3:-1000:-1]', random_string(7))
        check_indexing_str('[0:0:-1]', random_string(7))
        check_indexing_str('[0:-1000:-1]', random_string(7))
        check_indexing_str('[::-1]', random_string(0))
        check_indexing_str('[::-1]', random_string(7))
        check_indexing_str('[::-2]', random_string(7))
        check_indexing_str('[::2]', random_string(7))
        check_indexing_str('[::42]', random_string(7))
        check_indexing_str('[::-42]', random_string(7))
        check_indexing_str('[::42]', random_string(0))
        check_indexing_str('[::-42]', random_string(0))
        check_indexing_str('[::9223372036854775807]', random_string(42))
        check_indexing_str('[::-9223372036854775807]', random_string(42))
        with self.assertRaisesRegex(RuntimeError, 'out of bounds'):
            check_indexing_str('[::-9223372036854775808]', random_string(42))
        with self.assertRaisesRegex(RuntimeError, 'should have non-zero step'):
            check_indexing_str('[::0]', random_string(42))

    def test_module_copy_with_attributes(self):
        if False:
            return 10

        class Vocabulary(torch.jit.ScriptModule):

            def __init__(self, vocab_list):
                if False:
                    return 10
                super().__init__()
                self._vocab = torch.jit.Attribute(vocab_list, List[str])
                self.some_idx = torch.jit.Attribute(2, int)
                self.idx = torch.jit.Attribute({word: i for (i, word) in enumerate(vocab_list)}, Dict[str, int])

            @torch.jit.script_method
            def lookup_indices_1d(self, values):
                if False:
                    return 10
                result = torch.jit.annotate(List[int], [])
                for i in range(len(values)):
                    value = values[i]
                    result.append(self.idx.get(value, self.some_idx))
                return result

            @torch.jit.script_method
            def forward(self, values):
                if False:
                    for i in range(10):
                        print('nop')
                result = torch.jit.annotate(List[List[int]], [])
                for i in range(len(values)):
                    result.append(self.lookup_indices_1d(values[i]))
                return result
        v = Vocabulary(list('uabcdefg'))
        v.__copy__()

    def test_tuple_to_opt_list(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            return 1

        @torch.jit.script
        def tuple_call():
            if False:
                return 10
            return foo((1, 2))

    def test_keyword(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def func(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.sum(x, dim=0)
        x = torch.rand(10, dtype=torch.float, requires_grad=True)
        y = func(x)
        y2 = torch.sum(x, dim=0)
        self.assertEqual(y, y2)

    def test_constant_pooling_none(self):
        if False:
            return 10

        @torch.jit.script
        def typed_nones(a=None, b=None, c=None):
            if False:
                i = 10
                return i + 15
            return (a, b, c)

        @torch.jit.script
        def test(a):
            if False:
                i = 10
                return i + 15
            if a:
                print(typed_nones())
            else:
                print(typed_nones())
        graph_str = str(test.graph)
        self.assertTrue(graph_str.count('NoneType = prim::Constant') == 1)

    def test_constant_pooling_same_identity(self):
        if False:
            print('Hello World!')

        def foo():
            if False:
                i = 10
                return i + 15
            a = torch.tensor([4])
            b = (a,)
            index = len(a) - 1
            c = b[index]
            d = b[index]
            return (c, d)
        foo_script = torch.jit.script(foo)
        self.run_pass('constant_propagation', foo_script.graph)
        self.run_pass('constant_pooling', foo_script.graph)
        FileCheck().check_count('prim::Constant', 1, exactly=True).run(foo_script.graph)
        self.assertEqual(foo(), foo_script())

    def test_constant_pooling_introduce_aliasing(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def foo():
            if False:
                while True:
                    i = 10
            a = torch.tensor(1)
            b = torch.tensor(1)
            return (a, b)
        self.run_pass('constant_propagation', foo.graph)
        self.run_pass('constant_pooling', foo.graph)
        (a, b) = foo()
        self.assertIsNot(a, b)

    def test_literal(self):
        if False:
            return 10

        def func1(a, b):
            if False:
                return 10
            c = (a, b)
            (d, e) = c
            return d + e

        def func2(a, b):
            if False:
                while True:
                    i = 10
            c = (a, (a, b))
            (d, e) = c
            (f, g) = e
            return d + f + g

        def func3(a, b):
            if False:
                for i in range(10):
                    print('nop')
            c = (0.0, (0.0, 0.0))
            x = True
            while x:
                x = False
                c = (a, (a, b))
            (d, e) = c
            (f, g) = e
            return d + f + g
        a = torch.rand(1, requires_grad=True)
        b = torch.rand(1, requires_grad=True)
        self.checkScript(func1, (a, b), optimize=True)
        self.checkScript(func2, (a, b), optimize=True)
        self.checkScript(func3, (a.item(), b.item()), optimize=True)

    def test_expand(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def func(x, y):
            if False:
                while True:
                    i = 10
            return x + y
        x = torch.rand(2, 3, dtype=torch.float, requires_grad=True)
        y = torch.rand(3, dtype=torch.float, requires_grad=True)
        out = func(x, y)
        self.assertEqual(func(x, y), x + y)
        grad = torch.randn(2, 3, dtype=torch.float)
        out.backward(grad)
        self.assertEqual(x.grad, grad)
        self.assertEqual(y.grad, grad.sum(dim=0))

    def test_sum(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def func(x):
            if False:
                return 10
            return x.sum(dim=[4])

        @torch.jit.script
        def func2(x):
            if False:
                print('Hello World!')
            return x.sum(dim=4)
        self.run_pass('constant_propagation', func.graph)
        self.run_pass('constant_propagation', func2.graph)
        g = _propagate_shapes(func.graph, (torch.zeros(1, 1, 1, 1, 4),), False)
        g2 = _propagate_shapes(func2.graph, (torch.zeros(1, 1, 1, 1, 4),), False)

    def test_cat(self):
        if False:
            return 10
        with enable_profiling_mode_for_profiling_tests():

            @torch.jit.script
            def func(x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.cat((x, x), dim=0)
            x = torch.rand(10, dtype=torch.float, requires_grad=True)
            self.assertEqual(func(x, profile_and_replay=True), torch.cat((x, x), dim=0))

            @torch.jit.script
            def func2(x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.cat((x, x), y)
            with disable_autodiff_subgraph_inlining():
                for sizes in ((2, 2), (0, 2)):
                    x = torch.rand(sizes).requires_grad_()
                    y = torch.tensor(1)
                    output = func2(x, y, profile_and_replay=True)
                    output_ref = torch.cat((x, x), y)
                    self.assertEqual(output, output_ref)
                    if GRAPH_EXECUTOR != ProfilingMode.SIMPLE:
                        self.assertAutodiffNode(func2.graph_for(x, y), True, ['aten::cat'], [])
                        grad = torch.autograd.grad(output.sum(), x)
                        grad_ref = torch.autograd.grad(output_ref.sum(), x)
                        self.assertEqual(grad, grad_ref)

    def test_cat_lifts(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def foo(x):
            if False:
                print('Hello World!')
            return torch.cat([x, x], dim=1)

        @torch.jit.script
        def foo2(x):
            if False:
                while True:
                    i = 10
            return torch.cat([], dim=1)

        @torch.jit.script
        def foo3(x):
            if False:
                i = 10
                return i + 15
            return torch.cat([x], dim=1)
        for g in [foo.graph, foo2.graph, foo3.graph]:
            FileCheck().check('int =').check('ListConstruct').check('aten::cat').run(str(g))

    def test_stack(self):
        if False:
            while True:
                i = 10
        with enable_profiling_mode_for_profiling_tests():

            @torch.jit.script
            def func(x):
                if False:
                    while True:
                        i = 10
                return torch.stack((x, x), dim=1)
            x = torch.rand(10, 10)
            self.assertEqual(func(x, profile_and_replay=True), torch.stack((x, x), dim=1))

            @torch.jit.script
            def func2(x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.stack((x, y), dim=0)
            with disable_autodiff_subgraph_inlining():
                x = torch.randn([2, 2]).requires_grad_()
                y = torch.randn([2, 2]).requires_grad_()
                output = func2(x, y, profile_and_replay=True)
                output_ref = torch.stack((x, y), 0)
                self.assertEqual(output, output_ref)
                if GRAPH_EXECUTOR != ProfilingMode.SIMPLE:
                    self.assertAutodiffNode(func2.graph_for(x, y), True, ['aten::stack'], [])
                    grads = torch.autograd.grad(output.sum(), (x, y))
                    grads_ref = torch.autograd.grad(output_ref.sum(), (x, y))
                    self.assertEqual(grads, grads_ref)

    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, 'Profiling executor will be using different heuristics for constructing differentiable graphs')
    def test_unbind(self):
        if False:
            print('Hello World!')
        with enable_profiling_mode_for_profiling_tests():

            @torch.jit.script
            def func(x, y):
                if False:
                    return 10
                return torch.unbind(x, y)
            with disable_autodiff_subgraph_inlining():
                x = torch.rand([2, 2]).requires_grad_()
                y = 0
                outputs = func(x, y, profile_and_replay=True)
                outputs_ref = torch.unbind(x, dim=y)
                self.assertEqual(outputs, outputs_ref)
                self.assertAutodiffNode(func.graph_for(x, y), True, [], [])
                grad = torch.autograd.grad(_sum_of_list(outputs), x)
                grad_ref = torch.autograd.grad(_sum_of_list(outputs_ref), x)
                self.assertEqual(grad, grad_ref)

    @unittest.skipIf(GRAPH_EXECUTOR == ProfilingMode.PROFILING, 'Profiling executor fails to recognize that tensors in a list require gradients')
    def test_meshgrid(self):
        if False:
            for i in range(10):
                print('nop')
        with enable_profiling_mode_for_profiling_tests():

            @torch.jit.script
            def func(a):
                if False:
                    i = 10
                    return i + 15
                return torch.meshgrid(a)
            with disable_autodiff_subgraph_inlining():
                a = torch.tensor([1.0, 2, 3]).requires_grad_()
                b = torch.tensor([1.0, 2, 3, 4]).requires_grad_()
                inputs = [a, b]
                outputs_ref = torch.meshgrid(inputs)
                outputs = func(inputs, profile_and_replay=True)
                self.assertEqual(outputs, outputs_ref)
                if GRAPH_EXECUTOR != ProfilingMode.SIMPLE:
                    self.assertAutodiffNode(func.graph_for(inputs), True, [], [])
                    grads = torch.autograd.grad(_sum_of_list(outputs), inputs)
                    grads_ref = torch.autograd.grad(_sum_of_list(outputs_ref), inputs)
                    self.assertEqual(grads, grads_ref)

    def test_tensor_len(self):
        if False:
            print('Hello World!')

        def func(x):
            if False:
                return 10
            return len(x)
        self.checkScript(func, [torch.ones(4, 5, 6)])

    def test_func_call(self):
        if False:
            return 10

        def add(a, b):
            if False:
                return 10
            return a + b

        def mul(a, x):
            if False:
                return 10
            return a * x

        def func(alpha, beta, x, y):
            if False:
                return 10
            return add(mul(alpha, x), mul(beta, y))
        alpha = torch.rand(1, dtype=torch.float, requires_grad=True)
        beta = torch.rand(1, dtype=torch.float, requires_grad=True)
        x = torch.rand(3, dtype=torch.float, requires_grad=True)
        y = torch.rand(3, dtype=torch.float, requires_grad=True)
        self.checkScript(func, [alpha, beta, x, y], optimize=False)

    @unittest.skip('bailouts are being deprecated')
    def test_profiling_graph_executor(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def def_in_one_branch(x, z):
            if False:
                for i in range(10):
                    print('nop')
            y = x
            if z is False:
                y = x + 1
            return y.sum()
        a = torch.rand(2, 3)
        with enable_profiling_mode_for_profiling_tests():
            profiled_graph_str = str(def_in_one_branch.graph_for(a, True))
            FileCheck().check_count('prim::profile', 4).run(profiled_graph_str)
            def_in_one_branch(a, False)
            a = torch.ones(3)
            bailout_graph_str = str(def_in_one_branch.graph_for(a, True))
            FileCheck().check_count('prim::BailOut', 3).run(bailout_graph_str)
            self.assertEqual(def_in_one_branch(a, False), 6.0)
            self.assertEqual(def_in_one_branch(a, True), 3.0)

    @unittest.skip('bailouts are being deprecated')
    def test_maxpool_guard_elimination(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def my_maxpool(x):
            if False:
                print('Hello World!')
            return F.max_pool1d(x, kernel_size=[1]) + torch.ones([32, 32, 32])
        a = torch.rand(32, 32, 32)
        with enable_profiling_mode_for_profiling_tests():
            my_maxpool(a)
            bailout_graph_str = str(my_maxpool.graph_for(a))
            FileCheck().check_count('prim::BailOut', 1).run(bailout_graph_str)

    @unittest.skip('bailouts are being deprecated')
    def test_slice_guard_elimination(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def my_slice(x):
            if False:
                while True:
                    i = 10
            return x[0:16:2] + x[0:16:2]
        a = torch.rand(32, 4)
        with enable_profiling_mode_for_profiling_tests():
            my_slice(a)
            bailout_graph_str = str(my_slice.graph_for(a))
            FileCheck().check_count('prim::BailOut', 1).run(bailout_graph_str)

    @unittest.skip('bailouts are being deprecated')
    def test_unsqueeze_guard_elimination(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def my_unsqueeze(x):
            if False:
                return 10
            return torch.unsqueeze(x, 0) + torch.unsqueeze(x, 0)
        a = torch.rand(32, 4)
        with enable_profiling_mode_for_profiling_tests():
            my_unsqueeze(a)
            bailout_graph_str = str(my_unsqueeze.graph_for(a))
            FileCheck().check_count('prim::BailOut', 2).run(bailout_graph_str)

    def test_resize_input_ops(self):
        if False:
            for i in range(10):
                print('nop')

        def out_op_graph_input():
            if False:
                for i in range(10):
                    print('nop')

            @torch.jit.script
            def test(x, y, z):
                if False:
                    print('Hello World!')
                torch.mul(x, y, out=z)
                return z
            graph = _propagate_shapes(test.graph, (torch.zeros(2, 1), torch.zeros(1, 2), torch.zeros(1, 1, 1)), False)
            self.assertTrue(next(graph.outputs()).type() == TensorType.get())
        out_op_graph_input()

        def test_resize():
            if False:
                print('Hello World!')

            @torch.jit.script
            def test(x):
                if False:
                    return 10
                after_resize_alias = torch.zeros([2])
                for _i in range(5):
                    b = x + 1
                    f = [1]
                    before_resize_alias = b.sub_(1)
                    f.append(1)
                    b.resize_(f)
                    after_resize_alias = b.add_(1)
                return after_resize_alias
            self.run_pass('constant_propagation', test.graph)
            g = _propagate_shapes(test.graph, (torch.zeros(1, 1),), False)
            resize_node = g.findNode('aten::resize_')
            self.assertTrue(next(resize_node.inputs()).type() == TensorType.get())
            self.assertTrue(next(resize_node.outputs()).type() == TensorType.get())
            before_resize = g.findNode('aten::sub_')
            self.assertTrue(next(before_resize.outputs()).type() == TensorType.get())
            after_resize = g.findNode('aten::add_')
            self.assertTrue(next(after_resize.outputs()).type() == TensorType.get())
        test_resize()

        def test_resize_as():
            if False:
                i = 10
                return i + 15

            @torch.jit.script
            def test(x):
                if False:
                    i = 10
                    return i + 15
                b = torch.zeros([2, 2])
                b.resize_as_(x)
                return b
            g = test.graph
            self.run_pass('constant_propagation', g)
            g = _propagate_shapes(test.graph, (torch.zeros(1, 1),), False)
            self.assertTrue(next(g.inputs()).type() != TensorType.get())
            self.assertTrue(next(g.outputs()).type() == TensorType.get())
        test_resize_as()

    def test_uninitialized(self):
        if False:
            i = 10
            return i + 15
        graph_str = 'graph():\n          %1 : int = prim::Uninitialized()\n          %2 : int = prim::Constant[value=1]()\n          %3 : int = aten::add(%1, %2)\n          return (%3)\n        '
        g = parse_ir(graph_str)
        m = self.createFunctionFromGraph(g)
        self.getExportImportCopy(m)
        with self.assertRaisesRegex(RuntimeError, 'isInt'):
            m()

    @unittest.skipIf(GRAPH_EXECUTOR == ProfilingMode.SIMPLE, "Simple Executor doesn't use requires_grad information")
    @unittest.skipIf(GRAPH_EXECUTOR == ProfilingMode.PROFILING, 'Peeling is now disabled')
    def test_requires_grad_loop(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def test(x, y, z):
            if False:
                while True:
                    i = 10
            for _ in range(z):
                x = y
            return x
        inps = (torch.tensor(1.0, requires_grad=True), torch.tensor(1), 10)
        test(*inps, profile_and_replay=True)
        graph = test.graph_for(*inps)
        loop = graph.findNode('prim::Loop')
        loop_body = next(loop.blocks())
        loop_inputs = list(loop_body.inputs())
        loop_outputs = list(loop_body.outputs())
        if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
            index_of_x_in_peeled_unrolled_loop = -2
            self.assertTrue(loop_inputs[index_of_x_in_peeled_unrolled_loop].requires_grad())
            bailouts_in_outer_block = graph.findAllNodes('prim::BailOut', False)
            last_bailout_index_on_loops_output = -1
            self.assertFalse(bailouts_in_outer_block[last_bailout_index_on_loops_output].output().requires_grad())
        else:
            self.assertTrue(loop_inputs[1].requires_grad())
            self.assertTrue(loop.output().requires_grad())
            self.assertFalse(loop_outputs[1].requires_grad())

    def test_view_shape_prop(self):
        if False:
            while True:
                i = 10
        cu = torch.jit.CompilationUnit('\n        def test_view_shape_prop(a):\n            return a.view(size=[-1])\n        ')
        inputs = [torch.zeros(10, 10)]
        outputs = torch.zeros(100)
        real_outs = cu.test_view_shape_prop(*inputs)
        self.assertEqual(real_outs, outputs)

    @skipIfTorchDynamo('TorchDynamo fails with unknown reason')
    def test_view_listconstruct_shape_prop(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(x):
            if False:
                while True:
                    i = 10
            B = x.size(0)
            C = x.size(1)
            T = x.size(2)
            return x.view(T, B, C)
        x = torch.randn(3, 1, 5, requires_grad=True)
        fn = torch.jit.script(fn)
        graph = _propagate_shapes(fn.graph, (x,), False)
        self.assertTrue(next(graph.outputs()).type().scalarType() == 'Float')

    def test_shape_prop_promotion(self):
        if False:
            return 10

        @torch.jit.script
        def fn(x, y):
            if False:
                while True:
                    i = 10
            return x + y
        (x, y) = (torch.rand(3, 4, dtype=torch.float), torch.rand(3, 4, dtype=torch.double))
        graph = _propagate_shapes(fn.graph, (x, y), False)
        FileCheck().check('Double(*, *, device=cpu) = aten::add').run(graph)

    def test_shape_prop_promote_scalar_arg(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def fn(x):
            if False:
                while True:
                    i = 10
            return math.pi + x
        x = torch.zeros(3, 4, dtype=torch.long)
        graph = _propagate_shapes(fn.graph, (x,), False)
        default = torch.get_default_dtype()
        if default == torch.float:
            FileCheck().check('Float(*, *, requires_grad=0, device=cpu) = aten::add').run(graph)
        else:
            FileCheck().check('Double(*, *, requires_grad=0, device=cpu) = aten::add').run(graph)

    def test_integral_shape_inference(self):
        if False:
            return 10
        cu = torch.jit.CompilationUnit('\n        def test_integral_shape_inference(a):\n            return a * a\n        ')
        inputs = [torch.ones(10, 10, dtype=torch.long)]
        outputs = torch.ones(10, 10, dtype=torch.long)
        self.assertEqual(cu.test_integral_shape_inference(*inputs), outputs)

    @unittest.skipIf(RUN_CUDA, 'This tests the CPU fuser')
    @unittest.skipIf(IS_SANDCASTLE, 'NYI: fuser support for Sandcastle')
    @enable_cpu_fuser
    def test_batchnorm_fuser_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        code = '\n            graph(%3 : Tensor,\n                  %7 : Tensor,\n                  %12 : Float(*, *),\n                  %13 : Tensor,\n                  %25 : Tensor):\n                %23 : int = prim::Constant[value=1]()\n                %22 : float = prim::Constant[value=1e-05]()\n                %26 : Tensor = aten::sqrt(%25)\n                %24 : Tensor = aten::add(%26, %22, %23)\n                %20 : Tensor = aten::reciprocal(%24)\n                %norm_invstd : Tensor = aten::mul(%20, %23)\n                %15 : Tensor = aten::sub(%12, %13, %23)\n                %11 : Tensor = aten::mul(%15, %norm_invstd)\n                %8 : Tensor = aten::mul(%11, %7)\n                %5 : Tensor = aten::add(%8, %3, %23)\n                %1 : Float(*, *) = aten::relu(%5)\n                return (%1)\n        '
        graph = parse_ir(code)
        inputs = 5 * [torch.rand(26, 2048, dtype=torch.float)]
        code = torch._C._jit_fuser_get_fused_kernel_code(graph, inputs)
        FileCheck().check('sqrtf').run(code)

    @slowTest
    @unittest.skipIf(RUN_CUDA, 'This tests the CPU fuser')
    @unittest.skipIf(IS_SANDCASTLE, 'NYI: fuser support for Sandcastle')
    @enable_cpu_fuser
    def test_fuser_double_float_codegen(self):
        if False:
            print('Hello World!')
        fns = ['log', 'log10', 'log1p', 'log2', 'lgamma', 'exp', 'expm1', 'erf', 'erfc', 'cos', 'acos', 'cosh', 'sin', 'asin', 'sinh', 'tan', 'atan', 'tanh', 'sqrt', 'ceil', 'floor', 'round', 'trunc', 'frac']

        def lookup_c_equivalent_fn(aten_fn):
            if False:
                i = 10
                return i + 15
            return aten_fn

        def test_dispatch(op, expects, dtype, binary=False):
            if False:
                i = 10
                return i + 15
            if dtype == torch.double:
                dtype_str = 'Double'
            elif dtype == torch.float:
                dtype_str = 'Float'
            else:
                raise RuntimeError('Unknown dtype')
            if binary:
                code = f'\n                    graph(%3 : Tensor, %4 : Tensor):\n                        %2 : {dtype_str}(*, *) = aten::{op}(%3, %4)\n                        %1 : {dtype_str}(*, *) = aten::relu(%2)\n                        return (%1)\n                '
            else:
                code = f'\n                    graph(%3 : Tensor):\n                        %2 : {dtype_str}(*, *) = aten::{op}(%3)\n                        %1 : {dtype_str}(*, *) = aten::relu(%2)\n                        return (%1)\n                '
            graph = parse_ir(code)
            inputs = (2 if binary else 1) * [torch.rand(26, 2048, dtype=dtype)]
            code = torch._C._jit_fuser_get_fused_kernel_code(graph, inputs)
            FileCheck().check(expects).run(code)
        for fn in fns:
            test_dispatch(fn, lookup_c_equivalent_fn(fn) + '(', torch.double)
            test_dispatch(fn, lookup_c_equivalent_fn(fn) + 'f(', torch.float)
        binary_fns = ['pow']
        for fn in binary_fns:
            test_dispatch(fn, lookup_c_equivalent_fn(fn) + '(', torch.double, binary=True)
            test_dispatch(fn, lookup_c_equivalent_fn(fn) + 'f(', torch.float, binary=True)

    @unittest.skipIf(RUN_CUDA, 'This tests the CPU fuser')
    @unittest.skipIf(IS_SANDCASTLE, 'NYI: fuser support for Sandcastle')
    @enable_cpu_fuser
    def test_fuser_double_literal_precision(self):
        if False:
            while True:
                i = 10
        code = '\n        graph(%2 : Float(*, *)):\n            %4 : int = prim::Constant[value=1]()\n            %3 : float = prim::Constant[value=1.282549830161864]()\n            %5 : Float(*, *) = aten::add(%2, %3, %4)\n            %1 : Float(*, *) = aten::relu(%5)\n            return (%1)\n        '
        graph = parse_ir(code)
        code = torch._C._jit_fuser_get_fused_kernel_code(graph, [torch.rand(3, 4)])
        FileCheck().check('1.282549830161864').run(code)

    def test_fuser_multiple_blocks(self):
        if False:
            print('Hello World!')
        cu = torch.jit.CompilationUnit('\n        def test_fuser_multiple_blocks(this, that, theother, meme):\n            i = 0\n            while i < 20:\n                this = torch.cat([this, meme], dim=0)\n                that = torch.cat([that, meme], dim=0)\n                theother = torch.cat([theother, meme], dim=0)\n                i = i + 1\n            return this, that, theother\n        ')
        inputs = [torch.ones(0, 10, 10)] * 3
        inputs += [torch.ones(1, 10, 10)]
        outputs = [torch.ones(20, 10, 10)] * 3
        self.assertEqual(cu.test_fuser_multiple_blocks(*inputs), outputs)

    @unittest.skip('RuntimeError: VariableType::ID() not implemented')
    def test_cast(self):
        if False:
            i = 10
            return i + 15
        script = '\n        def to_int(x):\n            return int(x)\n        '
        x = Variable(torch.FloatTensor([1.1, 2.3]), requires_grad=True)
        out = Variable(torch.IntTensor([1, 2]), requires_grad=True)
        self.checkScript(script, [x], optimize=True, outputs=[out], func='to_int')

    def test_str_cast(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def to_str(x):
            if False:
                print('Hello World!')
            return str((x, x))
        self.assertEqual('(1, 1)', to_str(1))

    def test_int_cast(self):
        if False:
            return 10

        @torch.jit.script
        def to_int(x):
            if False:
                print('Hello World!')
            return int(x)
        self.assertEqual(5, to_int('5'))
        self.assertEqual(-5, to_int('-5'))
        self.assertEqual(2147483647, to_int('2147483647'))
        self.assertEqual(-2147483648, to_int('-2147483648'))
        with self.assertRaisesRegex(RuntimeError, 'invalid literal for int()'):
            to_int('0x20')
        with self.assertRaisesRegex(RuntimeError, 'invalid literal for int()'):
            to_int('0b0001')

    def test_python_frontend(self):
        if False:
            return 10

        def fn(x, y, z):
            if False:
                while True:
                    i = 10
            q = None
            q = x + y - z.sigmoid()
            print(q)
            w = -z
            if not x and (not y) and z:
                m = x if not z else y
            while x < y > z:
                q = x
            assert 1 == 1, 'hello'
            return x
        ast = torch.jit.frontend.get_jit_def(fn, fn.__name__)
        self.assertExpected(str(ast))

    def test_python_frontend_source_range(self):
        if False:
            print('Hello World!')

        def fn():
            if False:
                print('Hello World!')
            raise Exception('hello')
        ast = torch.jit.frontend.get_jit_def(fn, fn.__name__)
        FileCheck().check('SourceRange at:').check('def fn():').check('~~~~~~~~~').check('raise Exception("hello")').check('~~~~~~~~~~~~~~~~~ <--- HERE').run(str(ast.range()))

    def test_python_frontend_py3(self):
        if False:
            for i in range(10):
                print('nop')

        def fn():
            if False:
                print('Hello World!')
            raise Exception('hello')
        ast = torch.jit.frontend.get_jit_def(fn, fn.__name__)
        self.assertExpected(str(ast))

    def _make_scalar_vars(self, arr, dtype):
        if False:
            for i in range(10):
                print('nop')
        return [torch.tensor(val, dtype=dtype) for val in arr]

    def test_string_print(self):
        if False:
            i = 10
            return i + 15

        def func(a):
            if False:
                for i in range(10):
                    print('nop')
            print(a, 'abcd', 2, 1.5)
            return a
        inputs = self._make_scalar_vars([1], torch.int64)
        self.checkScript(func, inputs, capture_output=True)

    def test_while(self):
        if False:
            i = 10
            return i + 15

        def func(a, b, max):
            if False:
                while True:
                    i = 10
            while bool(a < max):
                a = a + 1
                b = b + 1
            c = a + b
            return c
        inputs = self._make_scalar_vars([1, 1, 10], torch.int64)
        self.checkScript(func, inputs, optimize=True)

    def test_fibb(self):
        if False:
            for i in range(10):
                print('nop')

        def func(lim):
            if False:
                for i in range(10):
                    print('nop')
            first = 1
            second = 1
            i = 1
            somenum = 5
            dontmutateme = 3
            third = 0
            while bool(i < lim):
                third = first + second
                first = second
                second = third
                j = 0
                while j < 10:
                    somenum = somenum * 2
                    j = j + 1
                i = i + j
                i = i + dontmutateme
            st = second + third
            fs = first + second
            return (third, st, fs)
        inputs = self._make_scalar_vars([10], torch.int64)
        self.checkScript(func, inputs, optimize=True)

    def test_fibb_totally_better(self):
        if False:
            while True:
                i = 10

        def fib(x):
            if False:
                while True:
                    i = 10
            prev = 1
            v = 1
            for i in range(0, x):
                save = v
                v = v + prev
                prev = save
            return v
        self.checkScript(fib, (10,))

    def test_if(self):
        if False:
            print('Hello World!')

        def func(a, b):
            if False:
                for i in range(10):
                    print('nop')
            d = 3
            if bool(a > 10):
                a = 3 + d
            else:
                b = 3 + d
                d = 4
            c = a + b
            return c
        inputs = self._make_scalar_vars([1, -1], torch.int64)
        self.checkScript(func, inputs, optimize=True)

    def test_if_for_in_range(self):
        if False:
            return 10

        def func(a, b):
            if False:
                return 10
            d = 3
            for _ in range(20):
                if bool(a > 10):
                    a = 3 + d
                else:
                    b = 3 + d
                    d = 4
                c = a + b
            return d
        inputs = self._make_scalar_vars([1, -1], torch.int64)
        self.checkScript(func, inputs, optimize=True)

    def test_if_noelse(self):
        if False:
            i = 10
            return i + 15

        def func(a, b):
            if False:
                i = 10
                return i + 15
            if bool(a > 10):
                a = 3 + b
            c = a + b
            return c
        inputs = self._make_scalar_vars([-1, 1], torch.int64)
        self.checkScript(func, inputs, optimize=True)

    def test_if_is_none_dispatch(self):
        if False:
            return 10

        @torch.jit.script
        def test_lhs_none_rhs_none():
            if False:
                print('Hello World!')
            if None is None:
                return 1
            elif None is not None:
                return 2
            else:
                return 3
        self.assertTrue(str(test_lhs_none_rhs_none.graph).count(': int = prim::Constant') == 1)

        @torch.jit.script
        def test_lhs_opt_rhs_none(lhs=None):
            if False:
                i = 10
                return i + 15
            if lhs is not None:
                return 2
            elif lhs is None:
                return 1
            else:
                return 3
        self.assertTrue(str(test_lhs_opt_rhs_none.graph).count(': int = prim::Constant') == 3)

        @torch.jit.script
        def test_lhs_none_rhs_opt(rhs=None):
            if False:
                for i in range(10):
                    print('nop')
            if None is rhs:
                return 1
            elif None is not rhs:
                return 2
            else:
                return 3
        self.assertTrue(str(test_lhs_opt_rhs_none.graph).count(': int = prim::Constant') == 3)

        @torch.jit.script
        def test_lhs_never_rhs_none(lhs):
            if False:
                i = 10
                return i + 15
            if lhs is None:
                return 1
            elif lhs is not None:
                return 2
            else:
                return 3
        self.assertTrue(str(test_lhs_never_rhs_none.graph).count(': int = prim::Constant') == 1)

        @torch.jit.script
        def test_lhs_none_rhs_never(rhs):
            if False:
                i = 10
                return i + 15
            if None is rhs:
                return 1
            elif None is not rhs:
                return 2
            else:
                return 3
        self.assertTrue(str(test_lhs_none_rhs_never.graph).count(': int = prim::Constant') == 1)

        @torch.jit.script
        def test_bool_arith_and(lhs):
            if False:
                print('Hello World!')
            if lhs is None and lhs is not None:
                return 1
            else:
                return 2
        self.assertEqual(test_bool_arith_and(torch.zeros(3)), 2)
        self.assertTrue(str(test_bool_arith_and.graph).count('if') == 0)

        @torch.jit.script
        def test_bool_arith_or(lhs):
            if False:
                i = 10
                return i + 15
            if lhs is None or lhs is not None:
                return 1
            else:
                return 2
        self.assertEqual(test_bool_arith_or(torch.zeros(3)), 1)
        self.assertTrue(str(test_bool_arith_or.graph).count('if') == 0)

        @torch.jit.script
        def test_bool_arith_not(lhs):
            if False:
                for i in range(10):
                    print('nop')
            if lhs is not None:
                return 1
            else:
                return 2
        self.assertEqual(test_bool_arith_not(torch.zeros(3)), 1)
        self.assertTrue(str(test_bool_arith_not.graph).count('if') == 0)

    def test_conditional_casting(self):
        if False:
            for i in range(10):
                print('nop')

        def test_bool_cast_tensor(x):
            if False:
                i = 10
                return i + 15
            if x:
                return 1
            else:
                return 0
        for make_one_dim in [True, False]:
            for inp_val in [0.1, 0.0, -0.0, -0.1, -1, 0, 1]:
                inp_val = [inp_val] if make_one_dim else inp_val
                self.checkScript(test_bool_cast_tensor, (torch.tensor(inp_val),))
        self.checkScriptRaisesRegex(test_bool_cast_tensor, (torch.tensor([1, 1]),), Exception, 'Boolean value of Tensor with more than one value')

        def test_not_cast(x):
            if False:
                print('Hello World!')
            if not x:
                return 1
            else:
                return 0
        self.checkScript(test_not_cast, (torch.tensor(1),))
        self.checkScript(test_not_cast, (torch.tensor(0),))
        with self.assertRaisesRegex(RuntimeError, 'Could not cast value of type Tuple\\[Tensor, Tensor\\]'):

            @torch.jit.script
            def test_mult(x, y):
                if False:
                    while True:
                        i = 10
                return not (x, y)

        def test_cast_int(x):
            if False:
                i = 10
                return i + 15
            if x:
                return 1
            else:
                return 0
        self.checkScript(test_cast_int, (1,))
        self.checkScript(test_cast_int, (0,))
        self.checkScript(test_cast_int, (-1,))

        def test_cast_float(x):
            if False:
                for i in range(10):
                    print('nop')
            if x:
                return 1
            else:
                return 0
        self.checkScript(test_cast_float, (1.0,))
        self.checkScript(test_cast_float, (0.0,))
        self.checkScript(test_cast_float, (-1.0,))
        with self.assertRaisesRegex(RuntimeError, 'Could not cast value of type Tuple\\[int, int\\] to bool'):

            @torch.jit.script
            def test_bad_conditional(x):
                if False:
                    return 10
                if (1, 2):
                    return
                else:
                    return 0

    def test_while_nonexistent_value(self):
        if False:
            return 10
        with self.assertRaisesRegex(RuntimeError, 'undefined value x'):
            torch.jit.CompilationUnit('\n            def test_while(a, b):\n                while bool(a < 10):\n                    a = a + x\n                    b = b + 1\n                return a + b\n            ')

    def test_while_nonexistent_cond_value(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(RuntimeError, 'undefined value x'):
            torch.jit.CompilationUnit('\n            def test_while(a, b):\n                while a < x:\n                    a = a + 1\n                    b = b + 1\n                return a + b\n            ')

        @torch.jit.script
        def test_ternary(x):
            if False:
                return 10
            x = x if x is not None else 2
            return x

        @torch.jit.script
        def test_not_none(x):
            if False:
                print('Hello World!')
            if x is not None:
                print(x + 1)

        @torch.jit.script
        def test_and(x, y):
            if False:
                return 10
            if x is not None and y is not None:
                print(x + y)

        @torch.jit.script
        def test_not(x, y):
            if False:
                i = 10
                return i + 15
            if not (x is not None and y is not None):
                pass
            else:
                print(x + y)

        @torch.jit.script
        def test_bool_expression(x):
            if False:
                while True:
                    i = 10
            if x is not None and x < 2:
                print(x + 1)

        @torch.jit.script
        def test_nested_bool_expression(x, y):
            if False:
                return 10
            if x is not None and x < 2 and (y is not None):
                x = x + y
            else:
                x = 5
            return x + 2

        @torch.jit.script
        def test_or(x, y):
            if False:
                print('Hello World!')
            if y is None or x is None:
                pass
            else:
                print(x + y)

        @torch.jit.script
        def test_manual_unwrap_opt(x):
            if False:
                for i in range(10):
                    print('nop')
            if x is None:
                x = 1
            else:
                x = torch.jit._unwrap_optional(x)
            return x
        with self.assertRaisesRegex(RuntimeError, 'Arguments for call are not valid'):

            @torch.jit.script
            def or_error(x, y):
                if False:
                    print('Hello World!')
                if x is None or y is None:
                    print(x + y)
        with self.assertRaisesRegex(RuntimeError, 'Arguments for call are not valid'):

            @torch.jit.script
            def and_error(x, y):
                if False:
                    while True:
                        i = 10
                if x is None and y is None:
                    pass
                else:
                    print(x + y)
        with self.assertRaisesRegex(RuntimeError, 'Arguments for call are not valid'):

            @torch.jit.script
            def named_var(x):
                if False:
                    print('Hello World!')
                x_none = x is not None
                if x_none:
                    print(x + 1)
        with self.assertRaisesRegex(RuntimeError, 'Arguments for call are not valid'):

            @torch.jit.script
            def named_var_and(x, y):
                if False:
                    return 10
                x_none = x is not None
                if y is not None and x_none:
                    print(x + y)

    def test_assertion_optional_refinement(self):
        if False:
            return 10

        @torch.jit.script
        def test(x, y):
            if False:
                i = 10
                return i + 15
            assert x is not None and y is not None
            return x + y
        self.assertEqual(test(2, 2), 4)
        with self.assertRaisesRegex(Exception, ''):
            test(1, None)

    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "the current version of Profiler doesn't profile/specialize Optionals")
    def test_optional_tensor(self):
        if False:
            return 10

        @torch.jit.script
        def fn(x, y):
            if False:
                for i in range(10):
                    print('nop')
            if x is None:
                return y
            else:
                return 0
        res = fn(None, 1)
        self.assertEqual(res, 1)
        g = torch.jit.last_executed_optimized_graph()
        first_input = next(g.inputs())
        self.assertEqual(first_input.type().kind(), 'OptionalType')
        self.assertEqual(first_input.uses(), [])
        t = torch.ones(1)
        res = fn(t, 1)
        self.assertEqual(res, 0)
        g = torch.jit.last_executed_optimized_graph()
        self.assertEqual(next(g.inputs()).type().kind(), 'TensorType')

        @torch.jit.script
        def fn(x, y, b):
            if False:
                i = 10
                return i + 15
            if b:
                res = y
            else:
                res = torch.jit._unwrap_optional(x)
            return res
        t2 = torch.zeros(1)
        res = fn(t, t2, True)
        self.assertEqual(res, t2)
        with self.assertRaisesRegex(RuntimeError, 'Unwrapping null optional'):
            res = fn(None, t2, False)
        res = fn(None, t2, True)
        g = torch.jit.last_executed_optimized_graph()
        self.assertIn(next(g.outputs()).type().str(), ('Tensor', 'Tensor(requires_grad=1)'))

    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "the current version of Profiler doesn't profile/specialize Optionals")
    def test_optional_list(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def fn(x, y):
            if False:
                return 10
            if x is None:
                return y
            else:
                res = 0
                for d in x:
                    res += d
                return res
        res = fn(None, 1)
        self.assertEqual(res, 1)
        g = torch.jit.last_executed_optimized_graph()
        first_input = next(g.inputs())
        self.assertEqual(first_input.type().kind(), 'OptionalType')
        self.assertEqual(first_input.uses(), [])
        l = [2, 3]
        res = fn(l, 1)
        self.assertEqual(res, 5)
        g = torch.jit.last_executed_optimized_graph()
        self.assertEqual(next(g.inputs()).type().kind(), 'ListType')

        @torch.jit.script
        def fn(x, y, b):
            if False:
                for i in range(10):
                    print('nop')
            if b:
                l = torch.jit._unwrap_optional(x)
            else:
                l = y
            return l
        l2 = [0, 1]
        res = fn(l, l2, True)
        self.assertEqual(res, l)
        with self.assertRaisesRegex(RuntimeError, 'Unwrapping null optional'):
            res = fn(None, l2, True)
        res = fn(None, l2, False)
        g = torch.jit.last_executed_optimized_graph()
        self.assertEqual(next(g.outputs()).type().str(), 'int[]')

    def test_alias_covariant_type_containers(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def foo(x):
            if False:
                while True:
                    i = 10
            if x:
                a = (None,)
            else:
                a = ([],)
            return a

        @torch.jit.script
        def foo2(x, li):
            if False:
                for i in range(10):
                    print('nop')
            if x:
                li = (None,)
            return li

    def test_while_write_outer_then_read(self):
        if False:
            while True:
                i = 10

        def func(a, b):
            if False:
                return 10
            while bool(a < 10):
                a = a + 1
                b = a + 1
            return a + b
        inputs = self._make_scalar_vars([42, 1337], torch.int64)
        self.checkScript(func, inputs, optimize=True)

    @skipIfTorchDynamo('TorchDynamo fails with unknown reason')
    def test_while_nest_if(self):
        if False:
            for i in range(10):
                print('nop')

        def func(a, b):
            if False:
                for i in range(10):
                    print('nop')
            c = 0
            while a < 10:
                a = a + 1
                b = b + 1
                if a > b:
                    c = -a
                else:
                    c = -b
            return c + 1
        inputs = self._make_scalar_vars([-1234, 4321], torch.int64)
        self.checkScript(func, inputs, optimize=True)

    def test_divmod(self):
        if False:
            for i in range(10):
                print('nop')

        def func_int(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return divmod(a, b)

        def func_float(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return divmod(a, b)

        def func_int_float(a, b):
            if False:
                i = 10
                return i + 15
            return divmod(a, b)

        def func_float_int(a, b):
            if False:
                print('Hello World!')
            return divmod(a, b)

        def divmod_test_iterator(func, num, den):
            if False:
                return 10
            for i in num:
                for j in den:
                    self.checkScript(func, (i, j), frames_up=2)
        num_int = [1024, -1024]
        den_int = [10, -10]
        num_float = [5.3, -5.3]
        den_float = [2.0, -2.0]
        divmod_test_iterator(func_int, num_int, den_int)
        divmod_test_iterator(func_float, num_float, den_float)
        divmod_test_iterator(func_int_float, num_int, den_float)
        divmod_test_iterator(func_float_int, num_float, den_int)
        with self.assertRaisesRegex(RuntimeError, 'ZeroDivisionError: integer division or modulo by zero'):
            cu = torch.jit.CompilationUnit(dedent(inspect.getsource(func_int)))
            cu.func_int(1024, 0)
        with self.assertRaisesRegex(RuntimeError, 'ZeroDivisionError: float divmod()'):
            cu = torch.jit.CompilationUnit(dedent(inspect.getsource(func_float)))
            cu.func_float(5.3, 0.0)
        with self.assertRaisesRegex(RuntimeError, 'ZeroDivisionError: float divmod()'):
            cu = torch.jit.CompilationUnit(dedent(inspect.getsource(func_int_float)))
            cu.func_int_float(1024, 0.0)
        with self.assertRaisesRegex(RuntimeError, 'ZeroDivisionError: float divmod()'):
            cu = torch.jit.CompilationUnit(dedent(inspect.getsource(func_float_int)))
            cu.func_float_int(5.3, 0)

    def test_math_ops(self):
        if False:
            print('Hello World!')

        def checkMathWrap(func_name, num_args=1, is_float=True, **args):
            if False:
                return 10
            if is_float:
                checkMath(func_name, num_args, True, **args)
                checkMath(func_name, num_args, False, **args)
            else:
                checkMath(func_name, num_args, is_float, **args)
        inf = float('inf')
        NaN = float('nan')
        mx_int = 2 ** 31 - 1
        mn_int = -2 ** 31
        float_vals = [inf, NaN, 0.0, 1.0, 2.2, -1.0, -0.0, -2.2, -inf, 1, 0, 2] + [10.0 ** i for i in range(5)] + [-10.0 ** i for i in range(5)]
        int_vals = list(range(-5, 5, 1)) + [mx_int + 5, mx_int * 2, mn_int - 5, mn_int * 2]

        def checkMath(func_name, num_args, is_float=True, ret_type='float', debug=False, vals=None, args_type=None):
            if False:
                print('Hello World!')
            funcs_template = dedent('\n            def func(a, b):\n                # type: {args_type} -> {ret_type}\n                return math.{func}({args})\n            ')
            if num_args == 1:
                args = 'a'
            elif num_args == 2:
                args = 'a, b'
            else:
                raise RuntimeError("Test doesn't support more than 2 arguments")
            if args_type is None:
                args_type = '(float, float)' if is_float else '(int, int)'
            funcs_str = funcs_template.format(func=func_name, args=args, args_type=args_type, ret_type=ret_type)
            scope = {}
            execWrapper(funcs_str, globals(), scope)
            cu = torch.jit.CompilationUnit(funcs_str)
            f_script = cu.func
            f = scope['func']
            if vals is None:
                vals = float_vals if is_float else int_vals
                vals = [(i, j) for i in vals for j in vals]
            for (a, b) in vals:
                res_python = None
                res_script = None
                try:
                    res_python = f(a, b)
                except Exception as e:
                    res_python = e
                try:
                    res_script = f_script(a, b)
                except Exception as e:
                    res_script = e
                if debug:
                    print('in: ', a, b)
                    print('out: ', res_python, res_script)
                if res_python != res_script:
                    if isinstance(res_python, Exception):
                        continue
                    if type(res_python) == type(res_script):
                        if isinstance(res_python, tuple) and math.isnan(res_python[0]) == math.isnan(res_script[0]):
                            continue
                        if isinstance(res_python, float) and math.isnan(res_python) and math.isnan(res_script):
                            continue
                    msg = f'Failed on {func_name} with inputs {a} {b}. Python: {res_python}, Script: {res_script}'
                    if sys.version_info >= (3, 11) and func_name == 'pow' and (a == 0.0) and (b == -math.inf):
                        self.assertTrue(res_python == math.inf and type(res_script) is RuntimeError)
                    else:
                        self.assertEqual(res_python, res_script, msg=msg, atol=0.0001 * max(abs(res_python), res_script), rtol=0)
        unary_float_ops = ['log', 'log1p', 'log10', 'exp', 'sqrt', 'gamma', 'lgamma', 'erf', 'erfc', 'expm1', 'fabs', 'acos', 'asin', 'atan', 'cos', 'sin', 'tan', 'asinh', 'atanh', 'acosh', 'sinh', 'cosh', 'tanh', 'degrees', 'radians']
        binary_float_ops = ['atan2', 'fmod', 'copysign']
        for op in unary_float_ops:
            checkMathWrap(op, 1)
        for op in binary_float_ops:
            checkMathWrap(op, 2)
        checkMath('modf', 1, ret_type='Tuple[float, float]')
        checkMath('frexp', 1, ret_type='Tuple[float, int]')
        checkMath('isnan', 1, ret_type='bool')
        checkMath('isinf', 1, ret_type='bool')
        checkMath('ldexp', 2, is_float=False, ret_type='float', args_type='(float, int)', vals=[(i, j) for i in float_vals for j in range(-10, 10)])
        checkMath('pow', 2, is_float=False, ret_type='float')
        checkMath('pow', 2, is_float=True, ret_type='float')
        checkMathWrap('floor', ret_type='int')
        checkMathWrap('ceil', ret_type='int')
        checkMathWrap('gcd', 2, is_float=False, ret_type='int')
        checkMath('isfinite', 1, ret_type='bool')
        checkMathWrap('remainder', 2)
        checkMathWrap('factorial', 1, is_float=False, ret_type='int', vals=[(i, 0) for i in range(-2, 10)])

    @skipIfTorchDynamo('TorchDynamo fails with unknown reason')
    def test_if_nest_while(self):
        if False:
            for i in range(10):
                print('nop')

        def func(a, b):
            if False:
                print('Hello World!')
            c = 0
            if a > b:
                while a > b:
                    b = b + 1
                    c = -b
            return c
        inputs = self._make_scalar_vars([4321, 1234], torch.int64)
        self.checkScript(func, inputs)

    def test_script_optional_none(self):
        if False:
            return 10

        def none_stmt(x):
            if False:
                print('Hello World!')
            output = None
            output = x
            return output

        def none_args(x):
            if False:
                while True:
                    i = 10
            return None
        self.checkScript(none_stmt, [torch.arange(0, 2)], optimize=True)
        self.checkScript(none_args, [None], optimize=True)

        def test_script_optional_tensor_none(x=None):
            if False:
                for i in range(10):
                    print('nop')
            res = torch.zeros(1, dtype=torch.int8)
            if x is None:
                res = res + 1
            else:
                res = x
            return res
        fn = test_script_optional_tensor_none
        scripted_fn = torch.jit.script(fn)
        self.assertEqual(fn(), scripted_fn())
        self.assertEqual(fn(torch.zeros(1)), scripted_fn(torch.zeros(1)))

        def test_script_optional_other_none(x=None):
            if False:
                i = 10
                return i + 15
            res = 2.0
            if x is None:
                res = res + 1.0
            else:
                res = x
            return res
        fn = test_script_optional_other_none
        scripted_fn = torch.jit.script(fn)
        self.assertEqual(fn(), scripted_fn())
        self.assertEqual(fn(1.0), scripted_fn(1.0))

    def test_script_clamp_none(self):
        if False:
            for i in range(10):
                print('nop')

        def test_script_clamp_max_none(x):
            if False:
                return 10
            return torch.clamp(x, min=2, max=None)

        def test_script_clamp_max(x):
            if False:
                while True:
                    i = 10
            return torch.clamp(x, max=2)

        def test_script_clamp_min_none(x):
            if False:
                i = 10
                return i + 15
            return torch.clamp(x, min=None, max=2)

        def test_script_clamp_min(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.clamp(x, min=2)
        input = [torch.arange(0, 3)]
        self.checkScript(test_script_clamp_max_none, input, optimize=True)
        self.checkScript(test_script_clamp_max, input, optimize=True)
        self.checkScript(test_script_clamp_min_none, input, optimize=True)
        self.checkScript(test_script_clamp_min, input, optimize=True)

    def test_script_bool_constant(self):
        if False:
            print('Hello World!')

        def test_script_bool_constant():
            if False:
                for i in range(10):
                    print('nop')
            a = True
            return a
        self.checkScript(test_script_bool_constant, [])

    def test_ternary(self):
        if False:
            return 10

        def func(a, b):
            if False:
                for i in range(10):
                    print('nop')
            c = 3
            c = a + b if bool(a > 3) else b
            return c
        inputs_true = self._make_scalar_vars([5, 2], torch.int64)
        inputs_false = self._make_scalar_vars([1, 0], torch.int64)
        self.checkScript(func, inputs_true, optimize=True)
        self.checkScript(func, inputs_false, optimize=True)

    def test_ternary_module_type_hint(self):
        if False:
            print('Hello World!')

        class M1(torch.nn.Module):

            def forward(self) -> Any:
                if False:
                    return 10
                return 'out' if self.training else {}

        class M2(torch.nn.Module):

            def forward(self) -> Any:
                if False:
                    print('Hello World!')
                out: Any = 'out' if self.training else {}
                return out

        class M3(torch.nn.Module):

            def forward(self) -> Optional[int]:
                if False:
                    return 10
                return None if self.training else 1
        for module in [M1, M2, M3]:
            self.checkModule(module().train(), ())
            self.checkModule(module().eval(), ())

    def test_ternary_static_if(self):
        if False:
            while True:
                i = 10

        class M1(torch.nn.Module):
            flag: torch.jit.Final[bool]

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.flag = True

            def forward(self) -> torch.Tensor:
                if False:
                    i = 10
                    return i + 15
                return torch.ones(3) if self.flag else {}

        class M2(torch.nn.Module):
            flag: torch.jit.Final[bool]

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.flag = False

            def forward(self) -> torch.Tensor:
                if False:
                    return 10
                return {} if self.flag else torch.ones(3)
        model1 = M1()
        model2 = M2()
        script_model_1 = torch.jit.script(model1)
        script_model_2 = torch.jit.script(model2)
        self.assertEqual(model1.forward(), script_model_1.forward())
        self.assertEqual(model2.forward(), script_model_2.forward())

    def test_ternary_right_associative(self):
        if False:
            while True:
                i = 10

        def plus_123(x: int):
            if False:
                for i in range(10):
                    print('nop')
            return x + 1 if x == 1 else x + 2 if x == 2 else x + 3
        self.checkScript(plus_123, (1,))
        self.checkScript(plus_123, (2,))
        self.checkScript(plus_123, (3,))

    @skipIfTorchDynamo('TorchDynamo fails with unknown reason')
    def test_print(self):
        if False:
            return 10

        def func(x, y):
            if False:
                for i in range(10):
                    print('nop')
            q = (x + y).sigmoid()
            print(q, 1, 2, [1, 2], [1.0, 2.0])
            w = -q
            return w * w
        x = torch.arange(4.0, requires_grad=True)
        y = torch.arange(0.0, 8, 2, requires_grad=True)
        self.checkScript(func, [x, y], optimize=True, capture_output=True)

    def test_format(self):
        if False:
            return 10

        def func(x):
            if False:
                while True:
                    i = 10
            print("{}, I'm a {}".format('Hello', 'test'))
            print('format blank'.format())
            print('stuff before {}'.format('hi'))
            print('{} stuff after'.format('hi'))
            return x + 1
        x = torch.arange(4.0, requires_grad=True)
        self.checkScript(func, [x], optimize=True, capture_output=True)

    def test_logical_short_circuit(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def testNoThrows(t):
            if False:
                return 10
            c1 = 1
            if False and bool(t[1]) or (True or bool(t[1])):
                c1 = 0
            return c1
        FileCheck().check_not('prim::If').run(testNoThrows.graph)
        self.assertEqual(0, testNoThrows(torch.randn(0)))
        self.assertEqual(0, testNoThrows(torch.randn([2, 3])))

        @torch.jit.script
        def throwsOr(t):
            if False:
                return 10
            c0 = False or bool(t[1])
            print(c0)

        @torch.jit.script
        def throwsAnd(t):
            if False:
                i = 10
                return i + 15
            c0 = True and bool(t[1])
            print(c0)
        t = torch.randn(0)
        with self.assertRaisesRegex(RuntimeError, 'index 1 out of range for tensor of size'):
            throwsOr(t)
        with self.assertRaisesRegex(RuntimeError, 'index 1 out of range for tensor of size'):
            throwsAnd(t)

    def test_type_cast(self):
        if False:
            for i in range(10):
                print('nop')
        template = dedent('\n        def func(v):\n            # type: ({from_type}) -> {to_type}\n            return {to_type}(v)\n        ')

        def check_cast(from_type, to_type, value, raises=False):
            if False:
                return 10
            code = template.format(from_type=from_type, to_type=to_type)
            self.checkScript(code, (value,))
        check_cast('int', 'float', 1)
        check_cast('int', 'bool', 1)
        check_cast('int', 'bool', 0)
        check_cast('float', 'int', 1.0)
        check_cast('float', 'bool', 1.0)
        check_cast('float', 'bool', 0.0)
        check_cast('bool', 'int', True)
        check_cast('bool', 'float', True)

    def test_multiple_assignment(self):
        if False:
            for i in range(10):
                print('nop')

        def outer_func(x):
            if False:
                print('Hello World!')
            return (x * 2, x + 2)

        @torch.jit.script
        def func(x):
            if False:
                i = 10
                return i + 15
            (y, z) = outer_func(x)
            return y + z
        x = torch.arange(4)
        self.assertEqual(func(x), x * 2 + x + 2)

    def test_literals(self):
        if False:
            i = 10
            return i + 15

        def func(a):
            if False:
                for i in range(10):
                    print('nop')
            return a.view(size=[1, 2, 3])
        a = torch.randn(6)
        self.checkScript(func, [a], optimize=True)

    def test_return(self):
        if False:
            i = 10
            return i + 15

        def no_return(a):
            if False:
                for i in range(10):
                    print('nop')
            a + 1

        def void_return(a):
            if False:
                for i in range(10):
                    print('nop')
            return

        def one_return(a):
            if False:
                while True:
                    i = 10
            return a + 1.0

        def multiple_returns(a):
            if False:
                i = 10
                return i + 15
            return (a * 1.0, a * 2.0, a * 3.0)
        a = torch.randn(1, dtype=torch.float)
        self.checkScript(no_return, [a], optimize=True)
        self.checkScript(void_return, [a], optimize=True)
        self.checkScript(one_return, [a], optimize=True)
        self.checkScript(multiple_returns, [a], optimize=True)
        with self.assertRaisesRegex(RuntimeError, 'does not return along all paths'):
            torch.jit.CompilationUnit('\n            def no_return_bad_annotation(a):\n                # type: (Tensor) -> Tensor\n                a + 1\n            ')

    def test_error(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def foo(a):
            if False:
                print('Hello World!')
            return a.t()
        s = Variable(torch.rand(5, 5, 5))
        with self.assertRaisesRegex(RuntimeError, 'failed in the TorchScript interpreter'):
            foo(s)

        @torch.jit.script
        def bar(c, b):
            if False:
                return 10
            return c + b
        with self.assertRaisesRegex(RuntimeError, 'failed in the TorchScript interpreter'):
            bar(Variable(torch.rand(10), requires_grad=True), Variable(torch.rand(9), requires_grad=True))

    def test_error_stacktrace(self):
        if False:
            return 10

        @torch.jit.script
        def baz(c, b):
            if False:
                while True:
                    i = 10
            return c + b

        @torch.jit.script
        def foo(c, b):
            if False:
                while True:
                    i = 10
            return baz(c, b)

        @torch.jit.script
        def bar(c, b):
            if False:
                for i in range(10):
                    print('nop')
            return foo(c, b)
        with self.assertRaises(RuntimeError) as cm:
            bar(torch.rand(10), torch.rand(9))
        FileCheck().check('The following operation failed in the TorchScript interpreter').check('Traceback').check('in foo').check('in baz').run(str(cm.exception))

    def test_error_stacktrace_interface(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def baz(c, b):
            if False:
                for i in range(10):
                    print('nop')
            return c + b

        @torch.jit.script
        def foo(c, b):
            if False:
                for i in range(10):
                    print('nop')
            return baz(c, b)

        @torch.jit.script
        def bar(c, b):
            if False:
                while True:
                    i = 10
            return foo(c, b)

        @torch.jit.script
        class Bar:

            def one(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return bar(x, y)

        @torch.jit.interface
        class IFace:

            def one(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                pass
        make_global(IFace)

        @torch.jit.script
        def as_interface(x):
            if False:
                i = 10
                return i + 15
            return x
        f = as_interface(Bar())
        with self.assertRaises(RuntimeError) as cm:
            x = f.one(torch.rand(10), torch.rand(9))
            bar(torch.rand(10), torch.rand(9))
        FileCheck().check('The following operation failed in the TorchScript interpreter').check('Traceback').check('in foo').check('in baz').run(str(cm.exception))

    def test_operator_precedence(self):
        if False:
            return 10

        def double(x):
            if False:
                i = 10
                return i + 15
            return 2 * x

        def complicated_arithmetic_operation():
            if False:
                for i in range(10):
                    print('nop')
            list = [0, 1, 2, 3]
            result = list[1:3][0] + double(4) + (-3 + 8) * 6 // 2 % 4 << 2 + 1 >> 1 | 23 & 16 + 3 ^ 4
            return result
        self.checkScript(complicated_arithmetic_operation, ())

    def test_in_operator_with_two_strings(self):
        if False:
            return 10

        def fn() -> bool:
            if False:
                for i in range(10):
                    print('nop')
            return 'a' in 'abcd'
        self.checkScript(fn, ())

    def test_bitwise_ops(self):
        if False:
            while True:
                i = 10

        def int_test():
            if False:
                return 10
            return (2 & 3, 2 ^ 3, 2 | 3, 2 << 3, 2 >> 3)
        self.checkScript(int_test, ())

        def bool_test(x, y):
            if False:
                return 10
            return (x & y, x ^ y, x | y)
        self.checkScript(bool_test, (True, False))
        self.checkScript(bool_test, (True, True))

        def tensor_test(x, y):
            if False:
                return 10
            return (x & y, x ^ y, x | y)

        def tensor_with_int_test(x, y):
            if False:
                return 10
            return (x << y, x >> y)
        x = torch.tensor(2)
        y = torch.tensor(3)
        self.checkScript(tensor_test, (x, y))
        self.checkScript(tensor_with_int_test, (x, 2))

        def not_test(x):
            if False:
                for i in range(10):
                    print('nop')
            return ~x
        self.checkScript(not_test, (torch.tensor([2, 4]),))

    def test_all(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def test_all_tensor(x):
            if False:
                return 10
            return all(x)
        self.assertFalse(test_all_tensor(torch.tensor([1, 0, 3], dtype=torch.uint8)))
        self.assertTrue(test_all_tensor(torch.tensor([3.14, 3, 99], dtype=torch.uint8)))
        self.assertTrue(test_all_tensor(torch.tensor([True, True], dtype=torch.uint8)))
        self.assertFalse(test_all_tensor(torch.tensor([True, False], dtype=torch.uint8)))

        @torch.jit.script
        def test_all_bool_list(x):
            if False:
                return 10
            return all(x)
        self.assertTrue(test_all_bool_list([True, True]))
        self.assertTrue(test_all_bool_list([True, 1]))
        self.assertFalse(test_all_bool_list([True, False]))
        self.assertFalse(test_all_bool_list([True, 0]))
        self.assertFalse(test_all_bool_list([False, 0]))
        self.assertTrue(test_all_bool_list([]))

        @torch.jit.script
        def test_all_int_list(x):
            if False:
                return 10
            return all(x)
        self.assertTrue(test_all_int_list([3, 6]))
        self.assertFalse(test_all_int_list([2, 0]))

        @torch.jit.script
        def test_all_float_list(x):
            if False:
                for i in range(10):
                    print('nop')
            return all(x)
        self.assertTrue(test_all_float_list([3.14, 8.1]))
        self.assertFalse(test_all_float_list([3.14, 0, 8.9]))

    def test_number_math(self):
        if False:
            i = 10
            return i + 15
        ops_template = dedent('\n        def func():\n            return {scalar1} {op} {scalar2}\n        ')
        ops = ['+', '-', '*', '%', '<', '<=', '>', '>=', '==', '!=', '//']
        funcs_template = dedent('\n        def func():\n            return {func}({scalar1}, {scalar2})\n        ')
        funcs = ['min', 'max']
        scalars = ['7', '2', '3', '-3', '3.14', '0.125', '-0.5', '2.0', '-2.0']
        scalar_pairs = [(scalar1, scalar2) for scalar1 in scalars for scalar2 in scalars]

        def run_test(code):
            if False:
                i = 10
                return i + 15
            scope = {}
            execWrapper(code, globals(), scope)
            cu = torch.jit.CompilationUnit(code)
            self.assertEqual(cu.func(), scope['func']())
        for (scalar1, scalar2) in scalar_pairs:
            for op in ops:
                code = ops_template.format(op=op, scalar1=scalar1, scalar2=scalar2)
                run_test(code)
            for func in funcs:
                code = funcs_template.format(func=func, scalar1=scalar1, scalar2=scalar2)
                run_test(code)
        for (scalar1, scalar2) in scalar_pairs:
            item1 = 'torch.tensor(' + scalar1 + ').item()'
            item2 = 'torch.tensor(' + scalar2 + ').item()'
            for op in ops:
                code = ops_template.format(op=op, scalar1=item1, scalar2=scalar2)
                run_test(code)
                code = ops_template.format(op=op, scalar1=scalar1, scalar2=item2)
                run_test(code)
                code = ops_template.format(op=op, scalar1=item1, scalar2=item2)
                run_test(code)
            for func in funcs:
                code = funcs_template.format(func=func, scalar1=item1, scalar2=scalar2)
                run_test(code)
                code = funcs_template.format(func=func, scalar1=scalar1, scalar2=item2)
                run_test(code)
                code = funcs_template.format(func=func, scalar1=item1, scalar2=item2)
                run_test(code)

    def test_number_abs(self):
        if False:
            while True:
                i = 10

        def func1(x):
            if False:
                while True:
                    i = 10
            return abs(x)

        def func2(x):
            if False:
                return 10
            return abs(x)

        def func3(x):
            if False:
                while True:
                    i = 10
            return abs(x)
        self.checkScript(func1, (-3.14,))
        self.checkScript(func1, (3.14,))
        self.checkScript(func2, (-10,))
        self.checkScript(func2, (10,))
        self.checkScript(func3, (torch.tensor([-5, -10, -20]),))
        self.checkScript(func3, (torch.tensor([5, 10, 20]),))
        self.checkScript(func3, (torch.tensor([-5, 10, -20]),))

    def test_number_div(self):
        if False:
            print('Hello World!')
        self.assertEqual(div_int_future(), torch.jit.script(div_int_future)())
        self.checkScript(div_float_future, ())
        self.checkScript(div_int_nofuture, ())
        self.checkScript(div_float_nofuture, ())

    def test_bool_augassign_bitwise_or(self):
        if False:
            for i in range(10):
                print('nop')

        def func(a: bool, b: bool) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            a |= b
            return a
        self.checkScript(func, (True, False), optimize=True)
        self.checkScript(func, (True, True), optimize=True)
        self.checkScript(func, (False, False), optimize=True)
        self.checkScript(func, (False, True), optimize=True)

    def test_bool_augassign_bitwise_and(self):
        if False:
            return 10

        def func(a: bool, b: bool) -> bool:
            if False:
                return 10
            a &= b
            return a
        self.checkScript(func, (True, False), optimize=True)
        self.checkScript(func, (True, True), optimize=True)
        self.checkScript(func, (False, False), optimize=True)
        self.checkScript(func, (False, True), optimize=True)

    def test_bool_augassign_bitwise_xor(self):
        if False:
            print('Hello World!')

        def func(a: bool, b: bool) -> bool:
            if False:
                while True:
                    i = 10
            a ^= b
            return a
        self.checkScript(func, (True, False), optimize=True)
        self.checkScript(func, (True, True), optimize=True)
        self.checkScript(func, (False, False), optimize=True)
        self.checkScript(func, (False, True), optimize=True)

    def test_number_augassign_bitwise_lshift(self):
        if False:
            i = 10
            return i + 15

        def func() -> int:
            if False:
                while True:
                    i = 10
            z = 8
            z <<= 2
            return z
        self.checkScript(func, (), optimize=True)

    def test_number_augassign_bitwise_rshift(self):
        if False:
            for i in range(10):
                print('nop')

        def func() -> int:
            if False:
                for i in range(10):
                    print('nop')
            z = 8
            z >>= 2
            return z
        self.checkScript(func, (), optimize=True)

    def test_number_augassign_bitwise_pow(self):
        if False:
            i = 10
            return i + 15

        def func() -> float:
            if False:
                for i in range(10):
                    print('nop')
            z = 8
            z **= 2
            return z
        self.checkScript(func, (), optimize=True)

    def test_number_augassign(self):
        if False:
            for i in range(10):
                print('nop')

        def func():
            if False:
                for i in range(10):
                    print('nop')
            z = 1
            z += 2
            return z
        self.checkScript(func, (), optimize=True)

    def test_nested_select_assign(self):
        if False:
            i = 10
            return i + 15

        class SubSubModule(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.abc = 11

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.abc

        class SubModule(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.a = 11
                self.nested = SubSubModule()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.a

        class TestModule(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.sub = SubModule()
                self.hi = 1

            def forward(self):
                if False:
                    return 10
                self.hi = 5
                self.sub.a = 1
                self.sub.nested.abc = 5
                return self.sub.a * 20 + self.sub.nested.abc * 3 + self.hi
        self.checkModule(TestModule(), ())

    def test_number_neg(self):
        if False:
            print('Hello World!')

        def func1():
            if False:
                print('Hello World!')
            return -8

        def func2():
            if False:
                while True:
                    i = 10
            return -3.14
        self.checkScript(func1, (), optimize=True)
        self.checkScript(func2, (), optimize=True)

    def test_compare_two_bool_inputs(self):
        if False:
            for i in range(10):
                print('nop')

        def compare_eq(a: bool, b: bool):
            if False:
                i = 10
                return i + 15
            return a == b

        def compare_ne(a: bool, b: bool):
            if False:
                return 10
            return a != b
        scripted_fn_eq = torch.jit.script(compare_eq)
        scripted_fn_ne = torch.jit.script(compare_ne)
        self.assertEqual(scripted_fn_eq(True, False), compare_eq(True, False))
        self.assertEqual(scripted_fn_eq(False, True), compare_eq(False, True))
        self.assertEqual(scripted_fn_eq(True, True), compare_eq(True, True))
        self.assertEqual(scripted_fn_eq(False, False), compare_eq(False, False))
        self.assertEqual(scripted_fn_ne(True, False), compare_ne(True, False))
        self.assertEqual(scripted_fn_ne(False, True), compare_ne(False, True))
        self.assertEqual(scripted_fn_ne(True, True), compare_ne(True, True))
        self.assertEqual(scripted_fn_ne(False, False), compare_ne(False, False))

    def _test_tensor_number_math(self, device='cpu'):
        if False:
            for i in range(10):
                print('nop')
        template = dedent('\n        def func(t):\n            return {lhs} {op} {rhs}\n        ')

        def test(op, tensor, const, swap_args, template=template):
            if False:
                while True:
                    i = 10
            args = ('t', const)
            if swap_args:
                args = (const, 't')
            code = template.format(lhs=args[0], rhs=args[1], op=op)
            scope = {}
            execWrapper(code, globals(), scope)
            cu = torch.jit.CompilationUnit(code)
            message = f'with code `{args[0]} {op} {args[1]}` and t={tensor}'
            res1 = cu.func(tensor)
            res2 = scope['func'](tensor)
            self.assertEqual(res1, res2, msg=message + '\nres1=' + str(res1) + '\nres2=' + str(res2))
            self.assertEqual(res1.dtype, res2.dtype, msg=message + '\nres1=' + str(res1) + '\nres2=' + str(res2))
        var_int = [2, -2]
        var_float = [1.4321, -1.2]
        ops = ['+', '-', '*', '%', '<', '<=', '>', '>=', '==', '!=', '/']
        float_tensor = torch.randn(5, 5, device=device)
        double_tensor = torch.randn(5, 5, dtype=torch.double, device=device)
        long_tensor = torch.randint(-5, 5, (5, 5), dtype=torch.long, device=device)
        long_tensor[long_tensor == 0] = 2
        tensors = [float_tensor, double_tensor, long_tensor]
        consts = var_int + var_float
        for (op, tensor, const, swap_args) in product(ops, tensors, consts, [True, False]):
            if op == '/' and tensor.data_ptr() == long_tensor.data_ptr():
                continue
            if op == '%' and swap_args is True:
                continue
            test(op, tensor, const, swap_args)

    def test_tensor_number_math(self):
        if False:
            i = 10
            return i + 15
        self._test_tensor_number_math()

    def test_torch_tensor_bad_input(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(RuntimeError, 'must be of ints, floats, or bools, got None'):

            @torch.jit.script
            def test():
                if False:
                    while True:
                        i = 10
                return torch.tensor([None])
            test()
        with self.assertRaisesRegex(RuntimeError, 'Empty lists default to List\\[Tensor\\]'):

            @torch.jit.script
            def tmp():
                if False:
                    return 10
                return torch.tensor([])
            tmp()

        @torch.jit.script
        def foo():
            if False:
                while True:
                    i = 10
            return torch.tensor([[2, 2], [1]])
        with self.assertRaisesRegex(RuntimeError, 'Expected sequence of length'):
            foo()

    @suppress_warnings
    def test_torch_tensor_as_tensor_empty_list(self):
        if False:
            for i in range(10):
                print('nop')
        tensor_template = dedent('\n        def func():\n            empty_list = torch.jit.annotate(List[int], [])\n            ten1 = torch.{tensor_op}({input})\n            return ten1\n        ')
        ops = ['tensor', 'as_tensor']
        inputs = ['empty_list', '[empty_list, empty_list]', '[[[empty_list]]]']
        for op in ops:
            for inp in inputs:
                code = tensor_template.format(tensor_op=op, input=inp)
                scope = {}
                exec(code, globals(), scope)
                cu = torch.jit.CompilationUnit(code)
                t1 = cu.func()
                t2 = scope['func']()
                if inp == 'empty_list':
                    self.assertNotEqual(t1.dtype, t2.dtype)
                self.assertEqual(t1, t2, exact_dtype=False)
                self.assertEqual(t1.device, t2.device)

    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, "Simple Executor doesn't have any shapes to propagate")
    def test_tensor_as_tensor_shape_prop(self):
        if False:
            while True:
                i = 10
        tensor_template = dedent('\n        def func():\n            return torch.{tensor_op}({input})\n        ')
        ops = ['tensor', 'as_tensor']
        inputs = ['[1]', '[False]', '[2.5]', '0.5', '1', 'False', '[[1]]', 'torch.jit.annotate(List[List[int]], [])']
        expected_shape = ['Long(*, device=cpu)', 'Bool(*, device=cpu)', 'Float(*, device=cpu)', 'Float(device=cpu)', 'Long(device=cpu)', 'Bool(device=cpu)', 'Long(*, *, device=cpu)']
        for op in ops:
            for (inp, expect) in zip(inputs, expected_shape):
                code = tensor_template.format(tensor_op=op, input=inp)
                scope = {}
                exec(code, globals(), scope)
                cu = torch.jit.CompilationUnit(code)
                torch._C._jit_pass_complete_shape_analysis(cu.func.graph, (), False)
                FileCheck().check(expect).check(f'aten::{op}').run(cu.func.graph)

        @torch.jit.script
        def test_dtype(inp_dtype: torch.dtype):
            if False:
                print('Hello World!')
            a = torch.tensor(1.0, dtype=torch.float, requires_grad=True)
            return (a, torch.tensor(1.0, dtype=inp_dtype))
        if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
            g = test_dtype.graph_for(5, profile_and_replay=True)
            FileCheck().check('Tensor = aten::tensor').check('Float(device=cpu) = prim::BailOut').check('Tensor = aten::tensor').check('Half(device=cpu) = prim::BailOut').run(g)
        else:
            g = test_dtype.graph_for(5)
            FileCheck().check('Float(requires_grad=1, device=cpu) = aten::tensor').check('Tensor(requires_grad=0) = aten::tensor').run(g)

        @torch.jit.script
        def test_as_tensor_tensor_input(input):
            if False:
                while True:
                    i = 10
            a = torch.as_tensor(input, dtype=input.dtype)
            return (a, torch.as_tensor(input, dtype=torch.float))
        if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
            g = test_as_tensor_tensor_input.graph_for(torch.ones(3, 4), profile_and_replay=True)
            FileCheck().check('Tensor = aten::as_tensor').check('Float(3, 4) = prim::BailOut').check('Tensor = aten::as_tensor').check('Float(3, 4) = prim::BailOut').run(g)
        else:
            g = test_as_tensor_tensor_input.graph_for(torch.ones(3, 4))
            FileCheck().check('Tensor = aten::as_tensor').check('Float(*, *, requires_grad=0, device=cpu) = aten::as_tensor').run(g)

    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, 'testing legacy behavior')
    def test_tensor_requires_grad(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def test(b):
            if False:
                while True:
                    i = 10
            a = torch.tensor(1.0, requires_grad=b)
            b = torch.tensor(1.0, requires_grad=True)
            c = torch.tensor(1.0, requires_grad=False)
            return (a, b, c)
        g = test.graph_for(True)
        out = next(g.outputs())
        out_inp = list(out.node().inputs())
        self.assertTrue(out_inp[0].requires_grad())
        self.assertTrue(out_inp[1].requires_grad())
        self.assertFalse(out_inp[2].requires_grad())

    def test_grad_from_script(self):
        if False:
            for i in range(10):
                print('nop')

        def test():
            if False:
                for i in range(10):
                    print('nop')
            a = torch.tensor(2.5, requires_grad=True)
            b = a * 2
            return (a, b)
        (a, b) = test()
        b.backward()
        (a_script, b_script) = torch.jit.script(test)()
        b_script.backward()
        self.assertEqual(a.grad, a_script.grad)

    def test_torch_tensor_as_tensor(self):
        if False:
            i = 10
            return i + 15
        tensor_template = dedent('\n        def func():\n            li = {list_create}\n            ten1 = torch.{tensor_op}(li {options})\n            return ten1\n        ')
        lists = ['2.5', '4', 'True', 'False', '[2]', '[-.5]', '[False, True, False]', '[2, 2]', '(1, 1)', 'torch.jit.annotate(List[List[int]], [])', 'torch.jit.annotate(List[int], [])', '[2.5, 2.5]', '[[2], [2]]', '[[-.5], [2.2]]', '[[False], [True]]']
        dtypes = ['', ', dtype=torch.float', ', dtype=torch.double', ', dtype=torch.half', ', dtype=torch.uint8', ', dtype=torch.int8', ', dtype=torch.short', ', dtype=torch.int', ', dtype=torch.long', ', dtype=torch.cfloat', ', dtype=torch.cdouble']
        ops = ['tensor', 'as_tensor']
        devices = ['', ", device='cpu'"]
        if RUN_CUDA:
            devices.append(", device='cuda'")
        option_pairs = [dtype + device for dtype in dtypes for device in devices]
        for op in ops:
            for li in lists:
                for option in option_pairs:
                    if 'annotate' in li and 'dtype' not in option:
                        continue
                    if sys.version_info[:2] >= (3, 10) and 'torch.uint8' in option and ('-' in li):
                        continue
                    code = tensor_template.format(list_create=li, tensor_op=op, options=option)
                    scope = {}
                    exec(code, globals(), scope)
                    cu = torch.jit.CompilationUnit(code)
                    t1 = cu.func()
                    t2 = scope['func']()
                    if t1.dtype == torch.float16:
                        self.assertTrue(str(t1) == str(t2))
                    else:
                        self.assertEqual(t1, t2)
                    self.assertEqual(t1.dtype, t2.dtype)
                    self.assertEqual(t1.device, t2.device)

        def test_as_tensor_tensor_input(input):
            if False:
                for i in range(10):
                    print('nop')
            return (torch.as_tensor(input, dtype=torch.cfloat), torch.as_tensor(input, dtype=torch.float), torch.as_tensor(input, dtype=torch.int32))
        inp = torch.randn(3, 4, dtype=torch.cfloat)
        self.checkScript(test_as_tensor_tensor_input, (inp,))

    def test_torch_tensor_dtype(self):
        if False:
            while True:
                i = 10

        def foo(s: float):
            if False:
                print('Hello World!')
            return (torch.tensor(s), torch.tensor([s, s]))
        with set_default_dtype(torch.double):
            self.assertEqual(torch.jit.script(foo)(1.0), foo(1.0), exact_dtype=True)
            if GRAPH_EXECUTOR == ProfilingMode.LEGACY:
                FileCheck().check('Double').check_same('aten::tensor').run(torch.jit.last_executed_optimized_graph())
        with set_default_dtype(torch.float):
            del torch.jit._state._jit_caching_layer[foo]
            self.assertEqual(torch.jit.script(foo)(1.0), foo(1.0), exact_dtype=True)
            if GRAPH_EXECUTOR == ProfilingMode.LEGACY:
                FileCheck().check('Float').check_same('aten::tensor').run(torch.jit.last_executed_optimized_graph())
        with set_default_dtype(torch.half):
            del torch.jit._state._jit_caching_layer[foo]
            self.assertEqual(torch.jit.script(foo)(1.0), foo(1.0), exact_dtype=True)
            if GRAPH_EXECUTOR == ProfilingMode.LEGACY:
                FileCheck().check('Half').check_same('aten::tensor').run(torch.jit.last_executed_optimized_graph())

    def test_shape_analysis_grad_property(self):
        if False:
            return 10

        @torch.jit.script
        def foo(x):
            if False:
                while True:
                    i = 10
            return torch.sub(x, torch.tanh(x))
        torch._C._jit_pass_complete_shape_analysis(foo.graph, (torch.tensor([0.39]),), False)
        self.assertTrue(foo.graph.findNode('aten::sub').output().requiresGrad() is None)

    def test_empty_like_memory_format_bc(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                i = 10
                return i + 15
            return torch.zeros_like(x, memory_format=None)
        scripted_f = torch.jit.script(f)
        x = torch.rand(3, 4)
        self.assertEqual(scripted_f(x), f(x))

    def test_multiline_string_dedents(self):
        if False:
            while True:
                i = 10

        def foo() -> None:
            if False:
                print('Hello World!')
            multiline_string_dedent_1 = '\nThis is a string dedent '
            multiline_string_dedent_2 = ' This is a\n  string dedent '
            multiline_string_dedent_3 = '\n            This is a string\ndedent '
            multiline_string_dedent_4 = ' This is a string dedent '
        scripted_foo = torch.jit.script(foo)
        self.assertEqual(scripted_foo(), foo())

    def test_class_with_comment_at_lower_indentation(self):
        if False:
            for i in range(10):
                print('nop')

        class Foo(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = torch.neg(x)
                return x
        torch.jit.script(Foo())

    def test_tensor_to(self):
        if False:
            i = 10
            return i + 15
        template = dedent('\n        def func(t):\n            cuda = "{cuda}"\n            device = "{device}"\n            non_blocking = {non_blocking}\n            return {to_str}\n        ')

        def s(t, to_str, non_blocking=None, device=None, cuda=None):
            if False:
                i = 10
                return i + 15
            device = device if device is not None else str(t.device)
            non_blocking = non_blocking if non_blocking is not None else False
            cuda = 'cuda' if cuda is None else cuda
            code = template.format(to_str=to_str, device=device, non_blocking=non_blocking, cuda=cuda)
            scope = {}
            cu = torch.jit.CompilationUnit(code)
            return cu.func(t, profile_and_replay=True)

        def test_copy_behavior(t, non_blocking=False):
            if False:
                for i in range(10):
                    print('nop')
            self.assertIs(t, s(t, 't.to(t, non_blocking=non_blocking)', non_blocking))
            self.assertIs(t, s(t, 't.to(t.dtype, non_blocking=non_blocking)', non_blocking))
            self.assertIs(t, s(t, 't.to(torch.empty_like(t), non_blocking=non_blocking)', non_blocking))
            self.assertIsNot(t, s(t, 't.to(t, non_blocking=non_blocking, copy=True)', non_blocking))
            self.assertIsNot(t, s(t, 't.to(t.dtype, non_blocking=non_blocking, copy=True)', non_blocking))
            self.assertIsNot(t, s(t, 't.to(torch.empty_like(t), non_blocking=non_blocking, copy=True)', non_blocking))
            devices = [t.device]
            if t.device.type == 'cuda':
                if t.device.index == -1:
                    devices.append(f'cuda:{torch.cuda.current_device()}')
                elif t.device.index == torch.cuda.current_device():
                    devices.append('cuda')
            for device in devices:
                self.assertIs(t, s(t, 't.to(device, non_blocking=non_blocking)', non_blocking, device))
                self.assertIs(t, s(t, 't.to(device, t.dtype, non_blocking=non_blocking)', non_blocking, device))
                self.assertIsNot(t, s(t, 't.to(device, non_blocking=non_blocking, copy=True)', non_blocking, device))
                self.assertIsNot(t, s(t, 't.to(device, t.dtype, non_blocking=non_blocking, copy=True)', non_blocking, device))
        t = torch.tensor(5)
        test_copy_behavior(t)
        self.assertEqual(t.device, s(t, "t.to('cpu')").device)
        self.assertEqual(t.device, s(t, "t.to('cpu', dtype=torch.float32)").device)
        self.assertIs(torch.float32, s(t, "t.to('cpu', dtype=torch.float32)").dtype)
        self.assertEqual(t.device, s(t, 't.to(torch.float32)').device)
        self.assertIs(torch.float32, s(t, 't.to(dtype=torch.float32)').dtype)
        self.assertEqual(t.data_ptr(), s(t, "t.to('cpu')").data_ptr())
        self.assertEqual(t.data_ptr(), s(t, 't.to(dtype=t.dtype, device=t.device, copy=False)').data_ptr())
        self.assertEqual(t.data_ptr(), s(t, "t.to('cpu', copy=False)").data_ptr())
        self.assertNotEqual(t.data_ptr(), s(t, "t.to('cpu', copy=True)").data_ptr())
        a = torch.tensor(5)
        if torch.cuda.is_available():
            for non_blocking in [True, False]:
                for cuda in ['cuda', 'cuda:0' if torch.cuda.device_count() == 1 else 'cuda:1']:
                    b = torch.tensor(5.0, device=cuda)
                    test_copy_behavior(b, non_blocking)
                    self.assertEqual(b.device, s(b, 't.to(cuda, non_blocking=non_blocking).device', cuda=cuda))
                    self.assertEqual(a.device, s(b, "t.to('cpu', non_blocking=non_blocking).device"))
                    self.assertEqual(b.device, s(b, 't.to(cuda, non_blocking=non_blocking).device', cuda=cuda))
                    self.assertIs(torch.int32, s(b, "t.to('cpu', dtype=torch.int32, non_blocking=non_blocking)").dtype)
                    self.assertEqual(a.device, s(b, "t.to('cpu', dtype=torch.int32, non_blocking=non_blocking)").device)
                    self.assertIs(torch.int32, s(b, 't.to(dtype=torch.int32)').dtype)
                    self.assertEqual(b.device, s(b, 't.to(dtype=torch.int32)').device)
        t = torch.tensor(5).float().requires_grad_()
        out_ref = t.to(torch.float32)
        out = s(t, 't.to(torch.float32)')
        self.assertEqual(out_ref, out)
        grad_ref = torch.autograd.grad(out_ref.sum(), t)
        grad = torch.autograd.grad(out.sum(), t)
        self.assertEqual(grad_ref, grad)
        out_ref = t.to('cpu')
        out = s(t, "t.to('cpu')")
        self.assertEqual(out_ref, out)
        grad_ref = torch.autograd.grad(out_ref.sum(), t)
        grad = torch.autograd.grad(out.sum(), t)
        self.assertEqual(grad_ref, grad)

        @torch.jit.script
        def func2(t, t_ref):
            if False:
                i = 10
                return i + 15
            return t.to(t_ref)
        with disable_autodiff_subgraph_inlining():
            t_ref = torch.tensor(4).double()
            out_ref = t.to(t_ref)
            out = func2(t, t_ref)
            grad_ref = torch.autograd.grad(out_ref.sum(), t)
            grad = torch.autograd.grad(out.sum(), t)
            self.assertEqual(grad_ref, grad)

    @unittest.skipIf(not RUN_CUDA, 'No CUDA')
    def test_tensor_number_math_cuda(self):
        if False:
            i = 10
            return i + 15
        self._test_tensor_number_math(device='cuda')

    def test_not(self):
        if False:
            i = 10
            return i + 15

        def test_not_op(a):
            if False:
                i = 10
                return i + 15
            return not bool(a > 1)
        self.checkScript(test_not_op, (torch.tensor(2),), optimize=True)

    def test_is_isnot(self):
        if False:
            while True:
                i = 10
        template = dedent('\n        def func():\n            # type: () -> bool\n            return {lhs} {op} {rhs}\n        ')

        def test(op, args):
            if False:
                for i in range(10):
                    print('nop')
            code = template.format(lhs=args[0], rhs=args[1], op=op)
            scope = {}
            execWrapper(code, globals(), scope)
            cu = torch.jit.CompilationUnit(code)
            self.assertEqual(cu.func(), scope['func'](), msg=f'Failed with op: {op}, lhs: {args[0]}, rhs: {args[1]}')
        ops = ['is', 'is not']
        type_literals = [True, False, None, [1, 1], 1, 2, 0.5, 1.5]
        for (op, lhs, rhs) in product(ops, type_literals, type_literals):
            test(op, [lhs, rhs])

    def test_isinstance_refinement(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def foo(a):
            if False:
                print('Hello World!')
            if isinstance(a, int):
                return a + 3
            else:
                return 4
        self.assertEqual(foo(4), 7)
        self.assertEqual(foo(None), 4)

        @torch.jit.script
        def foo2(a, b):
            if False:
                return 10
            if not isinstance(a, int) or not isinstance(b, int):
                return 0
            else:
                return a + b
        self.assertEqual(foo2(3, 4), 7)
        self.assertEqual(foo2(None, 4), 0)
        self.assertEqual(foo2(4, None), 0)

        @torch.jit.script
        def any_refinement(a, b):
            if False:
                while True:
                    i = 10
            if isinstance(a, int) and isinstance(b, int):
                return a + b
            return 0
        self.assertEqual(any_refinement(3, 4), 7)
        self.assertEqual(any_refinement(3, 'hi'), 0)

        @torch.jit.script
        def any_refinement2(a):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(a, Tensor):
                return a
            return torch.tensor(3)
        self.assertEqual(any_refinement2(3), torch.tensor(3))
        self.assertEqual(any_refinement2(torch.tensor(5)), torch.tensor(5))

    @unittest.skipIf(GRAPH_EXECUTOR == ProfilingMode.LEGACY, 'bug persists in deprecated executor')
    def test_unspecialized_any_binding(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def foo(x: Any):
            if False:
                return 10
            assert isinstance(x, Dict[str, torch.Tensor])
        foo({'1': torch.tensor(3)})
        with self.assertRaises(Exception):
            foo(2)

    def test_isinstance(self):
        if False:
            i = 10
            return i + 15
        template = dedent('\n        def func(x):\n            # type: ({type_hint}) -> bool\n            return isinstance(x, {typ})\n        ')

        def test(inp, typ, type_hint):
            if False:
                return 10
            code = template.format(typ=typ, type_hint=type_hint)
            scope = {}
            execWrapper(code, globals(), scope)
            cu = torch.jit.CompilationUnit(code)
            self.assertEqual(cu.func(inp), scope['func'](inp), msg=f'Failed with typ: {typ}')
        inputs = [True, 1, 1.0, torch.tensor(1), [1, 2], (1.0,), [1, 2], 1]
        type_literals = ['bool', 'int', 'float', 'torch.Tensor', 'list', 'tuple', '(list, tuple)', '(int, float, bool)']
        type_annotations = ['bool', 'int', 'float', 'Tensor', 'List[int]', 'Tuple[float]', 'List[int]', 'int']
        for (inp, typ, type_hint) in zip(inputs, type_literals, type_annotations):
            test(inp, typ, type_hint)

        @torch.jit.script
        def opt_func(x):
            if False:
                for i in range(10):
                    print('nop')
            return isinstance(x, int)
        self.assertTrue(opt_func(3))
        self.assertFalse(opt_func(None))

    def test_dropout_eval(self):
        if False:
            print('Hello World!')

        class ScriptedConv2d(torch.jit.ScriptModule):

            def __init__(self, in_channels, out_channels, **kwargs):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
                self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.conv(x)
                x = self.bn(x)
                return F.relu(x, inplace=True)

        class ScriptMod(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.Conv2d_1a_3x3 = ScriptedConv2d(3, 32, kernel_size=3, stride=2)

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.Conv2d_1a_3x3(x)
                return F.dropout(x, training=self.training)

        class EagerConv2d(torch.nn.Module):

            def __init__(self, in_channels, out_channels, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
                self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.conv(x)
                x = self.bn(x)
                return F.relu(x, inplace=True)

        class EagerMod(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.Conv2d_1a_3x3 = EagerConv2d(3, 32, kernel_size=3, stride=2)

            def forward(self, x):
                if False:
                    return 10
                x = self.Conv2d_1a_3x3(x)
                return F.dropout(x, training=self.training)
        script_input = torch.rand(4, 3, 299, 299)
        eager_input = script_input.clone()
        with freeze_rng_state():
            script_mod = ScriptMod()
            script_mod.eval()
            script_output = script_mod(script_input)
        with freeze_rng_state():
            eager_mod = EagerMod()
            eager_mod.eval()
            eager_output = eager_mod(eager_input)
        self.assertEqual(script_output, eager_output)
        with freeze_rng_state():
            script_mod = ScriptMod()
            script_mod.train()
            script_output = script_mod(script_input)
        with freeze_rng_state():
            eager_mod = EagerMod()
            eager_mod.train()
            eager_output = eager_mod(eager_input)
        self.assertEqual(script_output, eager_output)

    def test_nested_breaks(self):
        if False:
            while True:
                i = 10

        def no_bool_loop_outputs(g):
            if False:
                return 10
            loops = g.findAllNodes('prim::Loop')
            for loop in loops:
                for out in loop.outputs():
                    self.assertTrue(out.type() != BoolType.get())

        def test(y):
            if False:
                for i in range(10):
                    print('nop')
            ret = 0
            tensor = torch.tensor(0)
            while int(tensor.add_(1)) < 4:
                if y == 1:
                    continue
                for i in range(y):
                    continue
                    ret += 1
                ret += 1
            return (ret, int(tensor))
        self.assertEqual(torch.jit.script(test)(1), test(1))
        self.assertEqual(torch.jit.script(test)(2), test(2))
        no_bool_loop_outputs(torch.jit.script(test).graph)

        def foo():
            if False:
                i = 10
                return i + 15
            y = torch.tensor(0)
            z = 0
            while int(y.add_(1)) < 20:
                if int(y) < 10:
                    for i in range(6):
                        if i == 3:
                            continue
                        elif i > 3:
                            break
                        z += 2
                if int(y) == 18:
                    break
                if int(y) == 15:
                    continue
                z += 1
            return (int(y), z)
        no_bool_loop_outputs(torch.jit.script(foo).graph)
        self.checkScript(foo, ())

        def test_nested_two():
            if False:
                while True:
                    i = 10
            i = 0
            k = 0
            while i < 5:
                for j in range(5):
                    k += 1
                    if j == 3:
                        continue
                i += 1
                k += 1
                if i == 4:
                    break
            return (i, k)
        self.checkScript(test_nested_two, ())
        no_bool_loop_outputs(torch.jit.script(test_nested_two).graph)

    def test_breaks_continues(self):
        if False:
            return 10

        def foo_continue(cond):
            if False:
                while True:
                    i = 10
            j = 1
            for i in range(5):
                if i == cond:
                    continue
                j += 1
            return j

        def foo_break(cond):
            if False:
                i = 10
                return i + 15
            j = 1
            for i in range(5):
                if i == cond:
                    break
                j += 1
            return j
        for i in range(1, 4):
            self.checkScript(foo_continue, (i,))
            self.checkScript(foo_break, (i,))

        def test_refine_outside_loop():
            if False:
                return 10
            if 1 == 1:
                x = None
            else:
                x = 1
            i = 0
            j = 0
            while x is None or torch.jit._unwrap_optional(x) > 3:
                if i < 3:
                    if i < 3:
                        x = torch.jit.annotate(Optional[int], None)
                        i += 1
                        continue
                    x = 1
                else:
                    x = 1 if x is None else x
                x = x + 1
                j = x + x
            return (x, j)
        self.checkScript(test_refine_outside_loop, ())

        def assign_after_break(y):
            if False:
                while True:
                    i = 10
            x = 0
            for i in range(y):
                x = y * 2 + i
                break
                x = 4
            return x
        self.checkScript(assign_after_break, (1,))
        self.checkScript(assign_after_break, (2,))
        self.checkScript(assign_after_break, (3,))

        def assign_after_break_nested(y):
            if False:
                i = 10
                return i + 15
            x = 0
            for i in range(y):
                if y == 1:
                    x = 5
                    break
                    assert 1 == 2
                else:
                    x = x + 1
                    break
                    assert 1 == 2
                x = -30
                assert 1 == 2
            return x
        self.checkScript(assign_after_break_nested, (1,))
        self.checkScript(assign_after_break_nested, (2,))
        self.checkScript(assign_after_break_nested, (3,))

        def may_break(y):
            if False:
                while True:
                    i = 10
            x = 0
            for i in range(y):
                if y == 1:
                    x = 5
                else:
                    x = x + 1
                    break
                x = -30
            return x
        self.checkScript(may_break, (1,))
        self.checkScript(may_break, (2,))
        self.checkScript(may_break, (3,))

        def test(x, y):
            if False:
                print('Hello World!')
            a = 1
            while x > 0:
                if y == 3:
                    for i in range(y):
                        a += 1 % (i + 1)
                        x -= 1
                if x == 3:
                    a = x * 3
                    break
                if x < 3:
                    if x == 1:
                        a -= 2
                        x -= 1
                        break
                a -= 1
                x -= 3
            return (a, x)
        self.checkScript(test, (10, 3))
        self.checkScript(test, (10, 2))
        self.checkScript(test, (3, 2))
        self.checkScript(test, (5, 3))
        self.checkScript(test, (2, 3))

        def test_delete_after_break(x):
            if False:
                while True:
                    i = 10
            a = 1
            b = 1
            for i in range(x):
                a = i * 3
                break
                b = i * 5
            return (a, b)
        self.checkScript(test_delete_after_break, (0,))
        self.checkScript(test_delete_after_break, (1,))

        def test_will_break_after_guard(x):
            if False:
                for i in range(10):
                    print('nop')
            a = 1
            for i in range(x):
                if i == 4:
                    a = 3
                    break
                a -= 1
                break
                assert 1 == 2
                a -= -100
            return a
        self.checkScript(test_will_break_after_guard, (0,))
        self.checkScript(test_will_break_after_guard, (2,))
        self.checkScript(test_will_break_after_guard, (4,))

        def test_varexit(cond):
            if False:
                for i in range(10):
                    print('nop')
            m = 0
            for i in range(3):
                if cond == 2:
                    if cond == 2:
                        m = 2
                        break
                    k = 1
                else:
                    k = 2
                m += k
            return m
        self.checkScript(test_varexit, (3,))
        self.checkScript(test_varexit, (2,))

        def test_break_true():
            if False:
                while True:
                    i = 10
            i = 0
            while True:
                i += 1
                if i == 3:
                    break
            while False:
                i += 1
            return i
        self.checkScript(test_break_true, ())

    def test_break_continue_error(self):
        if False:
            return 10
        with self.assertRaisesRegex(RuntimeError, 'Syntax'):
            cu = torch.jit.CompilationUnit('\n            def other_func(a):\n                break\n                ')
        with self.assertRaisesRegex(RuntimeError, 'Syntax'):
            cu = torch.jit.CompilationUnit('\n            def other_func(a):\n                for i in range(5):\n                    def foo():\n                        break\n                ')
        with self.assertRaisesRegex(RuntimeError, 'do not support break or continue inside'):

            @torch.jit.script
            def foo(x):
                if False:
                    for i in range(10):
                        print('nop')
                i = 0
                for a in (1, '2', 1.5):
                    b = a
                    if x:
                        break
                return b

    def test_python_call(self):
        if False:
            return 10

        def pyfunc(a):
            if False:
                return 10
            return a * 3.0
        cu = torch.jit.CompilationUnit('\n        def other_func(a):\n            return a + a\n\n        def test_call_python(a):\n            b = pyfunc(a)\n            b = other_func(b)\n            i = 0\n            step = 1\n            while i < 10:\n                b = pyfunc(b)\n                if bool(b > 3.0):\n                    b = pyfunc(b)\n                i = 11\n            return b\n        ')
        inputs = self._make_scalar_vars([1], torch.float)
        outputs = self._make_scalar_vars([54], torch.float)
        self.assertEqual(cu.test_call_python(*inputs), outputs[0])

    def test_python_call_failure(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(RuntimeError, 'undefined value pyfunc2'):

            def pyfunc(a):
                if False:
                    while True:
                        i = 10
                return a * 3.0
            cu = torch.jit.CompilationUnit('\n            def other_func(a):\n                return a + a\n\n            def test_call_python(a):\n                b = pyfunc(a)\n                b = other_func(b)\n                i = 0\n                step = 1\n                while i < 10:\n                    b = pyfunc2(b)\n                    if b > 3.0:\n                        b = pyfunc(b)\n                    i = 11\n                return b\n            ')
            inputs = self._make_scalar_vars([1], torch.float)
            outputs = self._make_scalar_vars([54], torch.float)
            self.assertEqual(cu.test_call_python(*inputs), outputs)

    def test_type_call_in_script(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def fn(x):
            if False:
                i = 10
                return i + 15
            return type(x)
        with self.assertRaisesRegex(RuntimeError, 'value of type _TensorMeta'):
            fn(torch.tensor(0.5))

    def test_python_call_annotation(self):
        if False:
            for i in range(10):
                print('nop')

        def pyfunc(a):
            if False:
                for i in range(10):
                    print('nop')
            return a * 3.0

        @torch.jit.script
        def foo(a):
            if False:
                for i in range(10):
                    print('nop')
            return pyfunc(a) + pyfunc(a)
        inputs = self._make_scalar_vars([1], torch.float)
        outputs = self._make_scalar_vars([6], torch.float)
        self.assertEqual(foo(*inputs), outputs[0])

    def test_python_call_annoytation_failure(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(RuntimeError, 'undefined value pyfunc2'):

            def pyfunc(a):
                if False:
                    i = 10
                    return i + 15
                return a * 3.0

            @torch.jit.script
            def foo(a):
                if False:
                    for i in range(10):
                        print('nop')
                return pyfunc2(a) + pyfunc(a)
            inputs = self._make_scalar_vars([1], torch.float)
            outputs = self._make_scalar_vars([6], torch.float)
            self.assertEqual(foo(*inputs), outputs[0])

    def test_desugar_module(self):
        if False:
            i = 10
            return i + 15
        import torch.nn.functional as F

        def fn(x, slope):
            if False:
                while True:
                    i = 10
            a = torch.abs(x)
            b = torch.nn.functional.prelu(x, slope)
            c = F.prelu(x, slope)
            return (a, b, c)
        x = torch.arange(-3.0, 4)
        slope = torch.tensor([0.5])
        self.checkScript(fn, [x, slope], optimize=True)

    def test_script_docstring(self):
        if False:
            return 10

        @torch.jit.script
        def with_docstring(x):
            if False:
                return 10
            'test str'
            y = x
            'y is the same as x'
            return y
        self.assertEqual(with_docstring.__doc__, 'test str')

    def test_script_method_docstring(self):
        if False:
            print('Hello World!')

        class A(torch.jit.ScriptModule):

            @torch.jit.script_method
            def with_docstring(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                'test str'
                y = x
                'y is the same as x'
                return y
        a = A()
        self.assertEqual(a.with_docstring.__doc__, 'test str')

    def test_script_module(self):
        if False:
            return 10

        class M1(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.weight = nn.Parameter(torch.randn(2))

            @torch.jit.script_method
            def forward(self, thing):
                if False:
                    while True:
                        i = 10
                return self.weight + thing

        class PModule(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.a = nn.Parameter(torch.randn(2, 3))

            def forward(self, a):
                if False:
                    return 10
                return self.a.mm(a)

        class M2(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.sub = M1()
                self.sub2 = PModule()
                self.weight = nn.Parameter(torch.randn(2, 3))
                self.bias = nn.Parameter(torch.randn(2))
                self.define('\n                    def hi(self, a):\n                        return self.weight.mm(a)\n                ')

            @torch.jit.script_method
            def doit(self, input):
                if False:
                    while True:
                        i = 10
                return self.weight.mm(input)

            @torch.jit.script_method
            def doit2(self, input):
                if False:
                    return 10
                return self.weight.mm(input)

            @torch.jit.script_method
            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                a = self.doit(input)
                b = self.doit2(input)
                c = self.hi(input)
                d = self.sub2(input)
                return a + b + self.bias + self.sub(a) + c + d
        with torch.jit.optimized_execution(False):
            m2 = M2()
            input = torch.randn(3, 2)
            a = m2.weight.mm(input)
            b = m2.weight.mm(input)
            c = m2.weight.mm(input)
            d = m2.sub2.a.mm(input)
            ref = a + b + m2.bias + m2.sub.weight + a + c + d
            self.assertEqual(ref, m2.forward(input))
            m2.weight = nn.Parameter(torch.zeros_like(m2.weight))
            m2.bias = nn.Parameter(torch.zeros_like(m2.bias))
            m2.sub.weight = nn.Parameter(torch.zeros_like(m2.sub.weight))
            m2.sub2.a.data.zero_()
            self.assertEqual(torch.zeros(2, 2), m2.forward(torch.randn(3, 2)))

    def test_irparser(self):
        if False:
            return 10
        graph_str = 'graph(%0 : Double(5, 5)):\n          # CHECK: aten::relu\n          %1 : Double(5, 5) = aten::relu(%0)\n          return (%1)\n        '
        FileCheck().run(graph_str, parse_ir(graph_str))

    def test_parse_tensor_constants(self):
        if False:
            i = 10
            return i + 15

        def foo():
            if False:
                return 10
            return torch.zeros([4, 4])
        foo_s = torch.jit.script(foo)
        torch._C._jit_pass_constant_propagation(foo_s.graph)
        g = str(foo_s.graph)
        g_parsed = parse_ir(g, parse_tensor_constants=True)
        self.assertEqual(str(canonical(g_parsed)), str(canonical(foo_s.graph)))
        func = torch._C._create_function_from_graph('forward', g_parsed)
        out_parsed = func()
        out_func = foo()
        out_parsed[:] = 0
        out_func[:] = 0
        self.assertEqual(out_func, out_parsed)
        with self.assertRaises(RuntimeError):
            parse_ir(g, parse_tensor_constants=False)

    def test_parse_nested_names(self):
        if False:
            i = 10
            return i + 15
        g_str = '\n    graph(%x.1 : Tensor):\n        %3 : int = prim::Constant[value=1]()\n        %2 : int = prim::Constant[value=2]()\n        %hi.submod.value.5 : Tensor = aten::add(%x.1, %2, %3)\n        return (%hi.submod.value.5)\n        '
        g = parse_ir(g_str)
        round_trip_g = parse_ir(str(g))
        self.assertEqual(canonical(g), canonical(round_trip_g))
        func1 = torch._C._create_function_from_graph('forward', g)
        func2 = torch._C._create_function_from_graph('forward', round_trip_g)
        self.assertEqual(func1(torch.ones([2])), func2(torch.ones([2])))

    def test_is_after_use(self):
        if False:
            for i in range(10):
                print('nop')

        def sorted_input_use(g):
            if False:
                while True:
                    i = 10
            uses = list(next(g.inputs()).uses())
            return sorted(uses, key=functools.cmp_to_key(type(uses[0]).isAfter))

        @torch.jit.script
        def foo(x):
            if False:
                print('Hello World!')
            a = x + 1
            return (x, x, a)
        uses_sorted = sorted_input_use(foo.graph)
        self.assertFalse(uses_sorted[0].isAfter(uses_sorted[1]))
        self.assertTrue(uses_sorted[0].user.kind() == 'aten::add')
        self.assertEqual(uses_sorted[1].offset, 0)

        @torch.jit.script
        def foo(x, cond: bool):
            if False:
                print('Hello World!')
            if cond:
                return x + 3
            else:
                return x - 3
        uses_sorted = sorted_input_use(foo.graph)
        self.assertTrue(uses_sorted[0].user.kind() == 'aten::add')
        self.assertTrue(uses_sorted[1].user.kind() == 'aten::sub')

        @torch.jit.script
        def foo(x, cond: bool, cond2: bool):
            if False:
                for i in range(10):
                    print('nop')
            if cond:
                return x + 3
            elif cond2:
                return x - 3
            return x / 3
        graph1 = foo.graph

        @torch.jit.script
        def foo(x, cond: bool, cond2: bool):
            if False:
                while True:
                    i = 10
            if cond:
                return x + 3
            else:
                if cond2:
                    return x - 3
                return x / 3
        graph2 = foo.graph
        for graph in [graph1, graph2]:
            uses_sorted = sorted_input_use(graph)
            self.assertTrue(uses_sorted[0].user.kind() == 'aten::add')
            self.assertTrue(uses_sorted[1].user.kind() == 'aten::sub')
            self.assertTrue(uses_sorted[2].user.kind() == 'aten::div')

    def test_canonicalize_control_outputs(self):
        if False:
            print('Hello World!')

        def test_all_outputs(g):
            if False:
                while True:
                    i = 10
            ifs = g.findAllNodes('prim::If')
            loops = g.findAllNodes('prim::Loop')

            def contained_blocks(node):
                if False:
                    return 10
                return len(node.findAllNodes('prim::If')) * 2 + len(node.findAllNodes('prim::Loop'))
            for node in ifs + loops:
                outs = list(node.outputs())
                out_name = [x.debugName() for x in outs]
                if len(out_name) == 0:
                    continue
                fc = FileCheck()
                fc.check(out_name[-1] + ' : ')
                for i in range(contained_blocks(node)):
                    fc.check('->')
                if node.kind() == 'prim::If':
                    fc.check('->').check('->').check('\n')
                else:
                    fc.check('->').check('\n')
                for name in out_name:
                    fc.check(name)
                fc.run(g)

        @torch.jit.script
        def test(x):
            if False:
                print('Hello World!')
            b = 2
            a = 1
            if x:
                a = 1
                b = 2
                x = False
            if x:
                b = a
            else:
                a = b
            return (a, b)
        test_all_outputs(test.graph)

        @torch.jit.script
        def test2(x):
            if False:
                i = 10
                return i + 15
            b = 2
            a = 1
            if x:
                a = 1
                b = 2
                x = False
            if x:
                print(a)
            elif x:
                print(b)
            return (a, b)
        test_all_outputs(test2.graph)

        @torch.jit.script
        def test_loop(x, iter):
            if False:
                return 10
            a = 1
            b = 2
            c = 3
            for i in range(iter):
                a = 4
                b = 5
                c = 6
                x = True
            print(c)
            if x:
                print(a, b)
        test_all_outputs(test_loop.graph)

        @torch.jit.script
        def loop_unused(iter):
            if False:
                i = 10
                return i + 15
            a = 1
            b = 2
            c = 3
            for i in range(iter):
                c = c + 1
                b = b + 1
                a = a + 1
                print(a, b)
            print(c)
        FileCheck().check('%c : int, %a : int, %b : int').run(loop_unused.graph)

    def test_filecheck(self):
        if False:
            print('Hello World!')

        def test_check():
            if False:
                print('Hello World!')
            file = '232'
            FileCheck().check('2').check('3').check('2').run(file)
            FileCheck().check('232').run(file)
            with self.assertRaisesRegex(RuntimeError, 'Expected to find "22"'):
                FileCheck().check('22').run(file)
            with self.assertRaisesRegex(RuntimeError, 'CHECK: 3'):
                FileCheck().check('3').check('3').run(file)
        test_check()

        def test_check_count():
            if False:
                while True:
                    i = 10
            file = '22222'
            FileCheck().check_count('2', 5).run(file)
            FileCheck().check_count('22', 2).run(file)
            FileCheck().check_count('222', 1).run(file)
            with self.assertRaisesRegex(RuntimeError, 'Expected to not find'):
                FileCheck().check_count('2', 4, exactly=True).run(file)
            with self.assertRaisesRegex(RuntimeError, 'Expected to find "22"'):
                FileCheck().check_count('22', 3).run(file)
            with self.assertRaisesRegex(RuntimeError, 'CHECK-COUNT-6: 2'):
                FileCheck().check_count('2', 6).run(file)
        test_check_count()

        def test_check_same():
            if False:
                return 10
            file = '22\n33'
            FileCheck().check_same('22').run(file)
            with self.assertRaisesRegex(RuntimeError, 'Expected to not find'):
                FileCheck().check_same('33').run(file)
            file = '22  1  3'
            FileCheck().check('2').check_same('3').run(file)
            FileCheck().check_count('2', 2).check_same('3').run(file)
        test_check_same()

        def test_check_next():
            if False:
                for i in range(10):
                    print('nop')
            file = '\n1\n2\n3'
            FileCheck().check('1').check_next('2').check_next('3').run(file)
            FileCheck().check_next('1').check_next('2').check_next('3').run(file)
            with self.assertRaisesRegex(RuntimeError, 'Expected to find'):
                FileCheck().check('1').check_next('2').run('12')
            with self.assertRaisesRegex(RuntimeError, 'Expected to not find'):
                FileCheck().check('1').check_next('2').run('1\n\n2')
        test_check_next()

        def test_check_dag():
            if False:
                print('Hello World!')
            fc = FileCheck().check_dag('1').check_dag('2').check_not('2')
            fc.run('12')
            fc.run('21')
            fc = FileCheck()
            fc.check_not('3').check_dag('1').check_dag('2').check_not('3')
            fc.run('1 3 2')
            fc.run('2 3 1')
            fc = FileCheck().check_dag('1').check_dag('2').check('3')
            with self.assertRaisesRegex(RuntimeError, 'Expected to find "3" but did not find it'):
                fc.run('1 3 2')
        test_check_dag()

        def test_check_not():
            if False:
                return 10
            FileCheck().check_not('2').check('1').run('12')
            FileCheck().check('2').check_not('2').run('12')
            with self.assertRaisesRegex(RuntimeError, 'Expected to not find "2"'):
                FileCheck().check_not('2').check('1').run('21')
            with self.assertRaisesRegex(RuntimeError, 'Expected to not find "1"'):
                FileCheck().check('2').check_not('1').run('21')
            fb = FileCheck().check_count('2', 2).check_count('2', 2).check_not('2')
            with self.assertRaisesRegex(RuntimeError, 'Expected to not find "2"'):
                fb.run('22 2 22')
            fb = FileCheck().check_count('2', 2).check_not('1').check_count('2', 2)
            with self.assertRaisesRegex(RuntimeError, 'Expected to not find "1"'):
                fb.run('22 1 22')

    def _dtype_to_jit_name(self, dtype):
        if False:
            return 10
        if dtype == torch.float32:
            return 'Float'
        if dtype == torch.float64:
            return 'Double'
        if dtype == torch.int64:
            return 'Long'
        if dtype == torch.int32:
            return 'Int'
        if dtype == torch.bool:
            return 'Bool'
        raise RuntimeError('dtype not handled')

    def _dtype_to_expect(self, dtype, dim=0):
        if False:
            return 10
        param = ', '.join(['*'] * dim + ['device=cpu'])
        param = '(' + param + ')'
        jit_type = self._dtype_to_jit_name(dtype)
        if dim >= 0:
            return jit_type + param
        else:
            return jit_type.lower()

    def _test_dtype_op_shape(self, ops, args, input_dims=1):
        if False:
            i = 10
            return i + 15
        if input_dims < 1:
            raise RuntimeError('input dims must be at least 1')
        dtypes = [torch.float32, torch.float64, torch.int64, torch.int32]
        str_args = ', '.join([str(arg) for arg in args]) + (', ' if len(args) else '')
        tensor_data = '[' * input_dims + '1, 2, 3' + input_dims * ']'
        template = dedent('\n        def func():\n            return {return_line}\n        ')
        for op in ops:
            for dtype in dtypes + [None]:
                for tensor_type in dtypes:
                    if not tensor_type.is_floating_point or (dtype is not None and (not dtype.is_floating_point)):
                        if op in ['mean', 'softmax', 'log_softmax']:
                            continue
                    return_line = f'torch.tensor({tensor_data}, dtype={tensor_type}).{op}({str_args}dtype={dtype})'
                    code = template.format(return_line=return_line)
                    scope = {}
                    exec(code, globals(), scope)
                    cu = torch.jit.CompilationUnit(code)
                    graph = cu.func.graph
                    torch._C._jit_pass_complete_shape_analysis(graph, (), False)
                    input_array = [1, 2, 3]
                    for _ in range(1, input_dims):
                        input_array = [input_array]
                    t = torch.tensor(input_array, dtype=tensor_type)
                    attr = getattr(t, op)
                    kwargs = {'dtype': dtype}
                    result = attr(*args, **kwargs)
                    expect = self._dtype_to_expect(result.dtype, result.dim())
                    FileCheck().check('aten::tensor').check(expect).run(graph)

    def test_dtype_op_shape(self):
        if False:
            for i in range(10):
                print('nop')
        ops = ['prod']
        self._test_dtype_op_shape(ops, args=[])
        self._test_dtype_op_shape(ops, args=[0, False])
        self._test_dtype_op_shape(ops, args=[0, False])
        self._test_dtype_op_shape(ops, args=[0, True])

    def test_dtype_op_shape2(self):
        if False:
            i = 10
            return i + 15
        ops = ['cumprod', 'cumsum', 'softmax', 'log_softmax']
        self._test_dtype_op_shape(ops, args=[0])
        self._test_dtype_op_shape(ops, args=[1], input_dims=4)

    def _test_binary_op_shape(self, ops, input_dims=1):
        if False:
            while True:
                i = 10
        dtypes = [torch.float32, torch.float64, torch.int64, torch.int32, torch.bool]
        if input_dims == 0:
            shape = '1'
        else:
            shape = '[' + '1,' * 4 + ']'
            for _ in range(1, input_dims):
                shape = '[' + ','.join([shape] * 4) + ']'
        template = dedent('\n        def func():\n            arg1 = {}\n            arg2 = {}\n            return torch.{}(arg1, arg2)\n        ')
        args = []
        for dtype in dtypes:
            args = args + [f'torch.tensor({shape}, dtype={dtype})']
        args = args + [1, 1.5]

        def isBool(arg):
            if False:
                return 10
            return type(arg) == bool or (type(arg) == str and 'torch.bool' in arg)
        for op in ops:
            for first_arg in args:
                for second_arg in args:
                    if (op == 'sub' or op == 'div') and (isBool(first_arg) or isBool(second_arg)):
                        continue
                    if op == 'div' and (type(first_arg) != type(second_arg) or isinstance(first_arg, int) or (isinstance(first_arg, str) and 'int' in first_arg)):
                        continue
                    return_line = f'torch.{op}({first_arg}, {second_arg})'
                    code = template.format(first_arg, second_arg, op)
                    scope = {}
                    exec(code, globals(), scope)
                    non_jit_result = scope['func']()
                    cu = torch.jit.CompilationUnit(code)
                    graph = cu.func.graph
                    torch._C._jit_pass_complete_shape_analysis(graph, (), False)
                    dim = -1 if type(first_arg) != str and type(second_arg) != str else non_jit_result.dim()
                    dtype = non_jit_result.dtype
                    if dim < 0:
                        if dtype == torch.int64:
                            dtype = torch.int32
                        if dtype == torch.float64:
                            dtype = torch.float32
                    expect = self._dtype_to_expect(dtype, dim)
                    jit_output = next(graph.outputs())
                    check = FileCheck()
                    check.check(expect).run(str(jit_output))

    def test_binary_op_shape(self):
        if False:
            while True:
                i = 10
        self._test_binary_op_shape(['mul', 'div', 'add', 'sub'], 0)
        self._test_binary_op_shape(['mul', 'div', 'add', 'sub'], 3)

    def test_no_dtype_shape(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            scalar_number = x.item()
            return x.add(scalar_number)

        @torch.jit.script
        def foo2(x):
            if False:
                while True:
                    i = 10
            scalar_number = x.item()
            return torch.tensor(1).add(scalar_number)
        t = torch.tensor(5)
        g = foo.graph_for(t)
        type = next(g.outputs())
        self.assertTrue(type.type() == torch._C.TensorType.get())
        g2 = foo2.graph_for(t)
        type = next(g.outputs())
        self.assertTrue(type.type() == torch._C.TensorType.get())

    def test_filecheck_parse(self):
        if False:
            i = 10
            return i + 15

        def test_check():
            if False:
                print('Hello World!')
            file = '\n                # CHECK: 2\n                # CHECK: 3\n                # CHECK: 2\n                232\n                '
            FileCheck().run(checks_file=file, test_file=file)
            file = '\n                # CHECK: 232\n                232\n                '
            FileCheck().run(file, '232')
            with self.assertRaisesRegex(RuntimeError, 'Expected to find "232"'):
                FileCheck().run(file, '22')
            with self.assertRaisesRegex(RuntimeError, 'Expected to find "22"'):
                FileCheck().run('# CHECK: 22', '23')
        test_check()

        def test_check_count():
            if False:
                return 10
            file = '22222'
            FileCheck().run('# CHECK-COUNT-5: 2', file)
            FileCheck().run('# CHECK-COUNT-EXACTLY-5: 2', file)
            FileCheck().run('# CHECK-COUNT-2: 22', file)
            FileCheck().run('# CHECK-COUNT-1: 222', file)
            with self.assertRaisesRegex(RuntimeError, 'Expected to not find'):
                FileCheck().run('# CHECK-COUNT-EXACTLY-2: 2', file)
        test_check_count()

        def test_check_same():
            if False:
                print('Hello World!')
            file = '22\n33'
            FileCheck().run('# CHECK-SAME: 22', file)
            with self.assertRaisesRegex(RuntimeError, 'Expected to not find'):
                FileCheck().run('# CHECK-SAME: 33', file)
            file = '22  1  3'
            FileCheck().run('# CHECK: 2\n # CHECK-SAME: 3', file)
            FileCheck().run('# CHECK-COUNT-2: 2\n # CHECK-SAME: 3', file)
        test_check_same()

        def test_bad_input():
            if False:
                return 10
            with self.assertRaisesRegex(RuntimeError, 'Check for bad input'):
                FileCheck().run('', '1')
            with self.assertRaisesRegex(RuntimeError, 'Could not parse check'):
                FileCheck().run('# CHECK1', '')
        test_bad_input()

    def test_script_module_call_noscript(self):
        if False:
            print('Hello World!')

        class M(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.value = 1

            @torch.jit.ignore
            def foo(self):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.ones(2, 2) + self.value

            @torch.jit.script_method
            def forward(self, input):
                if False:
                    print('Hello World!')
                return input + self.foo()
        with torch.jit.optimized_execution(False):
            m = M()
            input = torch.randn(2, 2)
            o = m(input)
            self.assertEqual(o, input + torch.ones(2, 2) + 1)
            m.value = 2
            o = m(input)
            self.assertEqual(o, input + torch.ones(2, 2) + 2)

    def test_script_module_nochange_submodule(self):
        if False:
            while True:
                i = 10

        class M(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.sub = nn.Linear(5, 5)

            @torch.jit.script_method
            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return self.sub(input)
        with torch.jit.optimized_execution(False):
            m = M()
            input = torch.randn(1, 5, 5)
            o = m(input)
            self.assertEqual(o, m.sub(input))
            with self.assertRaisesRegex(RuntimeError, 'Cannot re-assign'):
                m.sub = nn.Linear(5, 5)

    def test_module_apis(self):
        if False:
            while True:
                i = 10

        class Sub(torch.nn.Module):

            def forward(self, thing):
                if False:
                    print('Hello World!')
                return thing - 2

        class Double(torch.nn.Module):

            def forward(self, thing):
                if False:
                    return 10
                return thing * 2

        class MyMod(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.mod = Sub()
                self.mod2 = Sub()
                self.mod3 = nn.Sequential(nn.Sequential(Sub()))
                self.mod4 = nn.Sequential(Sub(), Double())

            @torch.jit.export
            def method(self, x, x1, y, y1):
                if False:
                    print('Hello World!')
                mod_names = ''
                for (name, mod) in self.named_modules():
                    mod_names = mod_names + ' ' + name
                    x = mod(x)
                children_names = ''
                for (name, mod) in self.named_children():
                    children_names = children_names + ' ' + name
                    x1 = mod(x1)
                for mod in self.modules():
                    y = mod(y)
                for mod in self.children():
                    y1 = mod(y1)
                return (mod_names, children_names, x, x1, y, y1)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x + 2
        mod = torch.jit.script(MyMod())
        inps = tuple([torch.tensor(i) for i in range(1, 5)])
        self.assertEqual(mod.method(*inps), MyMod().method(*inps))

    def test_script_module_const(self):
        if False:
            return 10

        class M(torch.jit.ScriptModule):
            __constants__ = ['b', 'i', 'c', 's']

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.b = False
                self.i = 1
                self.c = 3.5
                self.s = ['hello']

            @torch.jit.script_method
            def forward(self):
                if False:
                    i = 10
                    return i + 15
                return (self.b, self.i, self.c)
        with torch.jit.optimized_execution(False):
            m = M()
            (o0, o1, o2) = m()
        self.assertEqual(o0, 0)
        self.assertEqual(o1, 1)
        self.assertEqual(o2, 3.5)

    def test_script_module_fail_exist(self):
        if False:
            print('Hello World!')

        class M(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x + self.whatisgoingon
        with self.assertRaisesRegex(RuntimeError, "Module 'M' has no attribute"):
            M()

    @unittest.skip("[module dedupe] currently NoneType refinement on optional attributes doesn't work.")
    def test_script_module_none_exist_fail(self):
        if False:
            while True:
                i = 10

        class M(torch.jit.ScriptModule):

            def __init__(self, my_optional):
                if False:
                    return 10
                super().__init__()
                self.my_optional = my_optional

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    while True:
                        i = 10
                if self.my_optional is not None:
                    return torch.neg(x) + self.my_optional
                return torch.neg(x)
        with self.assertRaisesRegex(RuntimeError, "has no attribute 'my_optional'"):
            x = torch.rand(3, 4)
            fb = M(None)
            fb(x)

    def test_script_module_invalid_consts(self):
        if False:
            for i in range(10):
                print('nop')

        class Foo(torch.jit.ScriptModule):
            __constants__ = ['invalid']

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.invalid = [nn.Linear(3, 4)]
        with self.assertRaisesRegex(TypeError, "Linear' object in attribute 'Foo.invalid' is not a valid constant"):
            Foo()

        class Foo2(torch.jit.ScriptModule):
            __constants__ = ['invalid']

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.invalid = int
        with self.assertRaisesRegex(TypeError, 'not a valid constant'):
            Foo2()

        class Foo3(torch.jit.ScriptModule):
            __constants__ = ['invalid']

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.invalid = (3, 4, {})
        with self.assertRaisesRegex(TypeError, 'not a valid constant'):
            Foo3()

        class Foo4(torch.jit.ScriptModule):
            __constants__ = ['invalid']

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.invalid = np.int64(5)
        with self.assertRaisesRegex(TypeError, 'numpy.int64'):
            Foo4()

    def test_script_module_param_buffer_mutation(self):
        if False:
            i = 10
            return i + 15

        class ModuleBufferMutate(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.register_buffer('running_var', torch.tensor(0, dtype=torch.long))

            @torch.jit.script_method
            def forward(self):
                if False:
                    print('Hello World!')
                if self.training:
                    self.running_var += 1
                return self.running_var
        with torch.jit.optimized_execution(False):
            m = ModuleBufferMutate()
            self.assertEqual(m(), 1)
            m.eval()
            self.assertEqual(m(), 1)

    def test_script_module_for(self):
        if False:
            return 10

        class M(torch.jit.ScriptModule):
            __constants__ = ['b']

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.b = [1, 2, 3, 4]

            @torch.jit.script_method
            def forward(self):
                if False:
                    return 10
                sum = 0
                for i in self.b:
                    sum += i
                return sum
        with torch.jit.optimized_execution(False):
            m = M()
            self.assertEqual(m(), 10)

    def test_override_magic(self):
        if False:
            print('Hello World!')

        class OverrideMagic(nn.Module):

            @torch.jit.export
            def __len__(self):
                if False:
                    i = 10
                    return i + 15
                return 10
        mod = OverrideMagic()
        self.assertEqual(len(mod), len(torch.jit.script(mod)))

        class OverrideMagicSeq(nn.Sequential):

            @torch.jit.export
            def __len__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 10
        mod = OverrideMagicSeq()
        self.assertEqual(len(mod), len(torch.jit.script(mod)))
        self.assertTrue(torch.jit.script(mod))

    def test_script_module_for2(self):
        if False:
            for i in range(10):
                print('nop')

        class Sub(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.weight = nn.Parameter(torch.randn(2))

            @torch.jit.script_method
            def forward(self, thing):
                if False:
                    for i in range(10):
                        print('nop')
                return self.weight + thing

        class M(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.mods = nn.ModuleList([Sub() for i in range(10)])

            @torch.jit.script_method
            def forward(self, v):
                if False:
                    print('Hello World!')
                for m in self.mods:
                    v = m(v)
                return v
        with torch.jit.optimized_execution(False):
            i = torch.empty(2)
            m = M()
            o = m(i)
            v = i
            for sub in m.mods:
                v = sub(v)
            self.assertEqual(o, v)
            with self.assertRaisesRegex(Exception, 'object is not iterable'):
                print(list(m))

    def test_attr_qscheme_script(self):
        if False:
            print('Hello World!')

        class Foo(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.qscheme = torch.per_tensor_affine

            def forward(self):
                if False:
                    i = 10
                    return i + 15
                if self.qscheme == torch.per_tensor_symmetric:
                    return 3
                else:
                    return 4
        f = Foo()
        scripted = torch.jit.script(f)
        self.assertEqual(f(), scripted())

    def test_script_module_const_submodule_fail(self):
        if False:
            while True:
                i = 10

        class Sub(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.weight = nn.Parameter(torch.randn(2))

            @torch.jit.script_method
            def forward(self, thing):
                if False:
                    i = 10
                    return i + 15
                return self.weight + thing

        class M(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.mods = [Sub() for _ in range(10)]

            @torch.jit.script_method
            def forward(self):
                if False:
                    print('Hello World!')
                for _ in self.mods:
                    print(1)
                return 4
        with self.assertRaisesRegex(RuntimeError, "has no attribute 'mods'"):
            M()

    class DerivedStateModule(torch.jit.ScriptModule):

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            super(TestScript.DerivedStateModule, self).__init__()
            self.param = torch.nn.Parameter(torch.ones(3, 4, dtype=torch.float))
            self.register_buffer('derived', torch.neg(self.param).detach().clone())
            self.register_buffer('pack_called', torch.zeros(1, dtype=torch.long))
            self.register_buffer('unpack_called', torch.zeros(1, dtype=torch.long))

        @torch.jit.script_method
        def _pack(self):
            if False:
                return 10
            self.pack_called.set_(torch.ones(1, dtype=torch.long))
            self.derived.set_(torch.rand(1).detach())

        @torch.jit.script_method
        def _unpack(self):
            if False:
                i = 10
                return i + 15
            self.unpack_called.set_(torch.ones(1, dtype=torch.long))
            self.derived.set_(torch.neg(self.param).detach())

        @torch.jit.script_method
        def forward(self, x):
            if False:
                while True:
                    i = 10
            return x + self.derived

    def test_pack_unpack_state(self):
        if False:
            return 10
        sm = TestScript.DerivedStateModule()
        x = torch.rand(3, 4)
        torch.testing.assert_close(sm(x), x + torch.neg(torch.ones(3, 4, dtype=torch.float)))
        self.assertFalse(sm.pack_called.item())
        self.assertFalse(sm.unpack_called.item())
        imported = self.getExportImportCopyWithPacking(sm)
        self.assertTrue(sm.pack_called.item())
        self.assertTrue(sm.unpack_called.item())
        torch.testing.assert_close(sm.derived, torch.neg(sm.param))
        self.assertTrue(imported.unpack_called.item())
        torch.testing.assert_close(imported(x), x + torch.neg(torch.ones(3, 4, dtype=torch.float)))

    @unittest.skipIf(not TEST_MKL, 'PyTorch is built without MKL support')
    @unittest.skipIf(True, 'Skipping while landing PR stack')
    def test_torch_functional(self):
        if False:
            for i in range(10):
                print('nop')

        def stft(input, n_fft):
            if False:
                for i in range(10):
                    print('nop')
            return torch.stft(input, n_fft, return_complex=True)
        inps = (torch.randn(10), 7)
        self.assertEqual(stft(*inps), torch.jit.script(stft)(*inps))

        def istft(input, n_fft):
            if False:
                print('Hello World!')
            return torch.istft(input, n_fft)
        inps2 = (stft(*inps), inps[1])
        self.assertEqual(istft(*inps2), torch.jit.script(istft)(*inps2))

        def lu_unpack(x):
            if False:
                while True:
                    i = 10
            (A_LU, pivots) = torch.linalg.lu_factor(x)
            return torch.lu_unpack(A_LU, pivots)
        for shape in ((3, 3), (5, 3, 3), (7, 3, 5, 5), (7, 5, 3, 3, 3)):
            a = torch.randn(*shape)
            self.checkScript(lu_unpack, (a,))

        def cdist_fn():
            if False:
                i = 10
                return i + 15
            a = torch.tensor([[0.9041, 0.0196], [-0.3108, -2.4423], [-0.4821, 1.059]])
            b = torch.tensor([[-2.1763, -0.4713], [-0.6986, 1.3702]])
            return torch.cdist(a, b, compute_mode='use_mm_for_euclid_dist')
        self.checkScript(cdist_fn, ())

        def norm():
            if False:
                for i in range(10):
                    print('nop')
            c = torch.tensor([[1, 2, 3], [-1, 1, 4]], dtype=torch.float)
            return (torch.norm(c, p='fro'), torch.norm(c, p='nuc'), torch.norm(c), torch.norm(c, p=0.5))
        self.checkScript(norm, ())

        def torch_unique(dim: Optional[int]):
            if False:
                while True:
                    i = 10
            ten = torch.unique(torch.tensor([[1, 3], [2, 3]], dtype=torch.long))
            a = torch.unique(ten, dim=dim)
            b = torch.unique(ten, return_counts=True, dim=dim)
            c = torch.unique(ten, return_inverse=True, dim=dim)
            d = torch.unique(ten, return_counts=True, return_inverse=True, dim=dim)
            return (a, b, c, d)
        self.checkScript(torch_unique, (None,))
        self.checkScript(torch_unique, (0,))

        def torch_unique_consecutive(dim: Optional[int]):
            if False:
                for i in range(10):
                    print('nop')
            ten = torch.unique(torch.tensor([[1, 3], [3, 2], [3, 2], [2, 3]], dtype=torch.long))
            a = torch.unique_consecutive(ten, dim=dim)
            b = torch.unique_consecutive(ten, return_counts=True, dim=dim)
            c = torch.unique_consecutive(ten, return_inverse=True, dim=dim)
            d = torch.unique_consecutive(ten, return_counts=True, return_inverse=True, dim=dim)
            return (a, b, c, d)
        self.checkScript(torch_unique_consecutive, (None,))
        self.checkScript(torch_unique_consecutive, (0,))

    def test_torch_functional_tensordot_int(self):
        if False:
            while True:
                i = 10

        def tensordot_dims_int(a: torch.Tensor, b: torch.Tensor, dims: int):
            if False:
                while True:
                    i = 10
            return torch.tensordot(a, b, dims=dims)
        a = torch.arange(120.0).reshape(2, 3, 4, 5)
        b = torch.arange(840.0).reshape(4, 5, 6, 7)
        dims = 2
        self.checkScript(tensordot_dims_int, (a, b, dims))
        for dims in [-1, 5]:
            try:
                tensordot_dims_int(a, b, dims)
            except RuntimeError as error:
                if dims < 0:
                    self.assertEqual(str(error), 'tensordot expects dims >= 0, but got dims=' + str(dims))
                if dims > min(a.dim(), b.dim()):
                    self.assertEqual(str(error), 'tensordot expects dims < ndim_a or ndim_b, but got dims=' + str(dims))

    def test_torch_functional_tensordot_tensor(self):
        if False:
            return 10

        def tensordot_dims_tensor(a: torch.Tensor, b: torch.Tensor, dims: torch.Tensor):
            if False:
                while True:
                    i = 10
            return torch.tensordot(a, b, dims=dims)
        a = torch.arange(120.0).reshape(2, 3, 4, 5)
        b = torch.arange(840.0).reshape(4, 5, 6, 7)
        dims = torch.tensor([2])
        self.checkScript(tensordot_dims_tensor, (a, b, dims))
        a = torch.arange(60.0).reshape(3, 4, 5)
        b = torch.arange(24.0).reshape(4, 3, 2)
        dims = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)
        self.checkScript(tensordot_dims_tensor, (a, b, dims))

    def test_torch_functional_tensordot_list(self):
        if False:
            while True:
                i = 10

        def tensordot_dims_list(a: torch.Tensor, b: torch.Tensor, dims: List[List[int]]):
            if False:
                for i in range(10):
                    print('nop')
            return torch.tensordot(a, b, dims=dims)
        a = torch.arange(60.0).reshape(3, 4, 5)
        b = torch.arange(24.0).reshape(4, 3, 2)
        dims = [[1, 0], [0, 1]]
        self.checkScript(tensordot_dims_list, (a, b, dims))

    def test_torch_functional_tensordot_tuple(self):
        if False:
            return 10

        def tensordot_dims_tuple(a: torch.Tensor, b: torch.Tensor, dims: Tuple[List[int], List[int]]):
            if False:
                for i in range(10):
                    print('nop')
            return torch.tensordot(a, b, dims=dims)
        a = torch.arange(60.0).reshape(3, 4, 5)
        b = torch.arange(24.0).reshape(4, 3, 2)
        dims = ([1, 0], [0, 1])
        self.checkScript(tensordot_dims_tuple, (a, b, dims))

    def test_missing_getstate(self):
        if False:
            for i in range(10):
                print('nop')

        class Foo(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.x = 1

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return x * self.x

            @torch.jit.export
            def __setstate__(self, state):
                if False:
                    return 10
                self.x = state[0]
                self.training = state[1]
        with self.assertRaisesRegex(RuntimeError, 'getstate'):
            scripted = torch.jit.script(Foo())

    def test_inlining_cleanup(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(x):
            if False:
                while True:
                    i = 10
            return F.linear(x, x)

        @torch.jit.script
        def fee(x):
            if False:
                return 10
            return foo(x)
        self.run_pass('inline', fee.graph)
        FileCheck().check_not('prim::If').run(fee.graph)

    @skipIfTorchDynamo('TorchDynamo fails with unknown reason')
    def test_pack_unpack_nested(self):
        if False:
            i = 10
            return i + 15

        class SubSubMod(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.register_buffer('buf', torch.ones(3, 4) * 3)

            @torch.jit.script_method
            def _pack(self):
                if False:
                    i = 10
                    return i + 15
                self.buf.set_(torch.zeros(1))

            @torch.jit.script_method
            def _unpack(self):
                if False:
                    while True:
                        i = 10
                self.buf.set_(torch.ones(3, 4) * 3)

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    return 10
                return x + self.buf

        class SubMod(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.register_buffer('buf', torch.ones(3, 4) * 2)
                self.ssm = SubSubMod()

            @torch.jit.script_method
            def _pack(self):
                if False:
                    print('Hello World!')
                self.buf.set_(torch.zeros(1))

            @torch.jit.script_method
            def _unpack(self):
                if False:
                    while True:
                        i = 10
                self.buf.set_(torch.ones(3, 4) * 2)

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.ssm(x + self.buf)

        class Mod(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.submod = SubMod()
                self.register_buffer('buf', torch.ones(3, 4) * 1)

            @torch.jit.script_method
            def _pack(self):
                if False:
                    i = 10
                    return i + 15
                self.buf.set_(torch.zeros(1))

            @torch.jit.script_method
            def _unpack(self):
                if False:
                    while True:
                        i = 10
                self.buf.set_(torch.ones(3, 4))

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.submod(x + self.buf)
        m = Mod()
        torch.testing.assert_close(m(torch.zeros(3, 4)), torch.ones(3, 4) * 6)
        m.apply(lambda s: s._pack())
        torch.testing.assert_close(m(torch.zeros(3, 4)), torch.zeros(3, 4))
        m.apply(lambda s: s._unpack())
        torch.testing.assert_close(m(torch.zeros(3, 4)), torch.ones(3, 4) * 6)

    def test_torch_any(self):
        if False:
            print('Hello World!')

        def fn(x):
            if False:
                i = 10
                return i + 15
            return torch.any(x)

        def fn1(x, dim: int):
            if False:
                for i in range(10):
                    print('nop')
            return torch.any(x, dim)
        self.checkScript(fn, (torch.randn(3, 4),))
        self.checkScript(fn, (torch.empty(3),))
        self.checkScript(fn, (torch.empty(1),))
        self.checkScript(fn, (torch.ones(3, 4),))
        self.checkScript(fn, (torch.zeros(5, 7, 1),))
        self.checkScript(fn1, (torch.empty(3, 4), -2))
        self.checkScript(fn1, (torch.randn(3, 8), 1))
        self.checkScript(fn1, (torch.zeros(3, 6, 9), -3))
        self.checkScript(fn1, (torch.empty(5), 0))

    def test_any(self):
        if False:
            i = 10
            return i + 15

        def fn(x: List[int]):
            if False:
                for i in range(10):
                    print('nop')
            return any(x)

        def fn1(x: List[float]):
            if False:
                while True:
                    i = 10
            return any(x)

        def fn2(x: List[bool]):
            if False:
                i = 10
                return i + 15
            return any(x)

        def fn3(x: List[str]):
            if False:
                return 10
            return any(x)
        self.checkScript(fn, ([0, 0, 0, 0],))
        self.checkScript(fn, ([0, 3, 0],))
        self.checkScript(fn, ([],))
        self.checkScript(fn1, ([1.0, 2.0, 3.0],))
        self.checkScript(fn1, ([0.0, 0.0, 0.0],))
        self.checkScript(fn1, ([0, 0, 0],))
        self.checkScript(fn1, ([],))
        self.checkScript(fn2, ([True, False, False],))
        self.checkScript(fn2, ([False, False, False],))
        self.checkScript(fn2, ([True, True, True, True],))
        self.checkScript(fn2, ([],))
        self.checkScript(fn3, (['', '', ''],))
        self.checkScript(fn3, (['', '', '', '-1'],))
        self.checkScript(fn3, ([],))

    def test_script_module_not_tuple(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.jit.ScriptModule):
            __constants__ = ['mods']

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.mods = 1

            @torch.jit.script_method
            def forward(self, v):
                if False:
                    for i in range(10):
                        print('nop')
                for m in self.mods:
                    print(m)
                return v
        with self.assertRaisesRegex(RuntimeError, "'int' object is not iterable"):
            M()

    def test_attr_module_constants(self):
        if False:
            for i in range(10):
                print('nop')

        class M2(torch.jit.ScriptModule):

            def __init__(self, mod_list):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.mods = mod_list

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.mods.forward(x)
        with torch.jit.optimized_execution(False):
            m = M2(nn.Sequential(nn.ReLU()))
            self.assertExportImportModule(m, (torch.randn(2, 2),))

    def test_script_sequential_for(self):
        if False:
            for i in range(10):
                print('nop')

        class Sub(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.weight = nn.Parameter(torch.randn(2))

            @torch.jit.script_method
            def forward(self, thing):
                if False:
                    print('Hello World!')
                return self.weight + thing

        class M(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.mods = nn.Sequential(Sub(), Sub(), Sub())

            @torch.jit.script_method
            def forward(self, v):
                if False:
                    while True:
                        i = 10
                for m in self.mods:
                    v = m(v)
                return v

            @torch.jit.script_method
            def forward2(self, v):
                if False:
                    print('Hello World!')
                return self.mods(v)
        with torch.jit.optimized_execution(False):
            i = torch.empty(2)
            m = M()
            o = m(i)
            v = i
            for sub in m.mods._modules.values():
                v = sub(v)
            self.assertEqual(o, v)
            o2 = m.forward2(i)
            self.assertEqual(o2, v)

    def test_script_sequential_sliced_iteration(self):
        if False:
            i = 10
            return i + 15

        class seq_mod(nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.layers = [nn.ReLU(), nn.ReLU(), nn.ReLU()]
                self.layers = nn.Sequential(*self.layers)

            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.layers[0].forward(input)
                for layer in self.layers[1:3]:
                    x = layer.forward(x)
                for layer in self.layers[2:]:
                    x = layer.forward(x)
                return x
        seq = seq_mod()
        self.checkModule(seq, [torch.tensor([-2, 1, -1, 2])])

    def test_script_sequential_orderdict(self):
        if False:
            print('Hello World!')

        class M(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.mods = nn.Sequential(OrderedDict([('conv', nn.Conv2d(1, 20, 5)), ('relu', nn.ReLU())]))

            @torch.jit.script_method
            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                return self.mods(input)
        m = M()
        self.assertTrue('mods.conv.weight' in m.state_dict().keys())

    def test_script_sequential_multi_output_fail(self):
        if False:
            return 10

        class Sub(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.weight = nn.Parameter(torch.randn(2))

            @torch.jit.script_method
            def forward(self, thing):
                if False:
                    return 10
                return self.weight + thing

        class ReturnMulti(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    print('Hello World!')
                return (x, x, x)

        class HaveSequential(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.someseq = nn.Sequential(Sub(), ReturnMulti(), Sub())

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.someseq(x)
        with self.assertRaisesRegex(RuntimeError, '(Tensor, Tensor, Tensor)'):
            with torch.jit.optimized_execution(False):
                hs = HaveSequential()
                i = torch.empty(2)
                hs(i)

    @_tmp_donotuse_dont_inline_everything
    def test_script_sequential_in_mod_list(self):
        if False:
            return 10

        class Sub(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.weight = nn.Parameter(torch.randn(2))

            @torch.jit.script_method
            def forward(self, thing):
                if False:
                    i = 10
                    return i + 15
                return self.weight + thing

        class M(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.mods = nn.ModuleList([Sub(), nn.Sequential(Sub(), nn.Sequential(Sub(), Sub()), Sub())])

            @torch.jit.script_method
            def forward(self, v):
                if False:
                    return 10
                for mod in self.mods:
                    v = mod(v)
                return v
        m = M()
        graph = str(m.graph)
        self.assertTrue(graph.count('prim::CallMethod') == 2)
        self.assertTrue('python' not in graph)

    @_tmp_donotuse_dont_inline_everything
    def test_script_nested_mod_list(self):
        if False:
            i = 10
            return i + 15

        class Sub(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.weight = nn.Parameter(torch.randn(2))

            @torch.jit.script_method
            def forward(self, thing):
                if False:
                    print('Hello World!')
                return self.weight + thing

        class M(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.mods = nn.ModuleList([nn.ModuleList([Sub()]), nn.Sequential(Sub()), nn.ModuleList([Sub(), Sub()])])

            @torch.jit.script_method
            def forward(self, v):
                if False:
                    print('Hello World!')
                for mod in self.mods:
                    for m in mod:
                        v = m(v)
                return v
        m = M()
        graph = str(m.graph)
        self.assertTrue(graph.count('prim::CallMethod') == 4)
        self.assertTrue('python' not in graph)

    def test_constant_as_attr(self):
        if False:
            while True:
                i = 10

        class M(torch.jit.ScriptModule):
            __constants__ = ['dim']

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.dim = 1

            @torch.jit.script_method
            def forward(self, v):
                if False:
                    return 10
                return torch.cat([v, v, v], dim=self.dim)
        v = torch.zeros(1, 1)
        with torch.jit.optimized_execution(False):
            self.assertEqual(torch.cat([v, v, v], dim=1), M()(v))

    class StarTestSumStarred(torch.nn.Module):

        def __init__(self):
            if False:
                while True:
                    i = 10
            super(TestScript.StarTestSumStarred, self).__init__()

        def forward(self, *inputs):
            if False:
                i = 10
                return i + 15
            output = inputs[0]
            for i in range(1, len(inputs)):
                output += inputs[i]
            return output

    class StarTestReturnThree(torch.nn.Module):

        def __init__(self):
            if False:
                i = 10
                return i + 15
            super(TestScript.StarTestReturnThree, self).__init__()

        def forward(self, rep):
            if False:
                return 10
            return (rep, rep, rep)

    def test_script_star_expr(self):
        if False:
            while True:
                i = 10

        class M2(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.m = torch.jit.trace(TestScript.StarTestSumStarred(), (torch.ones(4, 3), torch.ones(4, 3), torch.ones(4, 3)))
                self.g = torch.jit.trace(TestScript.StarTestReturnThree(), torch.ones(4, 3))

            @torch.jit.script_method
            def forward(self, rep):
                if False:
                    while True:
                        i = 10
                tup = self.g(rep)
                return self.m(*tup)
        m = M2()
        self.assertEqual(m(torch.zeros(4, 3)), 3 * torch.zeros(4, 3))

    def test_script_star_expr_string(self):
        if False:
            while True:
                i = 10

        class M2(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.m = torch.jit.trace(TestScript.StarTestSumStarred(), (torch.ones(4, 3), torch.ones(4, 3), torch.ones(4, 3)))
                self.g = torch.jit.trace(TestScript.StarTestReturnThree(), torch.ones(4, 3))
                self.define('\n            def forward(self, rep):\n                tup = self.g(rep)\n                return self.m(*tup)\n                ')
        m = M2()
        self.assertEqual(m(torch.zeros(4, 3)), 3 * torch.zeros(4, 3))

    class StarTestSumAndReturnThree(torch.nn.Module):

        def __init__(self):
            if False:
                while True:
                    i = 10
            super(TestScript.StarTestSumAndReturnThree, self).__init__()

        def forward(self, *inputs):
            if False:
                for i in range(10):
                    print('nop')
            output = inputs[0]
            for i in range(1, len(inputs)):
                output += inputs[i]
            return (output, output, output)

    def test_script_star_assign(self):
        if False:
            for i in range(10):
                print('nop')

        class M2(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.g = torch.jit.trace(TestScript.StarTestSumAndReturnThree(), torch.ones(4, 3))
                self.define('\n            def forward(self, rep):\n                head, *tail = self.g(rep)\n                return head\n                ')
        m = M2()
        self.assertEqual(m(torch.zeros(4, 3)), 3 * torch.zeros(4, 3))

    def test_script_module_star_assign2(self):
        if False:
            return 10

        class M2(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.g = torch.jit.trace(TestScript.StarTestSumAndReturnThree(), (torch.ones(4, 3), torch.ones(4, 3), torch.ones(4, 3)), _force_outplace=True)
                self.define('\n            def forward(self, rep):\n                *head, tail = self.g(rep, rep, rep)\n                return tail\n                ')
        m = M2()
        self.assertEqual(m(torch.ones(4, 3)), 3 * torch.ones(4, 3))

    def test_script_module_star_assign2_inplace(self):
        if False:
            while True:
                i = 10

        class M2(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.g = torch.jit.trace(TestScript.StarTestSumAndReturnThree(), (torch.ones(4, 3), torch.ones(4, 3), torch.ones(4, 3)), _force_outplace=False)
                self.define('\n            def forward(self, rep):\n                *head, tail = self.g(rep, rep, rep)\n                return tail\n                ')
        m = M2()
        self.assertEqual(m(torch.ones(4, 3)), 4 * torch.ones(4, 3))

    def test_script_module_star_assign_fail_pythonop(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(RuntimeError, 'cannot be used as a tuple'):

            class M2(torch.jit.ScriptModule):

                def __init__(self):
                    if False:
                        while True:
                            i = 10
                    super().__init__()

                    @torch.jit.ignore
                    def myfunc():
                        if False:
                            i = 10
                            return i + 15
                        return (torch.zeros(1, 2, 3), torch.zeros(1, 2, 3))
                    self.define('\n                def forward(self, rep):\n                    a, *b = myfunc()\n                    return a\n                    ')
            m = M2()
            m(torch.zeros(4, 3))

    def test_script_module_star_assign_fail_builtin(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(RuntimeError, 'cannot be used as a tuple'):

            class M2(torch.jit.ScriptModule):

                def __init__(self):
                    if False:
                        print('Hello World!')
                    super().__init__()
                    self.define('\n                def forward(self, rep):\n                    a, *b = torch.neg(rep)\n                    return a\n                    ')
            m = M2()
            m(torch.zeros(4, 3))

    def test_script_pack_padded_sequence(self):
        if False:
            return 10
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

        def pack_padded_pad_packed_script(x, seq_lens):
            if False:
                i = 10
                return i + 15
            x = pack_padded_sequence(x, seq_lens)
            (x, lengths) = pad_packed_sequence(x)
            return (x, lengths)
        (T, B, C) = (3, 5, 7)
        x = torch.ones((T, B, C))
        seq_lens = torch.tensor([3, 3, 2, 2, 1])
        for b in range(B):
            if seq_lens[b] < T:
                x[seq_lens[b]:, b, :] = 0
        (eager_seq, eager_lengths) = pack_padded_pad_packed_script(x, seq_lens)
        with torch._jit_internal._disable_emit_hooks():
            scripted_pack_padded_seq = torch.jit.script(pack_padded_pad_packed_script)
        (script_seq, script_lengths) = scripted_pack_padded_seq(x, seq_lens)
        self.assertEqual(eager_seq, script_seq)
        self.assertEqual(eager_lengths, script_lengths)

        class ExperimentalLSTM(torch.nn.Module):

            def __init__(self, input_dim, hidden_dim):
                if False:
                    i = 10
                    return i + 15
                super().__init__()

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                packed = pack_padded_sequence(input=input, lengths=torch.tensor([1, 2]), enforce_sorted=False)
                (output, lengths) = pad_packed_sequence(sequence=packed, total_length=2)
                return output[0]
        lstm = ExperimentalLSTM(input_dim=2, hidden_dim=2)
        with torch._jit_internal._disable_emit_hooks():
            self.checkModule(lstm, [torch.ones(2, 2)])

    def test_script_pad_sequence_pack_sequence(self):
        if False:
            for i in range(10):
                print('nop')
        from torch.nn.utils.rnn import pad_sequence, pack_sequence, pad_packed_sequence

        def pad_sequence_func(tensor_list, batch_first=False, padding_value=0.0):
            if False:
                for i in range(10):
                    print('nop')
            return pad_sequence(tensor_list, batch_first, padding_value)

        def pack_sequence_func(tensor_list, enforce_sorted=True):
            if False:
                for i in range(10):
                    print('nop')
            return pad_packed_sequence(pack_sequence(tensor_list, enforce_sorted))[0]
        ones3 = torch.ones(3, 5)
        ones4 = torch.ones(4, 5)
        ones5 = torch.ones(5, 5)
        tensor1 = torch.tensor([1, 2, 3])
        tensor2 = torch.tensor([4, 5])
        tensor3 = torch.tensor([6])
        with torch._jit_internal._disable_emit_hooks():
            self.checkScript(pad_sequence_func, ([ones3, ones4, ones5],))
            self.checkScript(pad_sequence_func, ([ones3, ones4, ones5], True))
            self.checkScript(pad_sequence_func, ([ones3, ones4, ones5], True, 2.5))
            self.checkScript(pack_sequence_func, ([tensor1, tensor2, tensor3],))
            self.checkScript(pack_sequence_func, ([tensor1, tensor2, tensor3], False))

    def test_script_get_tracing_state(self):
        if False:
            for i in range(10):
                print('nop')

        def test_if_tracing(x):
            if False:
                print('Hello World!')
            if torch._C._get_tracing_state():
                return x + 1
            else:
                return x - 1
        inp = torch.randn(3, 3)
        self.checkScript(test_if_tracing, (inp,))

    def test_script_is_tracing(self):
        if False:
            return 10

        def test_is_tracing(x):
            if False:
                return 10
            if torch.jit.is_tracing():
                return x + 1
            else:
                return x - 1
        inp = torch.randn(3, 3)
        self.checkScript(test_is_tracing, (inp,))

    def test_is_scripting(self):
        if False:
            i = 10
            return i + 15

        def foo():
            if False:
                for i in range(10):
                    print('nop')
            return torch.jit.is_scripting()
        self.assertFalse(foo())
        scripted = torch.jit.script(foo)
        self.assertTrue(scripted())

    def test_comment_ignore_indent(self):
        if False:
            while True:
                i = 10

        class Model(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()

            def forward(self):
                if False:
                    print('Hello World!')
                return 5
        self.checkModule(Model(), ())

    def test_script_outputs(self):
        if False:
            return 10
        with self.assertRaisesRegex(RuntimeError, 'cannot be used as a tuple'):

            @torch.jit.script
            def foo(a):
                if False:
                    print('Hello World!')
                (c, d) = a + a
                return c + d

        @torch.jit.script
        def return3():
            if False:
                for i in range(10):
                    print('nop')
            return (1, 2, 3)
        with self.assertRaisesRegex(RuntimeError, 'too many values to unpack'):

            @torch.jit.script
            def bind2():
                if False:
                    while True:
                        i = 10
                (a, b) = return3()
                print(a)
                print(b)

    @unittest.skipIf(not RUN_CUDA, 'requires CUDA')
    def test_script_get_device_cuda(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def foo(a):
            if False:
                return 10
            return a.get_device()
        v = torch.randn(1, device='cuda')
        self.assertEqual(foo(v), 0)

    def test_script_chunk(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def foo(a):
            if False:
                while True:
                    i = 10
            (b, c) = torch.chunk(a, dim=0, chunks=2)
            return b
        v = torch.rand(10, 3)
        self.assertEqual(torch.chunk(v, dim=0, chunks=2)[0], foo(v))

    def test_script_copy(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):
            __annotations__ = {'val': Optional[torch.Tensor]}

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.val = None

            def some_method(self):
                if False:
                    while True:
                        i = 10
                return 3

            def forward(self, x):
                if False:
                    return 10
                self.val = x + self.some_method()
                return x
        m = torch.jit.script(M())
        copy.copy(m)
        copy.deepcopy(m)

    def test_script_forward_method_replacement(self):
        if False:
            for i in range(10):
                print('nop')

        class LowLevelModule(torch.nn.Module):

            def forward(self, input: torch.Tensor):
                if False:
                    print('Hello World!')
                return self.forward_pytorch(input) * 2

        class TestModule(LowLevelModule):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.forward = types.MethodType(LowLevelModule.forward, self)

            def forward_pytorch(self, input: torch.Tensor):
                if False:
                    return 10
                return torch.tensor(123)

            def forward(self, input: torch.Tensor):
                if False:
                    i = 10
                    return i + 15
                raise AssertionError('This method should not be used')
                return self.forward_pytorch(input)
        m = TestModule()
        self.assertEqual(m(torch.tensor(1)), torch.tensor(246))
        m_scripted = torch.jit.script(m)
        self.assertEqual(m_scripted(torch.tensor(1)), torch.tensor(246))

    def test_python_call_non_tensor(self):
        if False:
            i = 10
            return i + 15

        def foo(a, b, c):
            if False:
                print('Hello World!')
            (d, e) = c
            return (b + e, a + d)

        @torch.jit.script
        def bar():
            if False:
                return 10
            x = torch.ones(3, 4)
            (a, b) = foo(x, 3, (x, 3))
            return (a, b)
        self.assertEqual((6, torch.ones(3, 4) + 1), bar())

    def test_python_call_non_tensor_wrong(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(RuntimeError, 'but instead got value of type tuple'):

            @torch.jit.ignore
            def foo():
                if False:
                    print('Hello World!')
                return ((3, 4),)

            @torch.jit.script
            def bar():
                if False:
                    return 10
                return foo()
            bar()

    def test_if_different_type(self):
        if False:
            return 10
        with self.assertRaisesRegex(RuntimeError, 'c0 is set to type int in the true branch and type float in the false branch'):

            @torch.jit.script
            def diff_type_used():
                if False:
                    return 10
                if 1 == 2:
                    c0 = 1
                else:
                    c0 = 1.0
                return c0
        with self.assertRaisesRegex(RuntimeError, "Variable 'c0' previously had type float"):

            @torch.jit.script
            def diff_existing_type(x):
                if False:
                    print('Hello World!')
                c0 = 1.0
                if 1 == 2:
                    c0 = 1
                    print(x)
                return x

        @torch.jit.script
        def diff_type_unused():
            if False:
                for i in range(10):
                    print('nop')
            if 1 == 1:
                c0 = 1
                print(c0)
            else:
                c0 = 1.0
                print(c0)
            return 1

    def test_if_not_defined_error(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(RuntimeError, 'c0 is not defined in the false branch'):

            @torch.jit.script
            def test():
                if False:
                    i = 10
                    return i + 15
                if 1 == 1:
                    c0 = 1
                return c0
        with self.assertRaisesRegex(RuntimeError, 'c0 is not defined in the true branch'):

            @torch.jit.script
            def test2():
                if False:
                    i = 10
                    return i + 15
                if 1 == 1:
                    pass
                else:
                    c0 = 1
                return c0

    def test_if_list_cat(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def test_list(x):
            if False:
                while True:
                    i = 10
            if bool(x.sum() < 1):
                c = [x, x]
            else:
                c = [x, x, x]
            return torch.cat(c)
        b = torch.zeros(2, 4)
        _propagate_shapes(test_list.graph, (b,), False)

    def test_if_supertype(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def tensor_unifying(x, y, z):
            if False:
                while True:
                    i = 10
            if bool(x):
                (x, y, z) = (x + 1, y, z)
            else:
                (x, y, z) = (x + 1, x, y)
            return (x, y, z)
        a = torch.zeros(2, 2, dtype=torch.float)
        b = torch.zeros(2, 4, dtype=torch.long)
        c = torch.zeros(2, 4, dtype=torch.float)
        graph = _propagate_shapes(tensor_unifying.graph, (a, b, c), False)
        if_outputs = list(graph.findNode('prim::If').outputs())
        self.assertTrue(if_outputs[0].type().str() == 'Float(*, *, requires_grad=0, device=cpu)')
        self.assertTrue(if_outputs[1].type().str() == 'Tensor(*, *, requires_grad=0, device=cpu)')
        self.assertTrue(if_outputs[2].type().str() == 'Tensor(*, *, requires_grad=0, device=cpu)')

    def test_list_unify(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(RuntimeError, 'int[] in the true branch and type None[]'):

            @torch.jit.script
            def list_optional_fails(x):
                if False:
                    print('Hello World!')
                if x:
                    y = [1]
                else:
                    y = [None]
                return y[0]

        @torch.jit.script
        def list_tensors(x):
            if False:
                while True:
                    i = 10
            if x:
                a = torch.zeros([1, 1])
                y = [a]
            else:
                a = torch.zeros([1, 2])
                y = [a]
            return (a, y)
        self.run_pass('constant_propagation', list_tensors.graph)
        m = self.createFunctionFromGraph(list_tensors.graph)
        self.getExportImportCopy(m)

    @skipIfTorchDynamo('Not a TorchDynamo suitable test')
    @_inline_everything
    def test_import_constants_not_specialized(self):
        if False:
            for i in range(10):
                print('nop')

        class Mod(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.cat(2 * [x], dim=0)

        class ScriptMod(torch.jit.ScriptModule):

            def __init__(self, mod):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                x = torch.zeros(1, 3)
                mod_fn = lambda : mod(x)
                self.mod = torch.jit.trace(mod_fn, tuple())

            @torch.jit.script_method
            def forward(self):
                if False:
                    i = 10
                    return i + 15
                return self.mod()
        cm = ScriptMod(Mod())
        FileCheck().check('Float(1, 3, strides=[3, 1], requires_grad=0, device=cpu)').run(cm.forward.graph)
        buffer = io.BytesIO()
        torch.jit.save(cm, buffer)
        buffer.seek(0)
        cm_load = torch.jit.load(buffer)
        FileCheck().check_not('Float(1, 3)').run(cm_load.forward.graph)

    @skipIfTorchDynamo('TorchDynamo fails with unknown reason')
    def test_type_annotations_repeated_list(self):
        if False:
            return 10

        @torch.jit.script
        def float_fn(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return y
        self.assertEqual(float_fn(2.0, 1.0), float_fn(2.0, [1.0, 1.0, 1.0]))
        self.assertEqual(float_fn(2.0, 1.0), float_fn(2.0, (1.0, 1.0, 1.0)))

        @torch.jit.script
        def float_fn_call():
            if False:
                while True:
                    i = 10
            print(float_fn(1.0, 1.0))
            print(float_fn(1.0, (1.0, 1.0, 1.0)))

        @torch.jit.script
        def int_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return x
        self.assertEqual(int_fn(1), int_fn([1, 1, 1]))
        self.assertEqual(int_fn(1), int_fn((1, 1, 1)))

        @torch.jit.script
        def int_fn_call():
            if False:
                for i in range(10):
                    print('nop')
            print(int_fn(1))
            print(int_fn((1, 1, 1)))
        with self.assertRaisesRegex(RuntimeError, 'must be a positive integer:'):

            @torch.jit.script
            def fn(x):
                if False:
                    i = 10
                    return i + 15
                return x
        with self.assertRaisesRegex(RuntimeError, 'Unknown type constructor'):
            cu = torch.jit.CompilationUnit('\n                def nested(x, y):\n                    # type: (int, Tuple[int, int[2]]) -> List[int]\n                    return x  # noqa: T484\n            ')

        @torch.jit.script
        def f(x: BroadcastingList2[int]):
            if False:
                i = 10
                return i + 15
            return x
        out = f(1)
        self.assertTrue(isinstance(out[0], int))
        self.assertEqual(out, [1, 1])

    def test_ntuple_builtins(self):
        if False:
            print('Hello World!')
        from torch.nn.modules.utils import _single, _pair, _triple, _quadruple

        def test_ints():
            if False:
                i = 10
                return i + 15
            return (_single(1), _pair(2), _triple(3), _quadruple(4))

        def test_floats():
            if False:
                print('Hello World!')
            return (_single(1), _pair(2.1), _triple(3.1), _quadruple(4.1))
        self.checkScript(test_ints, ())
        self.checkScript(test_floats, ())

    def test_embedding_renorm_grad_error(self):
        if False:
            while True:
                i = 10

        def embedding_norm(input, embedding_matrix, max_norm):
            if False:
                i = 10
                return i + 15
            F.embedding(input, embedding_matrix, max_norm=0.01)

        @torch.jit.script
        def embedding_norm_script(input, embedding_matrix, max_norm):
            if False:
                while True:
                    i = 10
            F.embedding(input, embedding_matrix, max_norm=0.01)
        for _ in [embedding_norm, embedding_norm_script]:
            input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
            embedding_matrix = torch.randn(10, 3)
            var1 = torch.randn(10, 3, requires_grad=True)
            var2 = var1.detach().requires_grad_()
            output1 = var1 * embedding_matrix
            output2 = var2 * embedding_matrix
            output1.sum().backward()
            ignore = F.embedding(input, embedding_matrix, max_norm=0.01)
            with self.assertRaisesRegex(RuntimeError, 'modified'):
                output2.sum().backward()

    def test_type_annotations(self):
        if False:
            while True:
                i = 10

        def fn(x, y):
            if False:
                while True:
                    i = 10
            return (x, x * 2, x * 3)
        with self.assertRaisesRegex(RuntimeError, 'need 4 values .* found only 3'):

            @torch.jit.script
            def script_fn(x):
                if False:
                    print('Hello World!')
                (x, y, z, w) = fn(x, x)
        with self.assertRaisesRegex(RuntimeError, 'too many values .* need 2 but found 3'):

            @torch.jit.script
            def script_fn2(x):
                if False:
                    while True:
                        i = 10
                (x, y) = fn(x, x)

        def fn_unpack(x):
            if False:
                i = 10
                return i + 15
            (y, z, w) = fn(x, x)
            return y

        def fn_index(x):
            if False:
                return 10
            q = fn(x, x)
            return x

        def fn_string(str, strpair):
            if False:
                i = 10
                return i + 15
            (str1, str2) = strpair
            return (str, 2, str1, str2)
        x = torch.ones(2, 2)
        self.checkScript(fn_unpack, (x,), optimize=True)
        self.checkScript(fn_index, (x,), optimize=True)
        self.checkScript(fn_string, ('1', ('3', '4')), optimize=True)

    def test_type_annotations_varargs(self):
        if False:
            return 10

        @torch.jit.ignore
        def fn_varargs(x, *args):
            if False:
                print('Hello World!')
            return args[0] if args else x

        def fn1(x, y, z):
            if False:
                print('Hello World!')
            return fn_varargs(x)

        def fn2(x, y, z):
            if False:
                print('Hello World!')
            return fn_varargs(x, y)

        def fn3(x, y, z):
            if False:
                i = 10
                return i + 15
            return fn_varargs(x, y, z)
        (x, y, z) = (torch.randn(2, 2) for _ in range(3))
        self.checkScript(fn1, (x, y, z), optimize=True)
        self.checkScript(fn2, (x, y, z), optimize=True)
        self.checkScript(fn3, (x, y, z), optimize=True)

    def test_type_annotation_py3(self):
        if False:
            while True:
                i = 10
        code = dedent('\n        import torch\n        from torch import Tensor\n        from typing import Tuple\n\n        def fn(x : torch.Tensor, y : Tensor, z) -> Tuple[Tensor, Tensor, Tensor]:\n            return (x, y + z, z)\n        ')
        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = os.path.join(tmp_dir, 'script.py')
            with open(script_path, 'w') as f:
                f.write(code)
            fn = get_fn('test_type_annotation_py3', script_path)
            fn = torch.jit.ignore(fn)
            with self.assertRaisesRegex(RuntimeError, "Expected a value of type 'Tensor' for argument 'x' but instead found type 'Tuple\\[Tensor,"):

                @torch.jit.script
                def bad_fn(x):
                    if False:
                        while True:
                            i = 10
                    (x, y) = fn((x, x), x, x)
                    return y
            with self.assertRaisesRegex(RuntimeError, 'too many values .* need 2 but found 3'):

                @torch.jit.script
                def bad_fn2(x):
                    if False:
                        print('Hello World!')
                    (x, y) = fn(x, x, x)
                    return y
            with self.assertRaisesRegex(RuntimeError, 'need 4 values .* found only 3'):

                @torch.jit.script
                def bad_fn3(x):
                    if False:
                        print('Hello World!')
                    (x, y, z, w) = fn(x, x, x)
                    return y

            def good_fn(x):
                if False:
                    while True:
                        i = 10
                (y, z, w) = fn(x, x, x)
                return (y, z, w)
            self.checkScript(good_fn, (torch.ones(2, 2),), optimize=True)

    def test_type_annotation_module(self):
        if False:
            while True:
                i = 10

        class BaseModule(torch.jit.ScriptModule):

            @torch.jit.ignore
            def foo(self, x):
                if False:
                    print('Hello World!')
                return x + 1

            @torch.jit.ignore
            def bar(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return (x + y, y)

            @torch.jit.ignore
            def baz(self, x, y):
                if False:
                    while True:
                        i = 10
                return x

        class ModuleTooMany(BaseModule):

            @torch.jit.script_method
            def method(self, x):
                if False:
                    return 10
                return self.foo(x, x)

        class ModuleTooFew(BaseModule):

            @torch.jit.script_method
            def method(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.bar(x)

        class ModuleTooManyAssign(BaseModule):

            @torch.jit.script_method
            def method(self, x):
                if False:
                    return 10
                (y, z, w) = self.bar(x, x)
                return x

        class ModuleDefault(BaseModule):

            @torch.jit.script_method
            def method(self, x):
                if False:
                    return 10
                y = self.baz(x)
                return x
        with self.assertRaisesRegex(RuntimeError, 'Expected at most 2 arguments but found 3'):
            ModuleTooMany()
        with self.assertRaisesRegex(RuntimeError, 'Argument y not provided'):
            ModuleTooFew()
        with self.assertRaisesRegex(RuntimeError, 'need 3 values .* found only 2'):
            ModuleTooManyAssign()
        with self.assertRaisesRegex(RuntimeError, 'Argument y not provided.'):
            ModuleDefault()

    def test_type_inferred_from_empty_annotation(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that the type inferred from an empty or missing annotation is Torch.Tensor wtih `inferred=true`\n        '

        @torch.jit.script
        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return x
        graph = fn.graph
        n = next(graph.inputs())
        self.assertTrue(n.type() == torch._C.TensorType.getInferred())
        with self.assertRaisesRegex(RuntimeError, "Inferred 'x' to be of type 'Tensor"):
            fn('1')

    def test_script_define_order(self):
        if False:
            return 10

        class M(torch.jit.ScriptModule):

            @torch.jit.script_method
            def call_foo(self, input):
                if False:
                    while True:
                        i = 10
                return self.foo(input)

            @torch.jit.script_method
            def foo(self, input):
                if False:
                    while True:
                        i = 10
                return input + 1
        m = M()
        self.assertEqual(2, m.call_foo(torch.ones((), dtype=torch.int64)))

    def test_script_define_order_recursive_fail(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.jit.ScriptModule):

            @torch.jit.script_method
            def call_foo(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return self.foo(input)

            @torch.jit.script_method
            def foo(self, input):
                if False:
                    while True:
                        i = 10
                self.call_foo(input)
        with self.assertRaisesRegex(RuntimeError, 'called recursively'):
            M()

    def test_script_kwargs_fn_call(self):
        if False:
            print('Hello World!')

        class M(torch.jit.ScriptModule):

            @torch.jit.script_method
            def call_foo(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return self.foo(input=input, bar=1)

            @torch.jit.script_method
            def foo(self, bar, input):
                if False:
                    i = 10
                    return i + 15
                return input + bar
        m = M()
        self.assertEqual(2, m.call_foo(torch.ones((), dtype=torch.int64)))

    def test_if_define(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def foo(a):
            if False:
                for i in range(10):
                    print('nop')
            if bool(a == 0):
                b = 1
            else:
                b = 0
            return b + 1

        @torch.jit.script
        def foo2(a):
            if False:
                print('Hello World!')
            b = 0
            if bool(a == 0):
                b = 1
            return b + 1

        @torch.jit.script
        def foo3(a):
            if False:
                i = 10
                return i + 15
            b = 1
            if bool(a == 0):
                c = 4
            else:
                b = 0
            return b + 1
        a = torch.ones(1, dtype=torch.long)
        b = torch.zeros(1, dtype=torch.long)
        self.assertEqual(1, foo(a))
        self.assertEqual(2, foo(b))
        self.assertEqual(1, foo2(a))
        self.assertEqual(2, foo2(b))
        self.assertEqual(1, foo3(a))
        self.assertEqual(2, foo3(b))

    def test_script_module_export_submodule(self):
        if False:
            i = 10
            return i + 15

        class M1(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.weight = nn.Parameter(torch.randn(2))

            @torch.jit.script_method
            def forward(self, thing):
                if False:
                    return 10
                return self.weight + thing

        class M2(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.sub = M1()
                self.weight = nn.Parameter(torch.randn(2, 3))
                self.bias = nn.Parameter(torch.randn(2))
                self.define('\n                    def hi(self, a):\n                        return self.weight.mm(a)\n                ')

            @torch.jit.script_method
            def doit(self, input):
                if False:
                    print('Hello World!')
                return self.weight.mm(input)

            @torch.jit.script_method
            def doit2(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return self.weight.mm(input)

            @torch.jit.script_method
            def doit3(self, input):
                if False:
                    i = 10
                    return i + 15
                return input + torch.ones([1], dtype=torch.double)

            @torch.jit.script_method
            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                a = self.doit(input)
                b = self.doit2(input)
                c = self.hi(input)
                return a + b + self.bias + c
        with torch.jit.optimized_execution(False):
            m_orig = M2()
            m_import = self.getExportImportCopy(m_orig)
            input = torch.randn(3, 2)
            self.assertEqual(m_orig.doit(input), m_import.doit(input))
            self.assertEqual(m_orig.hi(input), m_import.hi(input))
            self.assertEqual(m_orig.doit3(input), m_import.doit3(input))
            self.assertEqual(m_orig.forward(input), m_import.forward(input))

    @slowTest
    def test_compile_module_with_constant(self):
        if False:
            while True:
                i = 10

        class Double(nn.Module):

            def __init__(self, downsample=None):
                if False:
                    print('Hello World!')
                super().__init__()

            def forward(self, input):
                if False:
                    return 10
                return input * 2

        class Mod(nn.Module):
            __constants__ = ['downsample']

            def __init__(self, downsample=None):
                if False:
                    print('Hello World!')
                super().__init__()
                self.downsample = downsample

            def forward(self, input):
                if False:
                    return 10
                if self.downsample is not None:
                    return self.downsample(input)
                return input
        none_mod = torch.jit.script(Mod(None))
        double_mod = torch.jit.script(Mod(Double()))
        self.assertEqual(none_mod(torch.tensor(1)), torch.tensor(1))
        self.assertEqual(double_mod(torch.tensor(1)), torch.tensor(1) * 2)

    def test_device_kwarg(self):
        if False:
            for i in range(10):
                print('nop')
        from torch import device

        def f():
            if False:
                print('Hello World!')
            return (device(type='cuda'), torch.device(type='cpu'))
        self.checkScript(f, ())

    def test_script_module_export_tensor_type(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.jit.ScriptModule):

            def __init__(self, type):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.param = torch.nn.Parameter(torch.zeros((5, 5), dtype=type).random_())

            @torch.jit.script_method
            def foo(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.param
        with torch.jit.optimized_execution(False):
            for type in [torch.float, torch.double]:
                m_orig = M(type)
                m_import = self.getExportImportCopy(m_orig)
                self.assertTrue(m_orig.param.storage().size() == 25)
                self.assertEqual(m_orig.foo(), m_import.foo())
                self.assertTrue(m_orig.foo().dtype == m_import.foo().dtype)

    @unittest.skipIf(not RUN_CUDA, 'testing cuda tensors require CUDA')
    def test_script_module_export_tensor_cuda(self):
        if False:
            print('Hello World!')

        class M(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.param = torch.nn.Parameter(torch.zeros((5, 5), device='cuda:0').random_())

            @torch.jit.script_method
            def foo(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.param
        m_orig = M()
        m_import = self.getExportImportCopy(m_orig)
        self.assertTrue(m_orig.param.storage().size() == 25)
        self.assertTrue(m_import.foo().device == torch.device('cuda:0'))
        self.assertEqual(m_orig.foo(), m_import.foo())
        self.assertTrue(m_orig.foo().dtype == m_import.foo().dtype)

    def test_script_module_export_blocks(self):
        if False:
            i = 10
            return i + 15

        class M(torch.jit.ScriptModule):

            def __init__(self, n, m):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(n, m))

            @torch.jit.script_method
            def forward(self, input):
                if False:
                    while True:
                        i = 10
                if bool(input.sum() > 0):
                    output = self.weight.mv(input)
                else:
                    output = self.weight + input
                return output
        m_orig = M(200, 200)
        m_import = self.getExportImportCopy(m_orig)
        t = torch.rand(200)
        self.assertEqual(m_orig(t), m_import(t))

    def test_script_module_export_shared_storage(self):
        if False:
            i = 10
            return i + 15

        class M(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.param1 = torch.nn.Parameter(torch.rand(5, 5))
                self.param2 = torch.nn.Parameter(self.param1[3])
                self.param3 = torch.nn.Parameter(torch.rand(5, 5))
                self.param4 = torch.nn.Parameter(torch.rand(11, 5)[1:6])

            @torch.jit.script_method
            def foo(self):
                if False:
                    i = 10
                    return i + 15
                return self.param1 + self.param2 + self.param3 + self.param4
        with torch.jit.optimized_execution(False):
            m_orig = M()
            m_import = self.getExportImportCopy(m_orig)
            self.assertEqual(m_orig.foo(), m_import.foo())
            self.assertTrue(m_import.param1.storage().data_ptr() == m_import.param2.storage().data_ptr())
            self.assertTrue(m_import.param1.storage().data_ptr() != m_import.param3.storage().data_ptr())

    def test_sequential_intermediary_types(self):
        if False:
            print('Hello World!')

        class A(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x + 3

        class B(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return {'1': x}

        class C(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.foo = torch.nn.Sequential(A(), B())

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.foo(x)
        self.checkModule(C(), (torch.tensor(1),))

    def test_ellipsis_const_mid(self):
        if False:
            while True:
                i = 10

        def ellipsize(x):
            if False:
                return 10
            return x[2, Ellipsis, 0:4, 4:8].size()
        dummy = torch.zeros(8, 8, 8, 8, 8)
        self.checkScript(ellipsize, (dummy,), optimize=True)

    def test_ellipsis_const_mid_select(self):
        if False:
            while True:
                i = 10

        def ellipsize(x):
            if False:
                while True:
                    i = 10
            return x[2, Ellipsis, 4, 4, 4:8, 2].size()
        dummy = torch.zeros(8, 8, 8, 8, 8, 8, 8)
        self.checkScript(ellipsize, (dummy,), optimize=True)

    def test_ellipsis_const_start(self):
        if False:
            for i in range(10):
                print('nop')

        def ellipsize(x):
            if False:
                for i in range(10):
                    print('nop')
            return x[Ellipsis, 0:4, 4:8].size()
        dummy = torch.zeros(8, 8, 8, 8, 8)
        self.checkScript(ellipsize, (dummy,), optimize=True)

    def test_ellipsis_const_end(self):
        if False:
            while True:
                i = 10

        def ellipsize(x):
            if False:
                i = 10
                return i + 15
            return x[0:4, 2, Ellipsis].size()
        dummy = torch.zeros(8, 8, 8, 8, 8)
        self.checkScript(ellipsize, (dummy,), optimize=True)

    def test_ellipsis_mid(self):
        if False:
            for i in range(10):
                print('nop')

        def ellipsize(x):
            if False:
                print('Hello World!')
            return x[2, ..., 0:4, 4:8].size()
        dummy = torch.zeros(8, 8, 8, 8, 8)
        self.checkScript(ellipsize, (dummy,), optimize=True)

    def test_ellipsis_mid_select(self):
        if False:
            while True:
                i = 10

        def ellipsize(x):
            if False:
                i = 10
                return i + 15
            return x[2, ..., 4, 4, 4:8, 2].size()
        dummy = torch.zeros(8, 8, 8, 8, 8, 8, 8)
        self.checkScript(ellipsize, (dummy,), optimize=True)

    def test_ellipsis_start(self):
        if False:
            for i in range(10):
                print('nop')

        def ellipsize(x):
            if False:
                print('Hello World!')
            return x[..., 0:4, 4:8].size()
        dummy = torch.zeros(8, 8, 8, 8, 8)
        self.checkScript(ellipsize, (dummy,), optimize=True)

    def test_ellipsis_end(self):
        if False:
            for i in range(10):
                print('nop')

        def ellipsize(x):
            if False:
                for i in range(10):
                    print('nop')
            return x[0:4, 2, ...].size()
        dummy = torch.zeros(8, 8, 8, 8, 8)
        self.checkScript(ellipsize, (dummy,), optimize=True)

    def test_torch_manual_seed(self):
        if False:
            return 10
        with freeze_rng_state():

            def test():
                if False:
                    return 10
                torch.manual_seed(2)
                return torch.rand(1)
            script = torch.jit.script(test)
            self.assertEqual(test(), script())
            graph = script.graph_for()
            FileCheck().check('aten::manual_seed').run(graph)

    @skipIfTorchDynamo('Not a TorchDynamo suitable test')
    def test_index_select_shape_prop(self):
        if False:
            return 10

        @torch.jit.script
        def foo(x, y):
            if False:
                i = 10
                return i + 15
            return torch.index_select(x, index=y, dim=1)
        a = torch.zeros(2, 2)
        b = torch.zeros(4, dtype=torch.long)
        torch._C._jit_pass_complete_shape_analysis(foo.graph, (a, b), False)
        FileCheck().check('Float(2, 4, strides=[4, 1], requires_grad=0, device=cpu)').run(str(foo.graph))

    def test_shape_analysis_loop(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(a, b, x):
            if False:
                while True:
                    i = 10
            c = a
            for _ in range(2):
                a = c + b
                c = x
                b = x
            return a
        self.checkScript(foo, (torch.zeros(1), torch.zeros(4), torch.zeros(5)), optimize=False)

    def test_intlist_args(self):
        if False:
            for i in range(10):
                print('nop')

        def func_1(x):
            if False:
                while True:
                    i = 10
            return torch.nn.functional.adaptive_avg_pool1d(x, 1)

        def func_2(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.nn.functional.adaptive_avg_pool1d(x, output_size=1)

        def func_3(x):
            if False:
                while True:
                    i = 10
            return torch.nn.functional.adaptive_avg_pool1d(x, output_size=[1])
        x = torch.randn(8, 8, 8)
        self.checkScript(func_1, [x], optimize=True)
        self.checkScript(func_2, [x], optimize=True)
        self.checkScript(func_3, [x], optimize=True)

    def test_wrong_implicit_expand(self):
        if False:
            return 10

        @_trace(torch.zeros(3), torch.zeros(1))
        def foo(a, b):
            if False:
                while True:
                    i = 10
            return a + b
        a = torch.rand(4)
        b = torch.rand(4)
        self.assertEqual(a + b, foo(a, b))

    def test_builtin_args_fails(self):
        if False:
            return 10
        with self.assertRaisesRegex(RuntimeError, 'Argument self not provided'):

            @torch.jit.script
            def f1(a):
                if False:
                    while True:
                        i = 10
                torch.sum(foo=4)
        with self.assertRaisesRegex(RuntimeError, 'specified twice'):

            @torch.jit.script
            def f2(a):
                if False:
                    return 10
                torch.sum(a, self=a)
        with self.assertRaisesRegex(RuntimeError, 'not provided'):

            @torch.jit.script
            def f3(a):
                if False:
                    for i in range(10):
                        print('nop')
                torch.sum(dim=4)
        with self.assertRaisesRegex(RuntimeError, "for argument 'tensors' but instead found type 'Tensor"):

            @torch.jit.script
            def f4(a):
                if False:
                    for i in range(10):
                        print('nop')
                torch.cat(a)
        with self.assertRaisesRegex(RuntimeError, "argument \\'tensors\\' but instead found type \\'List\\[int\\]"):

            @torch.jit.script
            def f5(a):
                if False:
                    for i in range(10):
                        print('nop')
                torch.cat([3])
        with self.assertRaisesRegex(RuntimeError, "Expected a value of type \\'List\\[int\\]\\' for argument \\'size\\' but instead found type \\'List\\[Union\\[List\\[int\\], int\\]\\]"):

            @torch.jit.script
            def f6(a):
                if False:
                    return 10
                a.expand(size=[3, [4]])

    def test_builtin_args(self):
        if False:
            while True:
                i = 10

        def t0(a):
            if False:
                return 10
            return torch.cat([a, a])
        self.checkScript(t0, (torch.zeros(1, 1),))

        def t1(a):
            if False:
                i = 10
                return i + 15
            return torch.cat(dim=1, tensors=[a, a])
        self.checkScript(t1, (torch.zeros(1, 1, 2),))

        def t2(a):
            if False:
                print('Hello World!')
            if 1 == 1:
                b = 1
            else:
                b = 0
            return torch.sum(a, dim=b, keepdim=False)
        self.checkScript(t2, (torch.zeros(1, 1, 2),))

    def test_parser_type_annotations(self):
        if False:
            while True:
                i = 10
        cu = torch.jit.CompilationUnit('\n            def foo(x : Tensor, y : Tuple[Tuple[Tensor, Tensor], Tensor]) -> Tuple[Tensor, Tensor]:\n                return x, x\n        ')
        self.assertExpected(str(cu.foo.schema))

    def test_parser_type_annotations_comment(self):
        if False:
            print('Hello World!')
        cu = torch.jit.CompilationUnit('\n            def foo(x, y):\n                # type: (Tensor, Tuple[Tuple[Tensor, Tensor], Tensor]) -> Tuple[Tensor, Tensor]\n                return x, x\n        ')
        self.assertExpected(str(cu.foo.schema))

    def test_parser_type_annotations_unknown_type(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(RuntimeError, "Unknown type name 'Foo'"):
            cu = torch.jit.CompilationUnit('\n                def foo(x : Tensor, y : Tuple[Tuple[Foo, Tensor], Tensor]) -> Tuple[Tensor, Tensor]:\n                    return x, x\n            ')

    def test_parser_type_annotations_subscript_non_ident(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(RuntimeError, 'Subscripted type must be a type identifier'):
            cu = torch.jit.CompilationUnit('\n                def foo(x : Tensor, y : Tuple[Tensor, Tensor][Tensor]) -> Tuple[Tensor, Tensor]:\n                    return x, x\n            ')

    def test_parser_type_annotations_subscript_tensor(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(RuntimeError, 'Unknown type constructor Tensor'):
            cu = torch.jit.CompilationUnit('\n                def foo(x : Tensor, y : Tensor[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:\n                    return x, x\n            ')

    def test_parser_type_annotations_incompatible_expression(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(RuntimeError, 'Expression of type \\+ cannot be used in a type expression'):
            cu = torch.jit.CompilationUnit('\n                def foo(x : Tensor, y : Tuple[3 + 4, Tensor]) -> Tuple[Tensor, Tensor]:\n                    return x, x\n            ')

    def test_gather_dynamic_index(self):
        if False:
            for i in range(10):
                print('nop')

        def t(x):
            if False:
                print('Hello World!')
            gather1 = x[0]
            idx = 0 + 1
            gather2 = x[idx]
            return gather1 + gather2
        self.checkScript(t, (torch.zeros(3, 2, 3),))

    def test_torch_ignore_conversion_to_none(self):
        if False:
            for i in range(10):
                print('nop')

        class A(torch.nn.Module):

            @torch.jit.ignore
            def ignored(self, a: int) -> None:
                if False:
                    while True:
                        i = 10
                l: int = len([2 for i in range(a) if i > 2])
                return

            def forward(self) -> int:
                if False:
                    for i in range(10):
                        print('nop')
                a: int = 4
                b: int = 5
                self.ignored(a)
                return a + b

        class B(torch.nn.Module):

            @torch.jit.ignore
            def ignored(self, a: int):
                if False:
                    i = 10
                    return i + 15
                l: int = len([2 for i in range(a) if i > 2])
                return

            def forward(self) -> int:
                if False:
                    return 10
                a: int = 4
                b: int = 5
                self.ignored(a)
                return a + b
        modelA = torch.jit.script(A())
        self.assertEqual(modelA(), 9)
        modelB = torch.jit.script(B())
        self.assertEqual(modelB(), 9)

    def test_addmm_grad(self):
        if False:
            print('Hello World!')
        " This test checks several things:\n            1. An expand node was inserted before the addmm operating on the\n               bias term.\n            2. The fused form of addmm appears in the ultimate graph that's\n               executed.\n            3. A sum op was emitted for accumulating gradients along the 0th\n               (expanded) dimension of the bias term.\n            4. The correct symbolic representation for the backward pass of the\n               mm operator was emitted (x.t() -> mm)\n\n            TODO: we should actually check these conditions once we have a way\n            to dump the GraphExecutor state. Namely the processed forward graph\n            and the backward graph.\n        "

        @torch.jit.script
        def addmm_grad_test(b, x, w):
            if False:
                i = 10
                return i + 15
            return torch.addmm(b, x, w)
        w_init = torch.rand(2, 5)
        b_init = torch.rand(5)
        x = torch.rand(3, 2)
        b = b_init.clone()
        b.requires_grad_()
        w = w_init.clone()
        w.requires_grad_()
        y = addmm_grad_test(b, x, w)
        y.sum().backward()
        b_ref = b_init.clone()
        b_ref.requires_grad_()
        w_ref = w_init.clone()
        w_ref.requires_grad_()
        y_ref = torch.addmm(b_ref, x, w_ref)
        y_ref.sum().backward()
        self.assertEqual(w.grad, w_ref.grad)
        self.assertEqual(b.grad, b_ref.grad)

    @unittest.skipIf(not RUN_CUDA, 'running tests on cuda to verify cudnn fix')
    def test_batch_norm_inference_backward_cuda(self):
        if False:
            i = 10
            return i + 15
        with enable_profiling_mode_for_profiling_tests():

            class MyBatchNorm(torch.nn.Module):

                def __init__(self, num_features, affine, track_running_stats):
                    if False:
                        for i in range(10):
                            print('nop')
                    super().__init__()
                    self.bn = torch.nn.BatchNorm2d(num_features, 1e-05, affine=affine, track_running_stats=track_running_stats).float()

                def forward(self, x: torch.Tensor):
                    if False:
                        i = 10
                        return i + 15
                    o = self.bn(x)
                    o = torch.nn.functional.relu(o)
                    return o
            batch = 4
            c = 2
            hw = 3
            x_init = torch.randn(batch, c, hw, hw, dtype=torch.float).cuda()
            grad = torch.randn(batch, c, hw, hw, dtype=torch.float).cuda()
            training = False
            affine = True
            track_running_stats = True
            module = torch.jit.script(MyBatchNorm(c, affine, track_running_stats)).cuda()
            ref_module = MyBatchNorm(c, affine, track_running_stats).cuda()
            module.eval()
            ref_module.eval()
            jit_module = torch.jit.script(module)
            ref_module.load_state_dict(module.state_dict())
            x = x_init.detach().clone()
            x.requires_grad_()
            x_ref = x_init.detach().clone()
            x_ref.requires_grad_()
            for i in range(0, 3):
                y = jit_module(x)
                y.backward(grad)
            x.grad.zero_()
            module.bn.running_mean.zero_()
            module.bn.running_var.fill_(1.0)
            ref_module.bn.running_mean.zero_()
            ref_module.bn.running_var.fill_(1.0)
            y = jit_module(x)
            y.backward(grad)
            y_ref = ref_module(x_ref)
            y_ref.backward(grad)
            self.assertEqual(y_ref, y)
            self.assertEqual(x.grad, x_ref.grad)
            self.assertEqual(module.bn.running_mean, ref_module.bn.running_mean)
            self.assertEqual(module.bn.running_var, ref_module.bn.running_var)

    def test_zeros(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.jit.ScriptModule):
            __constants__ = ['d']

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.d = torch.device('cpu')

            @torch.jit.script_method
            def create(self):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.zeros([1, 1, 2], dtype=torch.float, device=self.d, layout=torch.strided)
        r = M().create()
        self.assertEqual(r.dtype, torch.float)
        self.assertEqual(torch.zeros([1, 1, 2], dtype=torch.float), r)

        def fn():
            if False:
                i = 10
                return i + 15
            return torch.zeros((1, 2, 3))
        self.checkScript(fn, ())

    def test_vararg_zeros(self):
        if False:
            return 10

        def foo():
            if False:
                print('Hello World!')
            return torch.zeros(3, 4, 5, dtype=torch.int)
        self.checkScript(foo, ())

    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.LEGACY, 'the original version of test_rand')
    def test_rand(self):
        if False:
            while True:
                i = 10

        def test_rand():
            if False:
                return 10
            a = torch.rand([3, 4])
            return a + 1.0 - a
        self.checkScript(test_rand, ())
        fn = torch.jit.script(test_rand)
        out = fn()
        self.assertEqual(out.dtype, torch.get_default_dtype())
        g = fn.graph_for()
        if GRAPH_EXECUTOR != ProfilingMode.SIMPLE:
            FileCheck().check('Double(*, *, requires_grad=0, device=cpu)').check_not('Float(*, *, requires_grad=0, device=cpu)').run(g)

        @torch.jit.script
        def randint():
            if False:
                return 10
            return torch.randint(0, 5, [1, 2])
        out = randint()
        self.assertEqual(out.dtype, torch.int64)
        if GRAPH_EXECUTOR != ProfilingMode.SIMPLE:
            FileCheck().check('Long(*, *, requires_grad=0, device=cpu)').check_not('Float(*, *, requires_grad=0, device=cpu)').check_not('Double(*, *, requires_grad=0, device=cpu)').run(randint.graph_for())

    @unittest.skipIf(not RUN_CUDA, 'no CUDA')
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, "skip if profiling isn't enabled")
    def test_autodiff_complex(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(x: torch.Tensor, y: torch.Tensor, W: torch.Tensor):
            if False:
                while True:
                    i = 10
            return torch.exp(torch.mm(torch.complex(x, y), W.cfloat()))

        @torch.jit.script
        def jitted_foo(x: torch.Tensor, y: torch.Tensor, W: torch.Tensor):
            if False:
                print('Hello World!')
            return torch.exp(torch.mm(torch.complex(x, y), W.cfloat()))
        x = torch.randn(128, 16, dtype=torch.float32, device='cuda:0')
        y = torch.randn(128, 16, dtype=torch.float32, device='cuda:0')
        W = torch.randn(16, 1, dtype=torch.float32, device='cuda:0', requires_grad=True)
        W.data /= 4
        with enable_profiling_mode_for_profiling_tests():
            for i in range(4):
                self.assertTrue((foo(x, y, W).grad_fn is None) == (jitted_foo(x, y, W).grad_fn is None))

    def test_linear_grad(self):
        if False:
            print('Hello World!')
        with enable_profiling_mode_for_profiling_tests():

            def t(x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.nn.functional.linear(x, w, b)
            x_init = torch.randn(4, 2)
            w_init = torch.randn(3, 2)
            b_init = torch.randn(3)
            grad = torch.randn(4, 3)
            with disable_autodiff_subgraph_inlining():
                jit_t = torch.jit.script(t)
                x = x_init.detach().requires_grad_()
                w = w_init.detach().requires_grad_()
                b = b_init.detach().requires_grad_()
                x_ref = x_init.detach().requires_grad_()
                w_ref = w_init.detach().requires_grad_()
                b_ref = b_init.detach().requires_grad_()
                jit_o = jit_t(x, w, b)
                jit_o.backward(grad)
                jit_o = jit_t(x, w, b)
                jit_o.backward(grad)
                x.grad.zero_()
                w.grad.zero_()
                b.grad.zero_()
                jit_o = jit_t(x, w, b)
                jit_o.backward(grad)
                o = t(x_ref, w_ref, b_ref)
                o.backward(grad)
                self.assertEqual(jit_o, o)
                self.assertEqual(x.grad, x_ref.grad)
                self.assertEqual(w.grad, w_ref.grad)
                self.assertEqual(b.grad, b_ref.grad)
                x.grad.zero_()
                w.grad.zero_()
                x_ref.grad.zero_()
                w_ref.grad.zero_()
                jit_o = jit_t(x, w, None)
                jit_o.backward(grad)
                o = t(x_ref, w_ref, None)
                o.backward(grad)
                self.assertEqual(jit_o, o)
                self.assertEqual(x.grad, x_ref.grad)
                self.assertEqual(w.grad, w_ref.grad)

    @skipIfTorchDynamo("TorchDynamo doesn't support profile")
    @unittest.skipIf(GRAPH_EXECUTOR != ProfilingMode.PROFILING, 'the profiling version of test_rand')
    def test_rand_profiling(self):
        if False:
            return 10

        def test_rand():
            if False:
                return 10
            a = torch.rand([3, 4])
            return a + 1.0 - a
        with enable_profiling_mode_for_profiling_tests():
            with num_profiled_runs(1):
                fn = torch.jit.script(test_rand)
                out = fn()
                graph_str = torch.jit.last_executed_optimized_graph()
                self.assertEqual(out.dtype, torch.float)
                FileCheck().check('Float(3, 4, strides=[4, 1], requires_grad=0, device=cpu)').check_not('Double(3, 4, strides=[4, 1], requires_grad=0, device=cpu)').run(graph_str)

        @torch.jit.script
        def randint():
            if False:
                i = 10
                return i + 15
            return torch.randint(0, 5, [1, 2])
        with enable_profiling_mode_for_profiling_tests():
            with num_profiled_runs(1):
                out = randint()
                graph_str = torch.jit.last_executed_optimized_graph()
                self.assertEqual(out.dtype, torch.int64)
                FileCheck().check('profiled_type=Long(1, 2, strides=[2, 1], requires_grad=0, device=cpu)').run(graph_str)

    def test_erase_number_types(self):
        if False:
            i = 10
            return i + 15

        def func(a):
            if False:
                while True:
                    i = 10
            b = 7 + 1 + 3
            c = a + b
            c += b
            return c
        graph = torch.jit.script(func).graph
        FileCheck().check('int = prim::Constant').check('aten::add_').run(str(graph))
        self.run_pass('erase_number_types', graph)
        FileCheck().check_not('int = prim::Constant').run(str(graph))

    def test_refine_tuple_types(self):
        if False:
            for i in range(10):
                print('nop')
        graph_str = '\n        graph(%a : Float(123), %b : Float(4, 5, 6)):\n          %c : (Tensor, Tensor) = prim::TupleConstruct(%a, %b)\n          return (%c)\n        '
        graph = parse_ir(graph_str)
        torch._C._jit_pass_refine_tuple_types(graph)
        self.assertTrue('(Float(123), Float(4, 5, 6))' in str(graph.findNode('prim::TupleConstruct').output()))

    def test_remove_dropout(self):
        if False:
            print('Hello World!')
        weight_0_shape = (20, 5)
        weight_1_shape = (20, 20)
        input_shape = (10, 5)

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.weight_0 = torch.nn.Parameter(torch.rand(weight_0_shape))
                self.weight_1 = torch.nn.Parameter(torch.rand(weight_1_shape))

            def forward(self, x):
                if False:
                    return 10
                o = F.linear(x, self.weight_0)
                o = F.dropout(o, training=self.training)
                o = F.linear(o, self.weight_1)
                return o
        data = torch.rand(input_shape)
        m = M()
        m = torch.jit.script(m)
        with self.assertRaisesRegex(RuntimeError, 'Dropout removal module in training mode is not yet supported'):
            torch._C._jit_pass_remove_dropout(m._c)
        m.eval()
        ref_res = m(data)
        from torch.jit._recursive import wrap_cpp_module
        m = wrap_cpp_module(torch._C._freeze_module(m._c))
        torch._C._jit_pass_remove_dropout(m._c)
        res = m(data)
        FileCheck().check_not('aten::dropout').run(str(m.graph))
        torch.testing.assert_close(ref_res, res, rtol=0.01, atol=0.001)

    def test_unfold_zero_dim(self):
        if False:
            return 10

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.unfold(0, 1, 1)
        graph = torch.jit.script(fn).graph
        torch._C._jit_pass_complete_shape_analysis(graph, (torch.tensor(0.39),), False)
        out_dims = fn(torch.tensor(0.3923)).ndim
        self.assertEqual(graph.findNode('aten::unfold').output().type().dim(), out_dims)

    def test_mm_batching(self):
        if False:
            i = 10
            return i + 15
        with enable_profiling_mode_for_profiling_tests():
            lstm_cell = torch.jit.script(LSTMCellS)

            def lstm(x, hx, cx, w_ih, w_hh, b_ih, b_hh):
                if False:
                    while True:
                        i = 10
                for i in range(x.size(0)):
                    (hx, cx) = lstm_cell(x[i], hx, cx, w_ih, w_hh, b_ih, b_hh)
                return hx
            slstm = torch.jit.script(lstm)
            inputs = get_lstm_inputs('cpu', training=True, seq_length=10)
            slstm(*inputs, profile_and_replay=True).sum().backward(retain_graph=True)
            if GRAPH_EXECUTOR == ProfilingMode.PROFILING:
                slstm(*inputs, profile_and_replay=True).sum().backward()
            fw_graph = slstm.graph_for(*inputs)
            if GRAPH_EXECUTOR == ProfilingMode.LEGACY:
                bw_graph = backward_graph(slstm, diff_graph_idx=0)
                self.assertTrue('prim::MMBatchSide' in str(fw_graph))
                self.assertTrue('prim::MMTreeReduce' in str(bw_graph))
            sout = slstm(*inputs)
            out = lstm(*inputs)
            self.assertEqual(sout, out)
            self.assertEqual(torch.autograd.grad(sout.sum(), inputs), torch.autograd.grad(out.sum(), inputs))

    def test_loop_unrolling(self):
        if False:
            while True:
                i = 10

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            y = 0
            for i in range(int(x)):
                y -= i
            return y
        graph = torch.jit.script(fn).graph
        self.run_pass('loop_unrolling', graph)
        unroll_factor = 8
        FileCheck().check('prim::Loop').check_count('aten::sub', unroll_factor).check('prim::Loop').check('aten::sub').run(str(graph))
        self.checkScript(fn, (torch.tensor(10),))

    def test_loop_unrolling_const(self):
        if False:
            i = 10
            return i + 15

        def fn():
            if False:
                for i in range(10):
                    print('nop')
            y = 0
            for _ in range(10):
                y -= 1
            return y

        def fn2():
            if False:
                while True:
                    i = 10
            y = 0
            for i in range(10):
                y -= i
            return y

        def check(fn, name):
            if False:
                print('Hello World!')
            graph = torch.jit.script(fn).graph
            self.run_pass('loop_unrolling', graph)
            FileCheck().check_not("prim::Loop'").run(str(graph))
            self.checkScript(fn, ())
        check(fn, 'add_const')
        check(fn2, 'add_iter')

    def test_loop_unrolling_nested(self):
        if False:
            print('Hello World!')

        def fn(x):
            if False:
                while True:
                    i = 10
            y = 0
            for _ in range(10):
                for j in range(int(x)):
                    y -= j
            return y
        graph = torch.jit.script(fn).graph
        self.run_pass('loop_unrolling', graph)
        unroll_factor = 8
        FileCheck().check('prim::Loop').check('prim::Loop').check_count('aten::sub', unroll_factor).check('prim::Loop').check('aten::sub').run(str(graph))
        self.checkScript(fn, (torch.tensor(10),))

    def test_loop_unroll_unused_counter(self):
        if False:
            return 10

        def fn(x):
            if False:
                return 10
            y = 0
            for _ in range(int(x)):
                y -= 1
            return y
        graph = torch.jit.script(fn).graph
        self.run_pass('loop_unrolling', graph)
        FileCheck().check('prim::Loop').check_not('aten::add').check('return').run(str(graph))

    def test_loop_unroll_negative(self):
        if False:
            while True:
                i = 10

        def fn(x):
            if False:
                print('Hello World!')
            y = 0
            for _ in range(int(x)):
                y += 1
            return y
        self.checkScript(fn, (torch.tensor(-20),))
        self.checkScript(fn, (torch.tensor(-2),))
        self.checkScript(fn, (torch.tensor(-1),))
        self.checkScript(fn, (torch.tensor(0),))
        self.checkScript(fn, (torch.tensor(1),))
        self.checkScript(fn, (torch.tensor(2),))

    def test_where(self):
        if False:
            i = 10
            return i + 15

        def fn(x, y):
            if False:
                return 10
            return torch.where(x > 0.0, x, y)
        self.checkScript(fn, (torch.randn(3, 2, dtype=torch.float), torch.ones(3, 2, dtype=torch.float)))

    def test_where_method(self):
        if False:
            while True:
                i = 10

        def fn(x, y):
            if False:
                return 10
            return x.where(x > 0.0, y)
        self.checkScript(fn, (torch.randn(3, 2, dtype=torch.float), torch.ones(3, 2, dtype=torch.float)))

    def test_union_to_number(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def fn(x: Union[int, complex, float], y: Union[int, complex, float]):
            if False:
                for i in range(10):
                    print('nop')
            return x + y
        FileCheck().check(': Scalar):').run(fn.graph)

    def test_reassign_module_lhs(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(RuntimeError, "Cannot re-assign 'self'"):

            class ReassignSelfLHS(torch.jit.ScriptModule):

                @torch.jit.script_method
                def forward(self, x):
                    if False:
                        return 10
                    for _ in range(20):
                        self = x
                    return self
            ReassignSelfLHS()

    def test_reassign_module_rhs(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(RuntimeError, "Cannot re-assign 'x' to a value of type module"):

            class ReassignSelfRHS(torch.jit.ScriptModule):

                @torch.jit.script_method
                def forward(self, x):
                    if False:
                        return 10
                    for _ in range(20):
                        x = self
                    return self
            ReassignSelfRHS()

    def test_unknown_builtin(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(RuntimeError, 'object has no attribute or method'):

            @torch.jit.script
            def unknown_builtin(x):
                if False:
                    while True:
                        i = 10
                return x.splork(3)

    def test_return_tuple(self):
        if False:
            for i in range(10):
                print('nop')

        def return_tuple(x):
            if False:
                i = 10
                return i + 15
            a = (x, x)
            return (a, x)
        self.checkScript(return_tuple, (torch.rand(4),))

    def test_add_tuple_optional(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(input: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
            if False:
                return 10
            changed_input = input[0] + 1
            value: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]] = (changed_input,) + input[1:]
            return value[2]
        inp: Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]] = (torch.rand(4), None, None)
        self.checkScript(foo, (inp,))

    def test_add_tuple_non_optional(self):
        if False:
            while True:
                i = 10

        def foo(input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
            if False:
                print('Hello World!')
            changed_input = input[0] + 1
            value: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = (changed_input,) + input[1:]
            return torch.sum(value[2]) + 4
        inp: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] = (torch.rand(4), torch.rand(4), torch.rand(4))
        self.checkScript(foo, (inp,))

    def test_add_tuple_different_types(self):
        if False:
            print('Hello World!')

        def foo(a: Tuple[int, float], b: Tuple[int]) -> int:
            if False:
                i = 10
                return i + 15
            c: Tuple[int, float, int] = a + b
            d: Tuple[int, float, int, int] = c + b
            return d[3] + 1
        a = (1, 2.0)
        b = (3,)
        self.checkScript(foo, (a, b))

    def test_add_tuple_same_types(self):
        if False:
            return 10

        def foo(a: Tuple[int, int], b: Tuple[int, int, int]) -> int:
            if False:
                return 10
            c: Tuple[int, int, int, int, int] = a + b
            d: Tuple[int, int, int, int, int, int, int, int] = c + b
            return d[6] - 2
        a = (1, 2)
        b = (3, 4, 5)
        self.checkScript(foo, (a, b))

    def test_method_no_self(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(RuntimeError, 'methods must have a self argument'):

            class MethodNoSelf(torch.jit.ScriptModule):

                @torch.jit.script_method
                def forward():
                    if False:
                        i = 10
                        return i + 15
                    return torch.zeros(3, 4)
            MethodNoSelf()

    def test_return_stmt_not_at_end(self):
        if False:
            return 10

        def return_stmt(x):
            if False:
                i = 10
                return i + 15
            if bool(x > 3):
                return x + 3
            else:
                return x
        self.checkScript(return_stmt, (torch.rand(1),))

    def test_for_in_range(self):
        if False:
            while True:
                i = 10

        def fn():
            if False:
                i = 10
                return i + 15
            c = 0
            for i in range(100):
                c += i
            return c
        self.checkScript(fn, ())

    def test_for_in_range_dynamic(self):
        if False:
            while True:
                i = 10

        def fn():
            if False:
                while True:
                    i = 10
            c = 0
            for i in range(100):
                acc = 0
                for j in range(i):
                    acc += j
                c += acc
            return c
        self.checkScript(fn, (), optimize=False)

    def test_for_in_range_ast(self):
        if False:
            for i in range(10):
                print('nop')

        def test_script_for_in_range_ast():
            if False:
                print('Hello World!')
            c = 0
            for i in range(100):
                acc = 0
                for j in range(i):
                    acc += j
                c += acc
            return c
        self.checkScript(test_script_for_in_range_ast, ())

    def test_for_in_range_if_ast(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def test_script_for_in_range_if_ast(x):
            if False:
                return 10
            output = x
            for i in range(20):
                if i == 0:
                    output = x.unsqueeze(0)
                else:
                    output = torch.cat((output, x.unsqueeze(0)), dim=0)
            return output
        inputs = self._make_scalar_vars([0], torch.int64)
        self.assertEqual(test_script_for_in_range_if_ast(*inputs).shape[0], 20)

    def test_for_in_range_start_end(self):
        if False:
            return 10

        def fn():
            if False:
                for i in range(10):
                    print('nop')
            x = 0
            for i in range(7, 100):
                x += i
            return x
        self.checkScript(fn, ())

    def test_for_in_range_start_end_step(self):
        if False:
            print('Hello World!')

        def fn(start, end, step):
            if False:
                print('Hello World!')
            x = 0
            for i in range(start, end, step):
                x += i
            return x
        self.checkScript(fn, (7, 100, 7))
        self.checkScript(fn, (7, 100, -7))
        self.checkScript(fn, (2, -11, -3))
        self.checkScript(fn, (2, -11, 3))
        self.checkScript(fn, (2, 10, 3))
        self.checkScript(fn, (-2, -10, -10))

    def test_for_in_range_zero_step(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def fn():
            if False:
                while True:
                    i = 10
            x = 0
            for i in range(2, -11, 0):
                x += i
            return x
        with self.assertRaisesRegex(RuntimeError, 'must not be zero'):
            fn()

    def test_range_args(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(RuntimeError, 'range expected at least 1 arguments, got 0'):

            @torch.jit.script
            def range_no_arg(x):
                if False:
                    return 10
                for _ in range():
                    x += 1
                return x
        with self.assertRaisesRegex(RuntimeError, 'found float'):

            @torch.jit.script
            def range_non_float():
                if False:
                    print('Hello World!')
                for i in range(0.5):
                    print(i)

    def test_parse_empty_tuple_annotation(self):
        if False:
            while True:
                i = 10
        cu = torch.jit.CompilationUnit('\n            def foo(x : Tuple[()]) -> Tuple[()]:\n                return x\n        ')
        foo_code = cu.find_function('foo').code
        FileCheck().check('Tuple[()]').check('Tuple[()]').run(foo_code)

    def test_parse_empty_tuple_annotation_element_error(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(RuntimeError, 'Tuple literal in Tuple type annotation must not have any elements'):
            cu = torch.jit.CompilationUnit('\n                def foo(x : Tuple[(int,)]) -> Tuple[(int,)]:\n                    return x\n            ')

    def test_parse_none_type_annotation(self):
        if False:
            while True:
                i = 10
        cu = torch.jit.CompilationUnit('\n            def foo(x : NoneType) -> NoneType:\n                return x\n        ')
        foo_code = cu.find_function('foo').code
        FileCheck().check(': NoneType').check('-> NoneType').run(foo_code)

    def test_empty_tuple_str(self):
        if False:
            print('Hello World!')
        empty_tuple_type = torch._C.TupleType([])
        g = {'Tuple': typing.Tuple}
        python_type = eval(empty_tuple_type.annotation_str, g)
        assert python_type is typing.Tuple[()]

    def test_tuple_str(self):
        if False:
            while True:
                i = 10
        tuple1_type = torch._C.TupleType([torch._C.StringType.get()])
        self.assertEqual(tuple1_type.annotation_str, 'Tuple[str]')
        tuple2_type = torch._C.TupleType([torch._C.StringType.get(), torch._C.StringType.get()])
        self.assertEqual(tuple2_type.annotation_str, 'Tuple[str, str]')

    def test_dict_str(self):
        if False:
            i = 10
            return i + 15
        dict_type = torch._C.DictType(torch._C.StringType.get(), torch._C.StringType.get())
        self.assertEqual(dict_type.annotation_str, 'Dict[str, str]')

    def test_none_type_str(self):
        if False:
            i = 10
            return i + 15
        none_type = torch._C.NoneType.get()
        g = {'NoneType': type(None)}
        python_type = eval(none_type.annotation_str, g)
        assert python_type is type(None)

    @skipIfTorchDynamo('TorchDynamo fails with unknown reason')
    def test_zip_enumerate_modulelist(self):
        if False:
            for i in range(10):
                print('nop')

        class Sub(torch.nn.Module):

            def forward(self, thing):
                if False:
                    while True:
                        i = 10
                return thing - 2

        class Double(torch.nn.Module):

            def forward(self, thing):
                if False:
                    i = 10
                    return i + 15
                return thing * 2

        class ZipModLists(torch.nn.Module):

            def __init__(self, mods, mods2):
                if False:
                    return 10
                super().__init__()
                self.mods = mods
                self.mods2 = mods2

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                iter = 0
                for (mod1, mod2) in zip(self.mods, self.mods2):
                    x = mod2(mod1(x))
                    iter += 1
                return (x, iter)

        class ZipWithValues(torch.nn.Module):
            __constants__ = ['tup_larger', 'tup_smaller']

            def __init__(self, mods, mods2):
                if False:
                    return 10
                super().__init__()
                self.mods = mods
                self.mods2 = mods2
                self.tup_larger = list(range(len(mods2) + 1))
                self.tup_smaller = list(range(max(len(mods2) + 1, 1)))

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                iter = 0
                x2 = x
                for (val, mod1, mod2) in zip(self.tup_larger, self.mods, self.mods2):
                    x = mod2(mod1(x)) + val
                    iter += 1
                for (val, mod1, mod2) in zip(self.tup_smaller, self.mods, self.mods2):
                    x2 = mod2(mod1(x2)) + val
                    iter += 1
                return (x, iter)
        mods = (nn.ModuleList([Double()]), nn.ModuleList([Double(), Sub(), Sub()]), nn.ModuleList([Sub(), Double()]))
        for i in range(len(mods)):
            for j in range(len(mods)):
                mod = ZipModLists(mods[i], mods[j])
                self.checkModule(mod, (torch.tensor(0.5),))
                mod2 = ZipWithValues(mods[i], mods[j])
                self.checkModule(mod2, (torch.tensor(0.5),))

    def test_enumerate_modlist_range(self):
        if False:
            while True:
                i = 10

        class Double(torch.nn.Module):

            def forward(self, thing):
                if False:
                    for i in range(10):
                        print('nop')
                return thing * 2

        class Mod(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.mods = nn.ModuleList([Double(), Double()])

            def forward(self, x):
                if False:
                    return 10
                x2 = x
                iter = 0
                for (val, mod) in enumerate(self.mods):
                    x2 = mod(x2) * val
                    iter += 1
                return (iter, x, x2)
        self.checkModule(Mod(), (torch.tensor(0.5),))

        class Mod2(Mod):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                for (val, mod) in zip(range(int(x)), self.mods):
                    x = mod(x) * val
                return x
        with self.assertRaisesRegex(Exception, 'that does not have a statically determinable length'):
            torch.jit.script(Mod2())

        class Mod3(Mod):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                for (val, mod) in zip(self.mods, range(int(x))):
                    x = mod(x) * val
                return x
        with self.assertRaisesRegex(Exception, 'that does not have a statically determinable length'):
            torch.jit.script(Mod3())

    def test_for_in_enumerate(self):
        if False:
            print('Hello World!')

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            sum = 0
            for (i, v) in enumerate(x):
                sum += i * v
            return sum
        self.checkScript(fn, ([1, 2, 3, 4, 5],))

        def fn_enumerate_start_arg(x):
            if False:
                return 10
            sum = 0
            for (i, v) in enumerate(x, 1):
                sum += i * v
            return sum
        self.checkScript(fn_enumerate_start_arg, ([1, 2, 3, 4, 5],))

        def fn_enumerate_start_kwarg(x):
            if False:
                i = 10
                return i + 15
            sum = 0
            for (i, v) in enumerate(x, start=1):
                sum += i * v
            return sum
        self.checkScript(fn_enumerate_start_kwarg, ([1, 2, 3, 4, 5],))

        def fn_nested_enumerate(x):
            if False:
                for i in range(10):
                    print('nop')
            sum = 0
            for (i, (j, v)) in enumerate(enumerate(x)):
                sum += i * j * v
            return sum
        self.checkScript(fn_nested_enumerate, ([1, 2, 3, 4, 5],))
        with self.assertRaisesRegex(RuntimeError, 'enumerate expected at least 1 arguments, got 0'):

            @torch.jit.script
            def enumerate_no_arg(x):
                if False:
                    while True:
                        i = 10
                sum = 0
                for _ in enumerate():
                    sum += 1
                return sum
        with self.assertRaisesRegex(RuntimeError, 'enumerate expected at most 2 arguments, got 3'):

            @torch.jit.script
            def enumerate_too_many_args(x):
                if False:
                    return 10
                sum = 0
                for _ in enumerate(x, x, x):
                    sum += 1
                return sum

    def test_list_comprehension_modulelist(self):
        if False:
            i = 10
            return i + 15

        class Inner(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x + 10

        class M(torch.nn.Module):

            def __init__(self, mod_list):
                if False:
                    return 10
                super().__init__()
                self.module_list = mod_list

            def forward(self, x):
                if False:
                    print('Hello World!')
                out = torch.jit.annotate(List[Tensor], [mod(x) for mod in self.module_list])
                return out
        mod = M(nn.ModuleList([Inner(), Inner()]))
        self.checkModule(mod, (torch.tensor(3),))
        mod = M(nn.ModuleList([]))
        torch.jit.script(mod)

        class M2(M):

            def __init__(self, mod_list):
                if False:
                    i = 10
                    return i + 15
                super().__init__(mod_list)

            def forward(self, x):
                if False:
                    return 10
                out = [mod(x) for mod in self.module_list]
                return out
        mod = M2(nn.ModuleList([Inner(), Inner()]))
        self.checkModule(mod, (torch.tensor(3),))
        mod = M2(nn.ModuleList([]))
        self.assertEqual(torch.jit.script(mod)(torch.tensor(0.5)), [])

        def bad_type_annotation():
            if False:
                while True:
                    i = 10
            out = torch.jit.annotate(int, [x for x in [1, 2, 3]])
            return out
        with self.assertRaisesRegex(Exception, 'Expected an annotation of type List'):
            torch.jit.script(bad_type_annotation)

    def test_list_comprehension_variable_write(self):
        if False:
            i = 10
            return i + 15

        def foo():
            if False:
                return 10
            i = 1
            x = [i if i != 5 else 3 for i in range(7)]
            return (i, x)
        self.assertEqual(foo(), torch.jit.script(foo)())

    def test_for_in_zip(self):
        if False:
            i = 10
            return i + 15

        def fn(x, y):
            if False:
                i = 10
                return i + 15
            sum = 0
            for (i, j) in zip(x, y):
                sum += i * j
            return sum
        self.checkScript(fn, ([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]))

        def fn_multi_inputs(x, y, z):
            if False:
                print('Hello World!')
            sum = 0
            for (i, j, k) in zip(x, y, z):
                sum += i * j * k
            return sum
        self.checkScript(fn_multi_inputs, ([1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]))

        def fn_nested_zip(x, y, z):
            if False:
                print('Hello World!')
            sum = 0
            for (i, (j, k)) in zip(x, zip(y, z)):
                sum += i * j * k
            return sum
        self.checkScript(fn_multi_inputs, ([1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]))
        with self.assertRaisesRegex(RuntimeError, 'zip expected at least 1 arguments, got 0'):

            @torch.jit.script
            def zip_no_arg(x):
                if False:
                    i = 10
                    return i + 15
                sum = 0
                for _ in zip():
                    sum += 1
                return sum
        with self.assertRaisesRegex(RuntimeError, 'too many values to unpack: need 2 but found 3'):

            @torch.jit.script
            def fn_nested_zip_wrong_target_assign(x, y, z):
                if False:
                    while True:
                        i = 10
                sum = 0
                for (i, (j, k)) in zip(x, y, z):
                    sum += i * j * k
                return sum

    def test_for_in_zip_enumerate(self):
        if False:
            print('Hello World!')

        def fn_zip_enumerate(x, y):
            if False:
                while True:
                    i = 10
            sum = 0
            for (i, (j, v), k) in zip(x, enumerate(y), range(0, 100)):
                sum += i * j * v * k
            return sum
        self.checkScript(fn_zip_enumerate, ([1, 2, 3, 4], [2, 3, 4, 5]))

        def fn_enumerate_zip(x, y):
            if False:
                print('Hello World!')
            sum = 0
            for (i, (j, v)) in enumerate(zip(x, y)):
                sum += i * j * v
            return sum
        self.checkScript(fn_enumerate_zip, ([1, 2, 3, 4], [2, 3, 4, 5]))

    def test_for_in_tensors(self):
        if False:
            while True:
                i = 10

        def test_sizes(x):
            if False:
                for i in range(10):
                    print('nop')
            sumz = 0
            for s in x:
                sumz += 1
            return sumz
        self.checkScript(test_sizes, (torch.rand(5, 4, 3, 2, 1),))
        self.checkScript(test_sizes, (torch.rand(777),))
        self.checkScript(test_sizes, (torch.rand(0),))

    def test_for_in_tensors_rank0(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(RuntimeError, 'of a 0-d tensor'):

            @torch.jit.script
            def test_sizes(x):
                if False:
                    for i in range(10):
                        print('nop')
                sumz = 0
                for s in x:
                    sumz += 1
                return sumz
            test_sizes(torch.tensor(1))

    def test_for_in_tensors_fail_scalar(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(RuntimeError, "'float' object is not iterable"):

            @torch.jit.script
            def test_sizes(x):
                if False:
                    i = 10
                    return i + 15
                sumz = 0
                for s in x:
                    sumz += 1
                return sumz
            test_sizes(0.0)

    def test_for_in_tensors_nested(self):
        if False:
            while True:
                i = 10

        def test_sizes(x):
            if False:
                i = 10
                return i + 15
            sumz = 0
            for n in x:
                for t in n:
                    sumz += 1
            return sumz
        self.checkScript(test_sizes, (torch.rand(5, 4, 3, 2, 1),))

    def get_sum_list_fn(self):
        if False:
            print('Hello World!')

        def sum_list(a):
            if False:
                print('Hello World!')
            sum = 0
            for i in a:
                sum += i
            return sum
        return sum_list

    def test_sum_list_diff_elms(self):
        if False:
            return 10
        self.checkScript(self.get_sum_list_fn(), ([1, 2, 3, 4, 5],))

    def test_sum_list_empty(self):
        if False:
            print('Hello World!')
        self.checkScript(self.get_sum_list_fn(), ([],))

    def test_sum_list_one(self):
        if False:
            return 10
        self.checkScript(self.get_sum_list_fn(), ([1],))

    def test_sum_list_literal(self):
        if False:
            return 10

        def sum_list():
            if False:
                print('Hello World!')
            sum = 0
            for i in [1, 2, 3, 4, 5]:
                sum += i
            return sum
        self.checkScript(sum_list, ())

    def test_sum_list_wrong_type(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(RuntimeError, "'int' object is not iterable"):

            @torch.jit.script
            def sum_list(a):
                if False:
                    i = 10
                    return i + 15
                sum = 0
                for i in a:
                    sum += i
                return sum
            sum_list(1)

    def test_list_iterables(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(RuntimeError, 'List of iterables is not supported currently'):
            cu = torch.jit.CompilationUnit('\n            def list_iterables(x):\n                for i, j in [2, 3, 4], [5, 6, 7]:\n                    x += i\n                    x += j\n                return x\n            ')

    def test_for_in_string(self):
        if False:
            while True:
                i = 10

        def test_strings(x):
            if False:
                return 10
            reverse = ''
            for c in x:
                reverse = c + reverse
            return reverse
        self.checkScript(test_strings, ('hello',))
        self.checkScript(test_strings, ('',))

        def test_list_strings(x):
            if False:
                print('Hello World!')
            result = ''
            for sub_str in x:
                result += sub_str
            return result
        self.checkScript(test_list_strings, (['hello', 'world'],))
        self.checkScript(test_list_strings, (['hello', ' ', 'world', ''],))

    def test_for_in_dict(self):
        if False:
            print('Hello World!')

        def test_dicts(x):
            if False:
                while True:
                    i = 10
            sum = 0
            for key in x:
                sum += x[key]
            return sum
        self.checkScript(test_dicts, ({'a': 1, 'b': 2, 'c': 3},))

        def test_dict_keys_values(x):
            if False:
                return 10
            key_str = ''
            sum = 0
            for key in x.keys():
                key_str += key
            for val in x.values():
                sum += val
            return (key_str, sum)
        self.checkScript(test_dicts, ({'a': 1, 'b': 2, 'c': 3},))

    def test_for_tuple_unpack(self):
        if False:
            return 10

        def for_tuple_unpack(x, y):
            if False:
                return 10
            for (i, j) in [[3, 4], [5, 6], [7, 8]]:
                x += i
                y += j
            return (x, y)
        self.checkScript(for_tuple_unpack, (torch.tensor(3), torch.tensor(5)))

        def nested_tuple_unpack(x, y):
            if False:
                print('Hello World!')
            sum = 0
            for (i, (j, k), v) in zip(x, enumerate(x), y):
                sum += i + j + k + v
            return sum
        self.checkScript(nested_tuple_unpack, ([1, 3, 5], [2, 4, 6]))

    def test_for_tuple_assign(self):
        if False:
            for i in range(10):
                print('nop')

        def test_simple_assign(x):
            if False:
                return 10
            sum = 0.0
            for a in x:
                sum += float(a)
            return sum
        self.checkScript(test_simple_assign, ((1, 2.5),))

        def test_tuple_assign(x):
            if False:
                while True:
                    i = 10
            sum = 0
            for a in x:
                sum += a[0]
                sum += a[1]
            return sum
        self.checkScript(test_tuple_assign, (((1, 2), (4, 7)),))

    def test_single_starred_lhs(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(RuntimeError, 'A Starred expression may only appear on the lhs within the presence of another non-starred expression'):
            cu = torch.jit.CompilationUnit('\n            def single_starred_lhs(x):\n                a = (x, x, x)\n                *b, = a\n                return b\n            ')

    def test_singleton_tuple_unpack(self):
        if False:
            print('Hello World!')

        def foo(a):
            if False:
                print('Hello World!')
            (b,) = (a,)
            return b + 1
        self.checkScript(foo, (torch.rand(3),))

    def test_tuple_assignments(self):
        if False:
            for i in range(10):
                print('nop')

        def var_tuple_assign(x, y):
            if False:
                print('Hello World!')
            ((a, b), c) = (x, y)
            return a + b + c
        tuple_inputs = (torch.randn(1, 4), torch.randn(3, 4))
        self.checkScript(var_tuple_assign, (tuple_inputs, torch.randn(3, 4)))

        def nested_tuple_assign(x, y, z):
            if False:
                for i in range(10):
                    print('nop')
            (a, (b, (c, d)), (e, f)) = (x, y, z)
            return a + b + c + d + e + f
        self.checkScript(nested_tuple_assign, (1, (2, (3, 4)), (5, 6)))

        def subscript_tuple_assign(a, x, i):
            if False:
                return 10
            (a[i], (x[i], b)) = (1, (2, 3))
            return (a[i] + 1, x + 5, b)
        self.checkScript(subscript_tuple_assign, ([12, 7, 9, 11], torch.tensor((3, 13, 17)), 0))

        def star_tuple_assign():
            if False:
                for i in range(10):
                    print('nop')
            (a, (b, *c), *d) = (1, (2, 3, 4), 5, 6)
            return (a, b, c, d)
        self.checkScript(star_tuple_assign, ())

        def subscript_tuple_augmented_assign(a):
            if False:
                print('Hello World!')
            a[0] += 1
            return a
        with self.assertRaisesRegex(RuntimeError, 'does not support augmented assign'):
            scripted_aug_assign = torch.jit.script(subscript_tuple_augmented_assign)

        class AttrTupleAssignmentTestClass:

            def __init__(self, a: int, b: int):
                if False:
                    i = 10
                    return i + 15
                self.a = a
                self.b = b

            def set_ab(self, a: int, b: int):
                if False:
                    return 10
                (self.a, self.b) = (a, b)

            def get(self) -> Tuple[int, int]:
                if False:
                    for i in range(10):
                        print('nop')
                return (self.a, self.b)
        make_global(AttrTupleAssignmentTestClass)

        @torch.jit.script
        def attr_tuple_assignment(o: AttrTupleAssignmentTestClass, a: int, b: int):
            if False:
                return 10
            o.set_ab(a, b)
            return o
        o = AttrTupleAssignmentTestClass(1, 2)
        self.assertEqual(attr_tuple_assignment(o, 3, 4).get(), (3, 4))

    def test_multiple_assign(self):
        if False:
            for i in range(10):
                print('nop')

        def test():
            if False:
                for i in range(10):
                    print('nop')
            a = (b, c) = (d, f) = (1, 1)
            ten = torch.tensor(1)
            ten1 = ten2 = ten.add_(1)
            x = 1
            y = 3
            (x, y) = (y, x + y)
            return (a, b, c, d, f, ten, ten1, ten2, x, y)
        self.checkScript(test, ())

    def test_multi_reduction(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(RuntimeError, 'augmented assignment can only have one LHS expression'):
            cu = torch.jit.CompilationUnit('\n            def multi_reduction(x):\n                a, b += x\n                return a, b\n            ')

    def test_invalid_call_arguments(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(RuntimeError, 'but instead found type '):

            @torch.jit.script
            def invalid_call_arguments(x):
                if False:
                    return 10
                return torch.unsqueeze(3, 4, 5, 6, 7, 8)

    def test_invalid_lhs_assignment(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(RuntimeError, 'unexpected expression'):
            cu = torch.jit.CompilationUnit('\n            def invalid_lhs_assignment(x):\n                x + 1 = x\n                return x\n            ')

    def test_multi_starred_expr_lhs(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(RuntimeError, 'Only one starred expression is allowed on the lhs'):
            cu = torch.jit.CompilationUnit('\n            def multi_starred_expr_lhs():\n                a, *b, *c = [1, 2, 3, 4, 5, 6]\n                return a\n            ')

    def test_pack_tuple_into_non_var(self):
        if False:
            return 10
        with self.assertRaisesRegex(RuntimeError, 'Cannot pack a tuple into a non-variable'):
            cu = torch.jit.CompilationUnit('\n            def pack_tuple_into_non_var(x):\n                a, *1 = (3, 4, 5)\n                return x\n            ')

    def test_print_kwargs(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(RuntimeError, "print doesn't accept any keyword arguments"):
            cu = torch.jit.CompilationUnit('\n            def print_kwargs(x):\n                print(x, flush=True)\n                return x\n            ')

    def test_builtin_use_as_value(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(RuntimeError, 'builtin cannot be used as a value'):

            @torch.jit.script
            def builtin_use_as_value(x):
                if False:
                    return 10
                return x.unsqueeze

    def test_wrong_use_as_tuple(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(RuntimeError, 'cannot be used as a tuple'):

            def test_fn():
                if False:
                    i = 10
                    return i + 15
                return 3

            @torch.jit.script
            def wrong_use_as_tuple(self):
                if False:
                    for i in range(10):
                        print('nop')
                (a, b) = test_fn
                return a

    def test_wrong_attr_lookup(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(RuntimeError, 'attribute lookup is not defined on builtin'):

            @torch.jit.script
            def wrong_attr_lookup(self, x):
                if False:
                    print('Hello World!')
                a = x.unsqueeze.myattr
                return a

    def test_wrong_use_as_callable(self):
        if False:
            return 10
        with self.assertRaisesRegex(RuntimeError, 'cannot call a value'):

            @torch.jit.script
            def wrong_use_as_callable(x):
                if False:
                    while True:
                        i = 10
                return x(3, 4, 5)

    def test_python_val_doesnt_have_attr(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(RuntimeError, 'object has no attribute abcd'):

            @torch.jit.script
            def python_val_doesnt_have_attr():
                if False:
                    return 10
                return shutil.abcd

    def test_wrong_module_attr_lookup(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(RuntimeError, "python value of type 'type' cannot be used as a value"):
            import io

            @torch.jit.script
            def wrong_module_attr_lookup():
                if False:
                    i = 10
                    return i + 15
                return io.BytesIO

    def test_wrong_method_call_inputs(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(RuntimeError, 'Argument y not provided'):

            class SomeModule(torch.jit.ScriptModule):

                @torch.jit.script_method
                def foo(self, x, y):
                    if False:
                        return 10
                    return x

                @torch.jit.script_method
                def forward(self, x, y):
                    if False:
                        for i in range(10):
                            print('nop')
                    return self.foo(x)
            SomeModule()

    def test_single_starred_expr_for_loop(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(RuntimeError, 'A Starred expression may only appear'):
            cu = torch.jit.CompilationUnit('\n            def test():\n                x = 0\n                for *a in [1, 2, 3]:\n                    x = x + 1\n                return x\n            ')

    def test_call_ge(self):
        if False:
            return 10
        with self.assertRaisesRegex(RuntimeError, 'Expected at most 1 arguments but found 3'):

            @_trace(torch.zeros(1, 2, 3))
            def foo(x):
                if False:
                    print('Hello World!')
                return x

            @torch.jit.script
            def test_fn():
                if False:
                    while True:
                        i = 10
                return foo(torch.full([1], 1), torch.full([1], 2), torch.full([1], 3))

    def test_wrong_return_type(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(RuntimeError, 'but instead got value of type tuple'):

            @torch.jit.ignore
            def somefunc():
                if False:
                    print('Hello World!')
                return (torch.zeros(3, 4), torch.zeros(4, 5))

            @torch.jit.script
            def wrong_return_type():
                if False:
                    for i in range(10):
                        print('nop')
                return somefunc()
            wrong_return_type()

    def test_call_python_fn_from_tracing_fn(self):
        if False:
            i = 10
            return i + 15

        def python_fn(x):
            if False:
                while True:
                    i = 10
            return torch.neg(x)

        @_trace(torch.rand(3, 4))
        def traced_fn(x):
            if False:
                while True:
                    i = 10
            return python_fn(x) + 1
        FileCheck().check('aten::neg').run(str(traced_fn.graph))

    def test_call_python_mod_from_tracing_fn(self):
        if False:
            print('Hello World!')

        class PythonMod(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3), requires_grad=False)

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.mm(x, self.param)
        pm = PythonMod()

        @_trace(torch.rand(3, 4))
        def traced_fn(x):
            if False:
                return 10
            return pm(x) + 1.0
        self.assertTrue(len(list(traced_fn.graph.inputs())) == 1)
        FileCheck().check('aten::mm').check('aten::add').run(str(traced_fn.graph))

    @_tmp_donotuse_dont_inline_everything
    def test_call_traced_fn_from_tracing_fn(self):
        if False:
            i = 10
            return i + 15

        @_trace(torch.rand(3, 4))
        def traced_fn1(x):
            if False:
                i = 10
                return i + 15
            return torch.neg(x)

        @_trace(torch.rand(3, 4))
        def traced_fn(x):
            if False:
                while True:
                    i = 10
            return traced_fn1(x) + 1
        FileCheck().check('traced_fn').check('prim::CallFunction').check('aten::add').run(str(traced_fn.graph))

    @unittest.skip('error in first class mode')
    def test_call_traced_mod_from_tracing_fn(self):
        if False:
            return 10

        class TracedModule(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3), requires_grad=False)

            def forward(self, x):
                if False:
                    return 10
                return torch.mm(x, self.param)
        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))
        with self.assertRaisesRegex(RuntimeError, 'must be registered as submodules'):

            @_trace(torch.rand(3, 4))
            def traced_fn(x):
                if False:
                    print('Hello World!')
                return tm(x) + 1.0

    @_tmp_donotuse_dont_inline_everything
    def test_call_script_fn_from_tracing_fn(self):
        if False:
            return 10

        @torch.jit.script
        def script_fn(x):
            if False:
                while True:
                    i = 10
            return torch.neg(x)

        @_trace(torch.rand(3, 4))
        def traced_fn(x):
            if False:
                while True:
                    i = 10
            return script_fn(x) + 1
        FileCheck().check('prim::CallFunction').check('aten::add').run(str(traced_fn.graph))

    @unittest.skip('error in first class mode')
    def test_call_script_mod_from_tracing_fn(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(RuntimeError, 'must be registered as submodules'):

            class ScriptMod(torch.jit.ScriptModule):

                def __init__(self):
                    if False:
                        i = 10
                        return i + 15
                    super().__init__()
                    self.param = torch.nn.Parameter(torch.rand(3, 4), requires_grad=False)

                @torch.jit.script_method
                def forward(self, x):
                    if False:
                        i = 10
                        return i + 15
                    for _i in range(4):
                        x += self.param
                    return x
            sm = ScriptMod()

            @_trace(torch.rand(3, 4))
            def traced_fn(x):
                if False:
                    print('Hello World!')
                return sm(x) + 1.0

    def test_call_python_fn_from_traced_module(self):
        if False:
            for i in range(10):
                print('nop')

        def python_fn(x):
            if False:
                i = 10
                return i + 15
            return torch.neg(x)

        class TracedModule(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3))

            def forward(self, x):
                if False:
                    return 10
                return torch.mm(python_fn(x), self.param)
        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))
        self.assertTrue(len(list(tm.graph.inputs())) == 2)
        FileCheck().check('aten::neg').check('aten::mm').run(str(tm.graph))

    def test_call_python_mod_from_traced_module(self):
        if False:
            print('Hello World!')

        class PythonModule(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(5, 7))

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.mm(x, self.param)

        class TracedModule(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 5))
                self.mod = PythonModule()

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.mod(torch.mm(x, self.param)) + 1.0
        tm = torch.jit.trace(TracedModule(), torch.rand(3, 4))
        FileCheck().check_not('value=<Tensor>').check('aten::mm').check('prim::CallMethod[name="forward"]').check('aten::add').run(str(tm.graph))
        FileCheck().check('aten::mm').run(str(tm.mod.graph))

    def test_op_dtype(self):
        if False:
            print('Hello World!')

        def check_equal_and_dtype(a, b):
            if False:
                print('Hello World!')
            self.assertEqual(a, b)
            self.assertEqual(a.dtype, b.dtype)

        def fn():
            if False:
                return 10
            a = torch.arange(10)
            b = torch.arange(10, dtype=torch.float)
            c = torch.arange(1, 10, 2)
            d = torch.arange(1, 10, 2, dtype=torch.float)
            e = torch.arange(1, 10.0, 2)
            f = torch.arange(1, 10.0, 2, dtype=torch.float)
            return (a, b, c, d, e, f)
        scripted_fn = torch.jit.script(fn)
        eager_out = fn()
        script_out = scripted_fn()
        for (a, b) in zip(eager_out, script_out):
            check_equal_and_dtype(a, b)

    def test_floor_div(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def foo(a, b):
            if False:
                print('Hello World!')
            return a // b
        for i in range(-8, 8):
            for j in range(-8, 8):
                if j != 0:
                    self.assertEqual(foo(i, j), i // j)

    def test_floordiv(self):
        if False:
            while True:
                i = 10
        funcs_template = dedent('\n        def fn():\n            ten = {a_construct}\n            ten_or_scalar = {b_construct}\n            return ten // ten_or_scalar, torch.floor_divide(ten, ten_or_scalar)\n        ')
        lhs = ['torch.tensor([5.5, 3.2])', 'torch.tensor([2, 2])', 'torch.tensor([3, 2])']
        rhs = ['1.5', '2', '4', '1.1'] + lhs
        for tensor in lhs:
            for tensor_or_scalar in rhs:
                funcs_str = funcs_template.format(a_construct=tensor, b_construct=tensor_or_scalar)
                scope = {}
                execWrapper(funcs_str, globals(), scope)
                cu = torch.jit.CompilationUnit(funcs_str)
                f_script = cu.fn
                f = scope['fn']
                self.assertEqual(f_script(), f())

    def test_call_python_fn_from_script_fn(self):
        if False:
            while True:
                i = 10

        @torch.jit.ignore
        def python_fn(x):
            if False:
                return 10
            return torch.neg(x)

        @torch.jit.script
        def script_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return python_fn(x) + 1
        a = torch.tensor(1)
        self.assertEqual(script_fn(a), torch.tensor(0))
        FileCheck().check('python_fn').run(str(script_fn.graph))

    def test_call_python_mod_from_script_fn(self):
        if False:
            print('Hello World!')

        class PythonModule(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(5, 7))

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.mm(x, self.param)
        pm = PythonModule()

        @torch.jit.script
        def script_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return pm(x) + 1
        FileCheck().check('python_value').check('aten::add').run(str(script_fn.graph))

    @_tmp_donotuse_dont_inline_everything
    def test_call_script_fn_from_script_fn(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def script_fn1(x):
            if False:
                i = 10
                return i + 15
            return torch.neg(x)

        @torch.jit.script
        def script_fn(x):
            if False:
                print('Hello World!')
            return script_fn1(x) + 1
        FileCheck().check('prim::CallFunction').run(str(script_fn.graph))

    def test_call_script_mod_from_script_fn(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(RuntimeError, 'Cannot call a ScriptModule that is not a submodule of the caller'):

            class ScriptMod(torch.jit.ScriptModule):

                @torch.jit.script_method
                def forward(self, x):
                    if False:
                        i = 10
                        return i + 15
                    return torch.mm(x, torch.zeros([4, 3]))
            sm = ScriptMod()

            @torch.jit.script
            def script_fn(x):
                if False:
                    while True:
                        i = 10
                return sm(x) + 1

    def test_call_python_fn_from_script_module(self):
        if False:
            print('Hello World!')

        @torch.jit.ignore
        def python_fn(x):
            if False:
                while True:
                    i = 10
            return torch.neg(x)

        class ScriptMod(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3))

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    return 10
                return python_fn(torch.mm(x, self.param))
        sm = ScriptMod()
        FileCheck().check('aten::mm').check('python_fn').run(str(sm.forward.graph))

    def test_call_python_mod_from_script_module(self):
        if False:
            while True:
                i = 10

        class PythonMod(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 5))

            @torch.jit.ignore
            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.mm(x, self.param)

        class ScriptMod(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3))
                self.pm = PythonMod()

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.pm(torch.mm(x, self.param))
        sm = ScriptMod()
        FileCheck().check('aten::mm').check('forward').run(str(sm.graph))

    @_tmp_donotuse_dont_inline_everything
    def test_call_script_fn_from_script_module(self):
        if False:
            return 10

        @torch.jit.script
        def script_fn(x):
            if False:
                print('Hello World!')
            return torch.neg(x)

        class ScriptMod(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3))

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return script_fn(torch.mm(x, self.param))
        sm = ScriptMod()
        graph = sm.forward.graph
        FileCheck().check('aten::mm').check('prim::CallFunction').run(str(graph))

    @_tmp_donotuse_dont_inline_everything
    def test_call_script_mod_from_script_module(self):
        if False:
            i = 10
            return i + 15

        class ScriptMod1(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 5))

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.mm(x, self.param)

        class ScriptMod(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 3))
                self.tm = ScriptMod1()

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.tm(torch.mm(x, self.param))
        sm = ScriptMod()
        FileCheck().check_count('%', 3).check(':').check_count('mm', 1).check('prim::CallMethod').run(str(sm.graph))

    def test_module_with_params_called_fails(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(RuntimeError, 'Cannot call a ScriptModule that is not a submodule of the caller'):

            class ScriptMod(torch.jit.ScriptModule):

                def __init__(self):
                    if False:
                        while True:
                            i = 10
                    super().__init__()
                    self.param = torch.nn.Parameter(torch.rand(3, 3))

                @torch.jit.script_method
                def forward(self, x):
                    if False:
                        i = 10
                        return i + 15
                    return torch.mm(x, self.param)
            sm = ScriptMod()

            @torch.jit.script
            def some_func(x):
                if False:
                    return 10
                return sm(x)

    def test_tuple_index_to_list(self):
        if False:
            for i in range(10):
                print('nop')

        def test_non_constant_input(a):
            if False:
                return 10
            if a:
                b = 1
            else:
                b = 0
            c = (0, 1)
            return c[b]
        self.checkScript(test_non_constant_input, (True,))
        self.checkScript(test_non_constant_input, (False,))
        with self.assertRaisesRegex(RuntimeError, 'because we cannot resolve the output type'):

            @torch.jit.script
            def test_non_constant_input(a):
                if False:
                    while True:
                        i = 10
                if a:
                    b = 1
                else:
                    b = 0
                c = (0, 1.1)
                print(c[b])

    def test_tuple_indexing(self):
        if False:
            i = 10
            return i + 15

        def tuple_index(a):
            if False:
                i = 10
                return i + 15
            if bool(a):
                b = (1, 2)
            else:
                b = (0, 2)
            return (b[-2], b[1])
        self.checkScript(tuple_index, (torch.tensor([0]),))
        self.checkScript(tuple_index, (torch.tensor([1]),))
        self.checkScript(tuple_index, (torch.tensor([1]),), optimize=True)
        tuple_comp = torch.jit.script(tuple_index)
        FileCheck().check_count('TupleIndex', 2, exactly=True).run(str(tuple_comp.graph))
        with self.assertRaisesRegex(RuntimeError, 'index must be an integer'):

            @torch.jit.script
            def test_indexing_float():
                if False:
                    for i in range(10):
                        print('nop')
                c = (1, 2)
                return c[0.1]

        def test_indexing_out_of_bounds_pos():
            if False:
                return 10
            c = (1, 2)
            return c[2]
        self.checkScriptRaisesRegex(test_indexing_out_of_bounds_pos, (), Exception, 'out of range')

        def test_indexing_out_of_bounds_neg():
            if False:
                print('Hello World!')
            c = (1, 2)
            return c[-3]
        self.checkScriptRaisesRegex(test_indexing_out_of_bounds_pos, (), Exception, 'out of range')

        def negative_index():
            if False:
                return 10
            tup = (1, 2, 3, 4)
            return tup[-1]
        self.checkScript(negative_index, [])

        def really_negative_index():
            if False:
                return 10
            tup = (1, 2, 3, 4)
            return tup[-100]
        self.checkScriptRaisesRegex(really_negative_index, [], Exception, 'index out of range')

        def negative_slice():
            if False:
                print('Hello World!')
            tup = (1, 2, 3, 4)
            return tup[-3:4]
        self.checkScript(negative_slice, [])

        def really_slice_out_of_bounds():
            if False:
                i = 10
                return i + 15
            tup = (1, 2, 3, 4)
            return tup[-300:4000]
        self.checkScript(really_slice_out_of_bounds, [])

    def test_namedtuple_attr(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                while True:
                    i = 10
            return x.max(dim=1).indices + torch.max(x, dim=1).indices
        self.checkScript(f, (torch.rand(20, 20, 20),), optimize=True)
        with self.assertRaisesRegex(RuntimeError, 'object has no attribute or method'):

            @torch.jit.script
            def g1(x):
                if False:
                    print('Hello World!')
                return x.max(dim=1).unknown_symbol
        with self.assertRaisesRegex(RuntimeError, 'object has no attribute or method'):

            @torch.jit.script
            def g2(x):
                if False:
                    for i in range(10):
                        print('nop')
                print((x, x, x).__doc__)
                return x

    def test_tuple_len(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def foo():
            if False:
                print('Hello World!')
            return len((1, 'str', None))
        self.assertEqual(foo(), 3)

        @torch.jit.script
        def test_indexing_end_out_of_bounds():
            if False:
                return 10
            c = (1, 2)
            return c[2:10]
        self.assertEqual(test_indexing_end_out_of_bounds(), ())

    def test_lower_nested_tuples(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def test():
            if False:
                while True:
                    i = 10
            return ((1, 2), 3)
        self.run_pass('constant_propagation', test.graph)
        FileCheck().check('prim::Constant').check_not('TupleConstruct').run(test.graph)
        self.run_pass('lower_all_tuples', test.graph)

    def test_unwrap_optional_builtin(self):
        if False:
            while True:
                i = 10

        def test(x):
            if False:
                i = 10
                return i + 15
            x = torch.jit._unwrap_optional(x)
            x = x + x
            return x
        self.checkScript(test, (3,))
        with self.assertRaisesRegex(AssertionError, 'Unwrapping null optional'):
            test(None)
        test_script = torch.jit.script(test)
        with self.assertRaisesRegex(RuntimeError, 'Unwrapping null optional'):
            test_script(None)

        @torch.jit.script
        def test_test():
            if False:
                print('Hello World!')
            return torch.jit._unwrap_optional(1)
        with self.assertRaisesRegex(RuntimeError, 'could not be inferred from actual type None'):

            @torch.jit.script
            def test_no_type():
                if False:
                    i = 10
                    return i + 15
                return torch.jit._unwrap_optional(None)

    def test_indexing_error(self):
        if False:
            return 10
        with self.assertRaisesRegex(RuntimeError, "'int' object is not subscriptable"):

            @torch.jit.script
            def test_wrong_type():
                if False:
                    i = 10
                    return i + 15
                a = 8
                return a[0]

    def test_unsupported_builtin_error(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(RuntimeError, 'Python builtin <built-in function hypot> is currently'):

            @torch.jit.script
            def test_unsupported(a):
                if False:
                    i = 10
                    return i + 15
                return math.hypot(a, 2.0)

    def test_annotated_script_fn(self):
        if False:
            return 10

        @torch.jit.script
        def foo(x, y, z):
            if False:
                while True:
                    i = 10
            return x
        self.assertExpected(str(foo.schema))

    def test_annotated_script_method(self):
        if False:
            for i in range(10):
                print('nop')

        class SM(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return (y, y, y)
        sm = SM()
        self.assertExpectedStripMangled(str(sm.forward.schema))

    def test_annotated_script_fn_return_mismatch(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(RuntimeError, 'but is actually of type'):

            @torch.jit.script
            def return_tup(x):
                if False:
                    for i in range(10):
                        print('nop')
                return (x, x)

    def test_annotated_script_fn_arg_mismatch(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(RuntimeError, 'Arguments for call are not valid'):

            @torch.jit.script
            def tuple_arg(x):
                if False:
                    i = 10
                    return i + 15
                return x + 1

    def test_script_non_tensor_args_outputs(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def fn(x, y):
            if False:
                return 10
            return float((x + y).sum())
        x = torch.ones(2, 2)
        z = fn(x, 1)
        self.assertIsInstance(z, float)
        self.assertEqual(z, 8.0)

    @unittest.skip('https://github.com/pytorch/pytorch/issues/9595')
    def test_inline_and_run_annotated_script_fn(self):
        if False:
            return 10

        @torch.jit.script
        def to_inline(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return y

        @torch.jit.script
        def some_func(x):
            if False:
                i = 10
                return i + 15
            return to_inline((x, x), x)
        x = torch.rand(3, 4)
        self.assertEqual(some_func(x), x)

    def _make_filereader_test_file(self):
        if False:
            for i in range(10):
                print('nop')
        filename = tempfile.mktemp()
        writer = torch._C.PyTorchFileWriter(filename)
        buffers = [os.urandom(size) for size in [random.randint(1, 100) for i in range(20)]]
        offsets = []
        for (i, buf) in enumerate(buffers):
            writer.write_record(str(i), buf, len(buf))
            offsets.append(i)
        serialized_offsets = pickle.dumps(offsets)
        writer.write_record('meta', serialized_offsets, len(serialized_offsets))
        writer.write_end_of_file()
        return (filename, buffers, serialized_offsets)

    def test_file_format_serialization(self):
        if False:
            while True:
                i = 10
        (filename, buffers, serialized_offsets) = self._make_filereader_test_file()
        reader = torch._C.PyTorchFileReader(filename)
        serialized_offsets_read = reader.get_record('meta')
        parsed_serialized_offsets = pickle.loads(serialized_offsets)
        for (i, offset) in enumerate(parsed_serialized_offsets):
            data = reader.get_record(str(offset))
            assert data == buffers[i]

    def test_file_reader_no_memory_leak(self):
        if False:
            print('Hello World!')
        num_iters = 10000
        (filename, _, _) = self._make_filereader_test_file()
        tracemalloc.start()
        for i in range(num_iters):
            torch._C.PyTorchFileReader(filename)
        (_, peak_from_string) = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracemalloc.start()
        with open(filename, 'rb') as f:
            for i in range(num_iters):
                f.seek(0)
                torch._C.PyTorchFileReader(f)
        (_, peak_from_file) = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert peak_from_file < peak_from_string * 500

    def type_input_return_pairs(self):
        if False:
            print('Hello World!')
        return [('Tensor', 'Tensor'), ('torch.Tensor', 'Tensor'), ('str', 'str'), ('int', 'int'), ('bool', 'bool'), ('BroadcastingList3[float]', 'List[float]'), ('BroadcastingList2[int]', 'List[int]'), ('List[int]', 'List[int]'), ('Optional[int]', 'Optional[int]')]

    def format_code(self, code, pair):
        if False:
            for i in range(10):
                print('nop')
        return code.format(input=pair[0], output=pair[1])

    def test_annot_string_py3_fn(self):
        if False:
            while True:
                i = 10
        code = '\n            def foo(x : {input}, y : Tuple[Tensor, Tensor]) -> Tuple[{output}, {output}]:\n                return x, x\n        '
        test_str = []
        for pair in self.type_input_return_pairs():
            cu = torch.jit.CompilationUnit(self.format_code(code, pair))
            test_str.append(str(cu.foo.schema))
        self.assertExpected('\n'.join(test_str) + '\n')

    def test_annot_string_py3_method(self):
        if False:
            i = 10
            return i + 15

        class TestModule(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
        code = '\n            def foo(self, x : {input}, y : Tuple[Tensor, Tensor]) -> Tuple[{output}, {output}]:\n                return x, x\n        '
        test_str = []
        for pair in self.type_input_return_pairs():
            jit_utils.clear_class_registry()
            tm = TestModule()
            tm.define(self.format_code(code, pair))
            test_str.append(str(tm.foo.schema))
        self.assertExpectedStripMangled('\n'.join(test_str) + '\n')

    def test_annot_string_mypy_fn(self):
        if False:
            i = 10
            return i + 15
        code = '\n            def foo(x, y):\n                # type: ({input}, Tuple[Tensor, Tensor]) -> Tuple[{output}, {output}]\n                return x, x\n        '
        test_str = []
        for pair in self.type_input_return_pairs():
            cu = torch.jit.CompilationUnit(self.format_code(code, pair))
            test_str.append(str(cu.foo.schema))
        self.assertExpectedStripMangled('\n'.join(test_str) + '\n')

    def test_annot_string_mypy_method(self):
        if False:
            print('Hello World!')

        class TestModule(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
        code = '\n        def foo(self, x, y):\n            # type: ({input}, Tuple[Tensor, Tensor]) -> Tuple[{output}, {output}]\n            return x, x\n        '
        test_str = []
        for pair in self.type_input_return_pairs():
            jit_utils.clear_class_registry()
            tm = TestModule()
            tm.define(self.format_code(code, pair))
            test_str.append(str(tm.foo.schema))
        self.assertExpectedStripMangled('\n'.join(test_str) + '\n')

    def test_annot_ast_py3_fn(self):
        if False:
            print('Hello World!')
        code = dedent('\n            from typing import Tuple, List, Optional\n            from torch import Tensor\n            from torch.jit.annotations import BroadcastingList2, BroadcastingList3\n            import torch\n            @torch.jit.script\n            def foo(x : {input}, y : Tuple[Tensor, Tensor]) -> Tuple[{output}, {output}]:\n                return x, x\n        ')
        test_str = []
        for pair in self.type_input_return_pairs():
            fn = jit_utils._get_py3_code(self.format_code(code, pair), 'foo')
            test_str.append(str(fn.schema))
        self.assertExpectedStripMangled('\n'.join(test_str) + '\n')

    def test_multiline_annot_ast_py3_fn(self):
        if False:
            print('Hello World!')
        code = dedent('\n            from typing import Tuple, List, Optional\n            from torch import Tensor\n            from torch.jit.annotations import BroadcastingList2, BroadcastingList3\n            import torch\n            @torch.jit.script\n            def foo(x,  # type: {input}\n                    y   # type: Tuple[Tensor, Tensor]\n                    ):\n                # type: (...) -> Tuple[{output}, {output}]\n                return x, x\n        ')
        test_str = []
        for pair in self.type_input_return_pairs():
            fn = jit_utils._get_py3_code(self.format_code(code, pair), 'foo')
            args = fn.schema.arguments
            returns = fn.schema.returns
            self.assertEqual(str(args[0].type), pair[1])
            self.assertEqual(str(args[1].type), 'Tuple[Tensor, Tensor]')
            self.assertEqual(str(returns[0].type), f'Tuple[{pair[1]}, {pair[1]}]')

    def test_bad_multiline_annotations(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(RuntimeError, 'Return type line'):

            @torch.jit.script
            def bad_type_line(a, b, c):
                if False:
                    i = 10
                    return i + 15
                return a + b + c
        with self.assertRaisesRegex(RuntimeError, 'Return type line'):

            @torch.jit.script
            def bad_return_line(a, b, c):
                if False:
                    while True:
                        i = 10
                return a + b + c
        with self.assertRaisesRegex(RuntimeError, 'Number of type annotations'):

            @torch.jit.script
            def missing_type(a, b, c):
                if False:
                    for i in range(10):
                        print('nop')
                return a + b + c

    def test_annot_ast_py3_method(self):
        if False:
            while True:
                i = 10
        code = dedent('\n            from typing import Tuple, List, Optional\n            from torch import Tensor\n            from torch.jit.annotations import BroadcastingList2, \\\n                BroadcastingList3\n            import torch\n            class FooModule(torch.jit.ScriptModule):\n                @torch.jit.script_method\n                def foo(self, x : {input}, y : Tuple[Tensor, Tensor]) -> Tuple[{output}, {output}]:\n                    return x, x\n            instance = FooModule()\n        ')
        test_str = []
        for pair in self.type_input_return_pairs():
            fn = jit_utils._get_py3_code(self.format_code(code, pair), 'instance')
            test_str.append(str(fn.foo.schema))
        self.assertExpectedStripMangled('\n'.join(test_str) + '\n')

    def test_annot_ast_mypy_fn(self):
        if False:
            i = 10
            return i + 15
        code = dedent('\n            import torch\n            @torch.jit.script\n            def foo(x, y):\n                # type: ({input}, Tuple[Tensor, Tensor]) -> Tuple[{output}, {output}]\n                return x, x\n        ')
        test_str = []
        for pair in self.type_input_return_pairs():
            fn = jit_utils._get_py3_code(self.format_code(code, pair), 'foo')
            test_str.append(str(fn.schema))
        self.assertExpected('\n'.join(test_str) + '\n')

    def test_annot_ast_mypy_method(self):
        if False:
            i = 10
            return i + 15
        code = dedent('\n            import torch\n            class FooModule(torch.jit.ScriptModule):\n                @torch.jit.script_method\n                def foo(self, x, y):\n                    # type: ({input}, Tuple[Tensor, Tensor]) -> Tuple[{output}, {output}]\n                    return x, x\n            instance = FooModule()\n        ')
        test_str = []
        for pair in self.type_input_return_pairs():
            fn = jit_utils._get_py3_code(self.format_code(code, pair), 'instance')
            test_str.append(str(fn.foo.schema))
        self.assertExpectedStripMangled('\n'.join(test_str) + '\n')

    def test_mypy_type_ignore(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            return x

        @torch.jit.script
        def bar(x):
            if False:
                for i in range(10):
                    print('nop')
            return x

    def test_method_casts_script(self):
        if False:
            return 10
        cast_types = ['byte', 'char', 'double', 'float', 'int', 'long', 'short']
        for cast_type in cast_types:
            cu = torch.jit.CompilationUnit(f'\n            def cast_to(x):\n                return x.{cast_type}()\n            ')
            x = torch.rand(3, 4, 5) * 128
            cu_result = cu.cast_to(x)
            reference = getattr(x, cast_type)()
            self.assertEqual(cu_result, reference)

    def test_string_frontend_elif(self):
        if False:
            for i in range(10):
                print('nop')
        code = '\n            def func(niter):\n                # type: (int)\n                rv = 0\n                for i in range(niter):\n                    if i % 3 == 0 and i % 5 == 0:\n                        rv += 35\n                    elif i % 3 == 0:\n                        rv += 3\n                    elif i % 5 == 0:\n                        rv += 5\n                    else:\n                        rv += i\n                return rv\n        '
        self.checkScript(dedent(code), (101,))

    def test_module_parameters_and_buffers(self):
        if False:
            print('Hello World!')
        weights = torch.randn(10, 10)
        bias = torch.randn(10)
        weights2 = torch.randn(10, 10)
        bias2 = torch.randn(10)

        class TestLinear(torch.nn.Module):

            def __init__(self, in_features, out_features):
                if False:
                    return 10
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
                self.bias = torch.nn.Parameter(torch.empty(out_features))
                self.register_buffer('counter', torch.ones(out_features))
                self.reset_parameters()

            def reset_parameters(self):
                if False:
                    return 10
                torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
                if self.bias is not None:
                    (fan_in, _) = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
                    bound = 1 / math.sqrt(fan_in)
                    torch.nn.init.uniform_(self.bias, -bound, bound)

            def forward(self, input):
                if False:
                    return 10
                return F.linear(input, self.weight, self.bias) + self.counter

        class Strong(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.fc1 = TestLinear(10, 10)
                self.fc1.weight = torch.nn.Parameter(weights)
                self.fc1.bias = torch.nn.Parameter(bias)
                self.fc2 = TestLinear(10, 10)
                self.fc2.weight = torch.nn.Parameter(weights2)
                self.fc2.bias = torch.nn.Parameter(bias2)

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x + self.fc1(x) + self.fc1(x) + self.fc2(x)
        strong_mod = Strong()
        inp = torch.ones(10)
        lin = torch.nn.Linear(10, 10)
        lin.weight = torch.nn.Parameter(weights)
        lin.bias = torch.nn.Parameter(bias)
        lin2 = torch.nn.Linear(10, 10)
        lin2.weight = torch.nn.Parameter(weights2)
        lin2.bias = torch.nn.Parameter(bias2)
        expected_result = inp + (lin(inp) + torch.ones(10)) * 2 + lin2(inp) + torch.ones(10)
        self.assertEqual(strong_mod(inp), expected_result)
        self.assertExportImportModule(strong_mod, (inp,))

    def test_module_copying(self):
        if False:
            i = 10
            return i + 15

        class Submodule(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return x + 100

        class Weak(torch.nn.Module):

            def __init__(self, in_features, out_features):
                if False:
                    print('Hello World!')
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(out_features, in_features))
                self.bias = torch.nn.Parameter(torch.ones(out_features))
                self.register_buffer('buffer', torch.ones(out_features))
                self.submodule = Submodule()

            def forward(self, x):
                if False:
                    print('Hello World!')
                return F.linear(x, self.weight, self.bias) + self.buffer + self.submodule(x)

        class Strong(torch.jit.ScriptModule):

            def __init__(self, weak):
                if False:
                    return 10
                super().__init__()
                self.weak = weak

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.weak(x)
        inp = torch.ones(5, 5) * 5
        weak_mod = Weak(5, 5)
        strong_mod = Strong(weak_mod)
        self.assertTrue(isinstance(strong_mod.weak, torch.jit.ScriptModule))
        self.assertFalse(isinstance(weak_mod, torch.jit.ScriptModule))
        self.assertIs(strong_mod.weak.weight, weak_mod.weight)
        self.assertIs(strong_mod.weak.buffer, weak_mod.buffer)
        self.assertIsNot(strong_mod.weak.submodule, weak_mod.submodule)
        weak_mod.weight.data += torch.ones(5, 5) * 100
        self.assertTrue(strong_mod(inp).allclose(weak_mod(inp)))
        weak_mod.weight = torch.nn.Parameter(torch.ones(5, 5) * 100)
        self.assertFalse(strong_mod(inp).allclose(weak_mod(inp)))

    def test_backend_cudnn_enabled(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def fn(x):
            if False:
                i = 10
                return i + 15
            if torch.backends.cudnn.enabled:
                x = x + 2
            else:
                x = x + 3
            return x

    def test_inplace_add(self):
        if False:
            i = 10
            return i + 15

        def foo(a, b):
            if False:
                return 10
            c = a + b
            c.add_(b)
            return c
        self.checkScript(foo, (torch.rand(3), torch.rand(3)))

    def test_add_out(self):
        if False:
            print('Hello World!')

        def foo(a, b):
            if False:
                print('Hello World!')
            c = a + b
            e = 2 * a
            torch.add(c, b, out=e)
            return e
        self.checkScript(foo, (torch.rand(3), torch.rand(3)))

    def test_tuple_error_msg(self):
        if False:
            print('Hello World!')

        def fn(t: Any):
            if False:
                print('Hello World!')
            if isinstance(t, tuple):
                (a, b) = t
            return a + b
        with self.assertRaisesRegexWithHighlight(RuntimeError, 'Provided tuple is not fully defined/refined', 't'):
            s = torch.jit.script(fn)

    def test_augmented_assign(self):
        if False:
            i = 10
            return i + 15

        def foo(a, b):
            if False:
                while True:
                    i = 10
            a += b
            a -= b
            a /= b
            a *= b
            return (a, b)
        self.checkScript(foo, (torch.rand(3), torch.rand(3)))

    def test_ignored_props(self):
        if False:
            while True:
                i = 10

        class A(nn.Module):
            __jit_ignored_attributes__ = ['ignored', 'ignored_return_val']

            @property
            def ignored(self):
                if False:
                    for i in range(10):
                        print('nop')
                raise ValueError("shouldn't be called")

            @property
            def ignored_return_val(self):
                if False:
                    return 10
                return 1

            @torch.jit.ignore
            def call(self):
                if False:
                    while True:
                        i = 10
                return self.ignored_return_val
        f = torch.jit.script(A())
        self.assertTrue(isinstance(f, torch.jit.ScriptModule))
        self.assertTrue(isinstance(f.call(), property))

    def test_pass(self):
        if False:
            print('Hello World!')

        def foo(x):
            if False:
                while True:
                    i = 10
            for _i in range(3):
                pass
            if x:
                pass
            else:
                pass
            return 3
        self.checkScript(foo, (True,))

    def test_lhs_indexing(self):
        if False:
            while True:
                i = 10

        def foo(a, b):
            if False:
                for i in range(10):
                    print('nop')
            a = a.clone()
            a[0] = b
            return a
        self.checkScript(foo, (torch.rand(2, 3), torch.rand(3)))

    def test_lhs_advanced_indexing_assignment(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(x, y):
            if False:
                for i in range(10):
                    print('nop')
            a = torch.exp(x)
            b = x == 1
            a[b] = y[b]
            return a
        self.checkScript(foo, (torch.ones(4, 3), torch.ones(4, 3)))

    def test_lhs_advanced_indexing_augmented_assignment(self):
        if False:
            i = 10
            return i + 15

        def foo(x, y):
            if False:
                while True:
                    i = 10
            a = torch.exp(x)
            b = x == 1
            a[b] += y[b]
            return a
        self.checkScript(foo, (torch.ones(4, 3), torch.ones(4, 3)))

    def test_lhs_indexing_list(self):
        if False:
            print('Hello World!')

        def foo(a, b):
            if False:
                while True:
                    i = 10
            ls = [a]
            ls[0] = b
            return ls
        self.checkScript(foo, (torch.rand(2, 3), torch.rand(3)))

    def test_inplace_copy_script(self):
        if False:
            return 10

        def foo(x):
            if False:
                i = 10
                return i + 15
            a = torch.rand(3, 4)
            a.copy_(x)
            return a
        self.checkScript(foo, (torch.rand(3, 4),))

    def test_lhs_indexing_increment(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(a, b):
            if False:
                print('Hello World!')
            a[0] += b
            return a
        self.checkScript(foo, (torch.rand(2, 3), torch.rand(3)))

    def test_lhs_indexing_increment_list(self):
        if False:
            print('Hello World!')

        def foo(a, b):
            if False:
                while True:
                    i = 10
            a = a.clone()
            ls = [a, b]
            ls[0] += b
            return ls
        self.checkScript(foo, (torch.rand(2, 3), torch.rand(3)))

    def test_lhs_indexing_increment_list_prim(self):
        if False:
            i = 10
            return i + 15

        def foo():
            if False:
                for i in range(10):
                    print('nop')
            ls = [1, 2, 3]
            ls[0] += 5
            return ls
        self.checkScript(foo, ())

    def test_lhs_indexing_multi(self):
        if False:
            return 10

        def foo(a, b):
            if False:
                while True:
                    i = 10
            a = a.clone()
            (foo, a[0], bar) = (1, b, 3)
            return (foo, a, bar)
        self.checkScript(foo, (torch.rand(2, 3), torch.rand(3)))

    def test_bool_dispatch(self):
        if False:
            i = 10
            return i + 15
        with torch._jit_internal._disable_emit_hooks():

            def kwarg_false(x):
                if False:
                    print('Hello World!')
                return F.max_pool1d(x, 1, 1, return_indices=False)
            self.checkScript(kwarg_false, (torch.randn(3, 3, 3),))

            def kwarg_true(x):
                if False:
                    while True:
                        i = 10
                return F.max_pool1d(x, 1, 1, return_indices=True)
            self.checkScript(kwarg_true, (torch.randn(3, 3, 3),))

            def full_kwarg_false(x):
                if False:
                    for i in range(10):
                        print('nop')
                return F.max_pool1d(x, 1, 1, ceil_mode=False, return_indices=False)
            self.checkScript(full_kwarg_false, (torch.randn(3, 3, 3),))

            def full_kwarg_true(x):
                if False:
                    return 10
                return F.max_pool1d(x, 1, 1, ceil_mode=False, return_indices=True)
            self.checkScript(full_kwarg_true, (torch.randn(3, 3, 3),))

            def use_default(x):
                if False:
                    i = 10
                    return i + 15
                return F.max_pool1d(x, 1, 1)
            self.checkScript(use_default, (torch.randn(3, 3, 3),))

            def arg_false(x):
                if False:
                    i = 10
                    return i + 15
                return F.max_pool1d(x, 1, 1, 0, 1, False, False)
            self.checkScript(arg_false, (torch.randn(3, 3, 3),))

            def arg_true(x):
                if False:
                    for i in range(10):
                        print('nop')
                return F.max_pool1d(x, 1, 1, 0, 1, False, True)
            self.checkScript(arg_true, (torch.randn(3, 3, 3),))

    def test_infer_size(self):
        if False:
            print('Hello World!')
        from torch._C import _infer_size

        def fn(x, y):
            if False:
                i = 10
                return i + 15
            return _infer_size(x.size(), y.size())
        self.checkScript(fn, (torch.ones(2, 4, 2), torch.ones(2, 4, 2)))

    def test_hash(self):
        if False:
            for i in range(10):
                print('nop')

        def tester(fn, inputs):
            if False:
                i = 10
                return i + 15
            for x in inputs:
                for y in inputs:
                    if x == y:
                        self.assertEqual(fn(x), fn(y))
                    else:
                        self.assertNotEqual(fn(x), fn(y))

        @torch.jit.script
        def int_hash(x):
            if False:
                print('Hello World!')
            return hash(x)

        @torch.jit.script
        def float_hash(x):
            if False:
                print('Hello World!')
            return hash(x)

        @torch.jit.script
        def str_hash(x):
            if False:
                for i in range(10):
                    print('nop')
            return hash(x)
        tester(int_hash, (20, 21, 22))
        tester(float_hash, (20.0, 21.00001, 22.443))
        tester(str_hash, ('', 'hello', 'a'))

    def test_id(self):
        if False:
            return 10
        with self.assertRaisesRegex(RuntimeError, 'Expected a value'):

            @torch.jit.script
            def test_id_scalars():
                if False:
                    while True:
                        i = 10
                return id(2) == id(None)

        @torch.jit.script
        class FooTest:

            def __init__(self, x):
                if False:
                    print('Hello World!')
                self.foo = x

            def getFooTest(self):
                if False:
                    print('Hello World!')
                return self.foo

        @torch.jit.script
        def test_id_class_types():
            if False:
                print('Hello World!')
            obj1 = FooTest(torch.tensor(3))
            obj2 = FooTest(torch.tensor(2))
            assert obj1 is not obj2
            assert id(obj1) != id(obj2)
            assert id(obj1) != id(None)
            return True
        self.assertTrue(test_id_class_types())

    def test_mutable_dce(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def foo():
            if False:
                return 10
            a = torch.rand(2, 3)
            a += torch.rand(2, 3)
            b = torch.rand(2, 3)
            b += torch.rand(2, 3)
            return a
        FileCheck().check_count('aten::rand', 2, exactly=True).check_count('aten::add', 1, exactly=True).run(str(foo.graph))

    def test_mutable_dce_block(self):
        if False:
            return 10

        @torch.jit.script
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            a = torch.rand(2, 3)
            a += torch.rand(2, 3)
            b = torch.rand(2, 3)
            if bool(a > torch.zeros(2, 3)):
                b += torch.rand(2, 3)
                a += torch.rand(2, 3)
            return b
        FileCheck().check('prim::If').check_count('aten::rand', 1, exactly=True).run(str(foo.graph))

    def test_mutable_dce_graph_input(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def foo(a):
            if False:
                i = 10
                return i + 15
            a += torch.rand(2, 3)
        FileCheck().check('aten::rand').check('aten::add').run(str(foo.graph))

    def test_mutable_dce_list(self):
        if False:
            return 10

        @torch.jit.script
        def foo(a):
            if False:
                for i in range(10):
                    print('nop')
            l = []
            l.append(a)
            c = l[0]
            b = torch.rand(2, 3)
            c += torch.rand(2, 3)
            return b
        FileCheck().check_count('aten::rand', 2, exactly=True).run(str(foo.graph))

    def test_mutable_dce_loop(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def foo(a):
            if False:
                print('Hello World!')
            l = []
            l.append(a)
            i = 0
            b = torch.rand(2, 3)
            while i < 1:
                dead = torch.rand(2, 3)
                c = l[0]
                c += torch.rand(2, 3)
                i += 1
            return b
        FileCheck().check('prim::Loop').check_not('aten::rand').check('aten::__getitem__').check_count('aten::rand', 1, exactly=True).run(str(foo.graph))

    def test_mutable_dce_indirect_wildcards(self):
        if False:
            i = 10
            return i + 15

        def fn():
            if False:
                for i in range(10):
                    print('nop')
            x = torch.ones(2, 3)
            x_1 = x.view(-1)
            l = []
            l.append(x_1)
            x_view = l[0]
            x.add_(torch.ones(2, 3))
            return x_view
        self.checkScript(fn, ())

    def test_mutable_dce_indirect_wildcard_write(self):
        if False:
            for i in range(10):
                print('nop')

        def fn():
            if False:
                print('Hello World!')
            indexes = torch.jit.annotate(List[Tensor], [])
            word_ids = torch.zeros(10, dtype=torch.int32)
            word_ids[1] = 1
            indexes.append(word_ids)
            return word_ids
        self.checkScript(fn, ())

    def test_mutable_dce_wildcards(self):
        if False:
            print('Hello World!')

        def fn():
            if False:
                for i in range(10):
                    print('nop')
            x = torch.ones(2, 3)
            l = []
            l.append(x)
            x_view = l[0]
            x.add_(torch.ones(2, 3))
            return x_view
        self.checkScript(fn, (), profiling=ProfilingMode.SIMPLE)

    def test_cpp_function_tensor_str(self):
        if False:
            print('Hello World!')
        x = torch.randn(2, 2)
        scale = torch.randn(2, 2, requires_grad=True)
        shift = torch.randn(2, 2, requires_grad=True)

        @torch.jit.script
        def fn(x, scale, shift):
            if False:
                i = 10
                return i + 15
            return scale * x + shift
        with self.capture_stdout() as captured:
            print(fn(x, scale, shift))

    def test_string_index(self):
        if False:
            return 10

        def fn(x):
            if False:
                print('Hello World!')
            return (x[2], x[-1])
        self.checkScript(fn, ('abcde',))

    def test_ord(self):
        if False:
            i = 10
            return i + 15

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return ord(x)
        self.checkScript(fn, 'h')
        self.checkScript(fn, 'y')

        def index_str_to_tensor(s):
            if False:
                i = 10
                return i + 15
            return torch.tensor(ord(s))
        s = '£'.encode()[:1]
        self.checkScript(index_str_to_tensor, (s,))

    def test_chr(self):
        if False:
            i = 10
            return i + 15

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return chr(x)
        self.checkScript(fn, (1,))
        self.checkScript(fn, (97,))

    def test_round(self):
        if False:
            print('Hello World!')

        def round_float(x):
            if False:
                while True:
                    i = 10
            return round(x)

        def round_int(x):
            if False:
                for i in range(10):
                    print('nop')
            return round(x)
        self.checkScript(round_float, (1.5,))
        self.checkScript(round_int, (2,))

    def test_convert_base(self):
        if False:
            for i in range(10):
                print('nop')

        def test_hex(x):
            if False:
                for i in range(10):
                    print('nop')
            return hex(x)

        def test_oct(x):
            if False:
                print('Hello World!')
            return oct(x)

        def test_bin(x):
            if False:
                for i in range(10):
                    print('nop')
            return bin(x)
        numbers = [-1000, -10, 0, 1, 10, 2343]
        for n in numbers:
            self.checkScript(test_bin, (n,))
            self.checkScript(test_oct, (n,))
            self.checkScript(test_hex, (n,))

    @unittest.skipIf(IS_WINDOWS or IS_SANDCASTLE, 'NYI: TemporaryFileName support for Windows or Sandcastle')
    def test_get_set_state(self):
        if False:
            i = 10
            return i + 15

        class Root(torch.jit.ScriptModule):
            __constants__ = ['number']

            def __init__(self, number):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.register_buffer('buffer1', torch.ones(2, 2))
                self.register_buffer('buffer2', torch.ones(2, 2))
                self.number = number

            @torch.jit.script_method
            def __getstate__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return (self.buffer1, self.buffer2, 74, self.training)

            @torch.jit.script_method
            def __setstate__(self, state):
                if False:
                    for i in range(10):
                        print('nop')
                self.buffer1 = state[0] + 10
                self.buffer2 = state[1] + 10
                self.training = state[3]

        class M(torch.jit.ScriptModule):
            __constants__ = ['number']

            def __init__(self, number, submodule):
                if False:
                    print('Hello World!')
                super().__init__()
                self.register_buffer('buffer1', torch.ones(2, 2))
                self.register_buffer('buffer2', torch.ones(2, 2))
                self.number = number
                self.submodule = submodule

            @torch.jit.script_method
            def __getstate__(self):
                if False:
                    i = 10
                    return i + 15
                return (self.buffer1, self.buffer2, 74, self.submodule, self.training)

            @torch.jit.script_method
            def __setstate__(self, state):
                if False:
                    for i in range(10):
                        print('nop')
                self.buffer1 = state[0] + 10
                self.buffer2 = state[1] + 10
                self.submodule = state[3]
                self.training = state[4]
        with TemporaryFileName() as fname:
            m = M(23, submodule=Root(99))
            m.save(fname)
            loaded = torch.jit.load(fname)
        self.assertEqual(m.buffer1, torch.ones(2, 2))
        self.assertEqual(m.buffer2, torch.ones(2, 2))
        self.assertEqual(loaded.buffer1, torch.ones(2, 2) + 10)
        self.assertEqual(loaded.buffer2, torch.ones(2, 2) + 10)
        self.assertEqual(loaded.submodule.buffer1, torch.ones(2, 2) + 10)
        self.assertEqual(loaded.submodule.buffer2, torch.ones(2, 2) + 10)

        class NoArgState(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.register_buffer('buffer1', torch.ones(2, 2))
                self.register_buffer('buffer2', torch.ones(2, 2))

            def forward(self):
                if False:
                    while True:
                        i = 10
                pass

            @torch.jit.export
            def __getstate__(self):
                if False:
                    print('Hello World!')
                return (5, self.training)

            @torch.jit.export
            def __setstate__(self, state):
                if False:
                    i = 10
                    return i + 15
                self.buffer1 = torch.ones(2, 2) + state[0]
                self.buffer2 = torch.ones(2, 2) + 10
                self.training = state[1]
        with TemporaryFileName() as fname:
            m = torch.jit.script(NoArgState())
            m.save(fname)
            loaded = torch.jit.load(fname)
            self.assertEqual(loaded.buffer1, torch.ones(2, 2) + 5)
            self.assertEqual(loaded.buffer2, torch.ones(2, 2) + 10)

    def test_string_slicing(self):
        if False:
            print('Hello World!')

        def fn1(x):
            if False:
                i = 10
                return i + 15
            return x[1:3]

        def fn2(x):
            if False:
                print('Hello World!')
            return x[-1:3]

        def fn3(x):
            if False:
                for i in range(10):
                    print('nop')
            return x[3:1]

        def fn4(x):
            if False:
                return 10
            return x[3:100]
        self.checkScript(fn1, ('abcdefghi',))
        self.checkScript(fn2, ('abcdefghi',))
        self.checkScript(fn3, ('abcdefghi',))
        self.checkScript(fn4, ('abcdefghi',))

    def test_early_return_closure(self):
        if False:
            return 10
        code = dedent('\n            def tanh(self):\n                output = torch.tanh(self)\n                def backward(grad_output):\n                    pass\n                return output, backward\n        ')
        cu = torch.jit.CompilationUnit(code)
        g = cu.tanh.graph
        FileCheck().check_count('prim::Closure_0', 2).check('NoneType = prim::Constant').check_next('return').run(g)
        code = dedent('\n            def tanh(self):\n                output = torch.tanh(self)\n                def backward(grad_output):\n                    a = 1\n                    if output:\n                        return 1\n                    else:\n                        a = 2\n                    return a\n                return output, backward\n        ')
        cu = torch.jit.CompilationUnit(code)
        g = cu.tanh.graph
        FileCheck().check_count('prim::Closure_0', 2).check('int = prim::If').run(g)
        code = dedent('\n            def loop_in_closure(self):\n                output = torch.tanh(self)\n                def backward(grad_output):\n                    for i in range(3):\n                        return 1\n                    return 4\n                return output, backward\n        ')
        cu = torch.jit.CompilationUnit(code)
        fc = FileCheck()
        fc.check('prim::Closure').check('(Tensor, NoneType) = prim::TupleConstruct')
        fc.check('prim::Closure').check('prim::Loop').check_count('prim::If', 2)
        fc.run(cu.loop_in_closure.graph)
        code = dedent('\n            def tanh(self):\n                output = torch.tanh(self)\n                def backward(grad_output):\n                    if 1 == 1:\n                        return 1\n                    else:\n                        return 1.\n                return output, backward\n        ')
        with self.assertRaisesRegex(RuntimeError, 'returned a value of type int but'):
            cu = torch.jit.CompilationUnit(code)

    @_inline_everything
    def test_early_return_fork_join(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def foo(x):
            if False:
                return 10
            if x.dim() == 2:
                return (torch.neg(x), x)
            else:
                return (torch.neg(x), x + 1)
        x = torch.rand(3, 4)

        @torch.jit.script
        def wait_script(x):
            if False:
                i = 10
                return i + 15
            fut = torch.jit._fork(foo, x)
            y_hat = foo(x)
            y = torch.jit._wait(fut)
            return (y, y_hat)
        FileCheck().check('with prim::fork').check('prim::If').check('return').run(wait_script.graph)

    def test_early_return_type_refinement(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def test(x):
            if False:
                i = 10
                return i + 15
            if x is None:
                return 1
            else:
                return x
        self.assertEqual(test(None), 1)
        self.assertEqual(test(2), 2)

    def test_exceptions_with_control_flow(self):
        if False:
            i = 10
            return i + 15

        def test_num_ifs(func, num_ifs):
            if False:
                for i in range(10):
                    print('nop')
            g = torch.jit.script(func).graph
            FileCheck().check_count('prim::If', num_ifs, exactly=True).run(g)

        def no_guard_ifs_added(x):
            if False:
                return 10
            if x == 1:
                return 1
            elif x == 2:
                raise RuntimeError('hi')
            else:
                raise RuntimeError('hi')
        self.checkScript(no_guard_ifs_added, (1,))
        self.checkScriptRaisesRegex(no_guard_ifs_added, (2,), Exception, '')
        test_num_ifs(no_guard_ifs_added, 2)

        def no_ifs_added(x):
            if False:
                return 10
            if x < 0:
                raise RuntimeError('hi')
            return x
        self.checkScript(no_ifs_added, (1,))
        self.checkScriptRaisesRegex(no_ifs_added, (-2,), Exception, '')
        test_num_ifs(no_ifs_added, 1)

        def test_if_might(x):
            if False:
                print('Hello World!')
            if x > 0:
                if x == 1:
                    return 1
                else:
                    a = 2
            else:
                raise RuntimeError('hi')
            return a + 2
        self.checkScript(test_if_might, (1,))
        self.checkScript(test_if_might, (3,))
        self.checkScriptRaisesRegex(no_ifs_added, (-2,), Exception, '')
        test_num_ifs(test_if_might, 3)

        def test_loop_no_escape(x):
            if False:
                i = 10
                return i + 15
            if x >= 0:
                for i in range(x):
                    raise RuntimeError('hi')
            else:
                return 5
            return x + 3
        self.checkScript(test_loop_no_escape, (0,))
        self.checkScript(test_loop_no_escape, (-1,))
        self.checkScriptRaisesRegex(test_loop_no_escape, (1,), Exception, '')
        test_num_ifs(test_loop_no_escape, 1)

        def test_loop_exception_with_continue(x):
            if False:
                while True:
                    i = 10
            i = 0
            for i in range(5):
                if i == x:
                    raise RuntimeError('hi')
                else:
                    continue
                print(i)
            return i + 5
        self.checkScript(test_loop_exception_with_continue, (-1,))
        self.checkScriptRaisesRegex(test_loop_exception_with_continue, (1,), Exception, '')
        test_num_ifs(test_loop_exception_with_continue, 1)

    def test_exception_exits_closure(self):
        if False:
            i = 10
            return i + 15
        code = dedent('\n            def no_return_func(self):\n                # type: (Tensor) -> Tensor\n                output = torch.tanh(self)\n                def backward(grad_output):\n                    raise RuntimeError("Hi")\n        ')
        with self.assertRaisesRegex(RuntimeError, 'does not return along all'):
            cu = torch.jit.CompilationUnit(code)
        code = dedent('\n            def test_exit_pair_reset(x):\n                # type: (int) -> int\n                if x > 0:\n                    a = 0\n                    def backward(grad_output):\n                        raise RuntimeError("Hi")\n                    a = a + 1\n                else:\n                    return x\n                return a + 1\n        ')
        func = torch.jit.CompilationUnit(code).test_exit_pair_reset
        self.assertEqual(func(1), 2)
        self.assertEqual(func(-1), -1)
        FileCheck().check_count('prim::If', 1, exactly=True).run(func.graph)

    def test_non_final_return(self):
        if False:
            print('Hello World!')

        def simple(x):
            if False:
                return 10
            if bool(x > 3):
                return x + 1
            else:
                return x + 2
            raise RuntimeError('nope')

        def nest(x):
            if False:
                print('Hello World!')
            x = x + 1
            if bool(x > 3):
                if bool(x > 4):
                    x += 1
                return x + 1
            else:
                return x + 2

        def early_ret(x):
            if False:
                print('Hello World!')
            x = x + 1
            if bool(x > 3):
                return x + 1
            x = x + 1
            return x + 2

        def nest_early_ret(x):
            if False:
                print('Hello World!')
            x = x + 1
            if bool(x > 3):
                if bool(x > 4):
                    return x + 2
                return x + 1
            x = x + 1
            return x + 2

        def not_early_ret(x):
            if False:
                for i in range(10):
                    print('nop')
            s = ''
            if bool(x > 3):
                if bool(x > 4):
                    return (1, s)
                s += 'foo'
            else:
                s += '5'
            s += 'hi'
            return (7, s)

        def not_total_ret(x):
            if False:
                while True:
                    i = 10
            s = ''
            if bool(x > 3):
                if bool(x > 4):
                    return (1, s)
                else:
                    return (2, s)
            else:
                s += '5'
            return (7, s)
        for i in range(3):
            for func in [simple, nest, early_ret, nest_early_ret, not_early_ret, not_total_ret]:
                self.checkScript(func, (torch.tensor(2.5 + i),))

        def vars_used_after_ret(x):
            if False:
                for i in range(10):
                    print('nop')
            if x == 0:
                return x
            else:
                y = 2
                z = 3
            return x + y * z
        self.checkScript(vars_used_after_ret, (1,))
        self.checkScript(vars_used_after_ret, (0,))

        def complicated(x):
            if False:
                while True:
                    i = 10
            if x:
                if x == 2:
                    return 1
                    assert 1 == 2
                elif x == 3:
                    return 2
                    assert 1 == 2
                else:
                    a = 2
                    b = 3
            else:
                a = 4
                b = 1
            return a + b
            assert 1 == 2
        for i in range(4):
            self.checkScript(complicated, (i,))

    def test_partial_returns(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(RuntimeError, 'does not return along all'):

            @torch.jit.script
            def no_ret():
                if False:
                    print('Hello World!')
                pass
        with self.assertRaisesRegex(RuntimeError, 'does not return along all'):

            @torch.jit.script
            def partial(x):
                if False:
                    i = 10
                    return i + 15
                if x:
                    return 1
        with self.assertRaisesRegex(RuntimeError, 'does not return along all'):

            @torch.jit.script
            def typed_none():
                if False:
                    for i in range(10):
                        print('nop')
                pass

        @torch.jit.script
        def none_ret():
            if False:
                while True:
                    i = 10
            pass
        self.assertIs(none_ret(), None)
        FileCheck().check(': None').run(none_ret.graph)

    def test_early_returns_loops(self):
        if False:
            return 10

        def nest_while_ret(x):
            if False:
                while True:
                    i = 10
            y = 4
            while x < 4:
                if x < 3:
                    return y
                else:
                    y = y + 1
                    break
                y = y + 2
            y = y + 1
            return y
        self.checkScript(nest_while_ret, (2,))
        self.checkScript(nest_while_ret, (3,))
        self.checkScript(nest_while_ret, (4,))

        def loop_ret(x, y):
            if False:
                print('Hello World!')
            i = 0
            for i in range(x):
                if x == y:
                    return x + y
                i = i + y
            i = i - 1
            return i
        self.checkScript(loop_ret, (3, 3))
        self.checkScript(loop_ret, (2, 3))
        self.checkScript(loop_ret, (3, 1))

        def test_will_ret(y):
            if False:
                return 10
            for i in range(y):
                return 2
            return 1
        self.checkScript(test_will_ret, (0,))
        self.checkScript(test_will_ret, (1,))

        def test_loop_nest_ret(y):
            if False:
                print('Hello World!')
            for i in range(y):
                for i in range(y - 2):
                    return 10
                return 5
            return 0
        self.checkScript(test_loop_nest_ret, (0,))
        self.checkScript(test_loop_nest_ret, (1,))
        self.checkScript(test_loop_nest_ret, (2,))

    def test_nn_init(self):
        if False:
            while True:
                i = 10
        tests = (('constant_', lambda : (torch.ones(2, 2), 2.5), 'Tensor, float'), ('ones_', lambda : (torch.ones(2, 2),), 'Tensor'), ('zeros_', lambda : (torch.ones(2, 2),), 'Tensor'), ('uniform_', lambda : (torch.ones(2, 2),), 'Tensor'), ('normal_', lambda : (torch.ones(2, 2),), 'Tensor'), ('xavier_normal_', lambda : (torch.ones(2, 2),), 'Tensor'), ('xavier_uniform_', lambda : (torch.ones(2, 2),), 'Tensor'))
        for (name, args_fn, type_str) in tests:
            arg_str = ', '.join([chr(i + ord('a')) for i in range(len(args_fn()))])
            code = dedent('\n                def test({arg_str}):\n                    # type: ({type_str})\n                    return torch.nn.init.{name}({arg_str})\n            ').format(arg_str=arg_str, type_str=type_str, name=name)
            cu = torch.jit.CompilationUnit(code)
            init_fn = getattr(torch.nn.init, name)
            script_out = self.runAndSaveRNG(cu.test, args_fn())
            eager_out = self.runAndSaveRNG(init_fn, args_fn())
            self.assertEqual(script_out, eager_out)
            FileCheck().check_not('prim::PythonOp').run(cu.test.graph)

    def test_early_return_rewrite(self):
        if False:
            return 10

        def test_foo(x: bool):
            if False:
                for i in range(10):
                    print('nop')
            if x:
                return 1
            return 2
        self.checkScript(test_foo, (True,))
        self.checkScript(test_foo, (False,))
        FileCheck().check_count('prim::If', 1, exactly=True).run(torch.jit.script(test_foo).graph)

        def test_multiple(x: int):
            if False:
                print('Hello World!')
            if x == 5:
                return x * x
            else:
                y = 2 * x
            z = y * 2
            if z == 8:
                return 1
            if z != 16:
                z = z - 2
                abc = 4
            else:
                return 3
            z = z * abc
            return z * z * z
        self.checkScript(test_multiple, (5,))
        self.checkScript(test_multiple, (2,))
        self.checkScript(test_multiple, (4,))
        self.checkScript(test_multiple, (3,))
        self.checkScript(test_multiple, (10,))
        graph = torch.jit.script(test_multiple).graph
        FileCheck().check_count('prim::If', 3, exactly=True).run(graph)

    def test_is_scripting_metacompile(self):
        if False:
            return 10

        @torch.jit.script
        def foo():
            if False:
                return 10
            if torch.jit.is_scripting():
                return 1
            else:
                print('hello') + 2
        self.assertEqual(foo(), 1)

    def test_boolean_literal_constant_metacompile(self):
        if False:
            while True:
                i = 10

        class Mod(torch.nn.Module):
            __constants__ = ['val']

            def __init__(self, val):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.val = val

            def forward(self):
                if False:
                    while True:
                        i = 10
                if self.val:
                    return 1
                else:
                    return '2'
        self.checkModule(Mod(True), ())
        self.checkModule(Mod(False), ())

        @torch.jit.script
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            if True:
                return 1
            else:
                return '2'
        self.assertEqual(foo(), 1)

    def test_assert_is_scripting_metacompile(self):
        if False:
            while True:
                i = 10

        def foo():
            if False:
                for i in range(10):
                    print('nop')
            assert not torch.jit.is_scripting(), 'TestErrorMsg'
            print('hello') + 2
        f = torch.jit.script(foo)
        with self.assertRaisesRegex(torch.jit.Error, 'TestErrorMsg'):
            f()

    def test_isinstance_metacompile(self):
        if False:
            return 10

        @torch.jit.script
        def test_primitive_type(x):
            if False:
                i = 10
                return i + 15
            if isinstance(x, int):
                return x + 1
            else:
                return x - 1
        self.assertEqual(test_primitive_type(1), 2)
        with self.assertRaisesRegex(Exception, 'Expected a value of type'):
            test_primitive_type(1.5)
        _MyNamedTuple = namedtuple('_MyNamedTuple', ['value'])

        @torch.jit.script
        def test_non_primitive_types(x):
            if False:
                while True:
                    i = 10
            if isinstance(1, _MyNamedTuple):
                return 10
            if isinstance(x, _MyNamedTuple):
                return x.value + 1
            else:
                return 1
        out = test_non_primitive_types(_MyNamedTuple(value=torch.tensor(5.0)))
        self.assertEqual(out, torch.tensor(6.0))

    def test_namedtuple_type_inference(self):
        if False:
            while True:
                i = 10
        _AnnotatedNamedTuple = NamedTuple('_NamedTupleAnnotated', [('value', int)])
        _UnannotatedNamedTuple = namedtuple('_NamedTupleUnAnnotated', ['value'])

        def test_check_named_tuple_value():
            if False:
                while True:
                    i = 10
            named_tuple = _AnnotatedNamedTuple(1)
            return named_tuple.value
        self.checkScript(test_check_named_tuple_value, ())

        def test_error():
            if False:
                i = 10
                return i + 15
            return _UnannotatedNamedTuple(1)
        with self.assertRaisesRegex(RuntimeError, "Expected a value of type \\'Tensor \\(inferred\\)\\' for argument \\'value\\' but instead found type \\'int\\'."):
            torch.jit.script(test_error)

    def test_namedtuple_default_values_simple_type(self):
        if False:
            for i in range(10):
                print('nop')

        class Point(NamedTuple):
            x: Optional[int] = None
            y: int = 2
        make_global(Point)

        class M(torch.nn.Module):

            def forward(self, point: Point):
                if False:
                    print('Hello World!')
                return point
        p = Point(x=3, y=2)
        self.checkModule(M(), (p,))
        self.checkModule(M(), (Point(),))
        m = torch.jit.script(M())
        FileCheck().check('NamedTuple(x : int? = None, y : int = 2))').run(m.graph)

    def test_namedtuple_default_values_missing(self):
        if False:
            print('Hello World!')

        class Point(NamedTuple):
            x: Optional[int]
            y: int
            z: int = 3
        make_global(Point)

        class M(torch.nn.Module):

            def forward(self, point: Point):
                if False:
                    return 10
                return point
        p1 = Point(x=3, y=2)
        p2 = Point(x=3, y=2, z=1)
        self.checkModule(M(), (p1,))
        self.checkModule(M(), (p2,))
        m = torch.jit.script(M())
        FileCheck().check('NamedTuple(x : int?, y : int, z : int = 3))').run(m.graph)

    def test_namedtuple_default_values_container_type(self):
        if False:
            print('Hello World!')

        class Point(NamedTuple):
            x: Optional[List[int]] = None
            y: List[int] = [1, 2, 3]
            z: Optional[Dict[str, int]] = {'a': 1}
        make_global(Point)

        class M(torch.nn.Module):

            def forward(self, point: Point):
                if False:
                    for i in range(10):
                        print('nop')
                return point
        p = Point(x=[4, 5, 6], y=[3, 2, 1], z={'b': 2})
        self.checkModule(M(), (p,))
        self.checkModule(M(), (Point(),))
        m = torch.jit.script(M())
        first_line = 'NamedTuple(x : int[]? = None, y : int[] = [1, 2, 3], z : Dict(str, int)? = {a: 1}))'
        FileCheck().check(first_line).run(m.graph)

    def test_namedtuple_default_values_Tensor_type(self):
        if False:
            return 10

        class Point(NamedTuple):
            x: torch.Tensor = torch.rand(2, 3)
        make_global(Point)

        class M(torch.nn.Module):

            def forward(self, point: Point):
                if False:
                    for i in range(10):
                        print('nop')
                return point
        p = Point(x=torch.rand(2, 3))
        with self.assertRaisesRegex(RuntimeError, 'Tensors are not supported as default NamedTuple fields'):
            m = torch.jit.script(M())
            m(p)

    def test_namedtuple_default_values_using_factory_constructor(self):
        if False:
            for i in range(10):
                print('nop')
        Pair = namedtuple('Pair', ['x', 'y'], defaults=(1, 2))
        make_global(Pair)

        @torch.jit.script
        def fn(x: Pair) -> Pair:
            if False:
                while True:
                    i = 10
            return x
        FileCheck().check('NamedTuple(x : Tensor = 1, y : Tensor = 2))').check_next('return (%x.1)').run(fn.graph)

    def test_isinstance_dynamic(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def foo(a):
            if False:
                for i in range(10):
                    print('nop')
            b = 0
            if isinstance(a, (int, (float,), list, str)):
                b += 1
            if isinstance(a, (int, str)):
                b += 1
            if isinstance(a, List[int]):
                b += 1
            return b
        self.assertEqual(foo([3, 4]), 2)
        self.assertEqual(foo(None), 0)

    def test_function_overloads(self):
        if False:
            print('Hello World!')

        @torch.jit._overload
        def test_simple(x1):
            if False:
                i = 10
                return i + 15
            pass

        @torch.jit._overload
        def test_simple(x1):
            if False:
                print('Hello World!')
            pass

        def test_simple(x1):
            if False:
                while True:
                    i = 10
            return x1

        def invoke_function():
            if False:
                return 10
            return (test_simple(1.0), test_simple(0.5))
        self.checkScript(invoke_function, ())
        compiled_fns_1 = torch.jit._script._get_overloads(test_simple)
        compiled_fns_2 = torch.jit._script._get_overloads(test_simple)
        for (a, b) in zip(compiled_fns_1, compiled_fns_2):
            self.assertIs(a.graph, b.graph)
        old_func = test_simple

        @torch.jit._overload
        def test_simple(x1):
            if False:
                i = 10
                return i + 15
            pass

        @torch.jit.script
        def my_func():
            if False:
                print('Hello World!')
            return old_func('hi')

        @torch.jit._overload
        def test_simple(a, b):
            if False:
                print('Hello World!')
            pass

        def test_simple(a, b):
            if False:
                print('Hello World!')
            return a + b

        @torch.jit.script
        def fn():
            if False:
                return 10
            return test_simple(3, 4)
        self.assertEqual(fn(), 7)

        @torch.jit._overload
        def identity(x1):
            if False:
                return 10
            pass

        @torch.jit._overload
        def identity(x1):
            if False:
                for i in range(10):
                    print('nop')
            pass

        def identity(x1=1.0):
            if False:
                return 10
            return x1

        def invoke():
            if False:
                return 10
            return (identity(), identity(0.5), identity('hi'))
        self.checkScript(invoke, ())

        def schema_match_failure():
            if False:
                for i in range(10):
                    print('nop')
            return identity((1, 2))
        thrown = False
        try:
            torch.jit.script(schema_match_failure)
        except Exception as e:
            thrown = True
            self.assertTrue("of type 'str'" in str(e) and "of type 'float" in str(e))
        self.assertTrue(thrown)
        with self.assertRaisesRegex(Exception, 'cannot be directly compiled'):
            torch.jit.script(identity)

        @torch.jit._overload
        def impl_compile_failure(x, y):
            if False:
                print('Hello World!')
            pass

        @torch.jit._overload
        def impl_compile_failure(x, y):
            if False:
                while True:
                    i = 10
            pass

        def impl_compile_failure(x, y):
            if False:
                return 10
            return x - y

        def test():
            if False:
                return 10
            impl_compile_failure('one', 'two')
        with self.assertRaisesRegex(Exception, 'Arguments for call are not valid'):
            torch.jit.script(test)

        @torch.jit._overload
        def good_overload(x=1):
            if False:
                i = 10
                return i + 15
            pass

        def good_overload(x=1):
            if False:
                i = 10
                return i + 15
            return x

        @torch.jit.script
        def foo():
            if False:
                while True:
                    i = 10
            return good_overload()
        self.assertEqual(foo(), 1)
        with self.assertRaisesRegex(Exception, 'must equal to the default parameter'):

            @torch.jit._overload
            def bad_default_on_overload(x, y=2):
                if False:
                    i = 10
                    return i + 15
                pass

            def bad_default_on_overload(x, y=1):
                if False:
                    i = 10
                    return i + 15
                pass

            @torch.jit.script
            def test():
                if False:
                    for i in range(10):
                        print('nop')
                return bad_default_on_overload(1, 2)

        @torch.jit._overload
        def diff_default(x):
            if False:
                for i in range(10):
                    print('nop')
            pass

        @torch.jit._overload
        def diff_default(x):
            if False:
                print('Hello World!')
            pass

        def diff_default(x='hi'):
            if False:
                i = 10
                return i + 15
            return x

        def test():
            if False:
                i = 10
                return i + 15
            return (diff_default(), diff_default(2), diff_default('abc'))
        self.assertEqual(test(), torch.jit.script(test)())

        @torch.jit._overload
        def diff_num_params(x):
            if False:
                for i in range(10):
                    print('nop')
            pass

        @torch.jit._overload
        def diff_num_params(x, y):
            if False:
                print('Hello World!')
            pass

        def diff_num_params(x, y=2, z=3):
            if False:
                while True:
                    i = 10
            return x + y + z

        def test():
            if False:
                return 10
            return (diff_num_params(1.0), diff_num_params(1, 2), diff_num_params(1), diff_num_params(1, 2, 3))
        self.assertEqual(test(), torch.jit.script(test)())

        @torch.jit._overload
        def diff_num_params_no_annot():
            if False:
                return 10
            pass

        def diff_num_params_no_annot(x=1):
            if False:
                return 10
            return x

        def test():
            if False:
                for i in range(10):
                    print('nop')
            return diff_num_params_no_annot(1.0)
        with self.assertRaisesRegex(Exception, 'Parameters not specified'):
            torch.jit.script(test)

    def test_function_overload_misuse(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(RuntimeError, 'Only `pass` statement or `...` can be the body'):

            @torch.jit._overload
            def wrong_decl_body(x: str) -> str:
                if False:
                    i = 10
                    return i + 15
                return x + '0'
        with self.assertRaisesRegex(RuntimeError, 'Only `pass` statement or `...` can be the body'):

            class MyClass:

                @torch.jit._overload_method
                def method(self):
                    if False:
                        i = 10
                        return i + 15
                    return 0

        @torch.jit._overload
        def null_overload(x: int) -> int:
            if False:
                while True:
                    i = 10
            ...

        @torch.jit._overload
        def null_overload(x: str) -> str:
            if False:
                i = 10
                return i + 15
            pass

        def null_overload_driver():
            if False:
                while True:
                    i = 10
            return null_overload(0)
        with self.assertRaisesRegex(RuntimeError, 'Implementation for the function ".+" is missing.'):
            torch.jit.script(null_overload_driver)

        class OverloadMisuse(torch.nn.Module):

            @torch.jit._overload_method
            def forward(self, x: int):
                if False:
                    return 10
                pass

            @torch.jit._overload_method
            def forward(self, x: Tensor):
                if False:
                    i = 10
                    return i + 15
                pass
        with self.assertRaisesRegex(RuntimeError, 'Implementation for the method ".+" is missing.'):
            m = torch.jit.script(OverloadMisuse())

    def test_script_method_torch_function_overload(self):
        if False:
            print('Hello World!')

        class MyCustomTensor(torch.Tensor):
            pass

        class MyCustomModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.relu(x)
        scripted_mod = torch.jit.script(MyCustomModule())
        t = torch.tensor([3.0])
        ref_out = scripted_mod(t)
        t_custom = MyCustomTensor([3.0])
        out1 = scripted_mod(t_custom)
        self.assertEqual(out1, ref_out)
        out2 = scripted_mod.forward(t_custom)
        self.assertEqual(out2, ref_out)

    def test_function_overloading_isinstance(self):
        if False:
            i = 10
            return i + 15

        @torch.jit._overload
        def my_conv(x, y):
            if False:
                while True:
                    i = 10
            pass

        @torch.jit._overload
        def my_conv(x, y):
            if False:
                return 10
            pass

        def my_conv(x, y=2.0):
            if False:
                print('Hello World!')
            if isinstance(y, str):
                if y == 'hi':
                    return 4.0 - x
                else:
                    return 5.0 - x
            else:
                return 2.0 + x

        def test_uses():
            if False:
                while True:
                    i = 10
            return (my_conv(1.5), my_conv(1.5, 'hi'), my_conv(1.5, 5.0))
        self.checkScript(test_uses, ())

    def test_method_overloading(self):
        if False:
            i = 10
            return i + 15

        class Over(torch.nn.Module):

            @torch.jit._overload_method
            def forward(self, x):
                if False:
                    print('Hello World!')
                pass

            @torch.jit._overload_method
            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                pass

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                if isinstance(x, Tensor):
                    return x + 20
                else:
                    return x[0] + 5

        class S(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.weak = Over()

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    return 10
                return self.weak(x) + self.weak((x, x))
        s_mod = S()
        x = torch.ones(1)
        self.assertEqual(s_mod(x), x + 20 + 5 + x)
        over = Over()
        self.assertEqual(over((x, x)), x + 5)
        self.assertEqual(over(x), x + 20)

        class Unannotated(torch.nn.Module):

            @torch.jit._overload_method
            def hello(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                pass

            @torch.jit._overload_method
            def hello(self, x):
                if False:
                    i = 10
                    return i + 15
                pass

            def hello(self, x):
                if False:
                    i = 10
                    return i + 15
                return x + 3

            def forward(self):
                if False:
                    while True:
                        i = 10
                return (self.hello(1), self.hello(0.5))
        w = Unannotated()
        with self.assertRaisesRegex(Exception, 'explicitly add type annotations to overloaded functions'):
            torch.jit.script(w)

        class CompileOverloadError(torch.nn.Module):

            @torch.jit._overload_method
            def hello(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                pass

            @torch.jit._overload_method
            def hello(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                pass

            def hello(self, x):
                if False:
                    print('Hello World!')
                return x + 1

            def forward(self):
                if False:
                    for i in range(10):
                        print('nop')
                return (self.hello('hi'), self.hello(0.5))
        w = CompileOverloadError()
        with self.assertRaisesRegex(Exception, "but instead found type 'str'"):
            torch.jit.script(w)
        with self.assertRaisesRegex(Exception, 'Overloads are not useable when a module'):

            class W3(torch.nn.Module):

                @torch.jit._overload_method
                def forward(self, x):
                    if False:
                        i = 10
                        return i + 15
                    pass

                @torch.jit._overload_method
                def forward(self, x):
                    if False:
                        return 10
                    pass

                def forward(self, x):
                    if False:
                        return 10
                    return x + 5
            a = W3()
            b = torch.jit.script(a)

            class W3(torch.nn.Module):

                def forward(self, x):
                    if False:
                        print('Hello World!')
                    return x + 5 + 10
            a = W3()
            b = torch.jit.script(a)

        class W2(torch.nn.Module):

            def hello(self, x1, x2):
                if False:
                    for i in range(10):
                        print('nop')
                return x1 + x2

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.hello(x, x)
        a = torch.jit.script(W2())
        self.assertEqual(a(torch.tensor(1)), torch.tensor(2))

        class W2(torch.nn.Module):

            @torch.jit._overload_method
            def hello(self, x):
                if False:
                    i = 10
                    return i + 15
                pass

            @torch.jit._overload_method
            def hello(self, x):
                if False:
                    while True:
                        i = 10
                pass

            def hello(self, x):
                if False:
                    return 10
                return x + 5 + 10

            def forward(self, x):
                if False:
                    print('Hello World!')
                return (self.hello(1), self.hello(x))
        with self.assertRaisesRegex(Exception, 'Overloads are not useable when a module'):
            a = torch.jit.script(W2())

    def test_narrow_copy(self):
        if False:
            print('Hello World!')

        def foo(a):
            if False:
                print('Hello World!')
            return a.narrow_copy(0, 0, 5)
        self.checkScript(foo, [torch.rand(10)])

    def test_select_after_chunk(self):
        if False:
            return 10

        def foo(x):
            if False:
                i = 10
                return i + 15
            chunked = torch.chunk(x, 1)
            foo = chunked[0]
            foo.add_(5)
            return x
        self.checkScript(foo, [torch.rand(2, 3)])

    def test_nn_LSTM_with_layers(self):
        if False:
            while True:
                i = 10

        class M(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.rnn = nn.LSTM(2, 3, 2, dropout=0)

            @torch.jit.script_method
            def forward(self, x, lengths, h0, c0):
                if False:
                    i = 10
                    return i + 15
                return self.rnn(x, (h0, c0))[0]

        class Eager(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.rnn = nn.LSTM(2, 3, 2, dropout=0)

            def forward(self, x, lengths, h0, c0):
                if False:
                    i = 10
                    return i + 15
                return self.rnn(x, (h0, c0))[0]
        inputs = (torch.randn(1, 1, 2), torch.LongTensor([7]), torch.randn(2, 1, 3), torch.randn(2, 1, 3))
        eager_out = self.runAndSaveRNG(lambda : Eager()(*inputs), ())[0]
        script_out = self.runAndSaveRNG(lambda : M()(*inputs), ())[0]
        self.assertEqual(eager_out, script_out)

    def test_nn_LSTM(self):
        if False:
            i = 10
            return i + 15
        input = torch.nn.utils.rnn.pack_sequence([torch.randn(5, 5)])

        class S(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.x = torch.nn.LSTM(5, 5)

            @torch.jit.script_method
            def forward(self, input: PackedSequence) -> Tuple[PackedSequence, Tuple[torch.Tensor, torch.Tensor]]:
                if False:
                    while True:
                        i = 10
                return self.x(input)
        eager_out = self.runAndSaveRNG(lambda x: torch.nn.LSTM(5, 5)(x), (input,))[0]
        script_out = self.runAndSaveRNG(lambda x: S()(x), (input,))[0]
        self.assertEqual(eager_out, script_out)

    def test_nn_GRU(self):
        if False:
            print('Hello World!')
        seq_input = torch.nn.utils.rnn.pack_sequence([torch.randn(5, 5)])
        tensor_input = torch.randn(5, 5, 5)

        class SeqLengthGRU(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.x = torch.nn.GRU(5, 5)

            @torch.jit.script_method
            def forward(self, input: PackedSequence) -> Tuple[PackedSequence, torch.Tensor]:
                if False:
                    print('Hello World!')
                return self.x(input)

        class TensorGRU(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.x = torch.nn.GRU(5, 5)

            @torch.jit.script_method
            def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                if False:
                    i = 10
                    return i + 15
                return self.x(input)
        seq_eager_out = self.runAndSaveRNG(lambda x: torch.nn.GRU(5, 5)(x), (seq_input,))[0]
        seq_script_out = self.runAndSaveRNG(lambda x: SeqLengthGRU()(x), (seq_input,))[0]
        tensor_eager_out = self.runAndSaveRNG(lambda x: torch.nn.GRU(5, 5)(x), (tensor_input,))[0]
        tensor_script_out = self.runAndSaveRNG(lambda x: TensorGRU()(x), (tensor_input,))[0]
        self.assertEqual(seq_eager_out, seq_script_out)
        self.assertEqual(tensor_eager_out, tensor_script_out)

    def test_torchscript_memoryformat(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def fn(x):
            if False:
                return 10
            return x.contiguous(memory_format=torch.channels_last)
        x = torch.randn(4, 3, 6, 6)
        y = fn(x)
        self.assertTrue(y.is_contiguous(memory_format=torch.channels_last))

    def test_torchscript_multi_head_attn(self):
        if False:
            return 10

        @torch.jit.script
        def jit_multihead_attn_forward(query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout, out_proj_weight, out_proj_bias, training=True, key_padding_mask=None, need_weights=True, attn_mask=None):
            if False:
                for i in range(10):
                    print('nop')
            return torch.nn.functional.multi_head_attention_forward(query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout, out_proj_weight, out_proj_bias, training, key_padding_mask, need_weights, attn_mask)
        src_l = 3
        bsz = 5
        embed_size = 8
        nhead = 2
        multi_head_attn = torch.nn.MultiheadAttention(embed_size, nhead)
        query = torch.rand((src_l, bsz, embed_size))
        key = torch.rand((src_l, bsz, embed_size))
        value = torch.rand((src_l, bsz, embed_size))
        mask = (torch.triu(torch.ones(src_l, src_l)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0).to(torch.get_default_dtype())
        jit_out = jit_multihead_attn_forward(query, key, value, embed_size, nhead, multi_head_attn.in_proj_weight, multi_head_attn.in_proj_bias, multi_head_attn.bias_k, multi_head_attn.bias_v, multi_head_attn.add_zero_attn, multi_head_attn.dropout, multi_head_attn.out_proj.weight, multi_head_attn.out_proj.bias, attn_mask=mask)[0]
        py_out = torch.nn.functional.multi_head_attention_forward(query, key, value, embed_size, nhead, multi_head_attn.in_proj_weight, multi_head_attn.in_proj_bias, multi_head_attn.bias_k, multi_head_attn.bias_v, multi_head_attn.add_zero_attn, multi_head_attn.dropout, multi_head_attn.out_proj.weight, multi_head_attn.out_proj.bias, attn_mask=mask)[0]
        self.assertEqual(jit_out, py_out, atol=0.0005, rtol=0.0001)

    def test_torchscript_multi_head_attn_fast_path(self):
        if False:
            print('Hello World!')
        src_l = 3
        bsz = 5
        embed_size = 8
        nhead = 2
        multi_head_attn = torch.nn.MultiheadAttention(embed_size, nhead, batch_first=True)
        multi_head_attn = multi_head_attn.eval()
        query = key = value = torch.rand((bsz, src_l, embed_size))
        with torch.no_grad():
            py_out = multi_head_attn(query, key, value)
            mha = torch.jit.script(multi_head_attn)
            jit_out = mha(query, key, value)
        torch.testing.assert_close(jit_out, py_out)

    @unittest.skipIf(not RUN_CUDA, 'no CUDA')
    def test_scriptmodule_multi_head_attn_cuda(self):
        if False:
            while True:
                i = 10

        class MyModule(torch.jit.ScriptModule):

            def __init__(self, embed_dim, num_heads):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                sample_q = torch.randn(3, 2, embed_dim)
                sample_kv = torch.randn(3, 2, embed_dim)
                attention = nn.MultiheadAttention(embed_dim, num_heads)
                attention.eval()
                self.mod = torch.jit.trace(attention, (sample_q, sample_kv, sample_kv))

            @torch.jit.script_method
            def forward(self, q, k, v):
                if False:
                    print('Hello World!')
                return self.mod(q, k, v)
        embed_dim = 8
        num_heads = 2
        sl = 3
        bs = 2
        model = MyModule(embed_dim, num_heads).cuda()
        q = torch.randn(sl, bs, embed_dim, device='cuda')
        kv = torch.randn(sl, bs, embed_dim, device='cuda')
        jit_out = model(q, kv, kv)[0]
        py_out = torch.nn.functional.multi_head_attention_forward(q, kv, kv, embed_dim, num_heads, model.mod.in_proj_weight, model.mod.in_proj_bias, None, None, None, 0.0, model.mod.out_proj.weight, model.mod.out_proj.bias)[0]
        self.assertEqual(jit_out, py_out, atol=0.0005, rtol=0.0001)

    @unittest.skipIf(not RUN_CUDA, 'no CUDA')
    def test_scriptmodule_transformer_cuda(self):
        if False:
            i = 10
            return i + 15

        class MyModule(torch.jit.ScriptModule):

            def __init__(self, transformer, sample_q, sample_kv):
                if False:
                    while True:
                        i = 10
                super().__init__()
                transformer.eval()
                self.mod = torch.jit.trace(transformer, (sample_q, sample_kv))

            @torch.jit.script_method
            def forward(self, q, k):
                if False:
                    i = 10
                    return i + 15
                return self.mod(q, k)
        d_model = 8
        nhead = 2
        num_encoder_layers = 2
        num_decoder_layers = 2
        dim_feedforward = 16
        bsz = 2
        seq_length = 5
        tgt_length = 3
        with torch.no_grad():
            src = torch.randn(seq_length, bsz, d_model)
            tgt = torch.randn(tgt_length, bsz, d_model)
            transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.0)
            model = MyModule(transformer, tgt, src)
            src = torch.randn(seq_length, bsz, d_model)
            tgt = torch.randn(tgt_length, bsz, d_model)
            jit_out = model(tgt, src)
            py_out = transformer(tgt, src)
        self.assertEqual(jit_out, py_out, atol=0.0005, rtol=0.0001)

    def test_list_python_op(self):
        if False:
            while True:
                i = 10

        def python_list_op(lst):
            if False:
                return 10
            return lst[0]

        def fn(lst):
            if False:
                i = 10
                return i + 15
            return python_list_op(lst)
        self.checkScript(fn, ([torch.ones(2) + 2, torch.ones(2)],))

    @unittest.skipIf(not RUN_CUDA, 'no CUDA')
    def test_weak_cuda(self):
        if False:
            while True:
                i = 10

        class M(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.lstm = torch.nn.LSTM(5, 5)
                self.lstm.cuda()

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.lstm(x)
        m = M()
        m.cuda()
        out = m(torch.ones(5, 5, 5).cuda())
        self.assertTrue(out[0].is_cuda)

    def test_ignore_decorator(self):
        if False:
            i = 10
            return i + 15
        with warnings.catch_warnings(record=True) as warns:

            class M(torch.jit.ScriptModule):

                def __init__(self):
                    if False:
                        return 10
                    super().__init__()
                    tensor = torch.zeros(1, requires_grad=False)
                    self.register_buffer('some_state', torch.nn.Parameter(tensor))

                @torch.jit.script_method
                def forward(self, x):
                    if False:
                        print('Hello World!')
                    self.ignored_code(x)
                    return x

                @torch.jit.ignore(drop_on_export=True)
                def ignored_code(self, x):
                    if False:
                        print('Hello World!')
                    self.some_state = torch.tensor((100,))
        FileCheck().check('TorchScript will now drop the function').run(str(warns[0]))
        m = M()
        m2 = self.getExportImportCopy(m)
        pp = str(m2.forward.code)
        self.assertNotIn('ignored_code', pp)
        with self.assertRaisesRegex(torch.jit.Error, 'annotated to be ignored and cannot be run'):
            m2.forward(torch.ones(1))

    def test_ignored_as_value(self):
        if False:
            print('Hello World!')

        class Model(nn.Module):

            @torch.jit.unused
            def tuple_ignored(self, x):
                if False:
                    return 10
                return (x, x)

            @torch.jit.unused
            def single_val_ignored(self, x, y):
                if False:
                    while True:
                        i = 10
                return x

            def forward(self, x, use_ignore_path):
                if False:
                    for i in range(10):
                        print('nop')
                if 1 == 2:
                    return self.tuple_ignored(x)
                if use_ignore_path:
                    return (self.single_val_ignored(x, x), self.single_val_ignored(x, x))
                return (x, x)
        original = Model()
        scripted = torch.jit.script(original)
        self.assertEqual(scripted(torch.tensor(0.5), False), (torch.tensor(0.5), torch.tensor(0.5)))
        buffer = io.BytesIO()
        torch.jit.save(scripted, buffer)
        buffer.seek(0)
        loaded = torch.jit.load(buffer)
        with self.assertRaisesRegex(torch.jit.Error, 'annotated to be ignored and cannot be run'):
            loaded(torch.tensor(0.5), True)

    def test_module_error(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModule(torch.nn.Module):

            def forward(self, foo):
                if False:
                    for i in range(10):
                        print('nop')
                return foo
        with self.assertRaisesRegex(RuntimeError, 'cannot be compiled since it inherits from nn.Module'):
            torch.jit.script(MyModule)

    def test_view_write(self):
        if False:
            print('Hello World!')

        def fn(x, y):
            if False:
                for i in range(10):
                    print('nop')
            l = []
            l.append(x)
            x_view = l[0]
            a = x + x
            x_view.add_(y)
            b = x + x
            return a == b
        self.checkScript(fn, (torch.rand(2, 3), torch.rand(2, 3)))

    def test_module_attrs(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.jit.ScriptModule):

            def __init__(self, table):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.table = torch.jit.Attribute(table, Dict[str, torch.Tensor])
                self.x = torch.nn.Parameter(torch.tensor([100.0]))

            @torch.jit.script_method
            def forward(self, key):
                if False:
                    i = 10
                    return i + 15
                return self.table[key] + self.x
        with torch._jit_internal._disable_emit_hooks():
            m = M({char: torch.ones(1) + ord(char) - ord('a') for char in 'abcdefg'})
            self.assertEqual(m('c'), torch.tensor([103.0]))

    def test_module_none_attrs(self):
        if False:
            while True:
                i = 10

        class MyMod(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.optional_value = None

            @torch.jit.script_method
            def forward(self):
                if False:
                    print('Hello World!')
                return self.optional_value
        graph = MyMod().forward.graph
        FileCheck().check('prim::GetAttr').run(graph)
        self.run_pass('peephole', graph)
        FileCheck().check_not('prim::GetAttr').run(graph)

    def test_tensor_import_export(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def foo(x):
            if False:
                while True:
                    i = 10
            a = torch.tensor(1)
            b = torch.tensor([1, 2])
            c = [a, b]
            return c
        self.run_pass('constant_propagation', foo.graph)
        m = self.createFunctionFromGraph(foo.graph)
        self.getExportImportCopy(m)

    def get_pickle_values(self):
        if False:
            i = 10
            return i + 15
        return (('dict', {'I': 'am', 'a test': 'test'}, Dict[str, str]), ('float', 2.3, float), ('int', 99, int), ('bool', False, bool), ('tuple', (1, 2, 3, 4), Tuple[int, int, int, int]), ('list', [(1, 2), (3, 4)], List[Tuple[int, int]]), ('tensor', torch.randn(2, 2), torch.Tensor), ('int_list', [1, 2, 3, 4], List[int]), ('tensor_list', [torch.ones(2, 2) + i for i in range(4)], List[torch.Tensor]), ('bool_list', [True, True, False, True], List[bool]), ('float_list', [1.0, 2.0, 3.0, 4.0], List[float]), ('str_list', ['hello', 'bye'], List[str]), ('none', None, Optional[int]), ('a_device', torch.device('cpu'), torch.device), ('another_device', torch.device('cuda:1'), torch.device))

    def test_attribute_serialization(self):
        if False:
            print('Hello World!')
        tester = self

        class M(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                for (name, value, the_type) in tester.get_pickle_values():
                    setattr(self, name, torch.jit.Attribute(value, the_type))

            @torch.jit.script_method
            def forward(self):
                if False:
                    i = 10
                    return i + 15
                return (self.dict, self.float, self.int, self.bool, self.tuple, self.list, self.int_list, self.tensor_list, self.bool_list, self.float_list, self.str_list, self.none)
        m = M()
        imported_m = self.getExportImportCopy(m)
        self.assertEqual(m(), imported_m())

    def test_string_len(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return len(x)
        self.checkScript(fn, ('',))
        self.checkScript(fn, ('h',))
        self.checkScript(fn, ('hello',))

    def test_multiline_optional_future_refinement(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def fun() -> int:
            if False:
                while True:
                    i = 10
            future: Optional[torch.jit.Future[Tuple[torch.Tensor]]] = None
            return 1
        self.assertEqual(fun(), 1)

    @unittest.skipIf(IS_WINDOWS or IS_SANDCASTLE, 'NYI: TemporaryFileName support for Windows or Sandcastle')
    def test_attribute_unpickling(self):
        if False:
            print('Hello World!')
        tensor = torch.randn(2, 2)
        tester = self

        class M(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                for (name, value, the_type) in tester.get_pickle_values():
                    setattr(self, '_' + name, torch.jit.Attribute(value, the_type))

            @torch.jit.script_method
            def forward(self):
                if False:
                    i = 10
                    return i + 15
                return (self._dict, self._float, self._int, self._bool, self._tuple, self._list, self._int_list, self._tensor_list, self._bool_list, self._float_list, self._str_list, self._none)
        with TemporaryFileName() as fname:
            M().save(fname)
            loaded = torch.jit.load(fname)

            def is_tensor_value(item):
                if False:
                    return 10
                if isinstance(item, torch.Tensor):
                    return True
                if isinstance(item, list):
                    return is_tensor_value(item[0])
                return False
            for (name, value, the_type) in self.get_pickle_values():
                if is_tensor_value(value):
                    continue
                self.assertEqual(value, getattr(loaded, '_' + name))

    @unittest.skipIf(IS_WINDOWS or IS_SANDCASTLE, 'NYI: TemporaryFileName support for Windows or Sandcastle')
    @unittest.skipIf(not BUILD_WITH_CAFFE2, 'PyTorch is build without Caffe2 support')
    def test_old_models_bc(self):
        if False:
            print('Hello World!')
        model = {'archive/version': b'1', 'archive/code/archive.py': b'\n                op_version_set = 0\n                def forward(self,\n                    _0: Tensor) -> Tensor:\n                  _1 = torch.zeros([10], dtype=6, layout=0, device=torch.device("cpu"))\n                  result = torch.to(torch.fill_(_1, 5), dtype=6, layout=0, device=torch.device("cpu"),\n                                    non_blocking=False, copy=False)\n                  result2 = torch.rand([10], dtype=6, layout=0, device=torch.device("cpu"))\n                  result3 = torch.rand_like(result2, dtype=6, layout=0, device=torch.device("cpu"))\n                  _2 = torch.add(torch.add(result, result2, alpha=1), result3, alpha=1)\n                  return _2\n                ', 'archive/attributes.pkl': b'\x80\x02](e.', 'archive/libs.py': b'op_version_set = 0\n', 'archive/model.json': b'\n                {\n                   "protoVersion":"2",\n                   "mainModule":{\n                      "torchscriptArena":{\n                         "key":"code/archive.py"\n                      },\n                      "name":"archive",\n                      "optimize":true\n                   },\n                   "producerName":"pytorch",\n                   "producerVersion":"1.0",\n                   "libs":{\n                      "torchscriptArena":{\n                         "key":"libs.py"\n                      }\n                   }\n                }'}
        with TemporaryFileName() as fname:
            archive_name = os.path.basename(os.path.normpath(fname))
            with zipfile.ZipFile(fname, 'w') as archive:
                for (k, v) in model.items():
                    archive.writestr(k, v)
            with open(fname, 'rb') as f:
                fn = torch.jit.load(f)
        x = torch.zeros(10)
        fn(x)

    def test_submodule_attribute_serialization(self):
        if False:
            i = 10
            return i + 15

        class S(torch.jit.ScriptModule):

            def __init__(self, list_data):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.table = torch.jit.Attribute({'I': 'am', 'a test': 'test'}, Dict[str, str])
                self.list = torch.jit.Attribute(list_data, List[Tuple[int, int]])

            @torch.jit.script_method
            def forward(self):
                if False:
                    i = 10
                    return i + 15
                return (self.table, self.list)

        class M(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.table = torch.jit.Attribute({'this': 'is', 'a different': 'dict'}, Dict[str, str])
                self.tensor = torch.jit.Attribute(torch.randn(2, 2), torch.Tensor)
                self.s1 = S([(1, 2)])
                self.s2 = S([(4, 5)])

            @torch.jit.script_method
            def forward(self):
                if False:
                    while True:
                        i = 10
                return (self.table, self.tensor, self.s1.table, self.s2.list, self.s1.list)
        m = M()
        imported_m = self.getExportImportCopy(m)
        self.assertEqual(m(), imported_m())

    def test_serialization_big_ints(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.int32_max = torch.jit.Attribute(2 ** 31 - 1, int)
                self.int32_min = torch.jit.Attribute(-2 ** 31, int)
                self.uint32_max = torch.jit.Attribute(2 ** 32, int)
                self.int64_max = torch.jit.Attribute(2 ** 63 - 1, int)
                self.int64_min = torch.jit.Attribute(-2 ** 63, int)
                self.tensor = torch.nn.Parameter(torch.ones(2, 2))

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    return 10
                return x + (self.int32_max + self.int32_min) + (self.int64_max + self.int64_min)
        m = M()
        imported = self.getExportImportCopy(m)
        self.assertEqual(m(10), imported(10))
        self.assertEqual(m.int32_max, imported.int32_max)
        self.assertEqual(m.int32_min, imported.int32_min)
        self.assertEqual(m.uint32_max, imported.uint32_max)
        self.assertEqual(m.int64_max, imported.int64_max)
        self.assertEqual(m.int64_min, imported.int64_min)

    def test_script_scope(self):
        if False:
            return 10
        scripted = torch.jit.script(torch.nn.functional.triplet_margin_loss)

    @unittest.skipIf(IS_WINDOWS, 'NYI: TemporaryFileName on Windows')
    def test_serialization_sharing(self):
        if False:
            return 10

        class M(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.list = torch.jit.Attribute([], List[str])

            @torch.jit.script_method
            def forward(self, key):
                if False:
                    i = 10
                    return i + 15
                self.list.append(key)
                self.list.append(key)
                self.list.append(key)
                return self.list
        m = M()
        s1 = 'a long string'
        s2 = 'a different, even longer string'
        self.assertEqual(m(s1), [s1] * 3)
        self.assertEqual(m(s2), [s1] * 3 + [s2] * 3)
        with TemporaryFileName() as fname:
            m.save(fname)
            archive_name = os.path.basename(os.path.normpath(fname))
            archive = zipfile.ZipFile(fname, 'r')
            pickled_data = archive.read(os.path.join(archive_name, 'data.pkl'))
            out = io.StringIO()
            pickletools.dis(pickled_data, out=out)
            disassembled = out.getvalue()
            FileCheck().check_count(s1, 1, exactly=True).check_count('BINGET', 2, exactly=True).check_count(s2, 1, exactly=True).check_count('BINGET', 2, exactly=True).run(out.getvalue())

    def test_sys_stdout_override(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def foo():
            if False:
                return 10
            print('foo')

        class Redirect:

            def __init__(self):
                if False:
                    while True:
                        i = 10
                self.s = ''

            def write(self, s):
                if False:
                    for i in range(10):
                        print('nop')
                self.s += s
        old_stdout = sys.stdout
        redirect = Redirect()
        try:
            sys.stdout = redirect
            foo()
        finally:
            sys.stdout = old_stdout
        FileCheck().check('foo').run(redirect.s)

    def test_dtype_attr(self):
        if False:
            while True:
                i = 10

        class Foo(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.dtype = torch.zeros([]).dtype

            def forward(self):
                if False:
                    while True:
                        i = 10
                return torch.zeros(3, 4, dtype=self.dtype)
        f = Foo()
        torch.jit.script(f)

    def test_named_buffers_are_iterable(self):
        if False:
            while True:
                i = 10

        class MyMod(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.mod = torch.nn.ReLU()
                self.mod2 = torch.nn.ReLU()
                self.mod3 = torch.nn.Sequential(torch.nn.Sequential(torch.nn.ReLU()))
                self.register_buffer('x', torch.zeros(3))
                self.register_buffer('y', torch.zeros(3))
                self.z = torch.zeros(3)

            def bleh(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.z + 4

            @torch.jit.export
            def method(self):
                if False:
                    for i in range(10):
                        print('nop')
                names = ['']
                vals = []
                for (name, buffer) in self.named_buffers():
                    names.append(name)
                    vals.append(buffer + 2)
                return (names, vals)

            def forward(self, x):
                if False:
                    return 10
                return x
        model = MyMod()
        x = torch.jit.script(model)
        z = self.getExportImportCopy(x)
        self.assertEqual(z.method(), x.method())
        self.assertEqual(z.method(), model.method())
        self.assertEqual(x.method(), model.method())
        names = x.method()
        for name in names:
            self.assertNotEqual('z', name)

    def test_static_if_prop(self):
        if False:
            for i in range(10):
                print('nop')

        class MaybeHasAttr(torch.nn.Module):

            def __init__(self, add_attr):
                if False:
                    return 10
                super().__init__()
                if add_attr:
                    self.maybe_attr = 1

            def forward(self):
                if False:
                    while True:
                        i = 10
                if hasattr(self, 'maybe_attr') and True:
                    return self.maybe_attr
                else:
                    return 0

        class MaybeHasAttr2(torch.nn.Module):

            def __init__(self, add_attr):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                if add_attr:
                    self.maybe_attr = 1

            def forward(self):
                if False:
                    for i in range(10):
                        print('nop')
                if not hasattr(self, 'maybe_attr') or False:
                    return 0
                else:
                    return self.maybe_attr
        torch.jit.script(MaybeHasAttr(True))
        torch.jit.script(MaybeHasAttr(False))
        torch.jit.script(MaybeHasAttr2(True))
        torch.jit.script(MaybeHasAttr2(False))

        class MyMod(torch.nn.Module):

            def forward(self):
                if False:
                    i = 10
                    return i + 15
                if hasattr(self, 'foo'):
                    return 1
                else:
                    return 0

            @torch.jit.export
            def fee(self):
                if False:
                    return 10
                return 1
        self.checkModule(MyMod(), ())

        class HasAttrMod(torch.nn.Module):
            __constants__ = ['fee']

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.fee = 3

            def forward(self):
                if False:
                    i = 10
                    return i + 15
                a = hasattr(self, 'fee')
                b = hasattr(self, 'foo')
                c = hasattr(self, 'hi')
                d = hasattr(self, 'nonexistant')
                return (a, b, c, d)

            def foo(self):
                if False:
                    print('Hello World!')
                return 1

            @torch.jit._overload_method
            def hi(self, x: Tensor):
                if False:
                    for i in range(10):
                        print('nop')
                ...

            def hi(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return 2
        self.checkModule(HasAttrMod(), ())

        @torch.jit.script
        class FooTest:

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.x = 1

            def foo(self, y):
                if False:
                    while True:
                        i = 10
                return self.x + y

        def foo():
            if False:
                return 10
            a = FooTest()
            val1 = (hasattr(a, 'foo'), hasattr(a, 'x'), hasattr(a, 'bla'))
            val2 = (hasattr(FooTest, 'foo'), hasattr(FooTest, 'a'))
            return (val1, val2)
        self.assertEqual(foo(), torch.jit.script(foo)())

    def _test_pickle_checkpoint(self, device):
        if False:
            for i in range(10):
                print('nop')
        with TemporaryFileName() as fname:

            class M(torch.jit.ScriptModule):
                __constants__ = ['fname']

                def __init__(self, tensor):
                    if False:
                        return 10
                    super().__init__()
                    self.fname = fname
                    self.tensor = torch.nn.Parameter(tensor)

                @torch.jit.script_method
                def forward(self, x):
                    if False:
                        return 10
                    y = self.tensor + x
                    torch.save(y, self.fname)
                    return y
            param = torch.randn(2, 2).to(device)
            input = torch.randn(2, 2).to(device)
            m = M(param)
            m(input)
            with open(fname, 'rb') as handle:
                loaded_tensor = torch.load(fname)
                self.assertEqual(loaded_tensor, input + param)

    def _test_pickle_checkpoint_views(self, device):
        if False:
            i = 10
            return i + 15
        with TemporaryFileName() as fname:

            class M(torch.jit.ScriptModule):
                __constants__ = ['fname']

                def __init__(self, tensor):
                    if False:
                        i = 10
                        return i + 15
                    super().__init__()
                    self.fname = fname
                    self.tensor = torch.nn.Parameter(tensor)

                @torch.jit.script_method
                def forward(self, x):
                    if False:
                        while True:
                            i = 10
                    y = self.tensor + x
                    y_view = y.view(4)
                    torch.save((y, y_view, y), self.fname)
                    return y
            param = torch.randn(2, 2).to(device)
            input = torch.randn(2, 2).to(device)
            m = M(param)
            m(input)
            with open(fname, 'rb') as handle:
                (loaded_y, loaded_y_view, loaded_y_2) = torch.load(fname)
                self.assertEqual(loaded_y, input + param)
                with torch.no_grad():
                    loaded_y_view[1] += 20
                    self.assertEqual(loaded_y.view(4), loaded_y_view)
                    self.assertEqual(loaded_y_2.view(4), loaded_y_view)

    @unittest.skipIf(not RUN_CUDA, 'no CUDA')
    def test_pickle_checkpoint_cuda(self):
        if False:
            i = 10
            return i + 15
        self._test_pickle_checkpoint('cuda')
        self._test_pickle_checkpoint_views('cuda')

    def test_pickle_checkpoint(self):
        if False:
            i = 10
            return i + 15
        self._test_pickle_checkpoint('cpu')
        self._test_pickle_checkpoint_views('cpu')

    def test_pickle_checkpoint_tup(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def foo(fname):
            if False:
                while True:
                    i = 10
            torch.save((3, 4), fname)
        with TemporaryFileName() as name:
            foo(name)
            self.assertEqual(torch.load(name), (3, 4))

    def test_string_list(self):
        if False:
            while True:
                i = 10

        def fn(string):
            if False:
                while True:
                    i = 10
            return list(string)
        self.checkScript(fn, ('abcdefgh',))

    def test_unicode_comments(self):
        if False:
            return 10

        @torch.jit.script
        def test(self, a):
            if False:
                i = 10
                return i + 15
            return torch.nn.functional.relu(a)

    def test_get_set_state_with_tensors(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.tensor = torch.randn(2, 2)

            @torch.jit.export
            def __getstate__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return (self.tensor, self.training)

            @torch.jit.export
            def __setstate__(self, state):
                if False:
                    return 10
                self.tensor = state[0]
                self.training = state[1]

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x + self.tensor
        with TemporaryFileName() as fname:
            m = torch.jit.script(M())
            m.save(fname)
            loaded = torch.jit.load(fname)
            self.assertEqual(loaded.tensor, m.tensor)

    def test_in_for_and_comp_expr(self):
        if False:
            while True:
                i = 10

        def fn(d):
            if False:
                while True:
                    i = 10
            out = [1]
            for i in range(d['hi'] if 'hi' in d else 6):
                out.append(i)
            return out
        self.checkScript(fn, ({'hi': 2, 'bye': 3},))
        self.checkScript(fn, ({'bye': 3},))

    def test_for_else(self):
        if False:
            i = 10
            return i + 15

        def fn():
            if False:
                print('Hello World!')
            c = 0
            for i in range(4):
                c += 10
            else:
                print('In else block of for...else')
        with self.assertRaisesRegex(torch.jit.frontend.NotSupportedError, "else branches of for loops aren't supported"):
            torch.jit.script(fn)

    def test_split(self):
        if False:
            i = 10
            return i + 15

        def split_two(tensor):
            if False:
                i = 10
                return i + 15
            (a, b, c) = torch.split(tensor, 2, dim=1)
            return (a, b, c)
        x = torch.randn(3, 6)
        y = torch.randn(3, 6)
        self.checkScript(split_two, [x + y])

    def test_conv_error(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def fn(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return F.conv2d(x, y)
        try:
            fn(torch.ones(2, 2), torch.ones(4, 4))
        except RuntimeError as e:
            self.assertFalse('frame' in str(e))

    def test_python_op_name(self):
        if False:
            i = 10
            return i + 15
        import random
        with self.assertRaisesRegex(RuntimeError, 'randint'):

            @torch.jit.script
            def fn():
                if False:
                    for i in range(10):
                        print('nop')
                return random.randint()

    def test_dir(self):
        if False:
            return 10

        class M(torch.jit.ScriptModule):

            def forward(self, t):
                if False:
                    i = 10
                    return i + 15
                return t
        self.assertTrue('forward' in dir(M()))

    def test_kwarg_expansion_error(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.ignore
        def something_else(h, i):
            if False:
                return 10
            pass

        def fn(x):
            if False:
                print('Hello World!')
            something_else(**x)
        with self.assertRaisesRegex(torch.jit.frontend.NotSupportedError, 'keyword-arg expansion is not supported'):
            torch.jit.script(fn)

    def test_kwargs_error_msg(self):
        if False:
            i = 10
            return i + 15

        def other(**kwargs):
            if False:
                i = 10
                return i + 15
            print(kwargs)

        def fn():
            if False:
                return 10
            return other()
        with self.assertRaisesRegex(torch.jit.frontend.NotSupportedError, 'variable number'):
            torch.jit.script(fn)

        def another_other(*args):
            if False:
                i = 10
                return i + 15
            print(args)

        def another_fn():
            if False:
                i = 10
                return i + 15
            return another_other()
        with self.assertRaisesRegex(torch.jit.frontend.NotSupportedError, 'variable number'):
            torch.jit.script(another_fn)

    def test_inferred_error_msg(self):
        if False:
            return 10
        '\n        Test that when we get a type mismatch on a function where we inferred\n        the type to be tensor, a good error message is given.\n        '

        @torch.jit.script
        def foo(a):
            if False:
                for i in range(10):
                    print('nop')
            return a
        with self.assertRaisesRegex(RuntimeError, "Expected a value of type \\'Tensor \\(inferred\\)\\'[\\S\\s]*Inferred \\'a\\' to be of type \\'Tensor\\'"):
            foo('1')

    def test_type_comments_in_body(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def foo(a, b):
            if False:
                i = 10
                return i + 15
            return a + b

        class M(torch.nn.Module):

            def __init__(self, a, b):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.a = a
                self.b = b
        torch.jit.script(M(2, 3))

    def test_input_keyword_in_schema(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                while True:
                    i = 10
            return torch.ceil(input=x)
        inp = torch.randn(10)
        self.checkScript(f, (inp,))

    def test_module_method_reassignment(self):
        if False:
            print('Hello World!')

        class Foo(torch.nn.Module):

            def _forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x
            forward = _forward
        sm = torch.jit.script(Foo())
        input = torch.ones(2, 2)
        self.assertEqual(input, sm(input))

    def test_script_module_tensor_subclass_argument(self):
        if False:
            return 10

        @torch.jit.script
        def parameter_script(x: torch.nn.Parameter):
            if False:
                for i in range(10):
                    print('nop')
            return x
        input = torch.ones(2, 2)
        self.assertEqual(input, parameter_script(input))

    def test_save_load_attr_error(self):
        if False:
            for i in range(10):
                print('nop')

        class Inner(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x

        class Wrapper(nn.Module):

            def __init__(self, inner):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.inner = inner

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.inner.b(x)
        inner_module = torch.jit.script(Inner())
        inner_module = self.getExportImportCopy(inner_module)
        wrapped = Wrapper(inner_module)
        with self.assertRaisesRegex(RuntimeError, 'has no attribute'):
            torch.jit.script(wrapped)

    def test_rescripting_loaded_modules(self):
        if False:
            while True:
                i = 10

        class InnerSubmod(nn.Module):
            __constants__ = ['my_constant']

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.register_buffer('foo', torch.ones(1))
                self.register_parameter('bar', torch.nn.Parameter(torch.ones(1)))
                self.baz = torch.ones(1)
                self.my_constant = 1

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x + x

        class Inner(nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.submod = InnerSubmod()

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.submod(x)

        class Wrapper(nn.Module):

            def __init__(self, inner):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.inner = inner

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                ret = self.inner.submod(x) + self.inner.submod.foo + self.inner.submod.bar + self.inner.submod.baz
                ret = ret + self.inner.submod.my_constant
                return ret
        inner_module = torch.jit.script(Inner())
        wrapped = Wrapper(inner_module)
        self.checkModule(wrapped, torch.ones(1))
        inner_module_loaded = self.getExportImportCopy(inner_module)
        wrapped_loaded = Wrapper(inner_module_loaded)
        self.assertEqual(wrapped(torch.ones(1)), wrapped_loaded(torch.ones(1)))

    def test_interpret_graph(self):
        if False:
            print('Hello World!')

        def fn(x):
            if False:
                return 10
            return x.unfold(0, 1, 1)
        graph_str = '\n        graph(%a : Tensor, %b : Tensor):\n          %c : Tensor = aten::mul(%a, %b)\n          return (%c)\n        '
        graph = parse_ir(graph_str)
        a = torch.rand(10)
        b = torch.rand(10)
        test = torch._C._jit_interpret_graph(graph, (a, b))
        ref = a * b
        self.assertEqual(test, ref)

    def test_signed_float_zero(self):
        if False:
            while True:
                i = 10

        class MyModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.div(x, -0.0)
        inp = torch.ones(1)
        self.checkModule(MyModule(), inp)

    def test_index_with_tuple(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return x[1,]
        self.checkModule(MyModule(), (torch.ones(2, 3),))

    def test_context_manager(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModule(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                p = x + y
                q = p + 2.0
                return q
        x = torch.randn(3, 2, dtype=torch.float)
        y = torch.randn(3, 2, dtype=torch.float)
        for fuser_name in ['fuser0', 'fuser1', 'none']:
            with torch.jit.fuser(fuser_name):
                self.checkModule(MyModule(), (x, y))
EXCLUDE_TRACED = {'test___getitem___adv_index', 'test___getitem___adv_index_beg', 'test___getitem___adv_index_comb', 'test___getitem___adv_index_dup', 'test___getitem___adv_index_sub', 'test___getitem___adv_index_sub_2', 'test___getitem___adv_index_sub_3', 'test___getitem___adv_index_var', 'test_to_sparse', 'test_to_sparse_dim'}
EXCLUDE_TYPE_CHECK = {'test_slogdet_1x1_neg_det', 'test_slogdet_1x1_pos_det', 'test_slogdet_distinct_singular_values', 'test_slogdet_neg_det', 'test_slogdet_pos_det', 'test_slogdet_symmetric', 'test_slogdet_symmetric_pd', 'test_slogdet_batched_1x1_neg_det', 'test_slogdet_batched_pos_det', 'test_slogdet_batched_symmetric', 'test_slogdet_batched_symmetric_pd', 'test_slogdet_batched_distinct_singular_values'}
EXCLUDE_SCRIPT_AD_CHECK = {'test_chunk', 'test_chunk_dim', 'test_chunk_dim_neg0', 'test_split_size_list', 'test_split_size_list_dim', 'test_split_size_list_dim_neg0', 'test_tensor_indices_sections', 'test_tensor_indices_sections_dim', 'test_tensor_indices_sections_dim_neg0', 'test_tensor_split_sections', 'test_tensor_split_sections_dim', 'test_tensor_split_sections_dim_neg0'}
EXCLUDE_PYTHON_PRINT = {'test_nn_max_unpool1d', 'test_nn_max_unpool2d', 'test_nn_max_unpool3d', 'test_nn_max_pool1d', 'test_nn_max_pool2d', 'test_nn_max_pool3d', 'test_nn_max_pool1d_with_indices'}
EXCLUDE_ALIAS = {'true_divide', 'lu'}

@skipIfTorchDynamo()
class TestJitGeneratedModule(JitTestCase):
    pass

@skipIfTorchDynamo()
class TestJitGeneratedFunctional(JitTestCase):
    pass
UBSAN_DISABLED_TESTS = ['test___rdiv___constant', 'test___rdiv___scalar_constant', 'test_addcdiv', 'test_addcdiv_broadcast_all', 'test_addcdiv_broadcast_rhs', 'test_addcdiv_scalar', 'test_addcdiv_scalar_broadcast_lhs', 'test_addcdiv_scalar_broadcast_rhs', 'test_addcdiv_scalar_scale', 'test_addcdiv_scalar_scale_broadcast_lhs', 'test_addcdiv_scalar_scale_broadcast_rhs', 'test_addcdiv_scale', 'test_addcdiv_scale_broadcast_all', 'test_addcdiv_scale_broadcast_rhs', 'test_add_broadcast_all', 'test_add_broadcast_lhs', 'test_add_broadcast_rhs', 'test_add_constant', 'test_add_scalar', 'test_add_scalar_broadcast_lhs', 'test_add_scalar_broadcast_rhs', 'test_div', 'test_div_broadcast_all', 'test_div_broadcast_lhs', 'test_div_broadcast_rhs', 'test_div_scalar', 'test_div_scalar_broadcast_lhs', 'test_div_scalar_broadcast_rhs', 'test_rsqrt', 'test_rsqrt_scalar', 'test_add', 'test_reciprocal', 'test_reciprocal_scalar']
L = 20
M = 10
S = 5

def add_nn_module_test(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    no_grad = False if 'no_grad' not in kwargs else kwargs['no_grad']
    if 'desc' in kwargs and 'eval' in kwargs['desc']:
        return
    test_name = get_nn_mod_test_name(**kwargs)

    @suppress_warnings
    def do_test(self):
        if False:
            return 10
        if test_name in EXCLUDE_SCRIPT_MODULES:
            return
        if not kwargs.get('check_jit', True):
            raise unittest.SkipTest('module test skipped on JIT')
        default_dtype = torch.get_default_dtype()
        if 'default_dtype' in kwargs and kwargs['default_dtype'] is not None:
            default_dtype = kwargs['default_dtype']
        module_name = get_nn_module_name_from_kwargs(**kwargs)
        if 'constructor' in kwargs:
            nn_module = kwargs['constructor']
        else:
            nn_module = getattr(torch.nn, module_name)
        if 'FunctionalModule' in str(nn_module):
            return
        with set_default_dtype(default_dtype):
            if 'constructor_args_fn' in kwargs:
                constructor_args = kwargs['constructor_args_fn']()
            else:
                constructor_args = kwargs.get('constructor_args', ())

            def create_script_module(*args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                'Construct a script module that passes arguments through to self.submodule'
                (formals, tensors, actuals) = get_script_args(args)
                method_args = ', '.join(['self'] + actuals)
                call_args_str = ', '.join(actuals)
                call = f'self.submodule({call_args_str})'
                script = script_method_template.format(method_args, call)
                submodule_constants = []
                if kwargs.get('is_constant'):
                    submodule_constants = ['submodule']

                class TheModule(torch.jit.ScriptModule):
                    __constants__ = submodule_constants

                    def __init__(self):
                        if False:
                            while True:
                                i = 10
                        super().__init__()
                        self.submodule = nn_module(*constructor_args)

                def make_module(script):
                    if False:
                        while True:
                            i = 10
                    module = TheModule()
                    str(module)
                    module.define(script)
                    return module
                module = make_module(script)
                self.assertExportImportModule(module, tensors)
                create_script_module.last_graph = module.graph
                mod = module(*args)
                return mod

            def create_nn_module(*args, **kwargs):
                if False:
                    print('Hello World!')
                module = nn_module(*constructor_args)
                return module(*args)
            dtype = torch.get_default_dtype()
            if 'input_fn' in kwargs:
                input = kwargs['input_fn']()
                if isinstance(input, Tensor):
                    input = (input,)
                if all((tensor.is_complex() for tensor in input)):
                    if dtype == torch.float:
                        dtype = torch.cfloat
                    elif dtype == torch.double:
                        dtype = torch.cdouble
                    else:
                        raise AssertionError(f'default_dtype {default_dtype} is not supported')
            else:
                input = (kwargs['input_size'],)
            if 'target_size' in kwargs:
                input = input + (kwargs['target_size'],)
            elif 'target_fn' in kwargs:
                if torch.is_tensor(input):
                    input = (input,)
                input = input + (kwargs['target_fn'](),)
            elif 'target' in kwargs:
                input = input + (kwargs['target'],)
            if 'extra_args' in kwargs:
                input = input + kwargs['extra_args']
            (args_variable, kwargs_variable) = create_input(input, dtype=dtype)
            f_args_variable = deepcopy(unpack_variables(args_variable))
            any_requires_grad = any((input.requires_grad for input in f_args_variable))
            check_against_reference(self, create_script_module, create_nn_module, lambda x: x, f_args_variable, no_grad=no_grad or not any_requires_grad)
    if 'slowTest' in kwargs:
        do_test = slowTest(do_test)
    post_add_test(test_name, (), do_test, TestJitGeneratedModule)

def post_add_test(test_name, skipTestIf, do_test, test_class):
    if False:
        i = 10
        return i + 15
    assert not hasattr(test_class, test_name), 'Two tests have the same name: ' + test_name
    for skip in skipTestIf:
        do_test = skip(do_test)
    if not (TEST_WITH_UBSAN and test_name in UBSAN_DISABLED_TESTS):
        setattr(test_class, test_name, do_test)

def normalize_check_ad(check_ad, name):
    if False:
        print('Hello World!')
    if len(check_ad) == 0:
        check_ad = [False, ['aten::' + name], []]
    elif len(check_ad) == 1:
        check_ad = [check_ad[0], ['aten::' + name], []]
    elif len(check_ad) == 2:
        check_ad = [check_ad[0], check_ad[1], []]
    elif len(check_ad) == 3:
        check_ad = list(check_ad)
    else:
        raise Exception('Invalid check_ad, requires (bool, str|List[str], str|List[str])')
    check_ad = [[t] if isinstance(t, str) else t for t in check_ad]
    return check_ad

class TestProducerVersion(TestCase):

    def test_version(self):
        if False:
            while True:
                i = 10
        self.assertTrue(torch.__version__.startswith(torch.onnx.producer_version))
for test in module_tests + new_module_tests + additional_module_tests:
    add_nn_module_test(**test)
for test in criterion_tests:
    test['no_grad'] = True
    add_nn_module_test(**test)
if __name__ == '__main__':
    TestCase._default_dtype_check_enabled = True
    run_tests()
    import jit.test_module_interface
    suite = unittest.findTestCases(jit.test_module_interface)
    unittest.TextTestRunner().run(suite)