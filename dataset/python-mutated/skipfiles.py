import _collections_abc
import _weakrefset
import abc
import collections
import contextlib
import copy
import copyreg
import dataclasses
import enum
import functools
import importlib
import inspect
import linecache
import logging
import multiprocessing
import operator
import os
import posixpath
import random
import re
import selectors
import signal
import tempfile
import threading
import tokenize
import traceback
import types
import typing
import unittest
import weakref
from typing import Optional
import torch
import torch._inductor.test_operators
import torch.distributed
import torch.utils._content_store
from .utils import getfile
from .variables.functions import NestedUserFunctionVariable, UserFunctionVariable, UserMethodVariable
"\nA note on skipfiles:\n\nDynamo consults this file to determine whether function should be inlined or skipped.\n\nA skip applies at the frame boundary, meaning dynamo either triggers a graph break\nat the beginning of the frame or attempts to trace/inline the whole frame. When skipping\na frame, recursively called frames are still traced by dynamo unless also skipped.\n\nSkipfiles (skipped at the file level instead of function level) still apply on a\nframe-by-frame boundary as dynamo traces, but apply to all functions in that file.\n\n@skip is a helper decorator that can be applied to your function to cause it to be\nincluded here.\n\nDynamo skip/inline rules & priorities are defined as follows:\n* Inline is the default behavior and will be used unless explicitly skipped.\n* Dynamo has two SKIPLIST: BUILTIN_SKIPLIST and THIRDPARTY_SKIPLIST.\n    * BUILTIN_SKIPLIST contains builtin python modules, such as abc, collections, etc.\n    * THIRDPARTY_SKIPLIST contains common third party libraries, such as numpy, pandas, etc.\n* Functions in these two SKIPLISTs are always skipped, except when they are explicitly\n    put into the two INLINELIST: FUNC_INLINELIST and MOD_INLINELIST.\n* PyTorch(torch) is in the BUILTIN_SKIPLIST by default, but there are many cases\n    where we want inline the functions under torch namespace. We should add them\n    into one of the two *_INLINELIST to make dynamo inline those functions.\n* If you call functions under skipped modules/files, Dynamo will wrap these functions\n    as SkipFilesVariable. There are a few functions(e.g, collections.OrderedDict) that\n    we have special handling at SkipFilesVariable.call_function.\n\nOverall: *_INLINELIST has precedence over *_SKIPLIST has precedence over DEFAULT (inline)\n\nTo figure out what the behavior is, check the following list in order:\n* FUNC_INLINELIST (Inline if YES)\n* MOD_INLINELIST (Inline if YES)\n* BUILTIN_SKIPLIST & THIRDPARTY_SKIPLIST (Skip if YES)\n* Inline by default\n\nIn general, if you want to force inline a function or module, please consider adding\nthe function's python module to MOD_INLINELIST first.\nUse the FUNC_INLINELIST only when there are other functions under the same module that\nyou don't want to inline them.\n"
BUILTIN_SKIPLIST = (abc, collections, contextlib, copy, copyreg, dataclasses, enum, functools, importlib, inspect, linecache, logging, multiprocessing, operator, os, posixpath, random, re, selectors, signal, tempfile, threading, tokenize, torch, traceback, types, typing, unittest, weakref, _collections_abc, _weakrefset)
THIRDPARTY_SKIPLIST = ('functorch', 'fx2trt_oss', 'networkx', 'numpy', 'omegaconf', 'onnx', 'onnxruntime', 'onnx_tf', 'pandas', 'sklearn', 'tabulate', 'tensorflow', 'tensorrt', 'torch2trt', 'tqdm', 'tree', 'tvm', 'xarray')

def _strip_init_py(s):
    if False:
        for i in range(10):
            print('nop')
    return re.sub('__init__.py$', '', s)

def _module_dir(m: types.ModuleType):
    if False:
        i = 10
        return i + 15
    return _strip_init_py(m.__file__)
FUNC_INLINELIST = {'torch._constrain_as_size', 'torch._constrain_as_value'}
LEGACY_MOD_INLINELIST = {'torch._dynamo.external_utils', 'torch._export.db.examples', 'torch._export.wrappers', 'torch._functorch.apis', 'torch._functorch.deprecated', 'torch._higher_order_ops.cond', 'torch.ao.quantization.pt2e.eval_utils', 'torch.ao.quantization.pt2e.qat_utils', 'torch.ao.quantization.pt2e.representation.rewrite', 'torch.ao.quantization.pt2e.utils', 'torch.ao.quantization.quantizer.xnnpack_quantizer', 'torch.optim'}
if torch.distributed.is_available():
    LEGACY_MOD_INLINELIST |= {'torch.distributed._tensor.api', 'torch.distributed._tensor.device_mesh', 'torch.distributed._device_mesh', 'torch.distributed.algorithms._checkpoint.checkpoint_wrapper', 'torch.distributed.tensor.parallel._data_parallel_utils', 'torch.distributed.tensor.parallel._utils', 'torch.distributed.tensor.parallel.style'}
MOD_INLINELIST = {'torch._refs', 'torch._prims', 'torch._decomp', 'torch._dynamo._trace_wrapped_higher_order_op', 'torch._dynamo.comptime', 'torch._dynamo.polyfill', 'torch._inductor.test_operators', 'torch.ao.nn', 'torch.distributions', 'torch.fx._pytree', 'torch.fx.passes.shape_prop', 'torch.nn', 'torch.random', 'torch.sparse', 'torch.testing', 'torch.utils._content_store', 'torch.utils._contextlib', 'torch.utils._foreach_utils', 'torch.utils._pytree', 'torch._tensor'}
if torch.distributed.is_available():
    MOD_INLINELIST.add('torch.distributed')
    MOD_INLINELIST.add('torch.distributed._functional_collectives')

@functools.lru_cache(None)
def get_func_inlinelist():
    if False:
        return 10
    inlinelist = set()
    for f in FUNC_INLINELIST:
        (module_name, fn_name) = f.rsplit('.', 1)
        m = importlib.import_module(module_name)
        fn = getattr(m, fn_name)
        inlinelist.add(fn.__code__)
    return inlinelist

@functools.lru_cache(None)
def get_legacy_mod_inlinelist():
    if False:
        print('Hello World!')
    inlinelist = set()
    for m in LEGACY_MOD_INLINELIST:
        inlinelist.add(_module_dir(torch) + m[len('torch.'):].replace('.', '/'))
    return inlinelist

@functools.lru_cache(None)
def get_mod_inlinelist():
    if False:
        for i in range(10):
            print('nop')
    inlinelist = set()
    for m in MOD_INLINELIST:
        inlinelist.add(_module_dir(torch) + m[len('torch.'):].replace('.', '/'))
    return inlinelist
SKIP_DIRS = ['<frozen importlib', '<__array_function__ internals>'] + [_module_dir(m) for m in BUILTIN_SKIPLIST]
SKIP_DIRS_RE = re.compile('match nothing^')
is_fbcode = importlib.import_module('torch._inductor.config').is_fbcode()
FBCODE_SKIP_DIRS = {'torchrec/distributed', 'torchrec/fb/distributed', 'caffe2/torch/fb/sparsenn/pooled_embeddings_modules.py'}
FBCODE_SKIP_DIRS_RE = re.compile(f".*({'|'.join(map(re.escape, FBCODE_SKIP_DIRS))})")

def _recompile_re():
    if False:
        while True:
            i = 10
    global SKIP_DIRS_RE
    SKIP_DIRS_RE = re.compile(f"^({'|'.join(map(re.escape, SKIP_DIRS))})")

def add(import_name: str):
    if False:
        i = 10
        return i + 15
    if isinstance(import_name, types.ModuleType):
        return add(import_name.__name__)
    assert isinstance(import_name, str)
    from importlib.util import find_spec
    module_spec = find_spec(import_name)
    if not module_spec:
        return
    origin = module_spec.origin
    if origin is None:
        return
    global SKIP_DIRS_RE
    SKIP_DIRS.append(_strip_init_py(origin))
    _recompile_re()

@dataclasses.dataclass
class SkipResult:
    skipped: bool
    reason: Optional[str]

def check_file(filename, allow_torch=False):
    if False:
        print('Hello World!')
    'Should skip this file?'
    if filename is None:
        return SkipResult(True, 'filename is None')
    if any((filename.startswith(d) for d in get_legacy_mod_inlinelist())):
        return SkipResult(False, 'inlined according skipfiles.LEGACY_MOD_INLINELIST')
    if allow_torch and is_torch_inline_allowed(filename):
        return SkipResult(False, 'inlined according skipfiles.MOD_INLINELIST')
    if is_fbcode and bool(FBCODE_SKIP_DIRS_RE.match(filename)):
        return SkipResult(True, 'skipped according skipfiles.FBCODE_SKIP_DIRS')
    if bool(SKIP_DIRS_RE.match(filename)):
        return SkipResult(True, 'skipped according skipfiles.SKIP_DIRS')
    else:
        return SkipResult(False, 'inlined by default')
"\nThis is the main entry point to determine whether an object (function) should be inlined or skipped.\nLet's illustrate the logic with an example:\n    @torch.compile\n    def f1(x, y):\n        ......\n        f2(x, y)\n        ......\n\n    def f2(x, y):\n        ......\n        f3(x, y)\n        ......\n\n    def f3(x, y):\n        ......\n\nThere are mainly three call sites of check/check_verbose:\n* The compile region entrance (like function f1), the correspoinding code is located at eval_frame.py.\n* When tracing the recursively called functions (like function f2 and f3).\n    * Dynamo decides inline/skip everytime it encounters a new recursively function call, and the call site\n      is in InliningInstructionTranslator.check_inlineable of symbolic_convert.py.\n    * If f2 is skipped by Dynamo, when evaluating the frame of f3, Dynamo need the inline/skip check again\n      and the call site is in catch_errors_wrapper.catch_errors of eval_frame.py.\n* For global variables and function arguments, Dynamo needs to decide if they are wrapped as SkipFilesVariable in builder.py.\n\nallow_torch is used to indicate whether we are checking the MOD_INLINELIST (torch modules), we only do this check when\nf2 is not skipped.\n"

def check_verbose(obj, allow_torch=False):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(obj, (UserFunctionVariable, UserMethodVariable, NestedUserFunctionVariable)):
        filename = obj.get_filename()
        obj = obj.get_code()
    elif isinstance(obj, types.CodeType):
        filename = obj.co_filename
    elif isinstance(obj, (types.FunctionType, types.MethodType)):
        filename = getfile(obj)
        obj = obj.__code__
    else:
        filename = getfile(obj)
    if obj in get_func_inlinelist():
        return SkipResult(False, 'inlined according skipfiles.FUNC_INLINELIST')
    return check_file(filename, allow_torch)

def check(obj, allow_torch=False):
    if False:
        for i in range(10):
            print('nop')
    return check_verbose(obj, allow_torch).skipped
for _name in THIRDPARTY_SKIPLIST:
    add(_name)
_recompile_re()

def is_torch_inline_allowed(filename):
    if False:
        while True:
            i = 10
    return any((filename.startswith(d) for d in get_mod_inlinelist()))

@functools.lru_cache(None)
def dynamo_dir():
    if False:
        return 10
    import torch._dynamo
    return _module_dir(torch._dynamo)

def is_torch(filename):
    if False:
        return 10
    if filename.startswith(dynamo_dir()):
        return False
    return filename.startswith(_module_dir(torch))