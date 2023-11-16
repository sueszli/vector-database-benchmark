import importlib
import inspect
import logging
import os
import unittest
from enum import Flag, auto
from functools import wraps
from pathlib import Path
import numpy as np
import paddle
from paddle import set_flags, static
from paddle.base import core
from paddle.jit.api import sot_mode_guard
'\n# Usage:\nclass MyTest(Dy2StTestBase):\n    @set_to_static_mode(\n        ToStaticMode.AST | ToStaticMode.SOT\n    )\n    @set_ir_mode(IrMode.LEGACY_IR | IrMode.PIR_EXE | IrMode.PIR_API)\n    def test_case1(self):\n        raise ValueError("MyTest 1")\n\n    def test_case2(self):\n        raise ValueError("MyTest 2")\n\n\nclass MyTest2(MyTest):\n    def test_case1(self):\n        raise ValueError("MyTest2 1")\n'
logger = logging.getLogger('Dygraph to static utils')
logger.setLevel(logging.WARNING)

class ToStaticMode(Flag):
    AST = auto()
    SOT = auto()

    def lower_case_name(self):
        if False:
            i = 10
            return i + 15
        return self.name.lower()

class IrMode(Flag):
    LEGACY_IR = auto()
    PIR_EXE = auto()
    PIR_API = auto()

    def lower_case_name(self):
        if False:
            i = 10
            return i + 15
        return self.name.lower()
DEFAULT_TO_STATIC_MODE = ToStaticMode.AST | ToStaticMode.SOT
DEFAULT_IR_MODE = IrMode.LEGACY_IR

def to_legacy_ast_test(fn):
    if False:
        print('Hello World!')
    '\n    convert run fall_back to ast\n    '

    @wraps(fn)
    def impl(*args, **kwargs):
        if False:
            return 10
        logger.info('[AST] running AST')
        with sot_mode_guard(False):
            fn(*args, **kwargs)
    return impl

def to_sot_test(fn):
    if False:
        return 10
    '\n    convert run fall_back to ast\n    '

    @wraps(fn)
    def impl(*args, **kwargs):
        if False:
            return 10
        logger.info('[SOT] running SOT')
        with sot_mode_guard(True):
            fn(*args, **kwargs)
    return impl

def to_legacy_ir_test(fn):
    if False:
        return 10

    def impl(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        logger.info('[LEGACY_IR] running legacy ir')
        return fn(*args, **kwargs)
    return impl

def to_pir_exe_test(fn):
    if False:
        return 10

    @wraps(fn)
    def impl(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        logger.info('[PIR_EXE] running pir exe')
        ir_outs = None
        if os.environ.get('FLAGS_use_stride_kernel', False):
            return
        with static.scope_guard(static.Scope()):
            with static.program_guard(static.Program()):
                pir_flag = 'FLAGS_enable_pir_in_executor'
                try:
                    os.environ[pir_flag] = 'True'
                    set_flags({pir_flag: True})
                    ir_outs = fn(*args, **kwargs)
                finally:
                    del os.environ[pir_flag]
                    set_flags({pir_flag: False})
        return ir_outs
    return impl

def to_pir_api_test(fn):
    if False:
        while True:
            i = 10

    @wraps(fn)
    def impl(*args, **kwargs):
        if False:
            return 10
        logger.info('[PIR_API] running pir api')
        ir_outs = None
        with paddle.pir_utils.IrGuard():
            paddle.disable_static()
            ir_outs = fn(*args, **kwargs)
        return ir_outs
    return impl

class Dy2StTestMeta(type):
    TO_STATIC_HANDLER_MAP = {ToStaticMode.SOT: to_sot_test, ToStaticMode.AST: to_legacy_ast_test}
    IR_HANDLER_MAP = {IrMode.LEGACY_IR: to_legacy_ir_test, IrMode.PIR_EXE: to_pir_exe_test, IrMode.PIR_API: to_pir_api_test}

    def __new__(cls, name, bases, attrs):
        if False:
            for i in range(10):
                print('nop')
        new_attrs = {}
        original_test_cases = {key: value for (key, value) in attrs.items() if key.startswith('test') and inspect.isfunction(value)}
        logger.info(f'[creating {name}]')
        new_attrs.update({key: value for (key, value) in attrs.items() if key not in original_test_cases})
        for (fn_name, fn) in original_test_cases.items():
            logger.info(f'Generating {fn_name}')
            for base in bases:
                for attr in dir(base):
                    if attr.startswith(f'{fn_name}__'):
                        new_attrs[attr] = None
            fn_to_static_modes = getattr(fn, 'to_static_mode', DEFAULT_TO_STATIC_MODE)
            fn_ir_modes = getattr(fn, 'ir_mode', DEFAULT_IR_MODE)
            fn_disabled_test_cases = getattr(fn, 'disabled_test_cases', [])
            logger.info(f'fn_to_static_modes: {fn_to_static_modes}')
            logger.info(f'fn_ir_modes: {fn_ir_modes}')
            logger.info(f'fn_disabled_test_cases: {fn_disabled_test_cases}')
            to_static_with_ir_modes = [(to_static_mode, ir_mode) for to_static_mode in ToStaticMode for ir_mode in IrMode if to_static_mode & fn_to_static_modes and ir_mode & fn_ir_modes]
            to_static_with_ir_modes = list(filter(lambda flags: flags not in fn_disabled_test_cases, to_static_with_ir_modes))
            for (to_static_mode, ir_mode) in to_static_with_ir_modes:
                new_attrs[Dy2StTestMeta.test_case_name(fn_name, to_static_mode, ir_mode)] = Dy2StTestMeta.convert_test_case(fn, to_static_mode, ir_mode)
        return type.__new__(cls, name, bases, new_attrs)

    @staticmethod
    def test_case_name(original_name: str, to_static_mode, ir_mode):
        if False:
            return 10
        return f'{original_name}__{to_static_mode.lower_case_name()}_{ir_mode.lower_case_name()}'

    @staticmethod
    def convert_test_case(fn, to_static_mode, ir_mode):
        if False:
            while True:
                i = 10
        fn = Dy2StTestMeta.IR_HANDLER_MAP[ir_mode](fn)
        fn = Dy2StTestMeta.TO_STATIC_HANDLER_MAP[to_static_mode](fn)
        return fn

class Dy2StTestBase(unittest.TestCase, metaclass=Dy2StTestMeta):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)

def set_to_static_mode(mode: ToStaticMode):
    if False:
        print('Hello World!')

    def decorator(fn):
        if False:
            return 10
        fn.to_static_mode = mode
        return fn
    return decorator

def set_ir_mode(mode: IrMode):
    if False:
        i = 10
        return i + 15

    def decorator(fn):
        if False:
            while True:
                i = 10
        fn.ir_mode = mode
        return fn
    return decorator

def disable_test_case(flags):
    if False:
        print('Hello World!')

    def decorator(fn):
        if False:
            return 10
        disabled_test_cases = getattr(fn, 'disabled_test_cases', [])
        disabled_test_cases.append(flags)
        fn.disabled_test_cases = disabled_test_cases
        return fn
    return decorator

def test_ast_only(fn):
    if False:
        for i in range(10):
            print('nop')
    fn = set_to_static_mode(ToStaticMode.AST)(fn)
    return fn

def test_sot_only(fn):
    if False:
        return 10
    fn = set_to_static_mode(ToStaticMode.SOT)(fn)
    return fn

def test_pir_only(fn):
    if False:
        i = 10
        return i + 15
    fn = set_ir_mode(IrMode.PIR_EXE)(fn)
    return fn

def test_pir_api_only(fn):
    if False:
        for i in range(10):
            print('nop')
    fn = set_ir_mode(IrMode.PIR_API)(fn)
    return fn

def test_legacy_and_pir(fn):
    if False:
        return 10
    fn = set_ir_mode(IrMode.LEGACY_IR | IrMode.PIR_EXE)(fn)
    return fn

def test_legacy_and_pir_api(fn):
    if False:
        for i in range(10):
            print('nop')
    fn = set_ir_mode(IrMode.LEGACY_IR | IrMode.PIR_API)(fn)
    return fn

def test_legacy_and_pir_exe_and_pir_api(fn):
    if False:
        return 10
    fn = set_ir_mode(IrMode.LEGACY_IR | IrMode.PIR_API | IrMode.PIR_EXE)(fn)
    return fn

def compare_legacy_with_pir(fn):
    if False:
        i = 10
        return i + 15

    @wraps(fn)
    def impl(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        outs = fn(*args, **kwargs)
        if core._is_bwd_prim_enabled() or core._is_fwd_prim_enabled():
            return outs
        ir_outs = to_pir_exe_test(fn)(*args, **kwargs)
        np.testing.assert_equal(outs, ir_outs, err_msg=f'Dy2St Unittest Check ({fn.__name__}) has diff \n' + f'Expect {outs}\n' + f'But Got {ir_outs}')
        return outs
    return impl

def show_all_test_cases(test_class):
    if False:
        while True:
            i = 10
    logger.info(f'[showing {test_class.__name__}]')
    for attr in dir(test_class):
        if attr.startswith('test'):
            fn = getattr(test_class, attr)
            logger.info(f'{attr}: {fn}')

def import_module_from_path(module_name, module_path):
    if False:
        while True:
            i = 10
    'A better way to import module from other directory than using sys.path.append'
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def import_legacy_test_utils():
    if False:
        while True:
            i = 10
    test_root = Path(__file__).parent.parent
    legacy_test_utils_path = test_root / 'legacy_test/utils.py'
    legacy_test_utils = import_module_from_path('legacy_test_utils', legacy_test_utils_path)
    return legacy_test_utils
legacy_test_utils = import_legacy_test_utils()
dygraph_guard = legacy_test_utils.dygraph_guard
static_guard = legacy_test_utils.static_guard