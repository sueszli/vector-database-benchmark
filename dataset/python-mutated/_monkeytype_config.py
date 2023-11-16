import inspect
import pathlib
import sys
import typing
from collections import defaultdict
from types import CodeType
from typing import Dict, Iterable, List, Optional
import torch
_IS_MONKEYTYPE_INSTALLED = True
try:
    import monkeytype
    from monkeytype import trace as monkeytype_trace
    from monkeytype.config import _startswith, LIB_PATHS
    from monkeytype.db.base import CallTraceStore, CallTraceStoreLogger, CallTraceThunk
    from monkeytype.tracing import CallTrace, CodeFilter
except ImportError:
    _IS_MONKEYTYPE_INSTALLED = False

def is_torch_native_class(cls):
    if False:
        for i in range(10):
            print('nop')
    if not hasattr(cls, '__module__'):
        return False
    parent_modules = cls.__module__.split('.')
    if not parent_modules:
        return False
    root_module = sys.modules.get(parent_modules[0])
    return root_module is torch

def get_type(type):
    if False:
        return 10
    'Convert the given type to a torchScript acceptable format.'
    if isinstance(type, str):
        return type
    elif inspect.getmodule(type) == typing:
        type_to_string = str(type)
        return type_to_string.replace(type.__module__ + '.', '')
    elif is_torch_native_class(type):
        return type.__module__ + '.' + type.__name__
    else:
        return type.__name__

def get_optional_of_element_type(types):
    if False:
        return 10
    'Extract element type, return as `Optional[element type]` from consolidated types.\n\n    Helper function to extracts the type of the element to be annotated to Optional\n    from the list of consolidated types and returns `Optional[element type]`.\n    TODO: To remove this check once Union support lands.\n    '
    elem_type = types[1] if type(None) == types[0] else types[0]
    elem_type = get_type(elem_type)
    return 'Optional[' + elem_type + ']'

def get_qualified_name(func):
    if False:
        return 10
    return func.__qualname__
if _IS_MONKEYTYPE_INSTALLED:

    class JitTypeTraceStoreLogger(CallTraceStoreLogger):
        """A JitTypeCallTraceLogger that stores logged traces in a CallTraceStore."""

        def __init__(self, store: CallTraceStore):
            if False:
                i = 10
                return i + 15
            super().__init__(store)

        def log(self, trace: CallTrace) -> None:
            if False:
                while True:
                    i = 10
            self.traces.append(trace)

    class JitTypeTraceStore(CallTraceStore):

        def __init__(self):
            if False:
                return 10
            super().__init__()
            self.trace_records: Dict[str, list] = defaultdict(list)

        def add(self, traces: Iterable[CallTrace]):
            if False:
                i = 10
                return i + 15
            for t in traces:
                qualified_name = get_qualified_name(t.func)
                self.trace_records[qualified_name].append(t)

        def filter(self, qualified_name: str, qualname_prefix: Optional[str]=None, limit: int=2000) -> List[CallTraceThunk]:
            if False:
                while True:
                    i = 10
            return self.trace_records[qualified_name]

        def analyze(self, qualified_name: str) -> Dict:
            if False:
                return 10
            records = self.trace_records[qualified_name]
            all_args = defaultdict(set)
            for record in records:
                for (arg, arg_type) in record.arg_types.items():
                    all_args[arg].add(arg_type)
            return all_args

        def consolidate_types(self, qualified_name: str) -> Dict:
            if False:
                for i in range(10):
                    print('nop')
            all_args = self.analyze(qualified_name)
            for (arg, types) in all_args.items():
                types = list(types)
                type_length = len(types)
                if type_length == 2 and type(None) in types:
                    all_args[arg] = get_optional_of_element_type(types)
                elif type_length > 1:
                    all_args[arg] = 'Any'
                elif type_length == 1:
                    all_args[arg] = get_type(types[0])
            return all_args

        def get_args_types(self, qualified_name: str) -> Dict:
            if False:
                while True:
                    i = 10
            return self.consolidate_types(qualified_name)

    class JitTypeTraceConfig(monkeytype.config.Config):

        def __init__(self, s: JitTypeTraceStore):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.s = s

        def trace_logger(self) -> JitTypeTraceStoreLogger:
            if False:
                while True:
                    i = 10
            'Return a JitCallTraceStoreLogger that logs to the configured trace store.'
            return JitTypeTraceStoreLogger(self.trace_store())

        def trace_store(self) -> CallTraceStore:
            if False:
                for i in range(10):
                    print('nop')
            return self.s

        def code_filter(self) -> Optional[CodeFilter]:
            if False:
                while True:
                    i = 10
            return jit_code_filter
else:

    class JitTypeTraceStoreLogger:

        def __init__(self):
            if False:
                print('Hello World!')
            pass

    class JitTypeTraceStore:

        def __init__(self):
            if False:
                return 10
            self.trace_records = None

    class JitTypeTraceConfig:

        def __init__(self):
            if False:
                while True:
                    i = 10
            pass
    monkeytype_trace = None

def jit_code_filter(code: CodeType) -> bool:
    if False:
        i = 10
        return i + 15
    "Codefilter for Torchscript to trace forward calls.\n\n    The custom CodeFilter is required while scripting a FX Traced forward calls.\n    FX Traced forward calls have `code.co_filename` start with '<' which is used\n    to exclude tracing of stdlib and site-packages in the default code filter.\n    Since we need all forward calls to be traced, this custom code filter\n    checks for code.co_name to be 'forward' and enables tracing for all such calls.\n    The code filter is similar to default code filter for monkeytype and\n    excludes tracing of stdlib and site-packages.\n    "
    if code.co_name != 'forward' and (not code.co_filename or code.co_filename[0] == '<'):
        return False
    filename = pathlib.Path(code.co_filename).resolve()
    return not any((_startswith(filename, lib_path) for lib_path in LIB_PATHS))