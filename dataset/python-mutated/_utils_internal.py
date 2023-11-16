import logging
import os
import sys
import tempfile
from typing import Any, Dict
import torch
log = logging.getLogger(__name__)
if torch._running_with_deploy():
    torch_parent = ''
elif os.path.basename(os.path.dirname(__file__)) == 'shared':
    torch_parent = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
else:
    torch_parent = os.path.dirname(os.path.dirname(__file__))

def get_file_path(*path_components: str) -> str:
    if False:
        i = 10
        return i + 15
    return os.path.join(torch_parent, *path_components)

def get_file_path_2(*path_components: str) -> str:
    if False:
        while True:
            i = 10
    return os.path.join(*path_components)

def get_writable_path(path: str) -> str:
    if False:
        while True:
            i = 10
    if os.access(path, os.W_OK):
        return path
    return tempfile.mkdtemp(suffix=os.path.basename(path))

def prepare_multiprocessing_environment(path: str) -> None:
    if False:
        return 10
    pass

def resolve_library_path(path: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    return os.path.realpath(path)

def throw_abstract_impl_not_imported_error(opname, module, context):
    if False:
        i = 10
        return i + 15
    if module in sys.modules:
        raise NotImplementedError(f'{opname}: We could not find the abstract impl for this operator. ')
    else:
        raise NotImplementedError(f"{opname}: We could not find the abstract impl for this operator. The operator specified that you may need to import the '{module}' Python module to load the abstract impl. {context}")

def signpost_event(category: str, name: str, parameters: Dict[str, Any]):
    if False:
        print('Hello World!')
    log.info('%s %s: %r', category, name, parameters)

def log_compilation_event(metrics):
    if False:
        for i in range(10):
            print('nop')
    log.info('%s', metrics)
TEST_MASTER_ADDR = '127.0.0.1'
TEST_MASTER_PORT = 29500
USE_GLOBAL_DEPS = True
USE_RTLD_GLOBAL_WITH_LIBTORCH = False