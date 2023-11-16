from enum import Enum
from typing import Callable
import ivy
import importlib

class BackendHandlerMode(Enum):
    WithBackend = 0
    SetBackend = 1

class WithBackendContext:

    def __init__(self, backend) -> None:
        if False:
            return 10
        self.backend = backend

    def __enter__(self):
        if False:
            print('Hello World!')
        return ivy.with_backend(self.backend)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            print('Hello World!')
        return
update_backend: Callable = ivy.utils.backend.ContextManager

class BackendHandler:
    _context = WithBackendContext
    _ctx_flag = 0

    @classmethod
    def _update_context(cls, mode: BackendHandlerMode):
        if False:
            print('Hello World!')
        if mode == BackendHandlerMode.WithBackend:
            cls._context = WithBackendContext
            cls._ctx_flag = 0
        elif mode == BackendHandlerMode.SetBackend:
            cls._context = ivy.utils.backend.ContextManager
            cls._ctx_flag = 1
        else:
            raise ValueError(f'Unknown backend handler mode! {mode}')

    @classmethod
    def update_backend(cls, backend):
        if False:
            return 10
        return cls._context(backend)

def get_frontend_config(frontend: str):
    if False:
        i = 10
        return i + 15
    config_module = importlib.import_module(f'ivy_tests.test_ivy.test_frontends.config.{frontend}')
    return config_module.get_config()