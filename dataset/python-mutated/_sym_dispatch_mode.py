from typing import List, Type
__all__ = ['SymDispatchMode', 'handle_sym_dispatch', 'sym_function_mode']
SYM_FUNCTION_MODE = None

class SymDispatchMode:

    def __sym_dispatch__(self, func, types, args, kwargs):
        if False:
            return 10
        raise NotImplementedError()

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        global SYM_FUNCTION_MODE
        old = SYM_FUNCTION_MODE
        if hasattr(self, 'inner'):
            raise RuntimeError(f'{self} has already been used as a mode. Please use a fresh version')
        else:
            self.inner = old
        SYM_FUNCTION_MODE = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            for i in range(10):
                print('nop')
        global SYM_FUNCTION_MODE
        SYM_FUNCTION_MODE = self.inner

def handle_sym_dispatch(func, args, kwargs):
    if False:
        while True:
            i = 10
    global SYM_FUNCTION_MODE
    mode = sym_function_mode()
    assert mode
    SYM_FUNCTION_MODE = mode.inner
    try:
        types: List[Type] = []
        return mode.__sym_dispatch__(func, types, args, kwargs)
    finally:
        SYM_FUNCTION_MODE = mode

def sym_function_mode():
    if False:
        print('Hello World!')
    return SYM_FUNCTION_MODE