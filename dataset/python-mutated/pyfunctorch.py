from abc import ABC, abstractmethod
import contextlib
from typing import Any
import torch
import torch.utils._pytree as pytree
from torch._C._functorch import TransformType, RandomnessType, CInterpreter, CGradInterpreterPtr, CFunctionalizeInterpreterPtr, CVmapInterpreterPtr, CJvpInterpreterPtr, pop_dynamic_layer_stack, push_dynamic_layer_stack
from torch.autograd.forward_ad import _set_fwd_grad_enabled
"\nThis file contains the functorch integration with PyDispatcher.\n\nPyDispatcher does not understand functorch's DynamicLayerStack dispatching\nlogic because it is entirely implemented in C++ in the fallbacks for two\ndispatch keys, FuncTorchDynamicLayer{Front, Back}Mode (PyDispatcher is unable\nto directly reuse C++ boxed fallbacks).\n\nInstead of trying to hammer PyDispatcher into understanding those fallbacks,\nwe re-implement the logic of peeking the top of the stack for an interpreter,\nselecting the interpreter to dispatch on, etc, in Python. This leads to a\nsimpler design.\n\nThe main difference between C++ functorch and PyDispatcher's functorch logic\nis that:\n- C++ functorch needs to manually tweak dispatch keys to ping-pong between\n  DynamicLayerFrontMode and DynamicLayerBackMode.\n- PyDispatcher's functorch logic pops an Interpreter from the top of the stack\n  and asks it to execute the rule associated with the Interpreter.\n\nIn C++ we do the ping-pong because e.g. vmap rules are associated with the\nbatched DispatchKey, but in PyDispatcher we are able to avoid this by asking\nthe user to register a batching rule directly to a transform that an\ninterpreter then invokes.\n"

class FuncTorchInterpreter(ABC):

    def __init__(self, cptr: Any):
        if False:
            print('Hello World!')
        self._cptr = cptr

    @abstractmethod
    def process(self, op, args, kwargs):
        if False:
            print('Hello World!')
        pass

    def lower(self):
        if False:
            while True:
                i = 10
        return temporarily_pop_interpreter_stack()

    def level(self):
        if False:
            print('Hello World!')
        return self._cptr.level()

    def key(self):
        if False:
            while True:
                i = 10
        return self._cptr.key()

@contextlib.contextmanager
def temporarily_pop_interpreter_stack():
    if False:
        return 10
    try:
        saved = pop_dynamic_layer_stack()
        yield
    finally:
        push_dynamic_layer_stack(saved)

class VmapInterpreter(FuncTorchInterpreter):

    def __init__(self, cdata: CInterpreter):
        if False:
            print('Hello World!')
        assert cdata.key() == TransformType.Vmap
        self._cdata = cdata
        self._cptr = CVmapInterpreterPtr(cdata)

    def process(self, op, args, kwargs):
        if False:
            while True:
                i = 10
        kernel = op.functorch_table[TransformType.Vmap]
        return kernel(self, *args, **kwargs)

    def batch_size(self):
        if False:
            i = 10
            return i + 15
        return self._cptr.batchSize()

    def randomness(self):
        if False:
            return 10
        typ = self._cptr.randomness()
        if typ == RandomnessType.Error:
            return 'error'
        elif typ == RandomnessType.Same:
            return 'same'
        elif typ == RandomnessType.Different:
            return 'different'
        raise RuntimeError(f'Unknown RandomnessType: {typ}')

@contextlib.contextmanager
def nested(*contexts):
    if False:
        return 10
    with contextlib.ExitStack() as stack:
        for ctx in contexts:
            stack.enter_context(ctx)
        yield contexts

class GradInterpreter(FuncTorchInterpreter):

    def __init__(self, cdata: CInterpreter):
        if False:
            while True:
                i = 10
        assert cdata.key() == TransformType.Grad
        self._cdata = cdata
        self._cptr = CGradInterpreterPtr(cdata)

    def lift(self, args, kwargs):
        if False:
            return 10
        (args, kwargs) = pytree.tree_map_only(torch.Tensor, self._cptr.lift, [args, kwargs])
        return (args, kwargs)

    def process(self, op, args, kwargs):
        if False:
            print('Hello World!')
        kernel = op.functorch_table[TransformType.Grad]
        (args, kwargs) = self.lift(args, kwargs)
        return kernel(self, *args, **kwargs)

    def lower(self):
        if False:
            while True:
                i = 10
        prev_grad_mode = self.prev_grad_mode()
        if not prev_grad_mode:
            return nested(torch.no_grad(), super().lower())
        return super().lower()

    def prev_grad_mode(self):
        if False:
            while True:
                i = 10
        return self._cptr.prevGradMode()

class JvpInterpreter(FuncTorchInterpreter):

    def __init__(self, cdata: CInterpreter):
        if False:
            while True:
                i = 10
        assert cdata.key() == TransformType.Jvp
        self._cdata = cdata
        self._cptr = CJvpInterpreterPtr(cdata)

    def lift(self, args, kwargs):
        if False:
            print('Hello World!')
        (args, kwargs) = pytree.tree_map_only(torch.Tensor, self._cptr.lift, [args, kwargs])
        return (args, kwargs)

    def process(self, op, args, kwargs):
        if False:
            for i in range(10):
                print('nop')
        kernel = op.functorch_table[TransformType.Jvp]
        (args, kwargs) = self.lift(args, kwargs)
        return kernel(self, *args, **kwargs)

    def lower(self):
        if False:
            i = 10
            return i + 15
        prev_fwd_grad_mode = self.prev_fwd_grad_mode()
        if not prev_fwd_grad_mode:
            return nested(_set_fwd_grad_enabled(False), super().lower())
        return super().lower()

    def prev_fwd_grad_mode(self):
        if False:
            i = 10
            return i + 15
        return self._cptr.prevFwdGradMode()

class FunctionalizeInterpreter(FuncTorchInterpreter):

    def __init__(self, cdata: CInterpreter):
        if False:
            i = 10
            return i + 15
        assert cdata.key() == TransformType.Functionalize
        self._cdata = cdata
        self._cptr = CFunctionalizeInterpreterPtr(cdata)

    def process(self, op, args, kwargs):
        if False:
            i = 10
            return i + 15
        kernel = op.functorch_table[TransformType.Functionalize]
        return kernel(self, *args, **kwargs)

    def functionalize_add_back_views(self):
        if False:
            for i in range(10):
                print('nop')
        return self._cptr.functionalizeAddBackViews()

def coerce_cinterpreter(cinterpreter: CInterpreter) -> FuncTorchInterpreter:
    if False:
        return 10
    key = cinterpreter.key()
    if key == TransformType.Grad:
        return GradInterpreter(cinterpreter)
    if key == TransformType.Vmap:
        return VmapInterpreter(cinterpreter)
    if key == TransformType.Jvp:
        return JvpInterpreter(cinterpreter)
    if key == TransformType.Functionalize:
        return FunctionalizeInterpreter(cinterpreter)
    raise RuntimeError(f'NYI: PyDispatcher has not implemented support for {key}')

def retrieve_current_functorch_interpreter():
    if False:
        while True:
            i = 10
    interpreter = torch._C._functorch.peek_interpreter_stack()
    assert interpreter is not None
    return coerce_cinterpreter(interpreter)

def dispatch_functorch(op, args, kwargs):
    if False:
        print('Hello World!')
    interpreter = retrieve_current_functorch_interpreter()
    (args, kwargs) = pytree.tree_map_only(torch.Tensor, torch._C._functorch.unwrap_if_dead, (args, kwargs))
    return interpreter.process(op, args, kwargs)