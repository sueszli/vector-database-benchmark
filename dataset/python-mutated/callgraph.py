from .space import AutoObject
from enum import Enum
from bigdl.nano.utils.common import invalidInputError

class CALLTYPE(Enum):
    """Type of Function Calls."""
    LAYER_CALL = 1
    FUNC_CALL = 2
    FUNC_SLICE = 3

class CallCache(object):
    """
    A data structure to cache the sequence of functional calls.

    Each autoobject contains a callcache object. The call sequences will
    be gradually passed down to the last function call.
    Internally, it containes a callqueue which collects the function calls in    order and a tensors table which is used to collect the resulting    tensors from function execution.
    """

    def __init__(self):
        if False:
            return 10
        'Create a Call Cache.'
        self.callqueue_ = []
        self.tensors_ = dict()
        self.skip = False

    def update_tensors(self, other):
        if False:
            print('Hello World!')
        'Merge the tensortable from another CallCache.'
        self.tensors_.update(other.tensors_)

    def update_calls(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Merge the callqueues from another CallCache.'
        self.callqueue_ += other.callqueue_

    def append_call(self, caller, arguments, type):
        if False:
            for i in range(10):
                print('nop')
        'Add one function call into the call queue.'
        self.callqueue_.append((type, caller, arguments))

    def add_tensor(self, node, tensor):
        if False:
            while True:
                i = 10
        'Update the resulting tensor a layer call execution.'
        self.tensors_[node] = tensor

    def get_tensor(self, n):
        if False:
            i = 10
            return i + 15
        'Get the resulting tensor of a layer call execution.'
        if isinstance(n, list):
            return [self.tensors_.get(one_n, None) for one_n in n]
        else:
            return self.tensors_.get(n, None)

    @property
    def calls(self):
        if False:
            i = 10
            return i + 15
        'Get all the call queue.'
        return self.callqueue_

    @staticmethod
    def create():
        if False:
            for i in range(10):
                print('nop')
        'Create a call Cache.'
        cache = CallCache()
        return cache

    @staticmethod
    def update(arguments, current, ctype=CALLTYPE.LAYER_CALL):
        if False:
            while True:
                i = 10
        "\n        Update the current autoobject's callcache from its input arguments.\n\n        If the argument is also an autoobject, merge the callcache of\n        the argument into current autoobject's callcache.\n\n        :param arguments: input arguments of current layers\n        :param current: the current autoobject\n        :param ctype: the type of current call. Defaults to CALLTYPE.LAYER_CALL\n        "

        def _update_cache_from_input(cache, inp):
            if False:
                print('Hello World!')
            'Loop over all arguments to find any autoobjects            in input and merge down the callcache.'
            if isinstance(inp, AutoObject):
                invalidInputError(inp._callgraph is not None, 'inp._callgraph cannot be none')
                input_callgraph = inp._callgraph
                if not input_callgraph.skip:
                    cache.update_tensors(input_callgraph)
                    cache.update_calls(input_callgraph)
                    input_callgraph.skip = True
            elif isinstance(inp, list) or isinstance(inp, tuple):
                for item in inp:
                    _update_cache_from_input(cache, item)
            elif isinstance(inp, dict):
                for (_, item) in inp.items():
                    _update_cache_from_input(cache, item)
            else:
                pass
        cur_cache = CallCache.create()
        if ctype == CALLTYPE.LAYER_CALL or ctype == CALLTYPE.FUNC_CALL:
            _update_cache_from_input(cur_cache, arguments)
            cur_cache.append_call(current, arguments, ctype)
        elif ctype == CALLTYPE.FUNC_SLICE:
            (source, slice_args) = arguments
            _update_cache_from_input(cur_cache, source)
            cur_cache.append_call(current, arguments, CALLTYPE.FUNC_SLICE)
        else:
            invalidInputError(False, 'Unexpected CallType: %s' % ctype)
        return cur_cache

    @staticmethod
    def execute(inputs, outputs, trial, backend):
        if False:
            print('Hello World!')
        '\n        Execute the function calls and construct the tensor graph.\n\n        :param inputs: model input\n        :param outputs: model outputs\n        :param trial: the current trial which provides the sampled\n            hyperparameters.\n        '

        def _replace_autoobj(n, cache):
            if False:
                print('Hello World!')
            if isinstance(n, AutoObject):
                new_n = cache.get_tensor(n)
            else:
                new_n = n
            return new_n

        def _process_arguments(arguments, cache):
            if False:
                while True:
                    i = 10
            if isinstance(arguments, list):
                new_arguments = [_process_arguments(arg, cache) for arg in arguments]
            elif isinstance(arguments, tuple):
                lst = [_process_arguments(arg, cache) for arg in arguments]
                new_arguments = tuple(lst)
            elif isinstance(arguments, dict):
                new_arguments = arguments.copy()
                for (name, arg) in new_arguments.items():
                    new_arg = _process_arguments(arg, cache)
                    new_arguments[name] = new_arg
            else:
                new_arguments = _replace_autoobj(arguments, cache)
            return new_arguments
        out_cache = outputs._callgraph
        for (call_type, caller, arguments) in out_cache.calls:
            if call_type == CALLTYPE.LAYER_CALL:
                new_arguments = _process_arguments(arguments, out_cache)
                invalidInputError(isinstance(caller, AutoObject), 'caller should be AutoObject')
                instance = backend.instantiate(trial, caller)
                out_tensor = instance(new_arguments)
            elif call_type == CALLTYPE.FUNC_SLICE:
                (source, slice_args) = arguments
                (slice_args, slice_kwargs) = slice_args
                source_tensor = out_cache.get_tensor(source)
                out_tensor = source_tensor.__getitem__(*slice_args, **slice_kwargs)
            elif call_type == CALLTYPE.FUNC_CALL:
                new_arguments = _process_arguments(arguments, out_cache)
                invalidInputError(isinstance(caller, AutoObject), 'caller should be AutoObject')
                (caller.args, caller.kwargs) = new_arguments
                out_tensor = backend.instantiate(trial, caller)
            else:
                invalidInputError(False, 'Unexpected CallType: %s' % type)
            out_cache.add_tensor(caller, out_tensor)
        out_tensors = out_cache.get_tensor(outputs)
        if isinstance(inputs, list):
            in_tensors = [out_cache.get_tensor(inp) for inp in inputs]
        else:
            in_tensors = out_cache.get_tensor(inputs)
        return (in_tensors, out_tensors)

    def plot(self, save_path=None):
        if False:
            while True:
                i = 10
        'Dump the call cache for debugging purpose.'
        print('dumping call cache...............start')
        print('===============dumpping call queue============')
        for call in self.callqueue_:
            print(call)
        print('===============dumpping tensors============')
        print(self.tensors_)
        print('dumping call cache...............end')