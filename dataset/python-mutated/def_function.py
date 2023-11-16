"""Experimental impl of tf.function using unified APIs, for testing only."""
from tensorflow.python.framework.experimental import _unified_api
from tensorflow.python.framework.experimental import context_stack as context_lib
from tensorflow.python.util import nest
NewTracingContext = _unified_api.NewTracingContext

class Function(object):
    """Helper for tf.function."""

    def __init__(self, func, name=None):
        if False:
            for i in range(10):
                print('nop')
        self._python_func = func
        self.name = name or func.__name__

    def __call__(self, *args, **kwargs):
        if False:
            return 10
        flat_args = nest.flatten(args, expand_composites=True)
        flat_kwargs = nest.flatten(kwargs, expand_composites=True)
        all_args = flat_args + flat_kwargs
        outer_ctx = context_lib.get_default()
        ctx = NewTracingContext(self.name)
        with context_lib.set_default(ctx):
            inputs = [ctx.AddParameter(arg.DataType()) for arg in all_args]
            structured_args = nest.pack_sequence_as(args, inputs[:len(flat_args)])
            structured_kwargs = nest.pack_sequence_as(kwargs, inputs[len(flat_args):])
            structured_outputs = self._python_func(*structured_args, **structured_kwargs)
            py_outputs = nest.flatten(structured_outputs, expand_composites=True)
            num_outputs = len(py_outputs)
            finalized_f = ctx.Finalize(py_outputs)
            outer_ctx.RegisterFunction(finalized_f)
        call_op = outer_ctx.CreateOperation(self.name, '')
        call_op.SetOpName(self.name)
        for arg in all_args:
            call_op.AddInput(arg)
        call_op_outputs = call_op.Execute(num_outputs)
        outer_ctx.RemoveFunction(self.name)
        return nest.pack_sequence_as(structured_outputs, call_op_outputs)

def function(func):
    if False:
        for i in range(10):
            print('nop')
    return Function(func)