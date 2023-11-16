"""Operations for automatic batching and unbatching."""
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import gen_batch_ops
from tensorflow.python.ops.gen_batch_ops import *
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

@tf_export('nondifferentiable_batch_function')
def batch_function(num_batch_threads, max_batch_size, batch_timeout_micros, allowed_batch_sizes=None, max_enqueued_batches=10, autograph=True, enable_large_batch_splitting=True):
    if False:
        return 10
    "Batches the computation done by the decorated function.\n\n  So, for example, in the following code\n\n  ```python\n  @batch_function(1, 2, 3)\n  def layer(a):\n    return tf.matmul(a, a)\n\n  b = layer(w)\n  ```\n\n  if more than one session.run call is simultaneously trying to compute `b`\n  the values of `w` will be gathered, non-deterministically concatenated\n  along the first axis, and only one thread will run the computation. See the\n  documentation of the `Batch` op for more details.\n\n  Assumes that all arguments of the decorated function are Tensors which will\n  be batched along their first dimension.\n\n  SparseTensor is not supported. The return value of the decorated function\n  must be a Tensor or a list/tuple of Tensors.\n\n  Args:\n    num_batch_threads: Number of scheduling threads for processing batches\n     of work. Determines the number of batches processed in parallel.\n    max_batch_size: Batch sizes will never be bigger than this.\n    batch_timeout_micros: Maximum number of microseconds to wait before\n     outputting an incomplete batch.\n    allowed_batch_sizes: Optional list of allowed batch sizes. If left empty,\n     does nothing. Otherwise, supplies a list of batch sizes, causing the op\n     to pad batches up to one of those sizes. The entries must increase\n     monotonically, and the final entry must equal max_batch_size.\n    max_enqueued_batches: The maximum depth of the batch queue. Defaults to 10.\n    autograph: Whether to use autograph to compile python and eager style code\n     for efficient graph-mode execution.\n    enable_large_batch_splitting: The value of this option doesn't affect\n     processing output given the same input; it affects implementation details\n     as stated below: 1. Improve batching efficiency by eliminating unnecessary\n     adding. 2.`max_batch_size` specifies the limit of input and\n     `allowed_batch_sizes` specifies the limit of a task to be processed. API\n     user can give an input of size 128 when 'max_execution_batch_size'\n     is 32 -> implementation can split input of 128 into 4 x 32, schedule\n     concurrent processing, and then return concatenated results corresponding\n     to 128.\n\n  Returns:\n    The decorated function will return the unbatched computation output Tensors.\n  "

    def decorator(fn):
        if False:
            for i in range(10):
                print('nop')

        def decorated(*args):
            if False:
                for i in range(10):
                    print('nop')

            @def_function.function(autograph=autograph)
            def computation(*computation_args):
                if False:
                    while True:
                        i = 10
                return fn(*computation_args)
            computation = computation.get_concrete_function(*[tensor.TensorSpec(dtype=x.dtype, shape=x.shape, name='batch_' + str(i)) for (i, x) in enumerate(args)])
            with ops.name_scope('batch') as name:
                for a in args:
                    if not isinstance(a, tensor.Tensor):
                        raise ValueError(f'All arguments to functions decorated with `batch_function`  are supposed to be Tensors; found {a!r}.')
                outputs = gen_batch_ops.batch_function(num_batch_threads=num_batch_threads, max_batch_size=max_batch_size, batch_timeout_micros=batch_timeout_micros, allowed_batch_sizes=allowed_batch_sizes, max_enqueued_batches=max_enqueued_batches, shared_name=name, enable_large_batch_splitting=enable_large_batch_splitting, f=computation, in_tensors=list(args), captured_tensors=computation.captured_inputs, Tout=[o.dtype for o in computation.outputs])
                return nest.pack_sequence_as(computation.structured_outputs, outputs, expand_composites=True)
        return decorated
    return decorator