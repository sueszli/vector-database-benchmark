"""Batch gather operations for RaggedTensors."""
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_gather_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch

@dispatch.dispatch_for_api(array_ops.batch_gather)
def batch_gather(params: ragged_tensor.RaggedOrDense, indices: ragged_tensor.RaggedOrDense, name=None):
    if False:
        while True:
            i = 10
    "Gathers slices from `params` according to `indices` with batch dims.\n\n  This operation is similar to `gather`, but it assumes that the leading `N`\n  dimensions of `indices` and `params` are batch dimensions, and performs a\n  gather within each batch.  In particular, when using this operation with `N`\n  batch dimensions `B1...BN`:\n\n  * `indices` has shape `[B1...BN, I]`\n  * `params` has shape `[B1...BN, P1...PM]`.\n  * `result` has shape `[B1...BN, I, P2...PM]`.\n  * `result[b1...bN, i, p2...pM] =\n    params[b1...bN, indices[b1...bN, i], p2...pM]`\n\n  Args:\n    params: A potentially ragged tensor with shape `[B1...BN, P1...PM]` (`N>=0`,\n      `M>0`).\n    indices: A potentially ragged tensor with shape `[B1...BN, I]` (`N>=0`).\n    name: A name for the operation (optional).\n\n  Returns:\n    A potentially ragged tensor with shape `[B1...BN, I, P2...PM]`.\n    `result.ragged_rank = max(indices.ragged_rank, params.ragged_rank)`.\n\n  #### Example:\n\n  >>> params = tf.ragged.constant([['a', 'b', 'c'], ['d'], [], ['e']])\n  >>> indices = tf.ragged.constant([[1, 2, 0], [], [], [0, 0]])\n  >>> tf.compat.v1.batch_gather(params, indices)\n  <tf.RaggedTensor [[b'b', b'c', b'a'], [], [], [b'e', b'e']]>\n  "
    return ragged_gather_ops.gather(params, indices, batch_dims=-1, name=name)