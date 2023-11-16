"""Distributed saving of a dataset to disk."""
from tensorflow.core.protobuf import snapshot_pb2
from tensorflow.python.ops import gen_experimental_dataset_ops
from tensorflow.python.saved_model import nested_structure_coder

def distributed_save(dataset, path, dispatcher_address, compression='AUTO'):
    if False:
        i = 10
        return i + 15
    'Initiates the process of distributedly saving a dataset to disk.\n\n  Args:\n    dataset: The `tf.data.Dataset` to save.\n    path: A string indicating the filepath of the directory to which to save\n      `dataset`.\n    dispatcher_address: A string indicating the address of the dispatcher for\n      the tf.data service instance used to save `dataset`.\n    compression: (Optional.) A string indicating whether and how to compress the\n      `dataset` materialization.  If `"AUTO"`, the tf.data runtime decides which\n      algorithm to use.  If `"GZIP"` or `"SNAPPY"`, that specific algorithm is\n      used.  If `None`, the `dataset` materialization is not compressed.\n\n  Returns:\n    An operation which when executed performs the distributed save.\n\n  Raises:\n    ValueError: If `dispatcher_address` is invalid.\n  '
    if not isinstance(dispatcher_address, str):
        raise ValueError(f'`dispatcher_address` must be a string, but is a {type(dispatcher_address)} ({dispatcher_address}')
    if not dispatcher_address:
        raise ValueError('`dispatcher_address` must not be empty')
    metadata = snapshot_pb2.DistributedSnapshotMetadata(element_spec=nested_structure_coder.encode_structure(dataset.element_spec).SerializeToString(), compression=compression)
    return gen_experimental_dataset_ops.distributed_save(dataset._variant_tensor, directory=path, address=dispatcher_address, metadata=metadata.SerializeToString())