"""Inputs and Readers.

See the [Inputs and
Readers](https://tensorflow.org/api_guides/python/io_ops) guide.
"""
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops.gen_io_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

def _save(filename, tensor_names, tensors, tensor_slices=None, name='save'):
    if False:
        i = 10
        return i + 15
    'Save a list of tensors to a file with given names.\n\n  Example usage without slice info:\n    Save("/foo/bar", ["w", "b"], [w, b])\n\n  Example usage with slices:\n    Save("/foo/bar", ["w", "w"], [slice0, slice1],\n         tensor_slices=["4 10 0,2:-", "4 10 2,2:-"])\n\n  Args:\n    filename: the file name of the sstable.\n    tensor_names: a list of strings.\n    tensors: the list of tensors to be saved.\n    tensor_slices: Optional list of strings to specify the shape and slices of\n      a larger virtual tensor that each tensor is a part of.  If not specified\n      each tensor is saved as a full slice.\n    name: string.  Optional name for the op.\n\n  Requires:\n    The length of tensors should match the size of tensor_names and of\n    tensor_slices.\n\n  Returns:\n    An Operation that saves the tensors.\n  '
    if tensor_slices is None:
        return gen_io_ops.save(filename, tensor_names, tensors, name=name)
    else:
        return gen_io_ops.save_slices(filename, tensor_names, tensor_slices, tensors, name=name)

def _restore_slice(file_pattern, tensor_name, shape_and_slice, tensor_type, name='restore_slice', preferred_shard=-1):
    if False:
        for i in range(10):
            print('nop')
    'Restore a tensor slice from a set of files with a given pattern.\n\n  Example usage:\n    RestoreSlice("/foo/bar-?????-of-?????", "w", "10 10 0,2:-", DT_FLOAT)\n\n  Args:\n    file_pattern: the file pattern used to match a set of checkpoint files.\n    tensor_name: the name of the tensor to restore.\n    shape_and_slice: the shape-and-slice spec of the slice.\n    tensor_type: the type of the tensor to restore.\n    name: string.  Optional name for the op.\n    preferred_shard: Int. Optional shard to open first in the checkpoint file.\n\n  Returns:\n    A tensor of type "tensor_type".\n  '
    base_type = dtypes.as_dtype(tensor_type).base_dtype
    return gen_io_ops.restore_slice(file_pattern, tensor_name, shape_and_slice, base_type, preferred_shard, name=name)

@tf_export('io.read_file', v1=['io.read_file', 'read_file'])
def read_file(filename, name=None):
    if False:
        while True:
            i = 10
    'Reads the contents of file.\n\n  This operation returns a tensor with the entire contents of the input\n  filename. It does not do any parsing, it just returns the contents as\n  they are. Usually, this is the first step in the input pipeline.\n\n  Example:\n\n  >>> with open("/tmp/file.txt", "w") as f:\n  ...   f.write("asdf")\n  ...\n  4\n  >>> tf.io.read_file("/tmp/file.txt")\n  <tf.Tensor: shape=(), dtype=string, numpy=b\'asdf\'>\n\n  Example of using the op in a function to read an image, decode it and reshape\n  the tensor containing the pixel data:\n\n  >>> @tf.function\n  ... def load_image(filename):\n  ...   raw = tf.io.read_file(filename)\n  ...   image = tf.image.decode_png(raw, channels=3)\n  ...   # the `print` executes during tracing.\n  ...   print("Initial shape: ", image.shape)\n  ...   image.set_shape([28, 28, 3])\n  ...   print("Final shape: ", image.shape)\n  ...   return image\n\n  Args:\n    filename: string. filename to read from.\n    name: string.  Optional name for the op.\n\n  Returns:\n    A tensor of dtype "string", with the file contents.\n  '
    return gen_io_ops.read_file(filename, name)

@tf_export('io.serialize_tensor', v1=['io.serialize_tensor', 'serialize_tensor'])
def serialize_tensor(tensor, name=None):
    if False:
        print('Hello World!')
    'Transforms a Tensor into a serialized TensorProto proto.\n\n  This operation transforms data in a `tf.Tensor` into a `tf.Tensor` of type\n  `tf.string` containing the data in a binary string in little-endian format.\n  This operation can transform scalar data and linear arrays, but it is most\n  useful in converting multidimensional arrays into a format accepted by binary\n  storage formats such as a `TFRecord` or `tf.train.Example`.\n\n  See also:\n  - `tf.io.parse_tensor`: inverse operation of `tf.io.serialize_tensor` that\n  transforms a scalar string containing a serialized Tensor in little-endian\n  format into a Tensor of a specified type.\n  - `tf.ensure_shape`: `parse_tensor` cannot statically determine the shape of\n  the parsed tensor. Use `tf.ensure_shape` to set the static shape when running\n  under a `tf.function`\n  - `.SerializeToString`, serializes a proto to a binary-string\n\n  Example of serializing scalar data:\n\n  >>> t = tf.constant(1)\n  >>> tf.io.serialize_tensor(t)\n  <tf.Tensor: shape=(), dtype=string, numpy=b\'\\x08...\\x00\'>\n\n  Example of storing non-scalar data into a `tf.train.Example`:\n\n  >>> t1 = [[1, 2]]\n  >>> t2 = [[7, 8]]\n  >>> nonscalar = tf.concat([t1, t2], 0)\n  >>> nonscalar\n  <tf.Tensor: shape=(2, 2), dtype=int32, numpy=\n  array([[1, 2],\n         [7, 8]], dtype=int32)>\n\n  Serialize the data using `tf.io.serialize_tensor`.\n\n  >>> serialized_nonscalar = tf.io.serialize_tensor(nonscalar)\n  >>> serialized_nonscalar\n  <tf.Tensor: shape=(), dtype=string, numpy=b\'\\x08...\\x00\'>\n\n  Store the data in a `tf.train.Feature`.\n\n  >>> feature_of_bytes = tf.train.Feature(\n  ...   bytes_list=tf.train.BytesList(value=[serialized_nonscalar.numpy()]))\n  >>> feature_of_bytes\n  bytes_list {\n    value: "\\010...\\000"\n  }\n\n  Put the `tf.train.Feature` message into a `tf.train.Example`.\n\n  >>> features_for_example = {\n  ...   \'feature0\': feature_of_bytes\n  ... }\n  >>> example_proto = tf.train.Example(\n  ...   features=tf.train.Features(feature=features_for_example))\n  >>> example_proto\n  features {\n    feature {\n      key: "feature0"\n      value {\n        bytes_list {\n          value: "\\010...\\000"\n        }\n      }\n    }\n  }\n\n  Args:\n    tensor: A `tf.Tensor`.\n    name: string.  Optional name for the op.\n\n  Returns:\n    A Tensor of dtype string.\n  '
    return gen_parsing_ops.serialize_tensor(tensor, name)

@tf_export(v1=['ReaderBase'])
class ReaderBase:
    """Base class for different Reader types, that produce a record every step.

  Conceptually, Readers convert string 'work units' into records (key,
  value pairs).  Typically the 'work units' are filenames and the
  records are extracted from the contents of those files.  We want a
  single record produced per step, but a work unit can correspond to
  many records.

  Therefore we introduce some decoupling using a queue.  The queue
  contains the work units and the Reader dequeues from the queue when
  it is asked to produce a record (via Read()) but it has finished the
  last work unit.

  @compatibility(eager)
  Readers are not compatible with eager execution. Instead, please
  use `tf.data` to get data into your model.
  @end_compatibility
  """

    def __init__(self, reader_ref, supports_serialize=False):
        if False:
            return 10
        'Creates a new ReaderBase.\n\n    Args:\n      reader_ref: The operation that implements the reader.\n      supports_serialize: True if the reader implementation can\n        serialize its state.\n\n    Raises:\n      RuntimeError: If eager execution is enabled.\n    '
        if context.executing_eagerly():
            raise RuntimeError('Readers are not supported when eager execution is enabled. Instead, please use tf.data to get data into your model.')
        self._reader_ref = reader_ref
        self._supports_serialize = supports_serialize

    @property
    def reader_ref(self):
        if False:
            while True:
                i = 10
        'Op that implements the reader.'
        return self._reader_ref

    def read(self, queue, name=None):
        if False:
            while True:
                i = 10
        'Returns the next record (key, value) pair produced by a reader.\n\n    Will dequeue a work unit from queue if necessary (e.g. when the\n    Reader needs to start reading from a new file since it has\n    finished with the previous file).\n\n    Args:\n      queue: A Queue or a mutable string Tensor representing a handle\n        to a Queue, with string work items.\n      name: A name for the operation (optional).\n\n    Returns:\n      A tuple of Tensors (key, value).\n      key: A string scalar Tensor.\n      value: A string scalar Tensor.\n    '
        if isinstance(queue, tensor_lib.Tensor):
            queue_ref = queue
        else:
            queue_ref = queue.queue_ref
        if self._reader_ref.dtype == dtypes.resource:
            return gen_io_ops.reader_read_v2(self._reader_ref, queue_ref, name=name)
        else:
            old_queue_op = gen_data_flow_ops.fake_queue(queue_ref)
            return gen_io_ops.reader_read(self._reader_ref, old_queue_op, name=name)

    def read_up_to(self, queue, num_records, name=None):
        if False:
            while True:
                i = 10
        'Returns up to num_records (key, value) pairs produced by a reader.\n\n    Will dequeue a work unit from queue if necessary (e.g., when the\n    Reader needs to start reading from a new file since it has\n    finished with the previous file).\n    It may return less than num_records even before the last batch.\n\n    Args:\n      queue: A Queue or a mutable string Tensor representing a handle\n        to a Queue, with string work items.\n      num_records: Number of records to read.\n      name: A name for the operation (optional).\n\n    Returns:\n      A tuple of Tensors (keys, values).\n      keys: A 1-D string Tensor.\n      values: A 1-D string Tensor.\n    '
        if isinstance(queue, tensor_lib.Tensor):
            queue_ref = queue
        else:
            queue_ref = queue.queue_ref
        if self._reader_ref.dtype == dtypes.resource:
            return gen_io_ops.reader_read_up_to_v2(self._reader_ref, queue_ref, num_records, name=name)
        else:
            old_queue_op = gen_data_flow_ops.fake_queue(queue_ref)
            return gen_io_ops.reader_read_up_to(self._reader_ref, old_queue_op, num_records, name=name)

    def num_records_produced(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns the number of records this reader has produced.\n\n    This is the same as the number of Read executions that have\n    succeeded.\n\n    Args:\n      name: A name for the operation (optional).\n\n    Returns:\n      An int64 Tensor.\n\n    '
        if self._reader_ref.dtype == dtypes.resource:
            return gen_io_ops.reader_num_records_produced_v2(self._reader_ref, name=name)
        else:
            return gen_io_ops.reader_num_records_produced(self._reader_ref, name=name)

    def num_work_units_completed(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns the number of work units this reader has finished processing.\n\n    Args:\n      name: A name for the operation (optional).\n\n    Returns:\n      An int64 Tensor.\n    '
        if self._reader_ref.dtype == dtypes.resource:
            return gen_io_ops.reader_num_work_units_completed_v2(self._reader_ref, name=name)
        else:
            return gen_io_ops.reader_num_work_units_completed(self._reader_ref, name=name)

    def serialize_state(self, name=None):
        if False:
            return 10
        'Produce a string tensor that encodes the state of a reader.\n\n    Not all Readers support being serialized, so this can produce an\n    Unimplemented error.\n\n    Args:\n      name: A name for the operation (optional).\n\n    Returns:\n      A string Tensor.\n    '
        if self._reader_ref.dtype == dtypes.resource:
            return gen_io_ops.reader_serialize_state_v2(self._reader_ref, name=name)
        else:
            return gen_io_ops.reader_serialize_state(self._reader_ref, name=name)

    def restore_state(self, state, name=None):
        if False:
            return 10
        'Restore a reader to a previously saved state.\n\n    Not all Readers support being restored, so this can produce an\n    Unimplemented error.\n\n    Args:\n      state: A string Tensor.\n        Result of a SerializeState of a Reader with matching type.\n      name: A name for the operation (optional).\n\n    Returns:\n      The created Operation.\n    '
        if self._reader_ref.dtype == dtypes.resource:
            return gen_io_ops.reader_restore_state_v2(self._reader_ref, state, name=name)
        else:
            return gen_io_ops.reader_restore_state(self._reader_ref, state, name=name)

    @property
    def supports_serialize(self):
        if False:
            for i in range(10):
                print('nop')
        'Whether the Reader implementation can serialize its state.'
        return self._supports_serialize

    def reset(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Restore a reader to its initial clean state.\n\n    Args:\n      name: A name for the operation (optional).\n\n    Returns:\n      The created Operation.\n    '
        if self._reader_ref.dtype == dtypes.resource:
            return gen_io_ops.reader_reset_v2(self._reader_ref, name=name)
        else:
            return gen_io_ops.reader_reset(self._reader_ref, name=name)
ops.NotDifferentiable('ReaderRead')
ops.NotDifferentiable('ReaderReadUpTo')
ops.NotDifferentiable('ReaderNumRecordsProduced')
ops.NotDifferentiable('ReaderNumWorkUnitsCompleted')
ops.NotDifferentiable('ReaderSerializeState')
ops.NotDifferentiable('ReaderRestoreState')
ops.NotDifferentiable('ReaderReset')

@tf_export(v1=['WholeFileReader'])
class WholeFileReader(ReaderBase):
    """A Reader that outputs the entire contents of a file as a value.

  To use, enqueue filenames in a Queue.  The output of Read will
  be a filename (key) and the contents of that file (value).

  See ReaderBase for supported methods.

  @compatibility(eager)
  Readers are not compatible with eager execution. Instead, please
  use `tf.data` to get data into your model.
  @end_compatibility
  """

    @deprecation.deprecated(None, 'Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.map(tf.read_file)`.')
    def __init__(self, name=None):
        if False:
            return 10
        'Create a WholeFileReader.\n\n    Args:\n      name: A name for the operation (optional).\n    '
        rr = gen_io_ops.whole_file_reader_v2(name=name)
        super(WholeFileReader, self).__init__(rr, supports_serialize=True)
ops.NotDifferentiable('WholeFileReader')

@tf_export(v1=['TextLineReader'])
class TextLineReader(ReaderBase):
    """A Reader that outputs the lines of a file delimited by newlines.

  Newlines are stripped from the output.
  See ReaderBase for supported methods.

  @compatibility(eager)
  Readers are not compatible with eager execution. Instead, please
  use `tf.data` to get data into your model.
  @end_compatibility
  """

    @deprecation.deprecated(None, 'Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.TextLineDataset`.')
    def __init__(self, skip_header_lines=None, name=None):
        if False:
            i = 10
            return i + 15
        'Create a TextLineReader.\n\n    Args:\n      skip_header_lines: An optional int. Defaults to 0.  Number of lines\n        to skip from the beginning of every file.\n      name: A name for the operation (optional).\n    '
        rr = gen_io_ops.text_line_reader_v2(skip_header_lines=skip_header_lines, name=name)
        super(TextLineReader, self).__init__(rr)
ops.NotDifferentiable('TextLineReader')

@tf_export(v1=['FixedLengthRecordReader'])
class FixedLengthRecordReader(ReaderBase):
    """A Reader that outputs fixed-length records from a file.

  See ReaderBase for supported methods.

  @compatibility(eager)
  Readers are not compatible with eager execution. Instead, please
  use `tf.data` to get data into your model.
  @end_compatibility
  """

    @deprecation.deprecated(None, 'Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.FixedLengthRecordDataset`.')
    def __init__(self, record_bytes, header_bytes=None, footer_bytes=None, hop_bytes=None, name=None, encoding=None):
        if False:
            for i in range(10):
                print('nop')
        'Create a FixedLengthRecordReader.\n\n    Args:\n      record_bytes: An int.\n      header_bytes: An optional int. Defaults to 0.\n      footer_bytes: An optional int. Defaults to 0.\n      hop_bytes: An optional int. Defaults to 0.\n      name: A name for the operation (optional).\n      encoding: The type of encoding for the file. Defaults to none.\n    '
        rr = gen_io_ops.fixed_length_record_reader_v2(record_bytes=record_bytes, header_bytes=header_bytes, footer_bytes=footer_bytes, hop_bytes=hop_bytes, encoding=encoding, name=name)
        super(FixedLengthRecordReader, self).__init__(rr)
ops.NotDifferentiable('FixedLengthRecordReader')

@tf_export(v1=['TFRecordReader'])
class TFRecordReader(ReaderBase):
    """A Reader that outputs the records from a TFRecords file.

  See ReaderBase for supported methods.

  @compatibility(eager)
  Readers are not compatible with eager execution. Instead, please
  use `tf.data` to get data into your model.
  @end_compatibility
  """

    @deprecation.deprecated(None, 'Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.TFRecordDataset`.')
    def __init__(self, name=None, options=None):
        if False:
            print('Hello World!')
        'Create a TFRecordReader.\n\n    Args:\n      name: A name for the operation (optional).\n      options: A TFRecordOptions object (optional).\n    '
        compression_type = python_io.TFRecordOptions.get_compression_type_string(options)
        rr = gen_io_ops.tf_record_reader_v2(name=name, compression_type=compression_type)
        super(TFRecordReader, self).__init__(rr)
ops.NotDifferentiable('TFRecordReader')

@tf_export(v1=['LMDBReader'])
class LMDBReader(ReaderBase):
    """A Reader that outputs the records from a LMDB file.

  See ReaderBase for supported methods.

  @compatibility(eager)
  Readers are not compatible with eager execution. Instead, please
  use `tf.data` to get data into your model.
  @end_compatibility
  """

    @deprecation.deprecated(None, 'Queue-based input pipelines have been replaced by `tf.data`. Use `tf.contrib.data.LMDBDataset`.')
    def __init__(self, name=None, options=None):
        if False:
            print('Hello World!')
        'Create a LMDBReader.\n\n    Args:\n      name: A name for the operation (optional).\n      options: A LMDBRecordOptions object (optional).\n    '
        del options
        rr = gen_io_ops.lmdb_reader(name=name)
        super(LMDBReader, self).__init__(rr)
ops.NotDifferentiable('LMDBReader')

@tf_export(v1=['IdentityReader'])
class IdentityReader(ReaderBase):
    """A Reader that outputs the queued work as both the key and value.

  To use, enqueue strings in a Queue.  Read will take the front
  work string and output (work, work).

  See ReaderBase for supported methods.

  @compatibility(eager)
  Readers are not compatible with eager execution. Instead, please
  use `tf.data` to get data into your model.
  @end_compatibility
  """

    @deprecation.deprecated(None, 'Queue-based input pipelines have been replaced by `tf.data`. Use `tf.data.Dataset.map(...)`.')
    def __init__(self, name=None):
        if False:
            while True:
                i = 10
        'Create a IdentityReader.\n\n    Args:\n      name: A name for the operation (optional).\n    '
        rr = gen_io_ops.identity_reader_v2(name=name)
        super(IdentityReader, self).__init__(rr, supports_serialize=True)
ops.NotDifferentiable('IdentityReader')