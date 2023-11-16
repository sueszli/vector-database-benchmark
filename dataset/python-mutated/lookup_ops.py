"""Lookup operations."""
import collections
import functools
import uuid
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.ops import lookup_grad
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.gen_lookup_ops import *
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import base as trackable_base
from tensorflow.python.trackable import resource
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.types import internal
from tensorflow.python.util import compat as compat_util
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export

@tf_export(v1=['initialize_all_tables'])
@deprecated(None, 'Use `tf.tables_initializer` instead.')
def initialize_all_tables(name='init_all_tables'):
    if False:
        while True:
            i = 10
    'Returns an Op that initializes all tables of the default graph.\n\n  Args:\n    name: Optional name for the initialization op.\n\n  Returns:\n    An Op that initializes all tables.  Note that if there are\n    not tables the returned Op is a NoOp.\n  '
    return tables_initializer(name)

@tf_export(v1=['initializers.tables_initializer', 'tables_initializer'])
def tables_initializer(name='init_all_tables'):
    if False:
        i = 10
        return i + 15
    "Returns an Op that initializes all tables of the default graph.\n\n  Args:\n    name: Optional name for the initialization op.\n\n  Returns:\n    An Op that initializes all tables.  Note that if there are\n    not tables the returned Op is a NoOp.\n\n  @compatibility(TF2)\n  `tf.compat.v1.tables_initializer` is no longer needed with eager execution and\n  `tf.function`. In TF2, when creating an initializable table like a\n  `tf.lookup.StaticHashTable`, the table will automatically be initialized on\n  creation.\n\n  #### Before & After Usage Example\n\n  Before:\n\n  >>> with tf.compat.v1.Session():\n  ...   init = tf.compat.v1.lookup.KeyValueTensorInitializer(['a', 'b'], [1, 2])\n  ...   table = tf.compat.v1.lookup.StaticHashTable(init, default_value=-1)\n  ...   tf.compat.v1.tables_initializer().run()\n  ...   result = table.lookup(tf.constant(['a', 'c'])).eval()\n  >>> result\n  array([ 1, -1], dtype=int32)\n\n  After:\n\n  >>> init = tf.lookup.KeyValueTensorInitializer(['a', 'b'], [1, 2])\n  >>> table = tf.lookup.StaticHashTable(init, default_value=-1)\n  >>> table.lookup(tf.constant(['a', 'c'])).numpy()\n  array([ 1, -1], dtype=int32)\n\n  @end_compatibility\n  "
    initializers = ops.get_collection(ops.GraphKeys.TABLE_INITIALIZERS)
    if initializers:
        return control_flow_ops.group(*initializers, name=name)
    return control_flow_ops.no_op(name=name)

def check_table_dtypes(table, key_dtype, value_dtype):
    if False:
        while True:
            i = 10
    "Check that the given key_dtype and value_dtype matches the table dtypes.\n\n  Args:\n    table: The table to check types against to.\n    key_dtype: The key data type to check.\n    value_dtype: The value data type to check.\n\n  Raises:\n    TypeError: when 'key_dtype' or 'value_dtype' doesn't match the table data\n      types.\n  "
    if key_dtype.base_dtype != table.key_dtype:
        raise TypeError(f'Invalid key dtype for table, expected {table.key_dtype} but got {key_dtype}.')
    if value_dtype.base_dtype != table.value_dtype:
        raise TypeError(f'Invalid value dtype for table, expected {table.value_dtype} but got {value_dtype}.')

class LookupInterface(resource.TrackableResource):
    """Represent a lookup table that persists across different steps."""

    def __init__(self, key_dtype, value_dtype):
        if False:
            for i in range(10):
                print('nop')
        'Construct a lookup table interface.\n\n    Args:\n      key_dtype: The table key type.\n      value_dtype: The table value type.\n    '
        self._key_dtype = dtypes.as_dtype(key_dtype)
        self._value_dtype = dtypes.as_dtype(value_dtype)
        super(LookupInterface, self).__init__()

    def _create_resource(self):
        if False:
            return 10
        raise NotImplementedError

    @property
    def key_dtype(self):
        if False:
            while True:
                i = 10
        'The table key dtype.'
        return self._key_dtype

    @property
    def value_dtype(self):
        if False:
            while True:
                i = 10
        'The table value dtype.'
        return self._value_dtype

    @property
    def name(self):
        if False:
            while True:
                i = 10
        'The name of the table.'
        return NotImplementedError

    def size(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Compute the number of elements in this table.'
        raise NotImplementedError

    def lookup(self, keys, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Looks up `keys` in a table, outputs the corresponding values.'
        raise NotImplementedError

    def __getitem__(self, keys):
        if False:
            while True:
                i = 10
        'Looks up `keys` in a table, outputs the corresponding values.'
        return self.lookup(keys)

class InitializableLookupTableBase(LookupInterface):
    """Initializable lookup table interface.

  An initializable lookup tables persist across different steps.
  """

    def __init__(self, default_value, initializer):
        if False:
            for i in range(10):
                print('nop')
        'Construct a table object from a table reference.\n\n    If requires a table initializer object (subclass of `TableInitializerBase`).\n    It provides the table key and value types, as well as the op to initialize\n    the table. The caller is responsible to execute the initialization op.\n\n    Args:\n      default_value: The value to use if a key is missing in the table.\n      initializer: The table initializer to use.\n    '
        super(InitializableLookupTableBase, self).__init__(initializer.key_dtype, initializer.value_dtype)
        self._default_value = ops.convert_to_tensor(default_value, dtype=self._value_dtype)
        self._default_value.get_shape().merge_with(tensor_shape.TensorShape([]))
        if isinstance(initializer, trackable_base.Trackable):
            self._initializer = self._track_trackable(initializer, '_initializer')
        with ops.init_scope():
            self._resource_handle = self._create_resource()
        if not context.executing_eagerly() and ops.get_default_graph()._get_control_flow_context() is not None:
            with ops.init_scope():
                self._init_op = self._initialize()
        else:
            self._init_op = self._initialize()

    def _initialize(self):
        if False:
            for i in range(10):
                print('nop')
        return self._initializer.initialize(self)

    @property
    def default_value(self):
        if False:
            for i in range(10):
                print('nop')
        'The default value of the table.'
        return self._default_value

    def size(self, name=None):
        if False:
            i = 10
            return i + 15
        'Compute the number of elements in this table.\n\n    Args:\n      name: A name for the operation (optional).\n\n    Returns:\n      A scalar tensor containing the number of elements in this table.\n    '
        with ops.name_scope(name, '%s_Size' % self.name, [self.resource_handle]):
            return gen_lookup_ops.lookup_table_size_v2(self.resource_handle)

    def lookup(self, keys, name=None):
        if False:
            print('Hello World!')
        "Looks up `keys` in a table, outputs the corresponding values.\n\n    The `default_value` is used for keys not present in the table.\n\n    Args:\n      keys: Keys to look up. May be either a `SparseTensor` or dense `Tensor`.\n      name: A name for the operation (optional).\n\n    Returns:\n      A `SparseTensor` if keys are sparse, a `RaggedTensor` if keys are ragged,\n      otherwise a dense `Tensor`.\n\n    Raises:\n      TypeError: when `keys` or `default_value` doesn't match the table data\n        types.\n    "
        key_tensor = keys
        if isinstance(keys, (sparse_tensor.SparseTensor, internal.RaggedTensor)):
            key_tensor = keys.values
        if keys.dtype.base_dtype != self._key_dtype:
            raise TypeError(f'Dtype of argument `keys` must be {self._key_dtype}, received: {keys.dtype}')
        with ops.name_scope(name, '%s_Lookup' % self.name, (self.resource_handle, key_tensor, self._default_value)):
            values = gen_lookup_ops.lookup_table_find_v2(self.resource_handle, key_tensor, self._default_value)
        values.set_shape(key_tensor.get_shape())
        if isinstance(keys, sparse_tensor.SparseTensor):
            return sparse_tensor.SparseTensor(keys.indices, values, keys.dense_shape)
        elif isinstance(keys, internal.RaggedTensor):
            return keys.with_values(values)
        else:
            return values

class InitializableLookupTableBaseV1(InitializableLookupTableBase):

    @property
    def initializer(self):
        if False:
            for i in range(10):
                print('nop')
        return self._init_op

@registration.register_tf_serializable(predicate=lambda obj: isinstance(obj, StaticHashTable))
@tf_export('lookup.StaticHashTable', v1=[])
class StaticHashTable(InitializableLookupTableBase):
    """A generic hash table that is immutable once initialized.

  Example usage:

  >>> keys_tensor = tf.constant(['a', 'b', 'c'])
  >>> vals_tensor = tf.constant([7, 8, 9])
  >>> input_tensor = tf.constant(['a', 'f'])
  >>> table = tf.lookup.StaticHashTable(
  ...     tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
  ...     default_value=-1)
  >>> table.lookup(input_tensor).numpy()
  array([ 7, -1], dtype=int32)

  Or for more pythonic code:

  >>> table[input_tensor].numpy()
  array([ 7, -1], dtype=int32)

  The result of a lookup operation has the same shape as the argument:

  >>> input_tensor = tf.constant([['a', 'b'], ['c', 'd']])
  >>> table[input_tensor].numpy()
  array([[ 7,  8],
         [ 9, -1]], dtype=int32)


  """

    def __init__(self, initializer, default_value, name=None, experimental_is_anonymous=False):
        if False:
            for i in range(10):
                print('nop')
        "Creates a non-initialized `HashTable` object.\n\n    Creates a table, the type of its keys and values are specified by the\n    initializer.\n    Before using the table you will have to initialize it. After initialization\n    the table will be immutable.\n\n    Args:\n      initializer: The table initializer to use. See `HashTable` kernel for\n        supported key and value types.\n      default_value: The value to use if a key is missing in the table.\n      name: A name for the operation (optional).\n      experimental_is_anonymous: Whether to use anonymous mode for the\n        table (default is False). In anonymous mode, the table\n        resource can only be accessed via a resource handle. It can't\n        be looked up by a name. When all resource handles pointing to\n        that resource are gone, the resource will be deleted\n        automatically.\n\n    Returns:\n      A `HashTable` object.\n    "
        self._initializer = initializer
        self._default_value = default_value
        self._is_anonymous = experimental_is_anonymous
        if not self._is_anonymous:
            self._shared_name = self._initializer._shared_name
            if not self._shared_name:
                self._shared_name = 'hash_table_%s' % (str(uuid.uuid4()),)
        self._name = name or 'hash_table'
        self._table_name = None
        super(StaticHashTable, self).__init__(default_value, initializer)
        self._value_shape = self._default_value.get_shape()

    def _create_resource(self):
        if False:
            for i in range(10):
                print('nop')
        if self._is_anonymous:
            table_ref = gen_lookup_ops.anonymous_hash_table(key_dtype=self._initializer.key_dtype, value_dtype=self._initializer.value_dtype, name=self._name)
        else:
            table_ref = gen_lookup_ops.hash_table_v2(shared_name=self._shared_name, key_dtype=self._initializer.key_dtype, value_dtype=self._initializer.value_dtype, name=self._name)
        if context.executing_eagerly():
            self._table_name = None
        else:
            self._table_name = table_ref.op.name.split('/')[-1]
        return table_ref

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return self._table_name

    def export(self, name=None):
        if False:
            return 10
        'Returns tensors of all keys and values in the table.\n\n    Args:\n      name: A name for the operation (optional).\n\n    Returns:\n      A pair of tensors with the first tensor containing all keys and the\n        second tensors containing all values in the table.\n    '
        with ops.name_scope(name, '%s_Export' % self.name, [self.resource_handle]):
            (exported_keys, exported_values) = gen_lookup_ops.lookup_table_export_v2(self.resource_handle, self._key_dtype, self._value_dtype)
        exported_values.set_shape(exported_keys.get_shape().concatenate(self._value_shape))
        return (exported_keys, exported_values)

    def _serialize_to_proto(self, **unused_kwargs):
        if False:
            for i in range(10):
                print('nop')
        return None

    def _add_trackable_child(self, name, value):
        if False:
            print('Hello World!')
        setattr(self, name, value)
        if isinstance(value, trackable_base.Trackable):
            self._track_trackable(value, name)

    @classmethod
    def _deserialize_from_proto(cls, **kwargs):
        if False:
            return 10

        class _RestoredStaticHashTable(resource.RestoredResource):

            @classmethod
            def _resource_type(cls):
                if False:
                    print('Hello World!')
                return 'RestoredStaticHashTable'
        return _RestoredStaticHashTable._deserialize_from_proto(**kwargs)

@tf_export(v1=['lookup.StaticHashTable'])
class StaticHashTableV1(StaticHashTable):
    """A generic hash table that is immutable once initialized.

  When running in graph mode, you must evaluate the tensor returned by
  `tf.tables_initializer()` before evaluating the tensor returned by
  this class's `lookup()` method. Example usage in graph mode:

  ```python
  keys_tensor = tf.constant([1, 2])
  vals_tensor = tf.constant([3, 4])
  input_tensor = tf.constant([1, 5])
  table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)
  out = table.lookup(input_tensor)
  with tf.Session() as sess:
      sess.run(tf.tables_initializer())
      print(sess.run(out))
  ```

  Note that in graph mode if you set `experimental_is_anonymous` to
  `True`, you should only call `Session.run` once, otherwise each
  `Session.run` will create (and destroy) a new table unrelated to
  each other, leading to errors such as "Table not initialized".
  You can do so like this:

  ```python
  keys_tensor = tf.constant([1, 2])
  vals_tensor = tf.constant([3, 4])
  input_tensor = tf.constant([1, 5])
  table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1,
      experimental_is_anonymous=True)
  with tf.control_dependencies([tf.tables_initializer()]):
    out = table.lookup(input_tensor)
  with tf.Session() as sess:
    print(sess.run(out))
  ```

  In eager mode, no special code is needed to initialize the table.
  Example usage in eager mode:

  ```python
  tf.enable_eager_execution()
  keys_tensor = tf.constant([1, 2])
  vals_tensor = tf.constant([3, 4])
  input_tensor = tf.constant([1, 5])
  table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)
  print(table.lookup(input_tensor))
  ```
  """

    @property
    def initializer(self):
        if False:
            print('Hello World!')
        return self._init_op

class HashTable(StaticHashTableV1):

    @property
    def init(self):
        if False:
            i = 10
            return i + 15
        return self.initializer

class TableInitializerBase(trackable_base.Trackable):
    """Base class for lookup table initializers."""

    def __init__(self, key_dtype, value_dtype):
        if False:
            return 10
        'Construct a table initializer object.\n\n    Args:\n      key_dtype: Type of the table keys.\n      value_dtype: Type of the table values.\n    '
        self._key_dtype = dtypes.as_dtype(key_dtype)
        self._value_dtype = dtypes.as_dtype(value_dtype)

    @property
    def key_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        'The expected table key dtype.'
        return self._key_dtype

    @property
    def value_dtype(self):
        if False:
            print('Hello World!')
        'The expected table value dtype.'
        return self._value_dtype

    def initialize(self, table):
        if False:
            print('Hello World!')
        'Returns the table initialization op.'
        raise NotImplementedError

    @property
    def _shared_name(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a shared name to be used by the table.'
        shared_name = ''
        if context.executing_eagerly():
            shared_name += str(ops.uid())
        return shared_name

@tf_export('lookup.KeyValueTensorInitializer')
class KeyValueTensorInitializer(TableInitializerBase):
    """Table initializers given `keys` and `values` tensors.

  >>> keys_tensor = tf.constant(['a', 'b', 'c'])
  >>> vals_tensor = tf.constant([7, 8, 9])
  >>> input_tensor = tf.constant(['a', 'f'])
  >>> init = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
  >>> table = tf.lookup.StaticHashTable(
  ...     init,
  ...     default_value=-1)
  >>> table.lookup(input_tensor).numpy()
  array([ 7, -1], dtype=int32)

  """

    def __init__(self, keys, values, key_dtype=None, value_dtype=None, name=None):
        if False:
            return 10
        'Constructs a table initializer object based on keys and values tensors.\n\n    Args:\n      keys: The tensor for the keys.\n      values: The tensor for the values.\n      key_dtype: The `keys` data type. Used when `keys` is a python array.\n      value_dtype: The `values` data type. Used when `values` is a python array.\n      name: A name for the operation (optional).\n    '
        if not context.executing_eagerly() and ops.get_default_graph()._get_control_flow_context() is not None:
            with ops.init_scope():
                self._keys = ops.convert_to_tensor(keys, dtype=key_dtype, name='keys')
                self._values = ops.convert_to_tensor(values, dtype=value_dtype, name='values')
        else:
            self._keys = ops.convert_to_tensor(keys, dtype=key_dtype, name='keys')
            self._values = ops.convert_to_tensor(values, dtype=value_dtype, name='values')
        self._name = name if name is not None else 'key_value_init'
        if context.executing_eagerly():
            self._name += str(ops.uid())
        super(KeyValueTensorInitializer, self).__init__(self._keys.dtype, self._values.dtype)

    def initialize(self, table):
        if False:
            while True:
                i = 10
        'Initializes the given `table` with `keys` and `values` tensors.\n\n    Args:\n      table: The table to initialize.\n\n    Returns:\n      The operation that initializes the table.\n\n    Raises:\n      TypeError: when the keys and values data types do not match the table\n      key and value data types.\n    '
        check_table_dtypes(table, self._keys.dtype, self._values.dtype)
        with ops.name_scope(self._name, values=(table.resource_handle, self._keys, self._values)):
            init_op = gen_lookup_ops.lookup_table_import_v2(table.resource_handle, self._keys, self._values)
        ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, init_op)
        return init_op

@tf_export('lookup.TextFileIndex')
class TextFileIndex:
    """The key and value content to get from each line.

  This class defines the key and value used for `tf.lookup.TextFileInitializer`.

  The key and value content to get from each line is specified either
  by the following, or a value `>=0`.
  * `TextFileIndex.LINE_NUMBER` means use the line number starting from zero,
    expects data type int64.
  * `TextFileIndex.WHOLE_LINE` means use the whole line content, expects data
    type string.

  A value `>=0` means use the index (starting at zero) of the split line based
      on `delimiter`.
  """
    WHOLE_LINE = -2
    LINE_NUMBER = -1

@tf_export('lookup.TextFileInitializer')
class TextFileInitializer(TableInitializerBase):
    """Table initializers from a text file.

  This initializer assigns one entry in the table for each line in the file.

  The key and value type of the table to initialize is given by `key_dtype` and
  `value_dtype`.

  The key and value content to get from each line is specified by
  the `key_index` and `value_index`.

  * `TextFileIndex.LINE_NUMBER` means use the line number starting from zero,
    expects data type int64.
  * `TextFileIndex.WHOLE_LINE` means use the whole line content, expects data
    type string.
  * A value `>=0` means use the index (starting at zero) of the split line based
      on `delimiter`.

  For example if we have a file with the following content:

  >>> import tempfile
  >>> f = tempfile.NamedTemporaryFile(delete=False)
  >>> content='\\n'.join(["emerson 10", "lake 20", "palmer 30",])
  >>> f.file.write(content.encode('utf-8'))
  >>> f.file.close()

  The following snippet initializes a table with the first column as keys and
  second column as values:

  * `emerson -> 10`
  * `lake -> 20`
  * `palmer -> 30`

  >>> init= tf.lookup.TextFileInitializer(
  ...    filename=f.name,
  ...    key_dtype=tf.string, key_index=0,
  ...    value_dtype=tf.int64, value_index=1,
  ...    delimiter=" ")
  >>> table = tf.lookup.StaticHashTable(init, default_value=-1)
  >>> table.lookup(tf.constant(['palmer','lake','tarkus'])).numpy()

  Similarly to initialize the whole line as keys and the line number as values.

  * `emerson 10 -> 0`
  * `lake 20 -> 1`
  * `palmer 30 -> 2`

  >>> init = tf.lookup.TextFileInitializer(
  ...   filename=f.name,
  ...   key_dtype=tf.string, key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
  ...   value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
  >>> table = tf.lookup.StaticHashTable(init, -1)
  >>> table.lookup(tf.constant('palmer 30')).numpy()
  2
  """

    def __init__(self, filename, key_dtype, key_index, value_dtype, value_index, vocab_size=None, delimiter='\t', name=None, value_index_offset=0):
        if False:
            for i in range(10):
                print('nop')
        "Constructs a table initializer object to populate from a text file.\n\n    It generates one key-value pair per line. The type of table key and\n    value are specified by `key_dtype` and `value_dtype`, respectively.\n    Similarly the content of the key and value are specified by the key_index\n    and value_index.\n\n    - TextFileIndex.LINE_NUMBER means use the line number starting from zero,\n      expects data type int64.\n    - TextFileIndex.WHOLE_LINE means use the whole line content, expects data\n      type string or int64.\n    - A value >=0 means use the index (starting at zero) of the split line based\n      on `delimiter`.\n\n    Args:\n      filename: The filename of the text file to be used for initialization. The\n        path must be accessible from wherever the graph is initialized (eg.\n        trainer or eval workers). The filename may be a scalar `Tensor`.\n      key_dtype: The `key` data type.\n      key_index: the index that represents information of a line to get the\n        table 'key' values from.\n      value_dtype: The `value` data type.\n      value_index: the index that represents information of a line to get the\n        table 'value' values from.'\n      vocab_size: The number of elements in the file, if known.\n      delimiter: The delimiter to separate fields in a line.\n      name: A name for the operation (optional).\n      value_index_offset: A number to add to all indices extracted from the file\n        This is useful for cases where a user would like to reserve one or more\n        low index values for control characters. For instance, if you would\n        like to ensure that no vocabulary item is mapped to index 0 (so you can\n        reserve 0 for a masking value), you can set value_index_offset to 1;\n        this will mean that the first vocabulary element is mapped to 1\n        instead of 0.\n\n    Raises:\n      ValueError: when the filename is empty, or when the table key and value\n      data types do not match the expected data types.\n    "
        if not isinstance(filename, tensor_lib.Tensor) and (not filename):
            raise ValueError('`filename` argument required for tf.lookup.TextFileInitializer')
        self._filename_arg = filename
        key_dtype = dtypes.as_dtype(key_dtype)
        value_dtype = dtypes.as_dtype(value_dtype)
        if key_index < -2:
            raise ValueError(f'`key_index` should be >= -2, received: {key_index}.')
        if key_index == TextFileIndex.LINE_NUMBER and key_dtype != dtypes.int64:
            raise ValueError(f'`key_dtype` must be int64 if `key_index` is {TextFileIndex.LINE_NUMBER}, received: {key_dtype}')
        if key_index == TextFileIndex.WHOLE_LINE and (not key_dtype.is_integer) and (key_dtype != dtypes.string):
            raise ValueError(f'`key_dtype` should be either integer or string for `key_index` {TextFileIndex.WHOLE_LINE}, received: {key_dtype}')
        if value_index < -2:
            raise ValueError(f'`value_index` should be >= -2, received: {value_index}')
        if value_index == TextFileIndex.LINE_NUMBER and value_dtype != dtypes.int64:
            raise ValueError(f'`value_dtype` must be int64 for `value_index` {TextFileIndex.LINE_NUMBER}, received: {value_dtype}')
        if value_index == TextFileIndex.WHOLE_LINE and (not value_dtype.is_integer) and (value_dtype != dtypes.string):
            raise ValueError(f'`value_dtype` should be either integer or string for `value_index` {TextFileIndex.WHOLE_LINE}, received: {value_dtype}')
        if vocab_size is not None and vocab_size <= 0:
            raise ValueError(f'`vocab_size` should be > 0, received: {vocab_size}')
        self._key_index = key_index
        self._value_index = value_index
        self._vocab_size = vocab_size
        self._delimiter = delimiter
        self._name = name
        self._filename = self._track_trackable(asset.Asset(filename), '_filename')
        self._offset = value_index_offset
        super(TextFileInitializer, self).__init__(key_dtype, value_dtype)

    def initialize(self, table):
        if False:
            return 10
        'Initializes the table from a text file.\n\n    Args:\n      table: The table to be initialized.\n\n    Returns:\n      The operation that initializes the table.\n\n    Raises:\n      TypeError: when the keys and values data types do not match the table\n      key and value data types.\n    '
        check_table_dtypes(table, self.key_dtype, self.value_dtype)
        with ops.name_scope(self._name, 'text_file_init', (table.resource_handle,)):
            filename = ops.convert_to_tensor(self._filename, dtypes.string, name='asset_filepath')
            init_op = gen_lookup_ops.initialize_table_from_text_file_v2(table.resource_handle, filename, self._key_index, self._value_index, -1 if self._vocab_size is None else self._vocab_size, self._delimiter, self._offset)
        ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, init_op)
        if not context.executing_eagerly() and constant_op.is_constant(filename):
            ops.add_to_collection(ops.GraphKeys.ASSET_FILEPATHS, filename)
        return init_op

    @property
    def _shared_name(self):
        if False:
            i = 10
            return i + 15
        if self._vocab_size:
            if self._offset:
                shared_name = 'hash_table_%s_%d_%s_%s_%s' % (self._filename_arg, self._vocab_size, self._key_index, self._value_index, self._offset)
            else:
                shared_name = 'hash_table_%s_%d_%s_%s' % (self._filename_arg, self._vocab_size, self._key_index, self._value_index)
        elif self._offset:
            shared_name = 'hash_table_%s_%s_%s_%s' % (self._filename_arg, self._key_index, self._value_index, self._offset)
        else:
            shared_name = 'hash_table_%s_%s_%s' % (self._filename_arg, self._key_index, self._value_index)
        return shared_name

class TextFileStringTableInitializer(TextFileInitializer):
    """Table initializer for `int64` IDs to string tables from a text file."""

    def __init__(self, filename, key_column_index=TextFileIndex.LINE_NUMBER, value_column_index=TextFileIndex.WHOLE_LINE, vocab_size=None, delimiter='\t', name='text_file_string_table_init'):
        if False:
            while True:
                i = 10
        'Constructs an initializer for an id-to-string table from a text file.\n\n    It populates a table that its key and value types are int64 and string,\n    respectively. It generates one key-value pair per line.\n    The content of the key and value are specified by `key_column_index`\n    and `value_column_index`.\n\n    - TextFileIndex.LINE_NUMBER means use the line number starting from zero,\n      expects data type int64.\n    - TextFileIndex.WHOLE_LINE means use the whole line content, expects data\n      type string or int64.\n    - A value >=0 means use the index (starting at zero) of the split line based\n      on `delimiter`.\n\n    Args:\n      filename: The filename of the text file to be used for initialization. The\n        path must be accessible from wherever the graph is initialized (eg.\n        trainer or eval workers). The filename may be a scalar `Tensor`.\n      key_column_index: The column index from the text file to get the keys\n        from. The default is to use the line number, starting from zero.\n      value_column_index: The column index from the text file to get the values\n        from. The default is to use the whole line content.\n      vocab_size: The number of elements in the file, if known.\n      delimiter: The delimiter to separate fields in a line.\n      name: Optional name for the op.\n\n    Raises:\n      TypeError: when the filename is empty, or when the table key and value\n      data types do not match the expected data types.\n    '
        super(TextFileStringTableInitializer, self).__init__(filename, dtypes.int64, key_column_index, dtypes.string, value_column_index, vocab_size=vocab_size, delimiter=delimiter, name=name)

class TextFileIdTableInitializer(TextFileInitializer):
    """Table initializer for string to `int64` IDs tables from a text file."""

    def __init__(self, filename, key_column_index=TextFileIndex.WHOLE_LINE, value_column_index=TextFileIndex.LINE_NUMBER, vocab_size=None, delimiter='\t', name='text_file_id_table_init', key_dtype=dtypes.string):
        if False:
            for i in range(10):
                print('nop')
        'Constructs an initializer for an string-to-id table from a text file.\n\n    It populates a table that its key and value types are string and int64,\n    respectively. It generates one key-value pair per line.\n    The content of the key and value are specified by the key_index\n    and value_index.\n\n    - TextFileIndex.LINE_NUMBER means use the line number starting from zero,\n      expects data type int64.\n    - TextFileIndex.WHOLE_LINE means use the whole line content, expects data\n      type string.\n    - A value >=0 means use the index (starting at zero) of the split line based\n      on `delimiter`.\n\n    Args:\n      filename: The filename of the text file to be used for initialization. The\n        path must be accessible from wherever the graph is initialized (eg.\n        trainer or eval workers). The filename may be a scalar `Tensor`.\n      key_column_index: The column index from the text file to get the `key`\n        values from. The default is to use the whole line content.\n      value_column_index: The column index from the text file to get the `value`\n        values from. The default is to use the line number, starting from zero.\n      vocab_size: The number of elements in the file, if known.\n      delimiter: The delimiter to separate fields in a line.\n      name: Optional name for the op.\n      key_dtype: The `key` data type.\n\n    Raises:\n      TypeError: when the filename is empty, or when the table key and value\n      data types do not match the expected data types.\n    '
        super(TextFileIdTableInitializer, self).__init__(filename, key_dtype, key_column_index, dtypes.int64, value_column_index, vocab_size=vocab_size, delimiter=delimiter, name=name)

class HasherSpec(collections.namedtuple('HasherSpec', ['hasher', 'key'])):
    """A structure for the spec of the hashing function to use for hash buckets.

  `hasher` is the name of the hashing function to use (eg. "fasthash",
  "stronghash").
  `key` is optional and specify the key to use for the hash function if
  supported, currently only used by a strong hash.

  Fields:
    hasher: The hasher name to use.
    key: The key to be used by the hashing function, if required.
  """
    __slots__ = ()
FastHashSpec = HasherSpec('fasthash', None)

class StrongHashSpec(HasherSpec):
    """A structure to specify a key of the strong keyed hash spec.

  The strong hash requires a `key`, which is a list of 2 unsigned integer
  numbers. These should be non-zero; random numbers generated from random.org
  would be a fine choice.

  Fields:
    key: The key to be used by the keyed hashing function.
  """
    __slots__ = ()

    def __new__(cls, key):
        if False:
            i = 10
            return i + 15
        if len(key) != 2:
            raise ValueError(f'`key` must have size 2, received {len(key)}')
        if not isinstance(key[0], compat_util.integral_types) or not isinstance(key[1], compat_util.integral_types):
            raise TypeError('Invalid key %s. Must be unsigned integer values.' % key)
        return super(cls, StrongHashSpec).__new__(cls, 'stronghash', key)

def _as_string(tensor):
    if False:
        i = 10
        return i + 15
    if dtypes.string == tensor.dtype.base_dtype:
        return tensor
    return string_ops.as_string(tensor)

class IdTableWithHashBuckets(LookupInterface):
    """String to Id table wrapper that assigns out-of-vocabulary keys to buckets.

  For example, if an instance of `IdTableWithHashBuckets` is initialized with a
  string-to-id table that maps:

  * `emerson -> 0`
  * `lake -> 1`
  * `palmer -> 2`

  The `IdTableWithHashBuckets` object will performs the following mapping:

  * `emerson -> 0`
  * `lake -> 1`
  * `palmer -> 2`
  * `<other term> -> bucket_id`, where bucket_id will be between `3` and
  `3 + num_oov_buckets - 1`, calculated by:
  `hash(<term>) % num_oov_buckets + vocab_size`

  If input_tensor is `["emerson", "lake", "palmer", "king", "crimson"]`,
  the lookup result is `[0, 1, 2, 4, 7]`.

  If `table` is None, only out-of-vocabulary buckets are used.

  Example usage:

  ```python
  num_oov_buckets = 3
  input_tensor = tf.constant(["emerson", "lake", "palmer", "king", "crimnson"])
  table = tf.IdTableWithHashBuckets(
      tf.StaticHashTable(
          tf.lookup.TextFileInitializer(
              filename,
              key_dtype=tf.string,
              key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
              value_dtype=tf.int64,
              value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
              delimiter="\\t"),
          default_value),
      num_oov_buckets)
  out = table.lookup(input_tensor).
  table.init.run()
  print(out.eval())
  ```

  The hash function used for generating out-of-vocabulary buckets ID is handled
  by `hasher_spec`.
  """

    def __init__(self, table, num_oov_buckets, hasher_spec=FastHashSpec, name=None, key_dtype=None):
        if False:
            i = 10
            return i + 15
        'Construct a `IdTableWithHashBuckets` object.\n\n    Args:\n      table: Table that maps `tf.string` or `tf.int64` keys to `tf.int64` ids.\n      num_oov_buckets: Number of buckets to use for out-of-vocabulary keys.\n      hasher_spec: A `HasherSpec` to specify the hash function to use for\n        assignation of out-of-vocabulary buckets  (optional).\n      name: A name for the operation (optional).\n      key_dtype: Data type of keys passed to `lookup`. Defaults to\n        `table.key_dtype` if `table` is specified, otherwise `tf.string`. Must\n        be string or integer, and must be castable to `table.key_dtype`.\n\n    Raises:\n      ValueError: when `table` in None and `num_oov_buckets` is not positive.\n      TypeError: when `hasher_spec` is invalid.\n    '
        if name:
            name = name.rstrip('/')
        if table:
            if key_dtype is None:
                key_dtype = table.key_dtype
            supported_table_key_dtypes = (dtypes.int64, dtypes.string)
            if table.key_dtype not in supported_table_key_dtypes:
                raise TypeError(f'Invalid `key_dtype`, expected one of {supported_table_key_dtypes}, received {key_dtype}.')
            if table.key_dtype.is_integer != key_dtype.is_integer:
                raise TypeError('Invalid `key dtype`, expected %s but got %s.' % ('integer' if key_dtype.is_integer else 'non-integer', table.key_dtype))
            if table.value_dtype != dtypes.int64:
                raise TypeError('Invalid `value_dtype`: expected int64 but got %s.' % table.value_dtype)
            self._table = table
            name = name or self._table.name
        else:
            if num_oov_buckets <= 0:
                raise ValueError('`oov_buckets` must be > 0 if no `table` is supplied.')
            key_dtype = dtypes.string if key_dtype is None else key_dtype
            self._table = None
            name = name or 'hash_bucket'
        if not key_dtype.is_integer and dtypes.string != key_dtype:
            raise TypeError(f'Invalid `key_dtype`, expected integer or string, got {key_dtype}.')
        self._num_oov_buckets = num_oov_buckets
        if not isinstance(hasher_spec, HasherSpec):
            raise TypeError(f'`hasher_spec` must be of type HasherSpec, got {type(hasher_spec)}.')
        self._hasher_spec = hasher_spec
        if name:
            self._table_name = name.split('/')[-1]
        else:
            self._table_name = None
        super(IdTableWithHashBuckets, self).__init__(key_dtype, dtypes.int64)

    def _create_resource(self):
        if False:
            while True:
                i = 10
        if self._table is not None:
            return self._table._create_resource()
        return None

    def _initialize(self):
        if False:
            while True:
                i = 10
        if self._table is not None:
            return self._table._initialize()
        with ops.name_scope(None, 'init'):
            return control_flow_ops.no_op()

    @property
    def initializer(self):
        if False:
            for i in range(10):
                print('nop')
        if self._table is not None:
            return self._table._init_op
        with ops.name_scope(None, 'init'):
            return control_flow_ops.no_op()

    @property
    @deprecated('2018-12-15', 'Use `initializer` instead.')
    def init(self):
        if False:
            while True:
                i = 10
        return self.initializer

    @property
    def resource_handle(self):
        if False:
            i = 10
            return i + 15
        if self._table is not None:
            return self._table.resource_handle
        return None

    @property
    def name(self):
        if False:
            print('Hello World!')
        return self._table_name

    def size(self, name=None):
        if False:
            print('Hello World!')
        'Compute the number of elements in this table.'
        with ops.name_scope(name, '%s_Size' % self.name):
            if self._table:
                tsize = self._table.size()
            else:
                tsize = ops.convert_to_tensor(0, dtype=dtypes.int64)
            return tsize + self._num_oov_buckets

    def _get_string_to_hash_bucket_fn(self, hasher_spec):
        if False:
            return 10
        'Returns the string_to_hash_bucket op to use based on `hasher_spec`.'
        if not isinstance(hasher_spec, HasherSpec):
            raise TypeError(f'`hasher_spec` must be of type HasherSpec, got {type(hasher_spec)}.')
        if hasher_spec.hasher == 'fasthash':
            return string_ops.string_to_hash_bucket_fast
        if hasher_spec.hasher == 'legacy':
            return string_ops.string_to_hash_bucket
        if hasher_spec.hasher == 'stronghash':
            return functools.partial(string_ops.string_to_hash_bucket_strong, key=hasher_spec.key)
        raise ValueError(f'Found unknown hasher {hasher_spec.hasher} in `hasher_spec`')

    def lookup(self, keys, name=None):
        if False:
            while True:
                i = 10
        "Looks up `keys` in the table, outputs the corresponding values.\n\n    It assigns out-of-vocabulary keys to buckets based in their hashes.\n\n    Args:\n      keys: Keys to look up. May be either a `SparseTensor` or dense `Tensor`.\n      name: Optional name for the op.\n\n    Returns:\n      A `SparseTensor` if keys are sparse, a `RaggedTensor` if keys are ragged,\n      otherwise a dense `Tensor`.\n\n    Raises:\n      TypeError: when `keys` doesn't match the table key data type.\n    "
        if keys.dtype.base_dtype != self._key_dtype:
            raise TypeError(f'Dtype of argument `keys` must be {self._key_dtype}, received: {keys.dtype}')
        values = keys
        if isinstance(keys, (sparse_tensor.SparseTensor, internal.RaggedTensor)):
            values = keys.values
        if self._table and self._table.key_dtype.base_dtype == dtypes.int64:
            values = math_ops.cast(values, dtypes.int64)
        if self._num_oov_buckets == 0:
            ids = self._table.lookup(values, name=name)
        else:
            with ops.name_scope(name, '%s_Lookup' % self.name):
                str_to_hash_bucket = self._get_string_to_hash_bucket_fn(self._hasher_spec)
                buckets = str_to_hash_bucket(_as_string(values), num_buckets=self._num_oov_buckets, name='hash_bucket')
                if self._table:
                    ids = self._table.lookup(values)
                    buckets = math_ops.add(buckets, self._table.size())
                    is_id_non_default = math_ops.not_equal(ids, self._table.default_value)
                    ids = array_ops.where_v2(is_id_non_default, ids, buckets)
                else:
                    ids = buckets
        if isinstance(keys, sparse_tensor.SparseTensor):
            return sparse_tensor.SparseTensor(keys.indices, ids, keys.dense_shape)
        elif isinstance(keys, internal.RaggedTensor):
            return keys.with_values(ids)
        return ids

@tf_export('lookup.StaticVocabularyTable', v1=[])
class StaticVocabularyTable(LookupInterface):
    """String to Id table that assigns out-of-vocabulary keys to hash buckets.

  For example, if an instance of `StaticVocabularyTable` is initialized with a
  string-to-id initializer that maps:

  >>> init = tf.lookup.KeyValueTensorInitializer(
  ...     keys=tf.constant(['emerson', 'lake', 'palmer']),
  ...     values=tf.constant([0, 1, 2], dtype=tf.int64))
  >>> table = tf.lookup.StaticVocabularyTable(
  ...    init,
  ...    num_oov_buckets=5)

  The `Vocabulary` object will performs the following mapping:

  * `emerson -> 0`
  * `lake -> 1`
  * `palmer -> 2`
  * `<other term> -> bucket_id`, where `bucket_id` will be between `3` and
  `3 + num_oov_buckets - 1 = 7`, calculated by:
  `hash(<term>) % num_oov_buckets + vocab_size`

  If input_tensor is:

  >>> input_tensor = tf.constant(["emerson", "lake", "palmer",
  ...                             "king", "crimson"])
  >>> table[input_tensor].numpy()
  array([0, 1, 2, 6, 7])

  If `initializer` is None, only out-of-vocabulary buckets are used.

  Example usage:

  >>> num_oov_buckets = 3
  >>> vocab = ["emerson", "lake", "palmer", "crimnson"]
  >>> import tempfile
  >>> f = tempfile.NamedTemporaryFile(delete=False)
  >>> f.write('\\n'.join(vocab).encode('utf-8'))
  >>> f.close()

  >>> init = tf.lookup.TextFileInitializer(
  ...     f.name,
  ...     key_dtype=tf.string, key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
  ...     value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
  >>> table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets)
  >>> table.lookup(tf.constant(["palmer", "crimnson" , "king",
  ...                           "tarkus", "black", "moon"])).numpy()
  array([2, 3, 5, 6, 6, 4])

  The hash function used for generating out-of-vocabulary buckets ID is
  Fingerprint64.

  Note that the out-of-vocabulary bucket IDs always range from the table `size`
  up to `size + num_oov_buckets - 1` regardless of the table values, which could
  cause unexpected collisions:

  >>> init = tf.lookup.KeyValueTensorInitializer(
  ...     keys=tf.constant(["emerson", "lake", "palmer"]),
  ...     values=tf.constant([1, 2, 3], dtype=tf.int64))
  >>> table = tf.lookup.StaticVocabularyTable(
  ...     init,
  ...     num_oov_buckets=1)
  >>> input_tensor = tf.constant(["emerson", "lake", "palmer", "king"])
  >>> table[input_tensor].numpy()
  array([1, 2, 3, 3])
  """

    def __init__(self, initializer, num_oov_buckets, lookup_key_dtype=None, name=None, experimental_is_anonymous=False):
        if False:
            return 10
        "Construct a `StaticVocabularyTable` object.\n\n    Args:\n      initializer: A `TableInitializerBase` object that contains the data used\n        to initialize the table. If None, then we only use out-of-vocab buckets.\n      num_oov_buckets: Number of buckets to use for out-of-vocabulary keys. Must\n        be greater than zero. If out-of-vocab buckets are not required, use\n        `StaticHashTable` instead.\n      lookup_key_dtype: Data type of keys passed to `lookup`. Defaults to\n        `initializer.key_dtype` if `initializer` is specified, otherwise\n        `tf.string`. Must be string or integer, and must be castable to\n        `initializer.key_dtype`.\n      name: A name for the operation (optional).\n      experimental_is_anonymous: Whether to use anonymous mode for the\n        table (default is False). In anonymous mode, the table\n        resource can only be accessed via a resource handle. It can't\n        be looked up by a name. When all resource handles pointing to\n        that resource are gone, the resource will be deleted\n        automatically.\n\n    Raises:\n      ValueError: when `num_oov_buckets` is not positive.\n      TypeError: when lookup_key_dtype or initializer.key_dtype are not\n        integer or string. Also when initializer.value_dtype != int64.\n    "
        if num_oov_buckets <= 0:
            raise ValueError('`num_oov_buckets` must be > 0; use StaticHashTable.')
        if name:
            name = name.rstrip('/')
        if initializer:
            if lookup_key_dtype is None:
                lookup_key_dtype = initializer.key_dtype
            supported_table_key_dtypes = (dtypes.int64, dtypes.string)
            if initializer.key_dtype not in supported_table_key_dtypes:
                raise TypeError('Invalid `key_dtype`, expected one of %s, but got %s.' % (supported_table_key_dtypes, initializer.key_dtype))
            if initializer.key_dtype.is_integer != lookup_key_dtype.is_integer:
                raise TypeError('Invalid `key_dtype`, expected %s but got %s.' % ('integer' if lookup_key_dtype.is_integer else 'non-integer', initializer.key_dtype))
            if initializer.value_dtype != dtypes.int64:
                raise TypeError('Invalid `value_dtype`, expected %s but got %s.' % (dtypes.int64, initializer.value_dtype))
            if isinstance(initializer, trackable_base.Trackable):
                self._initializer = self._track_trackable(initializer, '_initializer')
            self._table = HashTable(initializer, default_value=-1, experimental_is_anonymous=experimental_is_anonymous)
            name = name or self._table.name
        else:
            lookup_key_dtype = dtypes.string
            self._table = None
            name = name or 'hash_bucket'
        if not lookup_key_dtype.is_integer and dtypes.string != lookup_key_dtype:
            raise TypeError(f'Invalid `key_dtype`, expected integer or string, got {lookup_key_dtype}')
        self._num_oov_buckets = num_oov_buckets
        self._table_name = None
        if name is not None:
            self._table_name = name.split('/')[-1]
        super(StaticVocabularyTable, self).__init__(lookup_key_dtype, dtypes.int64)

    def _create_resource(self):
        if False:
            while True:
                i = 10
        if self._table is not None:
            return self._table._create_resource()
        return None

    def _initialize(self):
        if False:
            return 10
        if self._table is not None:
            return self._table._initialize()
        with ops.name_scope(None, 'init'):
            return control_flow_ops.no_op()

    @property
    def resource_handle(self):
        if False:
            while True:
                i = 10
        if self._table is not None:
            return self._table.resource_handle
        return None

    @property
    def name(self):
        if False:
            print('Hello World!')
        return self._table_name

    def size(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Compute the number of elements in this table.'
        with ops.name_scope(name, '%s_Size' % self.name):
            if self._table:
                tsize = self._table.size()
            else:
                tsize = ops.convert_to_tensor(0, dtype=dtypes.int64)
            return tsize + self._num_oov_buckets

    def lookup(self, keys, name=None):
        if False:
            return 10
        "Looks up `keys` in the table, outputs the corresponding values.\n\n    It assigns out-of-vocabulary keys to buckets based in their hashes.\n\n    Args:\n      keys: Keys to look up. May be either a `SparseTensor` or dense `Tensor`.\n      name: Optional name for the op.\n\n    Returns:\n      A `SparseTensor` if keys are sparse, a `RaggedTensor` if keys are ragged,\n      otherwise a dense `Tensor`.\n\n    Raises:\n      TypeError: when `keys` doesn't match the table key data type.\n    "
        if keys.dtype.base_dtype != self._key_dtype:
            raise TypeError(f'Dtype of argument `keys` must be {self._key_dtype}, received: {keys.dtype}')
        values = keys
        if isinstance(keys, (sparse_tensor.SparseTensor, internal.RaggedTensor)):
            values = keys.values
        if self._table and self._table.key_dtype.base_dtype == dtypes.int64:
            values = math_ops.cast(values, dtypes.int64)
        with ops.name_scope(name, '%s_Lookup' % self.name):
            buckets = string_ops.string_to_hash_bucket_fast(_as_string(values), num_buckets=self._num_oov_buckets, name='hash_bucket')
            if self._table:
                ids = self._table.lookup(values)
                buckets = math_ops.add(buckets, self._table.size())
                is_id_non_default = math_ops.not_equal(ids, self._table.default_value)
                ids = array_ops.where_v2(is_id_non_default, ids, buckets)
            else:
                ids = buckets
        if isinstance(keys, sparse_tensor.SparseTensor):
            return sparse_tensor.SparseTensor(keys.indices, ids, keys.dense_shape)
        elif isinstance(keys, internal.RaggedTensor):
            return keys.with_values(ids)
        return ids

@tf_export(v1=['lookup.StaticVocabularyTable'])
class StaticVocabularyTableV1(StaticVocabularyTable):

    @property
    def initializer(self):
        if False:
            return 10
        if self._table is not None:
            return self._table._init_op
        with ops.name_scope(None, 'init'):
            return control_flow_ops.no_op()

def index_table_from_file(vocabulary_file=None, num_oov_buckets=0, vocab_size=None, default_value=-1, hasher_spec=FastHashSpec, key_dtype=dtypes.string, name=None, key_column_index=TextFileIndex.WHOLE_LINE, value_column_index=TextFileIndex.LINE_NUMBER, delimiter='\t'):
    if False:
        print('Hello World!')
    'Returns a lookup table that converts a string tensor into int64 IDs.\n\n  This operation constructs a lookup table to convert tensor of strings into\n  int64 IDs. The mapping can be initialized from a vocabulary file specified in\n  `vocabulary_file`, where the whole line is the key and the zero-based line\n  number is the ID.\n\n  Any lookup of an out-of-vocabulary token will return a bucket ID based on its\n  hash if `num_oov_buckets` is greater than zero. Otherwise it is assigned the\n  `default_value`.\n  The bucket ID range is\n  `[vocabulary size, vocabulary size + num_oov_buckets - 1]`.\n\n  The underlying table must be initialized by calling\n  `session.run(tf.compat.v1.tables_initializer())` or\n  `session.run(table.init())` once.\n\n  To specify multi-column vocabulary files, use key_column_index and\n  value_column_index and delimiter.\n\n  - TextFileIndex.LINE_NUMBER means use the line number starting from zero,\n    expects data type int64.\n  - TextFileIndex.WHOLE_LINE means use the whole line content, expects data\n    type string.\n  - A value >=0 means use the index (starting at zero) of the split line based\n    on `delimiter`.\n\n  Sample Usages:\n\n  If we have a vocabulary file "test.txt" with the following content:\n\n  ```\n  emerson\n  lake\n  palmer\n  ```\n\n  ```python\n  features = tf.constant(["emerson", "lake", "and", "palmer"])\n  table = tf.lookup.index_table_from_file(\n      vocabulary_file="test.txt", num_oov_buckets=1)\n  ids = table.lookup(features)\n  ...\n  tf.compat.v1.tables_initializer().run()\n\n  ids.eval()  ==> [0, 1, 3, 2]  # where 3 is the out-of-vocabulary bucket\n  ```\n\n  Args:\n    vocabulary_file: The vocabulary filename, may be a constant scalar `Tensor`.\n    num_oov_buckets: The number of out-of-vocabulary buckets.\n    vocab_size: Number of the elements in the vocabulary, if known.\n    default_value: The value to use for out-of-vocabulary feature values.\n      Defaults to -1.\n    hasher_spec: A `HasherSpec` to specify the hash function to use for\n      assignation of out-of-vocabulary buckets.\n    key_dtype: The `key` data type.\n    name: A name for this op (optional).\n    key_column_index: The column index from the text file to get the `key`\n      values from. The default is to use the whole line content.\n    value_column_index: The column index from the text file to get the `value`\n      values from. The default is to use the line number, starting from zero.\n    delimiter: The delimiter to separate fields in a line.\n\n  Returns:\n    The lookup table to map a `key_dtype` `Tensor` to index `int64` `Tensor`.\n\n  Raises:\n    ValueError: If `vocabulary_file` is not set.\n    ValueError: If `num_oov_buckets` is negative or `vocab_size` is not greater\n      than zero.\n  '
    if vocabulary_file is None or (isinstance(vocabulary_file, str) and (not vocabulary_file)):
        raise ValueError('`vocabulary_file` must be specified and must not be empty.')
    if num_oov_buckets < 0:
        raise ValueError('num_oov_buckets must be greater or equal than 0, got %d.' % num_oov_buckets)
    if vocab_size is not None and vocab_size < 1:
        vocab_file_value = vocabulary_file
        if isinstance(vocabulary_file, tensor_lib.Tensor):
            vocab_file_value = tensor_util.constant_value(vocabulary_file) or '?'
        raise ValueError('`vocab_size` must be greater than 0, got %d for vocabulary_file: %s.' % (vocab_size, vocab_file_value))
    if not key_dtype.is_integer and dtypes.string != key_dtype.base_dtype:
        raise TypeError('Dtype for `keys` should be either integer or string.')
    with ops.name_scope(name, 'string_to_index'):
        table = None
        with ops.name_scope(None, 'hash_table'):
            init = TextFileIdTableInitializer(vocabulary_file, vocab_size=vocab_size, key_dtype=dtypes.int64 if key_dtype.is_integer else key_dtype, name='table_init', key_column_index=key_column_index, value_column_index=value_column_index, delimiter=delimiter)
            table = StaticHashTableV1(init, default_value)
        if num_oov_buckets:
            table = IdTableWithHashBuckets(table, num_oov_buckets=num_oov_buckets, hasher_spec=hasher_spec, key_dtype=key_dtype)
        return table

def index_table_from_tensor(vocabulary_list, num_oov_buckets=0, default_value=-1, hasher_spec=FastHashSpec, dtype=dtypes.string, name=None):
    if False:
        return 10
    'Returns a lookup table that converts a string tensor into int64 IDs.\n\n  This operation constructs a lookup table to convert tensor of strings into\n  int64 IDs. The mapping can be initialized from a string `vocabulary_list` 1-D\n  tensor where each element is a key and corresponding index within the tensor\n  is the value.\n\n  Any lookup of an out-of-vocabulary token will return a bucket ID based on its\n  hash if `num_oov_buckets` is greater than zero. Otherwise it is assigned the\n  `default_value`. The bucket ID range is\n  `[vocabulary list size, vocabulary list size + num_oov_buckets - 1]`.\n\n  The underlying table must be initialized by calling\n  `session.run(tf.compat.v1.tables_initializer())` or\n  `session.run(table.init())` once.\n\n  Elements in `vocabulary_list` cannot have duplicates, otherwise when executing\n  the table initializer op, it will throw a `FailedPreconditionError`.\n\n  Sample Usages:\n\n  ```python\n  vocabulary_list = tf.constant(["emerson", "lake", "palmer"])\n  table = tf.lookup.index_table_from_tensor(\n      vocabulary_list=vocabulary_list, num_oov_buckets=1, default_value=-1)\n  features = tf.constant(["emerson", "lake", "and", "palmer"])\n  ids = table.lookup(features)\n  ...\n  tf.compat.v1.tables_initializer().run()\n\n  ids.eval()  ==> [0, 1, 4, 2]\n  ```\n\n  Args:\n    vocabulary_list: A 1-D `Tensor` that specifies the mapping of keys to\n      indices. The type of this object must be castable to `dtype`.\n    num_oov_buckets: The number of out-of-vocabulary buckets.\n    default_value: The value to use for out-of-vocabulary feature values.\n      Defaults to -1.\n    hasher_spec: A `HasherSpec` to specify the hash function to use for\n      assignment of out-of-vocabulary buckets.\n    dtype: The type of values passed to `lookup`. Only string and integers are\n      supported.\n    name: A name for this op (optional).\n\n  Returns:\n    The lookup table to map an input `Tensor` to index `int64` `Tensor`.\n\n  Raises:\n    ValueError: If `vocabulary_list` is invalid.\n    ValueError: If `num_oov_buckets` is negative.\n  '
    if vocabulary_list is None:
        raise ValueError('`vocabulary_list` must be specified.')
    if num_oov_buckets < 0:
        raise ValueError('`num_oov_buckets` must be greater or equal than 0, got %d.' % num_oov_buckets)
    if not dtype.is_integer and dtypes.string != dtype.base_dtype:
        raise TypeError('`dtype` must either be integer or string.')
    with ops.name_scope(name, 'string_to_index'):
        keys = ops.convert_to_tensor(vocabulary_list)
        if keys.dtype.is_integer != dtype.is_integer:
            raise ValueError('Invalid `dtype`: Expected %s, got %s.' % ('integer' if dtype.is_integer else 'non-integer', keys.dtype))
        if not dtype.is_integer and keys.dtype.base_dtype != dtype:
            raise ValueError('Invalid `dtype`: Expected %s, got %s.' % (dtype, keys.dtype))
        num_elements = array_ops.size(keys)
        values = math_ops.cast(math_ops.range(num_elements), dtypes.int64)
        with ops.name_scope(None, 'hash_table'):
            table_keys = math_ops.cast(keys, dtypes.int64) if keys.dtype.is_integer else keys
            init = KeyValueTensorInitializer(table_keys, values, table_keys.dtype.base_dtype, dtypes.int64, name='table_init')
            table = StaticHashTableV1(init, default_value)
        if num_oov_buckets:
            table = IdTableWithHashBuckets(table, num_oov_buckets=num_oov_buckets, hasher_spec=hasher_spec, key_dtype=dtype)
        return table

def index_to_string_table_from_file(vocabulary_file, vocab_size=None, default_value='UNK', name=None, key_column_index=TextFileIndex.LINE_NUMBER, value_column_index=TextFileIndex.WHOLE_LINE, delimiter='\t'):
    if False:
        i = 10
        return i + 15
    'Returns a lookup table that maps a `Tensor` of indices into strings.\n\n  This operation constructs a lookup table to map int64 indices into string\n  values. The table is initialized from a vocabulary file specified in\n  `vocabulary_file`, where the whole line is the value and the\n  zero-based line number is the index.\n\n  Any input which does not have a corresponding index in the vocabulary file\n  (an out-of-vocabulary entry) is assigned the `default_value`\n\n  The underlying table must be initialized by calling\n  `session.run(tf.compat.v1.tables_initializer())` or\n  `session.run(table.init())` once.\n\n  To specify multi-column vocabulary files, use key_column_index and\n  value_column_index and delimiter.\n\n  - TextFileIndex.LINE_NUMBER means use the line number starting from zero,\n    expects data type int64.\n  - TextFileIndex.WHOLE_LINE means use the whole line content, expects data\n    type string.\n  - A value >=0 means use the index (starting at zero) of the split line based\n    on `delimiter`.\n\n  Sample Usages:\n\n  If we have a vocabulary file "test.txt" with the following content:\n\n  ```\n  emerson\n  lake\n  palmer\n  ```\n\n  ```python\n  indices = tf.constant([1, 5], tf.int64)\n  table = tf.lookup.index_to_string_table_from_file(\n      vocabulary_file="test.txt", default_value="UNKNOWN")\n  values = table.lookup(indices)\n  ...\n  tf.compat.v1.tables_initializer().run()\n\n  values.eval() ==> ["lake", "UNKNOWN"]\n  ```\n\n  Args:\n    vocabulary_file: The vocabulary filename, may be a constant scalar `Tensor`.\n    vocab_size: Number of the elements in the vocabulary, if known.\n    default_value: The value to use for out-of-vocabulary indices.\n    name: A name for this op (optional).\n    key_column_index: The column index from the text file to get the `key`\n      values from. The default is to use the line number, starting from zero.\n    value_column_index: The column index from the text file to get the `value`\n      values from. The default is to use the whole line content.\n    delimiter: The delimiter to separate fields in a line.\n\n  Returns:\n    The lookup table to map a string values associated to a given index `int64`\n    `Tensors`.\n\n  Raises:\n    ValueError: when `vocabulary_file` is empty.\n    ValueError: when `vocab_size` is invalid.\n  '
    if vocabulary_file is None or (isinstance(vocabulary_file, str) and (not vocabulary_file)):
        raise ValueError('`vocabulary_file` must be specified and must not be empty.')
    if vocab_size is not None and vocab_size < 1:
        raise ValueError(f'`vocab_size` must be greater than 0, got {vocab_size}.')
    with ops.name_scope(name, 'index_to_string'):
        init = TextFileStringTableInitializer(vocabulary_file, vocab_size=vocab_size, name='table_init', key_column_index=key_column_index, value_column_index=value_column_index, delimiter=delimiter)
        return StaticHashTableV1(init, default_value)

def index_to_string_table_from_tensor(vocabulary_list, default_value='UNK', name=None):
    if False:
        i = 10
        return i + 15
    'Returns a lookup table that maps a `Tensor` of indices into strings.\n\n  This operation constructs a lookup table to map int64 indices into string\n  values. The mapping is initialized from a string `vocabulary_list` 1-D\n  `Tensor` where each element is a value and the corresponding index within the\n  tensor is the key.\n\n  Any input which does not have a corresponding index in \'vocabulary_list\'\n  (an out-of-vocabulary entry) is assigned the `default_value`\n\n  The underlying table must be initialized by calling\n  `session.run(tf.compat.v1.tables_initializer())` or\n  `session.run(table.init())` once.\n\n  Elements in `vocabulary_list` cannot have duplicates, otherwise when executing\n  the table initializer op, it will throw a `FailedPreconditionError`.\n\n  Sample Usages:\n\n  ```python\n  vocabulary_list = tf.constant(["emerson", "lake", "palmer"])\n  indices = tf.constant([1, 5], tf.int64)\n  table = tf.lookup.index_to_string_table_from_tensor(\n      vocabulary_list, default_value="UNKNOWN")\n  values = table.lookup(indices)\n  ...\n  tf.compat.v1.tables_initializer().run()\n\n  values.eval() ==> ["lake", "UNKNOWN"]\n  ```\n\n  Args:\n    vocabulary_list: A 1-D string `Tensor` that specifies the strings to map\n      from indices.\n    default_value: The value to use for out-of-vocabulary indices.\n    name: A name for this op (optional).\n\n  Returns:\n    The lookup table to map a string values associated to a given index `int64`\n    `Tensors`.\n\n  Raises:\n    ValueError: when `vocabulary_list` is not set.\n  '
    if vocabulary_list is None:
        raise ValueError('`vocabulary_list` argument must be specified.')
    with ops.name_scope(name, 'index_to_string'):
        vocabulary_list = ops.convert_to_tensor(vocabulary_list, dtypes.string)
        num_elements = array_ops.size(vocabulary_list)
        keys = math_ops.cast(math_ops.range(num_elements), dtypes.int64)
        init = KeyValueTensorInitializer(keys, vocabulary_list, dtypes.int64, dtypes.string, name='table_init')
        return StaticHashTableV1(init, default_value)

@tf_export('lookup.experimental.MutableHashTable')
@saveable_compat.legacy_saveable_name('table')
class MutableHashTable(LookupInterface):
    """A generic mutable hash table implementation.

  Data can be inserted by calling the `insert` method and removed by calling the
  `remove` method. It does not support initialization via the init method.

  `MutableHashTable` requires additional memory during checkpointing and restore
  operations to create temporary key and value tensors.

  Example usage:

  >>> table = tf.lookup.experimental.MutableHashTable(key_dtype=tf.string,
  ...                                                 value_dtype=tf.int64,
  ...                                                 default_value=-1)
  >>> keys_tensor = tf.constant(['a', 'b', 'c'])
  >>> vals_tensor = tf.constant([7, 8, 9], dtype=tf.int64)
  >>> input_tensor = tf.constant(['a', 'f'])
  >>> table.insert(keys_tensor, vals_tensor)
  >>> table.lookup(input_tensor).numpy()
  array([ 7, -1])
  >>> table.remove(tf.constant(['c']))
  >>> table.lookup(keys_tensor).numpy()
  array([ 7, 8, -1])
  >>> sorted(table.export()[0].numpy())
  [b'a', b'b']
  >>> sorted(table.export()[1].numpy())
  [7, 8]
  """

    def __init__(self, key_dtype, value_dtype, default_value, name='MutableHashTable', checkpoint=True, experimental_is_anonymous=False):
        if False:
            while True:
                i = 10
        "Creates an empty `MutableHashTable` object.\n\n    Creates a table, the type of its keys and values are specified by key_dtype\n    and value_dtype, respectively.\n\n    Args:\n      key_dtype: the type of the key tensors.\n      value_dtype: the type of the value tensors.\n      default_value: The value to use if a key is missing in the table.\n      name: A name for the operation (optional).\n      checkpoint: if True, the contents of the table are saved to and restored\n        from checkpoints. If `shared_name` is empty for a checkpointed table, it\n        is shared using the table node name.\n      experimental_is_anonymous: Whether to use anonymous mode for the\n        table (default is False). In anonymous mode, the table\n        resource can only be accessed via a resource handle. It can't\n        be looked up by a name. When all resource handles pointing to\n        that resource are gone, the resource will be deleted\n        automatically.\n\n    Returns:\n      A `MutableHashTable` object.\n\n    Raises:\n      ValueError: If checkpoint is True and no name was specified.\n    "
        self._default_value = ops.convert_to_tensor(default_value, dtype=value_dtype)
        self._value_shape = self._default_value.get_shape()
        self._checkpoint = checkpoint
        self._key_dtype = key_dtype
        self._value_dtype = value_dtype
        self._name = name
        self._is_anonymous = experimental_is_anonymous
        if not self._is_anonymous:
            self._shared_name = None
            if context.executing_eagerly():
                self._shared_name = 'table_%d' % (ops.uid(),)
        super(MutableHashTable, self).__init__(key_dtype, value_dtype)
        self._resource_handle = self._create_resource()
        if checkpoint:
            saveable = MutableHashTable._Saveable(self, name)
            if not context.executing_eagerly():
                ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)

    def _create_resource(self):
        if False:
            for i in range(10):
                print('nop')
        if self._is_anonymous:
            if self._default_value.get_shape().ndims == 0:
                table_ref = gen_lookup_ops.anonymous_mutable_hash_table(key_dtype=self._key_dtype, value_dtype=self._value_dtype, name=self._name)
            else:
                table_ref = gen_lookup_ops.anonymous_mutable_hash_table_of_tensors(key_dtype=self._key_dtype, value_dtype=self._value_dtype, value_shape=self._default_value.get_shape(), name=self._name)
        else:
            use_node_name_sharing = self._checkpoint and self._shared_name is None
            if self._default_value.get_shape().ndims == 0:
                table_ref = gen_lookup_ops.mutable_hash_table_v2(shared_name=self._shared_name, use_node_name_sharing=use_node_name_sharing, key_dtype=self._key_dtype, value_dtype=self._value_dtype, name=self._name)
            else:
                table_ref = gen_lookup_ops.mutable_hash_table_of_tensors_v2(shared_name=self._shared_name, use_node_name_sharing=use_node_name_sharing, key_dtype=self._key_dtype, value_dtype=self._value_dtype, value_shape=self._default_value.get_shape(), name=self._name)
        if context.executing_eagerly():
            self._table_name = None
        else:
            self._table_name = table_ref.op.name.split('/')[-1]
        return table_ref

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return self._table_name

    def size(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Compute the number of elements in this table.\n\n    Args:\n      name: A name for the operation (optional).\n\n    Returns:\n      A scalar tensor containing the number of elements in this table.\n    '
        with ops.name_scope(name, '%s_Size' % self.name, [self.resource_handle]):
            with ops.colocate_with(self.resource_handle):
                return gen_lookup_ops.lookup_table_size_v2(self.resource_handle)

    def remove(self, keys, name=None):
        if False:
            return 10
        "Removes `keys` and its associated values from the table.\n\n    If a key is not present in the table, it is silently ignored.\n\n    Args:\n      keys: Keys to remove. Can be a tensor of any shape. Must match the table's\n        key type.\n      name: A name for the operation (optional).\n\n    Returns:\n      The created Operation.\n\n    Raises:\n      TypeError: when `keys` do not match the table data types.\n    "
        if keys.dtype != self._key_dtype:
            raise TypeError(f'Dtype of argument `keys` must be {self._key_dtype}, received: {keys.dtype}')
        with ops.name_scope(name, '%s_lookup_table_remove' % self.name, (self.resource_handle, keys, self._default_value)):
            op = gen_lookup_ops.lookup_table_remove_v2(self.resource_handle, keys)
        return op

    def lookup(self, keys, dynamic_default_values=None, name=None):
        if False:
            print('Hello World!')
        "Looks up `keys` in a table, outputs the corresponding values.\n\n    The `default_value` is used for keys not present in the table.\n\n    Args:\n      keys: Keys to look up. Can be a tensor of any shape. Must match the\n        table's key_dtype.\n      dynamic_default_values: The values to use if a key is missing in the\n        table. If None (by default), the `table.default_value` will be used.\n        Shape of `dynamic_default_values` must be same with\n        `table.default_value` or the lookup result tensor.\n        In the latter case, each key will have a different default value.\n\n        For example:\n\n          ```python\n          keys = [0, 1, 3]\n          dynamic_default_values = [[1, 3, 4], [2, 3, 9], [8, 3, 0]]\n\n          # The key '0' will use [1, 3, 4] as default value.\n          # The key '1' will use [2, 3, 9] as default value.\n          # The key '3' will use [8, 3, 0] as default value.\n          ```\n\n      name: A name for the operation (optional).\n\n    Returns:\n      A tensor containing the values in the same shape as `keys` using the\n        table's value type.\n\n    Raises:\n      TypeError: when `keys` do not match the table data types.\n    "
        with ops.name_scope(name, '%s_lookup_table_find' % self.name, (self.resource_handle, keys, self._default_value)):
            keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name='keys')
            with ops.colocate_with(self.resource_handle):
                values = gen_lookup_ops.lookup_table_find_v2(self.resource_handle, keys, dynamic_default_values if dynamic_default_values is not None else self._default_value)
        return values

    def insert(self, keys, values, name=None):
        if False:
            while True:
                i = 10
        "Associates `keys` with `values`.\n\n    Args:\n      keys: Keys to insert. Can be a tensor of any shape. Must match the table's\n        key type.\n      values: Values to be associated with keys. Must be a tensor of the same\n        shape as `keys` and match the table's value type.\n      name: A name for the operation (optional).\n\n    Returns:\n      The created Operation.\n\n    Raises:\n      TypeError: when `keys` or `values` doesn't match the table data\n        types.\n    "
        with ops.name_scope(name, '%s_lookup_table_insert' % self.name, [self.resource_handle, keys, values]):
            keys = ops.convert_to_tensor(keys, self._key_dtype, name='keys')
            values = ops.convert_to_tensor(values, self._value_dtype, name='values')
            with ops.colocate_with(self.resource_handle):
                op = gen_lookup_ops.lookup_table_insert_v2(self.resource_handle, keys, values)
        return op

    def export(self, name=None):
        if False:
            i = 10
            return i + 15
        'Returns tensors of all keys and values in the table.\n\n    Args:\n      name: A name for the operation (optional).\n\n    Returns:\n      A pair of tensors with the first tensor containing all keys and the\n        second tensors containing all values in the table.\n    '
        with ops.name_scope(name, '%s_lookup_table_export_values' % self.name, [self.resource_handle]):
            with ops.colocate_with(self.resource_handle):
                (exported_keys, exported_values) = gen_lookup_ops.lookup_table_export_v2(self.resource_handle, self._key_dtype, self._value_dtype)
        return (exported_keys, exported_values)

    def _serialize_to_tensors(self):
        if False:
            while True:
                i = 10
        'Implements checkpointing protocols for `Trackable`.'
        tensors = self.export()
        return {'-keys': tensors[0], '-values': tensors[1]}

    def _restore_from_tensors(self, restored_tensors):
        if False:
            for i in range(10):
                print('nop')
        'Implements checkpointing protocols for `Trackable`.'
        with ops.name_scope('%s_table_restore' % self._name):
            with ops.colocate_with(self.resource_handle):
                return gen_lookup_ops.lookup_table_import_v2(self.resource_handle, restored_tensors['-keys'], restored_tensors['-values'])

    def _copy_trackable_to_cpu(self, object_map):
        if False:
            return 10
        'Implements checkpointing protocols for `Trackable`.'
        if self not in object_map:
            object_map[self] = MutableHashTable(self._key_dtype, self._value_dtype, self._default_value, self._name, self._checkpoint, self._is_anonymous)
        serialized = self._serialize_to_tensors()
        object_map[self]._restore_from_tensors(serialized)

    class _Saveable(BaseSaverBuilder.SaveableObject):
        """SaveableObject implementation for DenseHashTable."""

        def __init__(self, table, name, table_name=None):
            if False:
                while True:
                    i = 10
            tensors = table.export()
            specs = [BaseSaverBuilder.SaveSpec(tensors[0], '', name + '-keys'), BaseSaverBuilder.SaveSpec(tensors[1], '', name + '-values')]
            self.table_name = table_name or name
            super(MutableHashTable._Saveable, self).__init__(table, specs, name)

        def restore(self, restored_tensors, restored_shapes):
            if False:
                for i in range(10):
                    print('nop')
            del restored_shapes
            with ops.name_scope('%s_table_restore' % self.table_name):
                with ops.colocate_with(self.op.resource_handle):
                    return gen_lookup_ops.lookup_table_import_v2(self.op.resource_handle, restored_tensors[0], restored_tensors[1])

@tf_export('lookup.experimental.DenseHashTable')
@saveable_compat.legacy_saveable_name('table')
class DenseHashTable(LookupInterface):
    """A mutable hash table with faster lookups and higher memory usage.

  Data can be inserted by calling the `insert` method and removed by calling the
  `remove` method. It does not support initialization via the init method.

  Compared to `MutableHashTable`, `DenseHashTable` offers generally faster
  `insert`, `remove` and `lookup` operations, in exchange for a higher overall
  memory footprint.

  It uses "open addressing" with quadratic reprobing to resolve collisions. This
  requires specifying two keys in the key space, `empty_key` and `deleted_key`,
  that can never inserted into the table.

  Unlike `MutableHashTable`, `DenseHashTable` does not require additional memory
  for temporary tensors created during checkpointing and restore operations.

  Example usage:

  >>> table = tf.lookup.experimental.DenseHashTable(
  ...     key_dtype=tf.string,
  ...     value_dtype=tf.int64,
  ...     default_value=-1,
  ...     empty_key='',
  ...     deleted_key='$')
  >>> keys = tf.constant(['a', 'b', 'c'])
  >>> values = tf.constant([0, 1, 2], dtype=tf.int64)
  >>> table.insert(keys, values)
  >>> table.remove(tf.constant(['c']))
  >>> table.lookup(tf.constant(['a', 'b', 'c','d'])).numpy()
  array([ 0,  1, -1, -1])
  """

    def __init__(self, key_dtype, value_dtype, default_value, empty_key, deleted_key, initial_num_buckets=None, name='MutableDenseHashTable', checkpoint=True, experimental_is_anonymous=False):
        if False:
            for i in range(10):
                print('nop')
        "Creates an empty `DenseHashTable` object.\n\n    Creates a table, the type of its keys and values are specified by key_dtype\n    and value_dtype, respectively.\n\n    Args:\n      key_dtype: the type of the key tensors.\n      value_dtype: the type of the value tensors.\n      default_value: The value to use if a key is missing in the table.\n      empty_key: the key to use to represent empty buckets internally. Must not\n        be used in insert, remove or lookup operations.\n      deleted_key: the key to use to represent deleted buckets internally. Must\n        not be used in insert, remove or lookup operations and be different from\n        the empty_key.\n      initial_num_buckets: the initial number of buckets (optional,\n        default to 2^17=131072). Note that the default value is\n        relatively large (~1MB), so if you are going to create many\n        tables (likely the case when `experimental_is_anonymous` is\n        `True`), you should set `initial_num_buckets` to a smaller\n        value to reduce memory usage.\n      name: A name for the operation (optional).\n      checkpoint: if True, the contents of the table are saved to and restored\n        from checkpoints. If `shared_name` is empty for a checkpointed table, it\n        is shared using the table node name.\n      experimental_is_anonymous: Whether to use anonymous mode for the\n        table (default is False). In anonymous mode, the table\n        resource can only be accessed via a resource handle. It can't\n        be looked up by a name. When all resource handles pointing to\n        that resource are gone, the resource will be deleted\n        automatically.\n\n    Returns:\n      A `DenseHashTable` object.\n\n    Raises:\n      ValueError: If checkpoint is True and no name was specified.\n    "
        self._default_value = ops.convert_to_tensor(default_value, dtype=value_dtype, name='default_value')
        self._key_dtype = key_dtype
        self._value_dtype = value_dtype
        self._initial_num_buckets = initial_num_buckets
        self._value_shape = self._default_value.get_shape()
        self._checkpoint = checkpoint
        self._name = name
        self._empty_key = empty_key
        self._deleted_key = deleted_key
        self._is_anonymous = experimental_is_anonymous
        if not self._is_anonymous:
            self._shared_name = None
            if context.executing_eagerly():
                self._shared_name = 'table_%d' % (ops.uid(),)
        super(DenseHashTable, self).__init__(key_dtype, value_dtype)
        self._resource_handle = self._create_resource()
        if checkpoint:
            saveable = DenseHashTable._Saveable(self, name)
            if not context.executing_eagerly():
                ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)

    def _create_resource(self):
        if False:
            return 10
        empty_key = ops.convert_to_tensor(self._empty_key, dtype=self._key_dtype, name='empty_key')
        deleted_key = ops.convert_to_tensor(self._deleted_key, dtype=self._key_dtype, name='deleted_key')
        if self._is_anonymous:
            table_ref = gen_lookup_ops.anonymous_mutable_dense_hash_table(empty_key=empty_key, deleted_key=deleted_key, value_dtype=self._value_dtype, value_shape=self._value_shape, initial_num_buckets=self._initial_num_buckets, name=self._name)
        else:
            use_node_name_sharing = self._checkpoint and self._shared_name is None
            table_ref = gen_lookup_ops.mutable_dense_hash_table_v2(empty_key=empty_key, deleted_key=deleted_key, shared_name=self._shared_name, use_node_name_sharing=use_node_name_sharing, value_dtype=self._value_dtype, value_shape=self._value_shape, initial_num_buckets=self._initial_num_buckets, name=self._name)
        if context.executing_eagerly():
            self._table_name = None
        else:
            self._table_name = table_ref.op.name.split('/')[-1]
        return table_ref

    @property
    def name(self):
        if False:
            i = 10
            return i + 15
        return self._table_name

    def size(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Compute the number of elements in this table.\n\n    Args:\n      name: A name for the operation (optional).\n\n    Returns:\n      A scalar tensor containing the number of elements in this table.\n    '
        with ops.name_scope(name, '%s_Size' % self.name, [self.resource_handle]):
            with ops.colocate_with(self.resource_handle):
                return gen_lookup_ops.lookup_table_size_v2(self.resource_handle)

    def lookup(self, keys, name=None):
        if False:
            while True:
                i = 10
        "Looks up `keys` in a table, outputs the corresponding values.\n\n    The `default_value` is used for keys not present in the table.\n\n    Args:\n      keys: Keys to look up. Can be a tensor of any shape. Must match the\n        table's key_dtype.\n      name: A name for the operation (optional).\n\n    Returns:\n      A tensor containing the values in the same shape as `keys` using the\n        table's value type.\n\n    Raises:\n      TypeError: when `keys` do not match the table data types.\n    "
        with ops.name_scope(name, '%s_lookup_table_find' % self.name, [self.resource_handle, keys]):
            keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name='keys')
            with ops.colocate_with(self.resource_handle):
                values = gen_lookup_ops.lookup_table_find_v2(self.resource_handle, keys, self._default_value)
        return values

    def insert_or_assign(self, keys, values, name=None):
        if False:
            return 10
        "Associates `keys` with `values`.\n\n    Args:\n      keys: Keys to insert. Can be a tensor of any shape. Must match the table's\n        key type.\n      values: Values to be associated with keys. Must be a tensor of the same\n        shape as `keys` and match the table's value type.\n      name: A name for the operation (optional).\n\n    Returns:\n      The created Operation.\n\n    Raises:\n      TypeError: when `keys` or `values` doesn't match the table data\n        types.\n    "
        with ops.name_scope(name, '%s_lookup_table_insert' % self.name, [self.resource_handle, keys, values]):
            keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name='keys')
            values = ops.convert_to_tensor(values, dtype=self._value_dtype, name='values')
            with ops.colocate_with(self.resource_handle):
                op = gen_lookup_ops.lookup_table_insert_v2(self.resource_handle, keys, values)
            return op

    def insert(self, keys, values, name=None):
        if False:
            i = 10
            return i + 15
        "Associates `keys` with `values`.\n\n    Args:\n      keys: Keys to insert. Can be a tensor of any shape. Must match the table's\n        key type.\n      values: Values to be associated with keys. Must be a tensor of the same\n        shape as `keys` and match the table's value type.\n      name: A name for the operation (optional).\n\n    Returns:\n      The created Operation.\n\n    Raises:\n      TypeError: when `keys` or `values` doesn't match the table data\n        types.\n    "
        return self.insert_or_assign(keys, values, name)

    def erase(self, keys, name=None):
        if False:
            print('Hello World!')
        "Removes `keys` and its associated values from the table.\n\n    If a key is not present in the table, it is silently ignored.\n\n    Args:\n      keys: Keys to remove. Can be a tensor of any shape. Must match the table's\n        key type.\n      name: A name for the operation (optional).\n\n    Returns:\n      The created Operation.\n\n    Raises:\n      TypeError: when `keys` do not match the table data types.\n    "
        if keys.dtype != self._key_dtype:
            raise TypeError('Signature mismatch. Keys must be dtype %s, got %s.' % (self._key_dtype, keys.dtype))
        with ops.name_scope(name, '%s_lookup_table_remove' % self.name, (self.resource_handle, keys, self._default_value)):
            op = gen_lookup_ops.lookup_table_remove_v2(self.resource_handle, keys)
        return op

    def remove(self, keys, name=None):
        if False:
            print('Hello World!')
        "Removes `keys` and its associated values from the table.\n\n    If a key is not present in the table, it is silently ignored.\n\n    Args:\n      keys: Keys to remove. Can be a tensor of any shape. Must match the table's\n        key type.\n      name: A name for the operation (optional).\n\n    Returns:\n      The created Operation.\n\n    Raises:\n      TypeError: when `keys` do not match the table data types.\n    "
        return self.erase(keys, name)

    def export(self, name=None):
        if False:
            i = 10
            return i + 15
        'Returns tensors of all keys and values in the table.\n\n    Args:\n      name: A name for the operation (optional).\n\n    Returns:\n      A pair of tensors with the first tensor containing all keys and the\n        second tensors containing all values in the table.\n    '
        with ops.name_scope(name, '%s_lookup_table_export_values' % self.name, [self.resource_handle]):
            with ops.colocate_with(self.resource_handle):
                (exported_keys, exported_values) = gen_lookup_ops.lookup_table_export_v2(self.resource_handle, self._key_dtype, self._value_dtype)
        return (exported_keys, exported_values)

    def _serialize_to_tensors(self):
        if False:
            print('Hello World!')
        'Implements checkpointing interface in `Trackable`.'
        tensors = self.export()
        return {'-keys': tensors[0], '-values': tensors[1]}

    def _restore_from_tensors(self, restored_tensors):
        if False:
            for i in range(10):
                print('nop')
        'Implements checkpointing interface in `Trackable`.'
        with ops.name_scope('%s_table_restore' % self._name):
            with ops.colocate_with(self.resource_handle):
                return gen_lookup_ops.lookup_table_import_v2(self.resource_handle, restored_tensors['-keys'], restored_tensors['-values'])

    def _copy_trackable_to_cpu(self, object_map):
        if False:
            i = 10
            return i + 15
        'Implements checkpointing protocols for `Trackable`.'
        if self not in object_map:
            object_map[self] = DenseHashTable(self._key_dtype, self._value_dtype, self._default_value, self._empty_key, self._deleted_key, self._initial_num_buckets, self._name, self._checkpoint, self._is_anonymous)
        serialized = self._serialize_to_tensors()
        object_map[self]._restore_from_tensors(serialized)

    class _Saveable(BaseSaverBuilder.SaveableObject):
        """SaveableObject implementation for DenseHashTable."""

        def __init__(self, table, name, table_name=None):
            if False:
                return 10
            tensors = table.export()
            specs = [BaseSaverBuilder.SaveSpec(tensors[0], '', name + '-keys'), BaseSaverBuilder.SaveSpec(tensors[1], '', name + '-values')]
            self.table_name = table_name or name
            super(DenseHashTable._Saveable, self).__init__(table, specs, name)

        def restore(self, restored_tensors, restored_shapes):
            if False:
                while True:
                    i = 10
            del restored_shapes
            with ops.name_scope('%s_table_restore' % self.table_name):
                with ops.colocate_with(self.op.resource_handle):
                    return gen_lookup_ops.lookup_table_import_v2(self.op.resource_handle, restored_tensors[0], restored_tensors[1])