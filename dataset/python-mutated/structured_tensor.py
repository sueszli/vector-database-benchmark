"""Structured Tensors."""
import re
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import dynamic_ragged_shape
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.row_partition import RowPartition
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
_FieldValue = Union[tensor.Tensor, ragged_tensor.RaggedTensor, 'StructuredTensor', extension_type.ExtensionType]
_FieldFn = Callable[[_FieldValue], _FieldValue]

@tf_export('experimental.StructuredTensor')
class StructuredTensor(extension_type.BatchableExtensionType):
    """A multidimensional collection of structures with the same schema.

  A **`StructuredTensor`** is a multi-dimensional collection of ***structures***
  with the same ***schema***, where:

  * A ***schema*** is a collection of fields, each of which has a name and type.
  * A ***structure*** maps each field in the schema to a tensor value (which
    could be a nested StructuredTensor).

  As an important special case, a 1D `StructuredTensor` encodes a 2D table,
  where columns are heterogeneous `Tensor`s, and rows are the aligned elements
  in each of those `Tensor`s.

  Internally, StructuredTensors use a "field-major" encoding: for each leaf
  field, there is a single tensor that stores the value of that field for all
  structures in the `StructuredTensor`.

  ### Examples

  >>> # A scalar StructuredTensor describing a single person.
  >>> s1 = tf.experimental.StructuredTensor.from_pyval(
  ...     {"age": 82, "nicknames": ["Bob", "Bobby"]})
  >>> s1.shape
  TensorShape([])
  >>> s1["age"]
  <tf.Tensor: shape=(), dtype=int32, numpy=82>

  >>> # A vector StructuredTensor describing three people.
  >>> s2 = tf.experimental.StructuredTensor.from_pyval([
  ...     {"age": 12, "nicknames": ["Josaphine"]},
  ...     {"age": 82, "nicknames": ["Bob", "Bobby"]},
  ...     {"age": 42, "nicknames": ["Elmo"]}])
  >>> s2.shape
  TensorShape([3])
  >>> s2[0]["age"]
  <tf.Tensor: shape=(), dtype=int32, numpy=12>


  ### Field Paths

  A *field path* is a tuple of field names, specifying the path to a nested
  field.
  """
    _fields: Mapping[str, _FieldValue]
    _ragged_shape: dynamic_ragged_shape.DynamicRaggedShape
    __name__ = 'tf.StructuredTensor'
    FieldName = Union[str, Sequence[str]]

    def __init__(self, fields: Mapping[str, _FieldValue], ragged_shape: dynamic_ragged_shape.DynamicRaggedShape):
        if False:
            print('Hello World!')
        self._fields = fields
        self._ragged_shape = ragged_shape

    @classmethod
    def _old_init(cls, fields, shape, nrows, row_partitions, internal=False):
        if False:
            print('Hello World!')
        'Private constructor -- use factory methods to create StructuredTensors.\n\n    This constructor builds a `StructuredTensor` from the given attributes,\n    performing minimal validation.\n\n    Args:\n      fields: A dictionary mapping from string to `Tensor`, `RaggedTensor`, or\n        `StructuredTensor`.  (This dict is not copied, so the caller must ensure\n        that it does not get mutated via leaked references.)\n      shape: `tf.TensorShape` with statically known rank.\n      nrows: scalar integer `tf.Tensor`, or `None` if `shape.rank==0`.\n      row_partitions: tuple of `RowPartition`s, with length `shape.rank-1`.\n      internal: ignored argument.\n\n    Returns:\n      a StructuredTensor.\n    '
        assert isinstance(fields, dict), fields
        assert isinstance(shape, tensor_shape.TensorShape), shape
        assert nrows is None or isinstance(nrows, tensor.Tensor), nrows
        assert row_partitions is None or isinstance(row_partitions, tuple), row_partitions
        return StructuredTensor(fields=fields, ragged_shape=_dynamic_ragged_shape_init(fields, shape, nrows, row_partitions))

    @classmethod
    def from_shape(cls, ragged_shape: dynamic_ragged_shape.DynamicRaggedShape) -> 'StructuredTensor':
        if False:
            for i in range(10):
                print('nop')
        'Creates a `StructuredTensor` with no fields and ragged_shape.\n\n    Args:\n      ragged_shape: the shape of the structured tensor.\n\n    Returns:\n      a StructuredTensor with no fields and ragged_shape.\n    '
        return StructuredTensor(fields={}, ragged_shape=ragged_shape)

    @classmethod
    def from_fields(cls, fields, shape=(), nrows=None, row_partitions=None, validate=False):
        if False:
            for i in range(10):
                print('nop')
        'Creates a `StructuredTensor` from a dictionary of fields.\n\n    Args:\n      fields: A dictionary mapping from string to `Tensor`, `RaggedTensor`, or\n        `StructuredTensor`, providing the values for individual fields in each\n        structure.  If `shape.rank > 0`, then every tensor in `fields` must have\n        the same shape in the first `shape.rank` dimensions; and that shape must\n        be compatible with `shape`; and `result[i1...iN][key] =\n        fields[key][i1...iN]` (where `N==shape.rank`).\n      shape: A `TensorShape`: static information about the shape of the\n        `StructuredTensor`.  Must have a known `rank`.  Defaults to scalar shape\n        (i.e. `rank=0`).\n      nrows: scalar integer tensor containing the number of rows in this\n        `StructuredTensor`.  Should only be specified if `shape.rank > 0`.\n        Default value is inferred from the `fields` values.  If `fields` is\n        empty, then this must be specified.\n      row_partitions: A list of `RowPartition`s describing the (possibly ragged)\n        shape of this `StructuredTensor`.  Should only be specified if\n        `shape.rank > 1`.  Default value is inferred from the `fields` values.\n        If `fields` is empty, then this must be specified.\n      validate: If true, then add runtime validation ops that check that the\n        field values all have compatible shapes in the outer `shape.rank`\n        dimensions.\n\n    Returns:\n      A `StructuredTensor`.\n\n    Examples:\n\n      >>> tf.experimental.StructuredTensor.from_fields({\'x\': 1, \'y\': [1, 2, 3]})\n      <StructuredTensor(\n        fields={\n          "x": tf.Tensor(1, shape=(), dtype=int32),\n          "y": tf.Tensor([1 2 3], shape=(3,), dtype=int32)},\n        shape=())>\n\n      >>> tf.experimental.StructuredTensor.from_fields(\n      ...     {\'foo\': [1, 2], \'bar\': [3, 4]}, shape=[2])\n      <StructuredTensor(\n        fields={\n          "bar": tf.Tensor([3 4], shape=(2,), dtype=int32),\n          "foo": tf.Tensor([1 2], shape=(2,), dtype=int32)},\n        shape=(2,))>\n    '
        shape = tensor_shape.as_shape(shape)
        rank = shape.rank
        if rank is None:
            raise ValueError("StructuredTensor's shape must have known rank.")
        if not isinstance(fields, dict):
            raise TypeError('fields must be a dictionary, got %s' % type(fields).__name__)
        if rank < 2 and row_partitions:
            raise ValueError('row_partitions must be None or [] if shape.rank<2')
        if rank == 0 and nrows is not None:
            raise ValueError('nrows must be None if shape.rank==0')
        if row_partitions is not None:
            row_partitions = tuple(row_partitions)
            if len(row_partitions) != max(0, rank - 1):
                raise ValueError('len(row_partitions) must be shape.rank-1')
        elif rank < 2:
            row_partitions = ()
        fields = dict(fields)
        with ops.name_scope(None, 'StructuredTensor', fields.values()):
            shape = _dynamic_ragged_shape_init(fields, shape, nrows, row_partitions)
            if shape.rank > 1:
                shape = shape._with_num_row_partitions(shape.rank - 1)
            for (key, value) in fields.items():
                if not isinstance(key, str):
                    raise TypeError(f'Unexpected type for key in `fields`: {key}')
                if not _FIELD_NAME_RE.match(key):
                    raise ValueError('Field name %r is not currently allowed.' % key)
                fields[key] = _convert_to_structured_field_value(value)
                fields = dict([(k, _replace_row_partitions(v, row_partitions)) for (k, v) in fields.items()])
            return cls(fields=fields, ragged_shape=shape)

    @classmethod
    def from_fields_and_rank(cls, fields: Mapping[str, _FieldValue], rank: int, validate: bool=False, dtype: Optional[dtypes.DType]=None) -> 'StructuredTensor':
        if False:
            for i in range(10):
                print('nop')
        'Creates a `StructuredTensor` from a nonempty dictionary of fields.\n\n    Note that if the shape dtype is not specified, the shape dtype will be\n    inferred from any fields that have a shape dtype. If fields differ, then\n    int64 will be preferred to int32, because coercing from int32 to int64 is\n    safer than coercing from int64 to int32.\n\n    If there are no ragged fields, then it will be int64 by default, but this\n    will be changed to int32 in the future.\n\n    Args:\n      fields: A dictionary mapping from string to `Tensor`, `RaggedTensor`, or\n        `StructuredTensor`, providing the values for individual fields in each\n        structure.  If `rank > 0`, then every tensor in `fields` must have the\n        same shape in the first `rank` dimensions. Cannot be empty.\n      rank: The rank of the resulting structured tensor.\n      validate: If true, then add runtime validation ops that check that the\n        field values all have compatible shapes in the outer `rank` dimensions.\n      dtype: If specified, then forces dtype of the shape to be this.\n\n    Returns:\n      A `StructuredTensor`.\n    Examples:\n      >>> tf.experimental.StructuredTensor.from_fields_and_rank(\n      ...     {\'x\': 1, \'y\': [1, 2, 3]}, 0)\n      <StructuredTensor(\n        fields={\n          "x": tf.Tensor(1, shape=(), dtype=int32),\n          "y": tf.Tensor([1 2 3], shape=(3,), dtype=int32)},\n        shape=())>\n      >>> StructuredTensor.from_fields_and_rank({\'foo\': [1, 2], \'bar\': [3, 4]},\n      ...                              1)\n      <StructuredTensor(\n        fields={\n          "bar": tf.Tensor([3 4], shape=(2,), dtype=int32),\n          "foo": tf.Tensor([1 2], shape=(2,), dtype=int32)},\n        shape=(2,))>\n    '
        if not fields:
            raise ValueError('Must provide at least one field')
        if not isinstance(rank, int):
            raise ValueError('rank must be an integer')
        if rank < 0:
            raise ValueError('rank must be nonnegative')
        fields = {k: _convert_to_structured_field_value(v) for (k, v) in fields.items()}
        if dtype is None:
            dtype = _find_shape_dtype(fields, None, None)
        fields = _fields_with_dtype(fields, dtype)
        shape = _shape_from_fields(fields, rank, dtype)
        if rank > 1:
            shape = shape._with_num_row_partitions(rank - 1)
        new_rp = shape._row_partitions
        fields = {k: _replace_row_partitions(v, new_rp) for (k, v) in fields.items()}
        return StructuredTensor(fields=fields, ragged_shape=shape)

    def with_updates(self, updates: Dict[FieldName, Union[_FieldValue, _FieldFn, None]], validate: bool=False) -> 'StructuredTensor':
        if False:
            for i in range(10):
                print('nop')
        'Creates a new `StructuredTensor` with the updated fields.\n\n    If this `StructuredTensor` is a scalar, and `k` is the `FieldName` being\n    updated and `v` the new value, then:\n\n    ```\n    result[k] = v              # If (k, v) is in updates and v is a FieldValue\n    result[k] = f(self[k])     # If (k, f) is in updates and f is a FieldFn\n    result[k] = self[k]        # If k is in self.field_names but not in updates\n    ```\n\n    If this `StructuredTensor` has rank `N` and shape `[D1...DN]`, then each\n    FieldValue `v` in `updates` must have shape `[D1...DN, ...]`, that is,\n    prefixed with the same shape as the `StructuredTensor`. Then the resulting\n    `StructuredTensor` will have:\n\n    ```\n    result[i1...iN][k] = v[i1...iN]                        # (k, v) in updates\n    result[i1...iN][k] = f(self.field_value(k))[i1...iN]   # (k, f) in updates\n    result[i1...iN][k] = self[i1...iN][k]                  # k not in updates\n    ```\n\n    Note that `result.shape` is always equal to `self.shape` (but the shapes\n    of nested StructuredTensors may be changed if they are updated with new\n    values).\n\n    Args:\n      updates: A dictionary mapping `FieldName` to either a `FieldValue` to be\n        used to update, or a `FieldFn` that will transform the value for the\n        given `FieldName`. `FieldName` can be a string for a direct field, or a\n        sequence of strings to refer to a nested sub-field. `FieldFn` is a\n        function that takes a `FieldValue` as input and should return a\n        `FieldValue`. All other fields are copied over to the new\n        `StructuredTensor`. New `FieldName` can be given (to add new fields),\n        but only to existing `StructuredTensor`, it won\'t automatically create\n        new nested structures -- but one can create a whole `StructureTensor`\n        sub-structure and set that into an existing structure. If the new value\n        is set to `None`, it is removed.\n      validate: If true, then add runtime validation ops that check that the\n        field values all have compatible shapes in the outer `shape.rank`\n        dimensions.\n\n    Returns:\n      A `StructuredTensor`.\n\n    Raises:\n      `ValueError`: If the any of the `FieldName` keys points to non-existent\n        sub-structures, if parent and child nodes are updated, if shapes\n        change, if a delete update is given for a non-existent field, or if a\n        `FieldFn` transforming function is given for a `FieldName` that doesn\'t\n        yet exist.\n\n    Examples:\n\n    >>> shoes_us = tf.experimental.StructuredTensor.from_pyval([\n    ...    {"age": 12, "nicknames": ["Josaphine"],\n    ...       "shoes": {"sizes": [8.0, 7.5, 7.5]}},\n    ...    {"age": 82, "nicknames": ["Bob", "Bobby"],\n    ...        "shoes": {"sizes": [11.0, 11.5, 12.0]}},\n    ...    {"age": 42, "nicknames": ["Elmo"],\n    ...        "shoes": {"sizes": [9.0, 9.5, 10.0]}}])\n    >>> def us_to_europe(t):\n    ...   return tf.round(t * 2.54 + 17.0)  # Rough approximation.\n    >>> shoe_sizes_key = ("shoes", "sizes")\n    >>> shoes_eu = shoes_us.with_updates({shoe_sizes_key: us_to_europe})\n    >>> shoes_eu.field_value(shoe_sizes_key)\n    <tf.RaggedTensor [[37.0, 36.0, 36.0], [45.0, 46.0, 47.0],\n    [40.0, 41.0, 42.0]]>\n    '
        updates_items = [(_normalize_field_name_to_tuple(name), value) for (name, value) in updates.items()]
        updates_items = sorted(updates_items)
        for i in range(1, len(updates_items)):
            name = updates_items[i][0]
            prev_name = updates_items[i - 1][0]
            if name[:len(prev_name)] == prev_name:
                raise ValueError('`StructuredTensor.with_updates` does not allow both parent and child nodes to be updated: parent={}, child={}. If needed you can update child nodes in the parent update value.'.format(prev_name, name))
        return self._with_updates_impl((), updates_items, validate)

    def _with_updates_impl(self, error_prefix: Tuple[str, ...], updates: List[Tuple[FieldName, Union[_FieldValue, _FieldFn]]], validate: bool) -> 'StructuredTensor':
        if False:
            i = 10
            return i + 15
        'Recursive part of `with_updates` implementation.'
        new_fields = dict(self._fields)

        def name_fullpath(name: Sequence[str]) -> str:
            if False:
                return 10
            return str(error_prefix + (name,))

        def apply_value(name: str, value: Union[_FieldValue, _FieldFn]) -> _FieldValue:
            if False:
                i = 10
                return i + 15
            if callable(value):
                if name not in new_fields:
                    raise ValueError('`StructuredTensor.with_updates` cannot update the field {} because a transforming function was given, but that field does not already exist.'.format(name_fullpath(name)))
                value = value(new_fields[name])
            return value
        for (name, value) in updates:
            if not name or not name[0]:
                raise ValueError('`StructuredTensor.with_updates` does not allow empty names {}.'.format(name_fullpath(name)))
            if len(name) == 1:
                name = name[0]
                if value is None:
                    if name not in new_fields:
                        raise ValueError('`StructuredTensor.with_updates` cannot delete field {} because it is not present.'.format(name_fullpath(name)))
                    new_fields.pop(name)
                else:
                    new_fields[name] = apply_value(name, value)
            else:
                prefix = name[0]
                suffix = name[1:]
                if prefix not in new_fields:
                    raise ValueError('`StructuredTensor.with_updates` cannot create new sub-field {} if parent field {} is not set.'.format(error_prefix + tuple(name), name_fullpath(prefix)))
                current_value = new_fields[prefix]
                if not isinstance(current_value, StructuredTensor):
                    raise ValueError('`StructuredTensor.with_updates` cannot create new sub-field {} if parent structure {} is not a `StructuredTensor` that can contain sub-structures -- it is a `{}`.'.format(error_prefix + tuple(name), name_fullpath(prefix), type(current_value)))
                one_update = [(suffix, value)]
                value = current_value._with_updates_impl(error_prefix + (prefix,), one_update, validate)
                new_fields[prefix] = value
        try:
            return StructuredTensor.from_fields(new_fields, shape=self.shape, row_partitions=self.row_partitions, nrows=self.nrows(), validate=validate)
        except ValueError as e:
            msg = '`StructuredTensor.with_updates` failed'
            if error_prefix:
                msg = '{} for field {}'.format(msg, error_prefix)
            raise ValueError(msg) from e

    def _promote_helper(self, source_path, new_parent_path):
        if False:
            return 10
        'Creates a promoted field without adding it to the structure.\n\n    Args:\n      source_path: the source path in the structured tensor.\n      new_parent_path: the new parent path. Must be a prefix of source_path.\n\n    Returns:\n      a composite tensor of source_path promoted.\n    Raises:\n      ValueError: if the shape of the field is unknown and the right strategy\n      cannot be determined.\n    '
        current_field = self.field_value(source_path)
        new_parent_rank = self.field_value(new_parent_path).rank
        parent_rank = self.field_value(source_path[:-1]).rank
        if new_parent_rank == parent_rank:
            return current_field
        current_field_rank = current_field.shape.rank
        if current_field_rank is None:
            raise ValueError('Cannot determine if dimensions should be merged.')
        inner_dim = min(parent_rank, current_field_rank - 1)
        if inner_dim <= new_parent_rank:
            return current_field
        return _merge_dims_generic(current_field, new_parent_rank, inner_dim)

    def promote(self, source_path, new_name):
        if False:
            while True:
                i = 10
        "Promotes a field, merging dimensions between grandparent and parent.\n\n    >>> d = [\n    ...  {'docs': [{'tokens':[1, 2]}, {'tokens':[3]}]},\n    ...  {'docs': [{'tokens':[7]}]}]\n    >>> st = tf.experimental.StructuredTensor.from_pyval(d)\n    >>> st2 =st.promote(('docs','tokens'), 'docs_tokens')\n    >>> st2[0]['docs_tokens']\n    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 2, 3], dtype=int32)>\n    >>> st2[1]['docs_tokens']\n    <tf.Tensor: shape=(1,), dtype=int32, numpy=array([7], dtype=int32)>\n\n    Args:\n      source_path: the path of the field or substructure to promote; must have\n        length at least 2.\n      new_name: the name of the new field (must be a string).\n\n    Returns:\n      a modified structured tensor with the new field as a child of the\n      grandparent of the source_path.\n\n    Raises:\n      ValueError: if source_path is not a list or a tuple or has a length\n        less than two, or new_name is not a string, or the rank\n        of source_path is unknown and it is needed.\n    "
        if not isinstance(new_name, str):
            raise ValueError('new_name is not a string')
        if not isinstance(source_path, (list, tuple)):
            raise ValueError('source_path must be a list or tuple')
        if len(source_path) < 2:
            raise ValueError('source_path must have length at least two')
        grandparent_path = source_path[:-2]
        new_field = self._promote_helper(source_path, grandparent_path)
        new_path = grandparent_path + (new_name,)
        return self.with_updates({new_path: new_field})

    @property
    def rank(self):
        if False:
            while True:
                i = 10
        'The rank of this StructuredTensor.  Guaranteed not to be `None`.'
        return self._ragged_shape.rank

    @property
    def shape(self):
        if False:
            print('Hello World!')
        'The static shape of this StructuredTensor.\n\n    The returned `TensorShape` is guaranteed to have a known rank, but the\n    individual dimension sizes may be unknown.\n\n    Returns:\n      `tf.TensorShape`\n    '
        return self._ragged_shape._to_tensor_shape()

    @property
    def _row_partitions(self):
        if False:
            i = 10
            return i + 15
        'Deprecated form of row_partitions.'
        return self.row_partitions

    @property
    def row_partitions(self):
        if False:
            while True:
                i = 10
        "A tuple of `RowPartition`s defining the shape of this `StructuredTensor`.\n\n    When `self.rank <= 1`, this tuple will be empty.\n\n    When `self.rank > 1`, these `RowPartitions` define the shape of the\n    `StructuredTensor` by describing how a flat (1D) list of structures can be\n    repeatedly partitioned to form a higher-dimensional object.  In particular,\n    the flat list is first partitioned into sublists using `row_partitions[-1]`,\n    and then those sublists are further partitioned using `row_partitions[-2]`,\n    etc.  The following examples show the row partitions used to describe\n    several different `StructuredTensor`, each of which contains 8 copies of\n    the same structure (`x`):\n\n    >>> x = {'a': 1, 'b': ['foo', 'bar', 'baz']}       # shape = [] (scalar)\n\n    >>> s1 = [[x, x, x, x], [x, x, x, x]]              # shape = [2, 4]\n    >>> tf.experimental.StructuredTensor.from_pyval(s1).row_partitions\n    (tf.RowPartition(row_splits=[0 4 8]),)\n\n    >>> s2 = [[x, x], [x, x], [x, x], [x, x]]          # shape = [4, 2]\n    >>> tf.experimental.StructuredTensor.from_pyval(s2).row_partitions\n    (tf.RowPartition(row_splits=[0 2 4 6 8]),)\n\n    >>> s3 = [[x, x, x], [], [x, x, x, x], [x]]        # shape = [2, None]\n    >>> tf.experimental.StructuredTensor.from_pyval(s3).row_partitions\n    (tf.RowPartition(row_splits=[0 3 3 7 8]),)\n\n    >>> s4 = [[[x, x], [x, x]], [[x, x], [x, x]]]      # shape = [2, 2, 2]\n    >>> tf.experimental.StructuredTensor.from_pyval(s4).row_partitions\n    (tf.RowPartition(row_splits=[0 2 4]),\n     tf.RowPartition(row_splits=[0 2 4 6 8]))\n\n\n    >>> s5 = [[[x, x], [x]], [[x, x]], [[x, x], [x]]]  # shape = [3, None, None]\n    >>> tf.experimental.StructuredTensor.from_pyval(s5).row_partitions\n    (tf.RowPartition(row_splits=[0 2 3 5]),\n     tf.RowPartition(row_splits=[0 2 3 5 7 8]))\n\n    Note that shapes for nested fields (such as `x['b']` in the above example)\n    are not considered part of the shape of a `StructuredTensor`, and are not\n    included in `row_partitions`.\n\n    If this `StructuredTensor` has a ragged shape (i.e., if any of the\n    `row_partitions` is not uniform in size), then all fields will be encoded\n    as either `RaggedTensor`s or `StructuredTensor`s with these `RowPartition`s\n    used to define their outermost `self.rank` dimensions.\n\n    Returns:\n      A `tuple` of `RowPartition` objects with length `self.rank - 1`\n      (or `0` if `self.rank < 2`)\n\n    "
        if self.rank < 2:
            return ()
        return self._ragged_shape._as_row_partitions()

    def nrows(self):
        if False:
            return 10
        'The number of rows in this StructuredTensor (if rank>0).\n\n    This means the length of the outer-most dimension of the StructuredTensor.\n\n    Notice that if `self.rank > 1`, then this equals the number of rows\n    of the first row partition. That is,\n    `self.nrows() == self.row_partitions[0].nrows()`.\n\n    Otherwise `self.nrows()` will be the first dimension of the field values.\n\n    Returns:\n      A scalar integer `Tensor` (or `None` if `self.rank == 0`).\n    '
        if self.rank == 0:
            return None
        return self._ragged_shape[0]

    def with_shape_dtype(self, dtype: dtypes.DType) -> 'StructuredTensor':
        if False:
            for i in range(10):
                print('nop')
        if dtype == self._ragged_shape.dtype:
            return self
        return StructuredTensor(fields=_fields_with_dtype(self._fields, dtype), ragged_shape=self._ragged_shape.with_dtype(dtype))

    def _is_eager(self):
        if False:
            print('Hello World!')
        'True if all fields are composed of eager tensors.'
        tensors = nest.flatten(self, expand_composites=True)
        return all((isinstance(t, ops.EagerTensor) for t in tensors))

    def field_names(self):
        if False:
            while True:
                i = 10
        'Returns the string field names for this `StructuredTensor`.'
        return tuple(self._fields.keys())

    def field_value(self, field_name):
        if False:
            return 10
        'Returns the tensor value for the specified field or path.\n\n    If `field_name` is a `string`, then it names a field directly owned by this\n    `StructuredTensor`.  If this `StructuredTensor` has shape `[D1...DN]`, then\n    the returned tensor will have shape `[D1...DN, V1...VM]`, where the slice\n    `result[d1...dN]` contains the field value for the structure at\n    `self[d1...dN]`.\n\n    If `field_name` is a `tuple` of `string`, then it specifies a path to a\n    field owned by nested `StructuredTensor`.  In particular,\n    `struct.field_value((f1, f2, ..., fN))` is equivalent to\n    `struct.field_value(f1).field_value(f2)....field_value(fN)`\n\n    Args:\n      field_name: `string` or `tuple` of `string`: The field whose values should\n        be returned.\n\n    Returns:\n      `Tensor`, `StructuredTensor`, or `RaggedTensor`.\n\n    Raises:\n      KeyError: If the given field_name is not found.\n    '
        if isinstance(field_name, (list, tuple)):
            value = self
            for f in field_name:
                if not isinstance(value, StructuredTensor):
                    raise KeyError('Field path {} not found in {}'.format(field_name, self))
                value = value.field_value(f)
            return value
        return self._fields[field_name]

    def __getitem__(self, key):
        if False:
            print('Hello World!')
        "Returns the specified piece of this StructuredTensor.\n\n    * If `struct_tensor` is scalar (i.e., a single structure), then\n      `struct_tensor[f]` returns the value of field `f` (where `f` must be a\n      string).\n\n    * If `struct_tensor` is non-scalar (i.e., a vector or higher-dimensional\n      tensor of structures), `struct_tensor[i]` selects an element or slice of\n      the tensor using standard Python semantics (e.g., negative values index\n      from the end).  `i` may have any of the following types:\n\n      * `int` constant\n      * `string` constant\n      * scalar integer `Tensor`\n      * `slice` containing integer constants and/or scalar integer\n        `Tensor`s\n\n    #### Multidimensional indexing\n\n    `StructuredTensor` supports multidimensional indexing.  I.e., `key` may be a\n    `tuple` of values, indexing or slicing multiple dimensions at once.  For\n    example, if `people` is a vector of structures, each of which has a vector-\n    valued `names` field, then `people[3, 'names', 0]` is equivalent to\n    `people[3]['names'][0]`; and `people[:, 'names', :]` will return a (possibly\n    ragged) matrix of names, with shape `[num_people, num_names_per_person]`.\n\n    Args:\n      key: Indicates which piece of the StructuredTensor to return.\n\n    Returns:\n      A `Tensor`, `StructuredTensor`, or `RaggedTensor`.\n    "
        if isinstance(key, list):
            key = tuple(key)
        elif not isinstance(key, tuple):
            key = (key,)
        if not key:
            return self
        if self.rank == 0:
            return self._scalar_getitem(key)
        else:
            return self._tensor_getitem(key)

    def _scalar_getitem(self, key):
        if False:
            print('Hello World!')
        if isinstance(key[0], slice) and key[0].start is None and (key[0].stop is None) and (key[0].step is None):
            fields = dict(((field_name, field_value.__getitem__(key[1:])) for (field_name, field_value) in self._fields.items()))
            return StructuredTensor.from_fields(fields, self.shape)
        elif not isinstance(key[0], compat.bytes_or_text_types):
            raise ValueError("Key for indexing a StructuredTensor must be a string or a full slice (':')")
        return self._fields[key[0]].__getitem__(key[1:])

    def _tensor_getitem(self, key):
        if False:
            for i in range(10):
                print('nop')
        rank = self.rank
        if len(key) <= rank:
            new_fields = dict(((field_name, field_value.__getitem__(key)) for (field_name, field_value) in self._fields.items()))
            result_shape = self.shape.as_list()
            for (d, k) in enumerate(key):
                if isinstance(k, slice):
                    if not (k.start is None and k.stop is None and (k.step is None)):
                        result_shape[d] = None
                elif isinstance(k, (int, tensor.Tensor)):
                    result_shape[d] = -1
                elif k is None:
                    raise ValueError('Slicing not supported for tf.newaxis')
                else:
                    raise ValueError('Slicing not supported for %r' % k)
            result_shape = [d for d in result_shape if d != -1]
            return StructuredTensor.from_fields(new_fields, result_shape)
        else:
            if not isinstance(key[rank], compat.bytes_or_text_types):
                raise ValueError('Key for indexing a StructuredTensor must be a string')
            return self._fields[key[rank]].__getitem__(key[:rank] + key[rank + 1:])

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        fields = sorted(self._fields.items())
        fields = ((k, str(v).replace('\n', '\n            ')) for (k, v) in fields)
        fields = ('"{}": {}'.format(k, v) for (k, v) in fields)
        dict_repr = ',\n        '.join(fields)
        return '<StructuredTensor(\n    fields={\n        %s},\n    shape=%s)>' % (dict_repr, self.shape)

    def to_pyval(self):
        if False:
            print('Hello World!')
        "Returns this StructuredTensor as a nested Python dict or list of dicts.\n\n    Converts this `StructuredTensor` to a nested python value:\n\n    * `StructTensors` with `rank=0` are converted into a dictionary, with an\n      entry for each field.  Field names are used as keys and field values are\n      converted to python values.  In particular:\n\n      * Scalar Tensor fields are converted to simple values (such as\n        `int` or `float` or `string`)\n      * Non-scalar Tensor fields and RaggedTensor fields are converted to\n        nested lists of simple values.\n      * StructuredTensor fields are converted recursively using `to_pyval`.\n\n    * `StructTensors` with `rank>0` are converted to nested python `list`s,\n      containing one dictionary for each structure (where each structure's\n      dictionary is defined as described above).\n\n    Requires that all fields are Eager tensors.\n\n    >>> tf.experimental.StructuredTensor.from_fields(\n    ...     {'a': [1, 2, 3]}, [3]).to_pyval()\n    [{'a': 1}, {'a': 2}, {'a': 3}]\n\n    Note that `StructuredTensor.from_pyval(pyval).to_pyval() == pyval`.\n\n    Returns:\n      A nested Python dict or list of dicts.\n    "
        if not self._is_eager():
            raise ValueError('StructuredTensor.to_pyval() is only supported in eager mode.')
        result = {}
        for (key, value) in self._fields.items():
            if isinstance(value, ops.EagerTensor):
                value = value.numpy()
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, ragged_tensor.RaggedTensor):
                value = value.to_list()
            elif isinstance(value, StructuredTensor):
                value = value.to_pyval()
            result[key] = value
        if len(self.shape) > 0:
            if not result:
                return _empty_dict_pylist_from_row_partitions(self.row_partitions, self.nrows())
            return _pyval_field_major_to_node_major(list(result.keys()), list(result.values()), self.rank)
        else:
            return result

    @classmethod
    def from_pyval(cls, pyval, typespec=None):
        if False:
            return 10
        'Constructs a StructuredTensor from a nested Python structure.\n\n    >>> tf.experimental.StructuredTensor.from_pyval(\n    ...     {\'a\': [1, 2, 3], \'b\': [[4, 5], [6, 7]]})\n    <StructuredTensor(\n        fields={\n          "a": tf.Tensor([1 2 3], shape=(3,), dtype=int32),\n          "b": <tf.RaggedTensor [[4, 5], [6, 7]]>},\n        shape=())>\n\n    Note that `StructuredTensor.from_pyval(pyval).to_pyval() == pyval`.\n\n    Args:\n      pyval: The nested Python structure that should be used to create the new\n        `StructuredTensor`.\n      typespec: A `StructuredTensor.Spec` specifying the expected type for each\n        field. If not specified, then all nested dictionaries are turned into\n        StructuredTensors, and all nested lists are turned into Tensors (if\n        rank<2) or RaggedTensors (if rank>=2).\n\n    Returns:\n      A `StructuredTensor`.\n    '
        return cls._from_pyval(pyval, typespec, ())

    @classmethod
    def _from_pyval(cls, pyval, typespec, path_so_far):
        if False:
            for i in range(10):
                print('nop')
        'Helper function for from_pyval.\n\n\n    Args:\n      pyval: The nested Python structure that should be used to create the new\n        `StructuredTensor`.\n      typespec: A `StructuredTensor.Spec` specifying the expected type for each\n        field. If not specified, then all nested dictionaries are turned into\n        StructuredTensors, and all nested lists are turned into Tensors (if\n        rank<2) or RaggedTensors (if rank>=2).\n      path_so_far: the path of fields that led here (for error messages).\n\n    Returns:\n      A `StructuredTensor`.\n    '
        if isinstance(pyval, dict):
            return cls._from_pydict(pyval, typespec, path_so_far)
        elif isinstance(pyval, (list, tuple)):
            keys = set()
            rank = _pyval_find_struct_keys_and_depth(pyval, keys)
            if rank is not None:
                return cls._from_pylist_of_dict(pyval, keys, rank, typespec, path_so_far)
            else:
                return cls._from_pylist_of_value(pyval, typespec, path_so_far)
        else:
            return cls._from_pyscalar(pyval, typespec, path_so_far)

    @classmethod
    def _from_pydict(cls, pyval, typespec, path_so_far):
        if False:
            print('Hello World!')
        'Converts python dictionary `pyval` to a StructuredTensor with rank=0.'
        if typespec is None:
            fields = dict(((k, cls._from_pyval(v, None, path_so_far + (k,))) for (k, v) in pyval.items()))
        else:
            spec_shape = typespec._shape
            field_specs = typespec._field_specs
            if not (isinstance(typespec, StructuredTensor.Spec) and spec_shape.rank == 0 and (set(pyval) == set(field_specs))):
                raise ValueError('Value at %r does not match typespec: %r vs %r' % (path_so_far, pyval, typespec))
            fields = dict(((k, cls._from_pyval(v, field_specs[k], path_so_far + (k,))) for (k, v) in pyval.items()))
        return StructuredTensor.from_fields(fields=fields, shape=(), validate=False)

    @classmethod
    def _from_pylist_of_dict(cls, pyval, keys, rank, typespec, path_so_far):
        if False:
            while True:
                i = 10
        'Converts python list `pyval` to a StructuredTensor with rank>1.'
        fields = dict(((key, []) for key in keys))
        for child in pyval:
            _pyval_update_fields(child, fields, 1)
        if typespec is None:
            shape = tensor_shape.TensorShape([None] * rank)
            for (key, target) in fields.items():
                fields[key] = cls._from_pyval(target, None, path_so_far + (key,))
        else:
            field_specs = typespec._fields
            if not isinstance(typespec, StructuredTensor.Spec) or set(fields) - set(field_specs):
                raise ValueError('Value at %r does not match typespec: %r vs %r' % (path_so_far, pyval, typespec))
            shape = typespec._shape
            if shape.rank < rank:
                raise ValueError('Value at %r does not match typespec (rank mismatch): %r vs %r' % (path_so_far, pyval, typespec))
            for (key, spec) in field_specs.items():
                fields[key] = cls._from_pyval(fields.get(key, []), spec, path_so_far + (key,))
        try:
            if not fields and typespec is None:
                return StructuredTensor._from_pylist_of_empty_dict(pyval, rank)
            return StructuredTensor.from_fields(fields=fields, shape=shape, validate=False)
        except Exception as exc:
            raise ValueError('Error parsing path %r' % (path_so_far,)) from exc

    @classmethod
    def _from_pylist_of_empty_dict(cls, pyval, rank):
        if False:
            i = 10
            return i + 15
        'Converts a pylist of empty dictionaries to StructuredTensors.'
        if rank == 0:
            return StructuredTensor.from_fields(fields={}, shape=(), validate=False)
        elif rank == 1:
            nrows = len(pyval)
            shape = (nrows,)
            return StructuredTensor.from_fields(fields={}, shape=shape, nrows=nrows)
        elif rank > 1:
            ragged_zeros = ragged_factory_ops.constant(_dicts_to_zeros(pyval))
            nrows = len(pyval)
            shape = tensor_shape.TensorShape([len(pyval)] + [None] * (rank - 1))
            return StructuredTensor.from_fields(fields={}, shape=shape, row_partitions=ragged_zeros._nested_row_partitions, nrows=nrows)

    @classmethod
    def _from_pylist_of_value(cls, pyval, typespec, path_so_far):
        if False:
            i = 10
            return i + 15
        'Converts python list `pyval` to a Tensor or RaggedTensor with rank>1.'
        if typespec is None:
            try:
                return ragged_factory_ops.constant(pyval)
            except Exception as exc:
                raise ValueError('Error parsing path %r' % (path_so_far,)) from exc
        elif isinstance(typespec, tensor.TensorSpec):
            try:
                result = constant_op.constant(pyval, typespec.dtype)
            except Exception as exc:
                raise ValueError('Error parsing path %r' % (path_so_far,)) from exc
            if not typespec.shape.is_compatible_with(result.shape):
                raise ValueError('Value at %r does not match typespec: %r vs %r' % (path_so_far, typespec, pyval))
            return result
        elif isinstance(typespec, ragged_tensor.RaggedTensorSpec):
            try:
                return ragged_factory_ops.constant(pyval, dtype=typespec._dtype, ragged_rank=typespec._ragged_rank, row_splits_dtype=typespec._row_splits_dtype, inner_shape=typespec._shape[typespec._ragged_rank + 1:])
            except Exception as exc:
                raise ValueError('Error parsing path %r' % (path_so_far,)) from exc
        elif isinstance(typespec, StructuredTensor.Spec):
            empty_rank = _pyval_empty_list_depth(pyval)
            if empty_rank is None:
                raise ValueError('Value at %r does not match typespec: %r vs %r' % (path_so_far, typespec, pyval))
            else:
                return cls._from_pylist_of_dict(pyval, set(), empty_rank, typespec, path_so_far)
        else:
            raise ValueError('Value at %r does not match typespec: %r vs %r' % (path_so_far, typespec, pyval))

    @classmethod
    def _from_pyscalar(cls, pyval, typespec, path_so_far):
        if False:
            i = 10
            return i + 15
        'Converts python scalar value `pyval` to a Tensor.'
        if typespec is None:
            try:
                return constant_op.constant(pyval)
            except Exception as exc:
                raise ValueError('Error parsing path %r' % (path_so_far,)) from exc
        else:
            if not (isinstance(typespec, tensor.TensorSpec) and typespec.shape.rank == 0):
                raise ValueError('Value at %r does not match typespec: %r vs %r' % (path_so_far, typespec, pyval))
            return constant_op.constant(pyval, typespec.dtype)

    def partition_outer_dimension(self, row_partition):
        if False:
            while True:
                i = 10
        'Partitions the outer dimension of this StructuredTensor.\n\n    Returns a new `StructuredTensor` with the same values as `self`, where\n    the outer dimension is partitioned into two (possibly ragged) dimensions.\n    Requires that this StructuredTensor have an outer dimension (i.e.,\n    `self.shape.rank > 0`).\n\n    >>> st = tf.experimental.StructuredTensor.from_pyval(\n    ...     [{\'foo\': 12}, {\'foo\': 33}, {\'foo\': 99}])\n    >>> partition = RowPartition.from_row_lengths([2, 0, 1])\n    >>> st.partition_outer_dimension(partition)\n    <StructuredTensor(\n      fields={\n        "foo": <tf.RaggedTensor [[12, 33], [], [99]]>},\n      shape=(3, None))>\n\n    Args:\n      row_partition: A `RowPartition`.\n\n    Returns:\n      A `StructuredTensor` with rank `values.rank + 1`.\n    '
        if not isinstance(row_partition, RowPartition):
            raise TypeError('row_partition must be a RowPartition.')
        if self.shape.rank == 0:
            raise ValueError('Shape %s must have rank at least 1' % self.shape)
        return _partition_outer_dimension(self, row_partition)

    def merge_dims(self, outer_axis, inner_axis):
        if False:
            print('Hello World!')
        'Merges outer_axis...inner_axis into a single dimension.\n\n    Returns a copy of this RaggedTensor with the specified range of dimensions\n    flattened into a single dimension, with elements in row-major order.\n\n    >>> st = tf.experimental.StructuredTensor.from_pyval(\n    ...     [[{\'foo\': 12}, {\'foo\': 33}], [], [{\'foo\': 99}]])\n    >>> st.merge_dims(0, 1)\n    <StructuredTensor(\n      fields={\n        "foo": tf.Tensor([12 33 99], shape=(3,), dtype=int32)},\n      shape=(3,))>\n\n    Args:\n      outer_axis: `int`: The first dimension in the range of dimensions to\n        merge. May be negative (to index from the last dimension).\n      inner_axis: `int`: The last dimension in the range of dimensions to merge.\n        May be negative (to index from the last dimension).\n\n    Returns:\n      A copy of this tensor, with the specified dimensions merged into a\n      single dimension.  The shape of the returned tensor will be\n      `self.shape[:outer_axis] + [N] + self.shape[inner_axis + 1:]`, where `N`\n      is the total number of slices in the merged dimensions.\n    '
        outer_axis = array_ops.get_positive_axis(outer_axis, self.shape.rank, axis_name='outer_axis', ndims_name='rank(self)')
        inner_axis = array_ops.get_positive_axis(inner_axis, self.shape.rank, axis_name='inner_axis', ndims_name='rank(self)')
        if not outer_axis <= inner_axis:
            raise ValueError('Expected outer_axis (%d) to be less than or equal to inner_axis (%d)' % (outer_axis, inner_axis))
        return _merge_dims(self, outer_axis, inner_axis)

    class Spec:
        """A spec for StructuredTensor."""

        def __validate__(self):
            if False:
                for i in range(10):
                    print('nop')
            assert self._ragged_shape is not None

        @classmethod
        def _from_fields_and_rank(cls, fields, rank):
            if False:
                print('Hello World!')
            'Creates a spec of a StructuredTensor with fields and rank.'
            shape = None
            for (k, v) in fields.items():
                field_shape_untruncated = _dynamic_ragged_shape_spec_from_spec(v)
                if field_shape_untruncated is None:
                    raise ValueError(f'Cannot convert spec of {k}.')
                untruncated_rank = field_shape_untruncated.rank
                if untruncated_rank is not None and untruncated_rank < rank:
                    raise ValueError(f'Rank of field {k} is {untruncated_rank}, but must be at least {rank}.')
                field_shape = field_shape_untruncated._truncate(rank)
                if shape is None:
                    shape = field_shape
                else:
                    shape = shape._merge_with(field_shape)
            return StructuredTensor.Spec(_ragged_shape=shape, _fields=fields)

        @classmethod
        def _from_shape(cls, shape: dynamic_ragged_shape.DynamicRaggedShape) -> 'StructuredTensor.Spec':
            if False:
                return 10
            'Creates the spec of an empty StructuredTensor.'
            return StructuredTensor.Spec(_ragged_shape=shape, _fields={})

        @property
        def _shape(self) -> tensor_shape.TensorShape:
            if False:
                for i in range(10):
                    print('nop')
            return self._ragged_shape._to_tensor_shape()

        @property
        def _field_specs(self) -> Dict[str, type_spec.TypeSpec]:
            if False:
                while True:
                    i = 10
            return self._fields

        @property
        def shape(self) -> tensor_shape.TensorShape:
            if False:
                print('Hello World!')
            return self._shape

        @property
        def rank(self):
            if False:
                return 10
            return self._ragged_shape.rank
_FIELD_NAME_RE = re.compile('^[a-zA-Z][a-zA-Z0-9_]*$')

def _convert_to_structured_field_value(value):
    if False:
        print('Hello World!')
    'Converts `value` to a Tensor, RaggedTensor, or StructuredTensor.'
    if isinstance(value, (tensor.Tensor, ragged_tensor.RaggedTensor, StructuredTensor)):
        return value
    elif ragged_tensor.is_ragged(value):
        return ragged_tensor.convert_to_tensor_or_ragged_tensor(value)
    elif isinstance(value, extension_type.ExtensionType):
        return value
    else:
        try:
            return ops.convert_to_tensor(value)
        except (ValueError, TypeError) as e:
            raise TypeError('Unexpected type for value in `fields`: %r' % value) from e

def _find_shape_dtype(fields: Mapping[str, _FieldValue], nrows: Optional[tensor.Tensor], row_partitions: Optional[Sequence[RowPartition]]) -> dtypes.DType:
    if False:
        return 10
    'Return a consistent dtype for fields, nrows, & row_partitions.\n\n  In the future, the default will switch from int64 to int32, but for now,\n  we stick with int64.\n\n  Args:\n    fields: the fields of the StructuredTensor.\n    nrows: the nrows of the StructuredTensor\n    row_partitions: the row_partitions of the StructuredTensor.\n\n  Returns:\n    If anything requires int64, then return int64.\n    If int32 is explicitly specified, return int32. Otherwise, return int64.\n  '
    field_dtypes = [_field_shape_dtype(v) for v in fields.values()]
    nrows_dtypes = [nrows.dtype] if isinstance(nrows, tensor.Tensor) else []
    rp_dtypes = [] if row_partitions is None else [rp.dtype for rp in row_partitions]
    all_dtypes = field_dtypes + nrows_dtypes + rp_dtypes
    if dtypes.int64 in all_dtypes:
        return dtypes.int64
    if dtypes.int32 in all_dtypes:
        return dtypes.int32
    return dtypes.int64

def _merge_nrows(nrows, static_nrows, value, dtype, validate):
    if False:
        print('Hello World!')
    'Merges `nrows` with `nrows(value)`.\n\n  Checks that `value` has the expected number of rows (`nrows`), and returns\n  `nrows`.  If `validate` is true, then add validation ops that check that\n  the `nrows` values match.\n\n  Args:\n    nrows: scalar integer Tensor.\n    static_nrows: tf.Dimension: static value of nrows, if known.\n    value: Tensor or RaggedTensor or StructuredTensor\n    dtype: dtype for `nrows`.\n    validate: bool -- whether to add validation ops.\n\n  Returns:\n    A tuple `(nrows, static_nrows)`.\n  '
    static_value_nrows = tensor_shape.dimension_at_index(value.shape, 0)
    if isinstance(value, tensor.Tensor):
        value_nrows = array_ops.shape(value, out_type=dtype)[0]
    else:
        value_nrows = value.nrows()
    if nrows is None:
        nrows = value_nrows
    elif static_value_nrows.value is not None and static_nrows.value is not None:
        if not static_value_nrows.is_compatible_with(static_nrows):
            raise ValueError('fields have incompatible nrows')
        nrows = value_nrows
    elif validate:
        nrows = control_flow_ops.with_dependencies([check_ops.assert_equal(nrows, value_nrows, message='fields have incompatible nrows')], nrows)
    return (nrows, static_nrows._merge_with(static_value_nrows))

def _merge_row_partitions(row_partitions, value, rank, dtype, validate):
    if False:
        for i in range(10):
            print('nop')
    'Merges `row_partitions` with `row_partitions(value)`.'
    if isinstance(value, tensor.Tensor):
        value_row_partitions = _row_partitions_for_tensor(value, rank, dtype)
    elif isinstance(value, ragged_tensor.RaggedTensor):
        value_row_partitions = _row_partitions_for_ragged_tensor(value, rank, dtype)
    else:
        assert isinstance(value, StructuredTensor), type(value)
        value_row_partitions = value.row_partitions[:rank - 1]
    assert len(value_row_partitions) == rank - 1
    if row_partitions is None:
        return tuple(value_row_partitions)
    else:
        return tuple([p1._merge_precomputed_encodings(p2, validate) for (p1, p2) in zip(row_partitions, value_row_partitions)])

def _row_partitions_for_tensor(value, rank, dtype):
    if False:
        print('Hello World!')
    'Returns the row partitions for a tf.Tensor.'
    shape = array_ops.shape(value, out_type=dtype)
    return _row_partitions_for_uniform_shape(shape, rank)

def _row_partitions_for_ragged_tensor(value, rank, dtype):
    if False:
        for i in range(10):
            print('nop')
    'Returns the row partitions for a tf.RaggedTensor.'
    assert rank > 1
    value_row_partitions = value._nested_row_partitions[:rank - 1]
    if len(value_row_partitions) < rank - 1:
        value_row_partitions += _row_partitions_for_tensor(value.flat_values, rank - len(value_row_partitions), dtype)
    assert len(value_row_partitions) == rank - 1
    return value_row_partitions

def _row_partitions_for_uniform_shape(shape, rank):
    if False:
        print('Hello World!')
    'Returns row partitions for the given shape Tensor.\n\n  Args:\n    shape: A vector describing a uniform shape.\n    rank: The number of dimensions to generate row partitions for\n\n  Returns:\n    A list of (rank-1) `RowPartition`s with uniform row length.\n  '
    shape_cumprod = math_ops.cumprod(shape[:rank])
    return tuple([RowPartition.from_uniform_row_length(uniform_row_length=shape[i + 1], nvals=shape_cumprod[i + 1], nrows=shape_cumprod[i]) for i in range(rank - 1)])

def _pyval_field_major_to_node_major(keys, values, depth):
    if False:
        i = 10
        return i + 15
    'Regroup each field (k, v) from dict-of-list to list-of-dict.\n\n  Given a "field-major" encoding of the StructuredTensor (which maps each key to\n  a single nested list containing the values for all structs), return a\n  corresponding "node-major" encoding, consisting of a nested list of dicts.\n\n  Args:\n    keys: The field names (list of string).  Must not be empty.\n    values: The field values (list of python values).  Must have the same length\n      as `keys`.\n    depth: The list depth at which dictionaries should be created.\n\n  Returns:\n    A nested list of dict, with depth `depth`.\n  '
    assert keys
    if depth == 0:
        return dict(zip(keys, values))
    nvals = len(values[0])
    assert all((nvals == len(values[i]) for i in range(1, len(values))))
    return [_pyval_field_major_to_node_major(keys, value_slice, depth - 1) for value_slice in zip(*values)]

def _empty_dict_pylist_from_row_partitions(row_partitions, nrows):
    if False:
        print('Hello World!')
    'Returns a python list of empty dicts from the given row partitions.\n\n  Args:\n    row_partitions: The row-partitions describing the ragged shape of the\n      result.\n    nrows: The number of rows in the outermost row-partition.  (Or if\n      `len(row_partitions)==0`, then the number of empty dicts to return.)\n\n  Returns:\n    A nested python list whose leaves (if any) are empty python dicts.\n  '
    if not row_partitions:
        return [{} for _ in range(nrows)]
    else:
        values = _empty_dict_pylist_from_row_partitions(row_partitions[1:], row_partitions[0].row_splits()[-1])
        splits = row_partitions[0].row_splits()
        return [values[splits[i]:splits[i + 1]] for i in range(len(splits) - 1)]

def _pyval_find_struct_keys_and_depth(pyval, keys):
    if False:
        while True:
            i = 10
    'Finds the keys & depth of nested dictionaries in `pyval`.\n\n  Args:\n    pyval: A nested structure of lists, tuples, and dictionaries.\n    keys: (output parameter) A set, which will be updated with any keys that are\n      found in the nested dictionaries.\n\n  Returns:\n    The nesting depth of dictionaries in `pyval`, or `None` if `pyval` does\n    not contain any dictionaries.\n  Raises:\n    ValueError: If dictionaries have inconsistent depth.\n  '
    if isinstance(pyval, dict):
        keys.update(pyval.keys())
        return 0
    elif isinstance(pyval, (list, tuple)):
        depth = None
        for child in pyval:
            child_depth = _pyval_find_struct_keys_and_depth(child, keys)
            if child_depth is not None:
                if depth is None:
                    depth = child_depth + 1
                elif depth != child_depth + 1:
                    raise ValueError('Inconsistent depth of dictionaries')
        return depth
    else:
        return None

def _pyval_update_fields(pyval, fields, depth):
    if False:
        i = 10
        return i + 15
    "Append the field values from `pyval` to `fields`.\n\n  Args:\n    pyval: A python `dict`, or nested list/tuple of `dict`, whose value(s)\n      should be appended to `fields`.\n    fields: A dictionary mapping string keys to field values.  Field values\n      extracted from `pyval` are appended to this dictionary's values.\n    depth: The depth at which `pyval` should be appended to the field values.\n  "
    if not isinstance(pyval, (dict, list, tuple)):
        raise ValueError('Expected dict or nested list/tuple of dict')
    for (key, target) in fields.items():
        for _ in range(1, depth):
            target = target[-1]
        target.append(pyval[key] if isinstance(pyval, dict) else [])
    if isinstance(pyval, (list, tuple)):
        for child in pyval:
            _pyval_update_fields(child, fields, depth + 1)

def _pyval_empty_list_depth(pyval):
    if False:
        while True:
            i = 10
    'Find the max depth for nested empty lists.\n\n  Args:\n    pyval: A nested python list.\n\n  Returns:\n    The maximum depth of empty lists in `pyval`, or None if `pyval` contains\n    anything other than nested empty lists.\n  '
    if isinstance(pyval, list):
        if not pyval:
            return 1
        depths = [_pyval_empty_list_depth(v) for v in pyval]
        if any((depth is None for depth in depths)):
            return None
        else:
            return max(depths) + 1
    else:
        return None

def _replace_row_partitions(value, new_partitions):
    if False:
        for i in range(10):
            print('nop')
    "Updates `value` to use `new_partitions` as its (outer) row partitions.\n\n  This is used to ensure that all fields in a `StructuredTensor` use identical\n  `RowPartition` objects for the shared dimensions.  In particular,\n  `StructuredTensor.from_fields` first merges all of the row partitions from\n  any fields, and then replaces the outer row partitions of all fields with\n  the merged row partitions (using this function).\n\n  Args:\n    value: A `Tensor`, `RaggedTensor`, or `StructuredTensor`.\n    new_partitions: A list of row-partitions that should be used by `value`.\n      Must be equivalent to `value`'s current row partitions.\n\n  Returns:\n    A value that is equivalent to `value`, where outer row partitions have been\n    replaced by `new_partitions`.\n  "
    if isinstance(value, tensor.Tensor) or not new_partitions:
        return value
    elif isinstance(value, ragged_tensor.RaggedTensor):
        return ragged_tensor.RaggedTensor._from_row_partition(values=_replace_row_partitions(value.values, new_partitions[1:]), row_partition=new_partitions[0])
    else:
        assert isinstance(value, StructuredTensor)
        new_fields = dict(((k, _replace_row_partitions(v, new_partitions)) for (k, v) in value._fields.items()))
        return StructuredTensor._old_init(fields=new_fields, shape=value.shape, nrows=value.nrows(), row_partitions=tuple(new_partitions) + tuple(value.row_partitions[len(new_partitions):]))

def _partition_outer_dimension(value, row_partition):
    if False:
        return 10
    'Partitions the outer dimension of `value` using `row_partitions`.\n\n  Examples:\n\n    >>> partition = RowPartition.from_row_lengths([2, 0, 1])\n    >>> _partition_outer_dimension(tf.constant([1, 2, 3]), partition)\n    <tf.RaggedTensor [[1, 2], [], [3]]>\n\n    >>> struct_value = tf.experimental.StructuredTensor.from_pyval(\n    ...     [{\'x\': 1}, {\'x\': 2}, {\'x\': 3}])\n    >>> _partition_outer_dimension(struct_value, partition)\n    <StructuredTensor(\n      fields={\n        "x": <tf.RaggedTensor [[1, 2], [], [3]]>},\n      shape=(3, None))>\n\n  Args:\n    value: Tensor, RaggedTensor, or StructuredTensor\n    row_partition: RowPartition\n\n  Returns:\n    A value with the same type as `value`, where\n    `result.rank = value.rank + 1`.\n  '
    is_ragged = row_partition.uniform_row_length() is None
    if isinstance(value, tensor.Tensor) and (not is_ragged):
        new_shape = array_ops.concat([[row_partition.nrows(), row_partition.uniform_row_length()], array_ops.shape(value, out_type=row_partition.dtype)[1:]], axis=0)
        return array_ops.reshape(value, new_shape)
    elif isinstance(value, (tensor.Tensor, ragged_tensor.RaggedTensor)):
        return ragged_tensor.RaggedTensor._from_row_partition(value, row_partition)
    else:
        assert isinstance(value, StructuredTensor)
        nrows = row_partition.static_nrows
        ncols = row_partition.static_uniform_row_length
        shape = tensor_shape.TensorShape([nrows, ncols]).concatenate(value.shape[1:])
        fields = dict(((k, _partition_outer_dimension(v, row_partition)) for (k, v) in value._fields.items()))
        return StructuredTensor._old_init(fields, shape, row_partition.nrows(), (row_partition,) + value.row_partitions)

def _merge_dims(value, outer_axis, inner_axis):
    if False:
        while True:
            i = 10
    'Merges `outer_axis...inner_axis` of `value` into a single dimension.'
    assert outer_axis < inner_axis
    if isinstance(value, (tensor.Tensor, ragged_tensor.RaggedTensor)):
        return ragged_tensor.merge_dims(value, outer_axis, inner_axis)
    else:
        assert isinstance(value, StructuredTensor)
        fields = dict(((k, _merge_dims(v, outer_axis, inner_axis)) for (k, v) in value._fields.items()))
        ragged_shape = value._ragged_shape._merge_dims(outer_axis, inner_axis)
        return StructuredTensor(fields, ragged_shape)
_structured_tensor_factory_key = object()

def _dynamic_ragged_shape_spec_from_spec(spec: Union[dynamic_ragged_shape.DynamicRaggedShape.Spec, ragged_tensor.RaggedTensorSpec, StructuredTensor.Spec, tensor.TensorSpec]) -> dynamic_ragged_shape.DynamicRaggedShape.Spec:
    if False:
        for i in range(10):
            print('nop')
    if isinstance(spec, StructuredTensor.Spec):
        return spec._ragged_shape
    else:
        return dynamic_ragged_shape.DynamicRaggedShape.Spec._from_spec(spec)

def _normalize_field_name_to_tuple(name: 'FieldName') -> Sequence[str]:
    if False:
        print('Hello World!')
    'FieldName can be given also as string, this normalizes it to a tuple.'
    if isinstance(name, str):
        return (name,)
    if isinstance(name, list):
        return tuple(name)
    assert isinstance(name, tuple)
    return name

def _dicts_to_zeros(pyval):
    if False:
        while True:
            i = 10
    'Replaces dictionaries zeros in a pylist.'
    if isinstance(pyval, dict):
        return 0
    return [_dicts_to_zeros(x) for x in pyval]

def _merge_dims_generic(source, outer, inner):
    if False:
        return 10
    'Merges outer_axis...inner_axis into a single dimension.\n\n  If outer == inner, this is a NOOP. If inner < outer, then this fials.\n  If inner >= source.shape.rank, then the behavior is undefined.\n\n  Args:\n    source: a tensor, ragged tensor, or structured tensor.\n    outer: a python int, indicating the first dimension to compress (must be\n      nonnegative).\n    inner: a python int, indicating the first dimension to keep (of the tail)\n      (must be nonnegative).\n\n  Returns:\n    source with outer_axis...inner_axis merged into a single dimension.\n\n  '
    if isinstance(source, StructuredTensor):
        return source.merge_dims(outer, inner)
    else:
        return ragged_tensor.merge_dims(source, outer, inner)

def _dynamic_ragged_shape_from_tensor(field, dtype=None) -> dynamic_ragged_shape.DynamicRaggedShape:
    if False:
        for i in range(10):
            print('nop')
    'Extension of DynamicRaggedShape.from_tensor to support StructuredTensor.'
    if isinstance(field, StructuredTensor):
        return field._ragged_shape
    shape = array_ops.shape_v2(field, out_type=dtype)
    if isinstance(shape, tensor.Tensor):
        return dynamic_ragged_shape.DynamicRaggedShape(row_partitions=[], inner_shape=shape)
    elif isinstance(shape, dynamic_ragged_shape.DynamicRaggedShape):
        return shape
    raise TypeError(f'Expected shape tf.shape({field}) to return a Tensor or a DynamicRaggedShape. Instead, got: {shape}.')

def _merge_with_optional(a: Optional[dynamic_ragged_shape.DynamicRaggedShape], b: Optional[dynamic_ragged_shape.DynamicRaggedShape]) -> Optional[dynamic_ragged_shape.DynamicRaggedShape]:
    if False:
        for i in range(10):
            print('nop')
    if a is None:
        return b
    if b is None:
        return a
    return a._merge_with(b)

def _shape_from_fields(fields, rank: int, dtype: dtypes.DType) -> Optional[dynamic_ragged_shape.DynamicRaggedShape]:
    if False:
        for i in range(10):
            print('nop')
    'Given fields, rank, and dtype, create a shape.'
    field_shape = None
    for (k, field) in fields.items():
        try:
            next_field_shape_raw = _dynamic_ragged_shape_from_tensor(field, dtype=dtype)
            next_field_shape = next_field_shape_raw[:rank]
            field_shape = _merge_with_optional(field_shape, next_field_shape)
        except Exception as err:
            raise ValueError(f'Error in shape of {k}') from err
    return field_shape

def _field_shape_dtype(field: _FieldValue) -> Optional[dtypes.DType]:
    if False:
        i = 10
        return i + 15
    if isinstance(field, ragged_tensor.RaggedTensor):
        return field._row_partition.dtype
    if isinstance(field, StructuredTensor):
        return field._ragged_shape.dtype
    return None

def _field_with_shape_dtype(field: _FieldValue, dtype: dtypes.DType) -> _FieldValue:
    if False:
        return 10
    if isinstance(field, ragged_tensor.RaggedTensor):
        return field.with_row_splits_dtype(dtype)
    if isinstance(field, StructuredTensor):
        return field.with_shape_dtype(dtype)
    return field

def _fields_with_dtype(fields: Mapping[str, _FieldValue], dtype: dtypes.DType) -> Mapping[str, _FieldValue]:
    if False:
        for i in range(10):
            print('nop')
    return {k: _field_with_shape_dtype(v, dtype) for (k, v) in fields.items()}

def _dynamic_ragged_shape_init(fields, shape, nrows, row_partitions):
    if False:
        print('Hello World!')
    'Produce a DynamicRaggedShape for StructuredTensor.'
    assert isinstance(fields, dict), fields
    assert isinstance(shape, tensor_shape.TensorShape), shape
    assert nrows is None or isinstance(nrows, tensor.Tensor) or isinstance(nrows, int), nrows
    assert row_partitions is None or isinstance(row_partitions, tuple), row_partitions
    rank = shape.rank
    if rank is None:
        raise TypeError("StructuredTensor's shape must have known rank.")
    dtype = _find_shape_dtype(fields, nrows, row_partitions)
    fields = _fields_with_dtype(fields, dtype)
    result = None
    if shape.is_fully_defined():
        result = dynamic_ragged_shape.DynamicRaggedShape._from_inner_shape(shape.as_list(), dtype=dtype)
    if rank == 0:
        return dynamic_ragged_shape.DynamicRaggedShape._from_inner_shape(array_ops.zeros((0,), dtype=dtype))
    result = _merge_with_optional(result, _shape_from_fields(fields, rank, dtype))
    if rank == 1:
        alt_value = tensor_shape.dimension_value(shape[0])
        if alt_value is not None:
            nrows = alt_value
        if nrows is not None:
            result = _merge_with_optional(result, dynamic_ragged_shape.DynamicRaggedShape._from_inner_shape([nrows], dtype=dtype))
        if result is None:
            raise ValueError('Must specify `nrows`, a fully specified `shape`,' + ' or have `fields` if `rank=1`')
        return result
    if row_partitions:
        result = _merge_with_optional(result, dynamic_ragged_shape.DynamicRaggedShape.from_row_partitions(row_partitions, dtype=dtype))
    if result is None:
        raise ValueError('Must specify row_partitions, a fully specified shape, ' + 'or have fields if rank > 1')
    return result

def StructuredTensorSpec(shape, field_specs):
    if False:
        print('Hello World!')
    'A placeholder for the old StructuredTensorSpec.'
    if not isinstance(field_specs, dict):
        raise TypeError('field_specs must be a dictionary.')
    for k in field_specs.keys():
        if not isinstance(k, str):
            raise TypeError('field_specs must be a dictionary with string keys.')
    for v in field_specs.values():
        if not isinstance(v, type_spec.TypeSpec):
            raise TypeError('field_specs must be a dictionary with TypeSpec values.')
    shape = dynamic_ragged_shape.DynamicRaggedShape.Spec._from_tensor_shape(tensor_shape.as_shape(shape), 0, dtypes.int32)
    rank = shape.rank
    if rank is None:
        raise TypeError("StructuredTensor's shape must have known rank.")
    for (k, v) in field_specs.items():
        field_shape_untruncated = _dynamic_ragged_shape_spec_from_spec(v)
        if field_shape_untruncated is None:
            raise ValueError(f'Cannot convert spec of {k}.')
        untruncated_rank = field_shape_untruncated.rank
        if untruncated_rank is not None and untruncated_rank < rank:
            raise ValueError(f'Rank of field {k} is {untruncated_rank}, but must be at least {rank}.')
        field_shape = field_shape_untruncated._truncate(rank)
        shape = shape._merge_with(field_shape)
    return StructuredTensor.Spec(_ragged_shape=shape, _fields=field_specs)