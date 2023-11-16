"""Utility methods for handling nests.

This module encapsulates different semantics of handling nests by the public
tf.nest APIs and internal tf.data APIs. The difference in semantics exists for
historic reasons and reconciliation would require a non-backwards compatible
change.

The implementation of the different semantics use a common utility to
avoid / minimize further divergence between the two APIs over time.
"""
import collections as _collections
import enum
import six as _six
import wrapt as _wrapt
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util.compat import collections_abc as _collections_abc
from tensorflow.python.util.custom_nest_protocol import CustomNestProtocol
_is_mapping_view = _pywrap_utils.IsMappingView
_is_attrs = _pywrap_utils.IsAttrs
_is_composite_tensor = _pywrap_utils.IsCompositeTensor
_is_type_spec = _pywrap_utils.IsTypeSpec
_is_mutable_mapping = _pywrap_utils.IsMutableMapping
_is_mapping = _pywrap_utils.IsMapping
_tf_data_is_nested = _pywrap_utils.IsNestedForData
_tf_data_flatten = _pywrap_utils.FlattenForData
_tf_core_is_nested = _pywrap_utils.IsNested
_is_nested_or_composite = _pywrap_utils.IsNestedOrComposite
same_namedtuples = _pywrap_utils.SameNamedtuples
STRUCTURES_HAVE_MISMATCHING_TYPES = "The two structures don't have the same sequence type. Input structure has type {input_type}, while shallow structure has type {shallow_type}."
STRUCTURES_HAVE_MISMATCHING_LENGTHS = "The two structures don't have the same sequence length. Input structure has length {input_length}, while shallow structure has length {shallow_length}."
INPUT_TREE_SMALLER_THAN_SHALLOW_TREE = 'The input_tree has fewer items than the shallow_tree. Input structure has length {input_size}, while shallow structure has length {shallow_size}.'
SHALLOW_TREE_HAS_INVALID_KEYS = "The shallow_tree's keys are not a subset of the input_tree's keys. The shallow_tree has the following keys that are not in the input_tree: {}."

class Modality(enum.Enum):
    """Modality/semantic used for treating nested structures.

  - Modality.CORE follows tensorflow_core/tf.nest semantics.

    The following collection types are recognized by `tf.nest` as nested
    structures:

    * `collections.abc.Sequence` (except `string` and `bytes`).
      This includes `list`, `tuple`, and `namedtuple`.
    * `collections.abc.Mapping` (with sortable keys).
      This includes `dict` and `collections.OrderedDict`.
    * `collections.abc.MappingView` (with sortable keys).
    * [`attr.s` classes](https://www.attrs.org/).

    Any other values are considered **atoms**.  Not all collection types are
    considered nested structures.  For example, the following types are
    considered atoms:

    * `set`; `{"a", "b"}` is an atom, while `["a", "b"]` is a nested structure.
    * [`dataclass` classes](https://docs.python.org/library/dataclasses.html)
    * `tf.Tensor`
    * `numpy.array`

  - Modality.DATA follows tf.data's nest semantics.

  This modality makes two changes:
  1. It removes support for lists as a level of nesting in nested structures.
  2. It adds support for `SparseTensorValue` as an atomic element.

  The motivation for this change is twofold:

  1. It seems more natural for lists to be treated (e.g. in Dataset
  constructors)
    as tensors, rather than lists of (lists of...) tensors.
  2. This is needed because `SparseTensorValue` is implemented as a `namedtuple`
    that would normally be flattened and we want to be able to create sparse
    tensor from `SparseTensorValue's similarly to creating tensors from numpy
    arrays.
  """
    CORE = 'CORE'
    DATA = 'DATA'

class _DotString(object):
    __slots__ = []

    def __str__(self):
        if False:
            print('Hello World!')
        return '.'

    def __repr__(self):
        if False:
            print('Hello World!')
        return '.'
_DOT = _DotString()

def is_nested(modality, structure):
    if False:
        return 10
    'Returns true if its input is a nested structure.\n\n  For Modality.CORE refer to\n  [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)\n  for the definition of a nested structure.\n\n  Args:\n    modality: enum value of supported modality [Modality.CORE or Modality.DATA]\n    structure: the value to test.\n\n  Returns:\n    True if the input is a nested structure.\n  '
    if modality == Modality.CORE:
        return _tf_core_is_nested(structure)
    elif modality == Modality.DATA:
        return _tf_data_is_nested(structure)
    else:
        raise ValueError('Unknown modality used {} for nested structure'.format(modality))

def is_namedtuple(instance, strict=False):
    if False:
        return 10
    'Returns True iff `instance` is a `namedtuple`.\n\n  Args:\n    instance: An instance of a Python object.\n    strict: If True, `instance` is considered to be a `namedtuple` only if it is\n      a "plain" namedtuple. For instance, a class inheriting from a `namedtuple`\n      will be considered to be a `namedtuple` iff `strict=False`.\n\n  Returns:\n    True if `instance` is a `namedtuple`.\n  '
    return _pywrap_utils.IsNamedtuple(instance, strict)

def sequence_like(instance, args):
    if False:
        i = 10
        return i + 15
    'Converts the sequence `args` to the same type as `instance`.\n\n  Args:\n    instance: an instance of `tuple`, `list`, `namedtuple`, `dict`,\n      `collections.OrderedDict`, or `composite_tensor.Composite_Tensor` or\n      `type_spec.TypeSpec`.\n    args: items to be converted to the `instance` type.\n\n  Returns:\n    `args` with the type of `instance`.\n  '
    if _is_mutable_mapping(instance):
        result = dict(zip(_tf_core_sorted(instance), args))
        instance_type = type(instance)
        if instance_type == _collections.defaultdict:
            d = _collections.defaultdict(instance.default_factory)
        else:
            d = instance_type()
        for key in instance:
            d[key] = result[key]
        return d
    elif _is_mapping(instance):
        result = dict(zip(_tf_core_sorted(instance), args))
        instance_type = type(instance)
        if not getattr(instance_type, '__supported_by_tf_nest__', False):
            tf_logging.log_first_n(tf_logging.WARN, 'Mapping types may not work well with tf.nest. Prefer using MutableMapping for {}'.format(instance_type), 1)
        try:
            return instance_type(((key, result[key]) for key in instance))
        except TypeError as err:
            raise TypeError('Error creating an object of type {} like {}. Note that it must accept a single positional argument representing an iterable of key-value pairs, in addition to self. Cause: {}'.format(type(instance), instance, err))
    elif _is_mapping_view(instance):
        return list(args)
    elif is_namedtuple(instance) or _is_attrs(instance):
        if isinstance(instance, _wrapt.ObjectProxy):
            instance_type = type(instance.__wrapped__)
        else:
            instance_type = type(instance)
        return instance_type(*args)
    elif _is_composite_tensor(instance):
        assert len(args) == 1
        spec = instance._type_spec
        return spec._from_components(args[0])
    elif _is_type_spec(instance):
        assert len(args) == 1
        return instance._from_components(args[0])
    elif isinstance(instance, _six.moves.range):
        return sequence_like(list(instance), args)
    elif isinstance(instance, _wrapt.ObjectProxy):
        return type(instance)(sequence_like(instance.__wrapped__, args))
    elif isinstance(instance, CustomNestProtocol):
        metadata = instance.__tf_flatten__()[0]
        return instance.__tf_unflatten__(metadata, tuple(args))
    else:
        return type(instance)(args)

def _get_attrs_items(obj):
    if False:
        while True:
            i = 10
    "Returns a list of (name, value) pairs from an attrs instance.\n\n  TODO(b/268078256): check if this comment is valid, and if so, ensure it's\n  handled in the function below.\n  The list will be sorted by name.\n\n  Args:\n    obj: an object.\n\n  Returns:\n    A list of (attr_name, attr_value) pairs, sorted by attr_name.\n  "
    attrs = getattr(obj.__class__, '__attrs_attrs__')
    attr_names = (a.name for a in attrs)
    return [(attr_name, getattr(obj, attr_name)) for attr_name in attr_names]

def _tf_core_sorted(dict_):
    if False:
        i = 10
        return i + 15
    'Returns a sorted list of the dict keys, with error if keys not sortable.'
    try:
        return sorted(dict_.keys())
    except TypeError:
        raise TypeError('nest only supports dicts with sortable keys.')

def _tf_data_sorted(dict_):
    if False:
        for i in range(10):
            print('nop')
    'Returns a sorted list of the dict keys, with error if keys not sortable.'
    try:
        return sorted(list(dict_))
    except TypeError as e:
        raise TypeError(f'nest only supports dicts with sortable keys. Error: {e.message}')

def yield_value(modality, iterable):
    if False:
        while True:
            i = 10
    'Yield elements of `iterable` in a deterministic order.\n\n  Args:\n    modality: enum value of supported modality [Modality.CORE or Modality.DATA]\n    iterable: an iterable.\n\n  Yields:\n    The iterable elements in a deterministic order.\n  '
    if modality == Modality.CORE:
        yield from _tf_core_yield_value(iterable)
    elif modality == Modality.DATA:
        yield from _tf_data_yield_value(iterable)
    else:
        raise ValueError('Unknown modality used {} for nested structure'.format(modality))

def _tf_core_yield_value(iterable):
    if False:
        for i in range(10):
            print('nop')
    for (_, v) in _tf_core_yield_sorted_items(iterable):
        yield v

def yield_sorted_items(modality, iterable):
    if False:
        for i in range(10):
            print('nop')
    if modality == Modality.CORE:
        return _tf_core_yield_sorted_items(iterable)
    else:
        raise ValueError('Unknown modality used {} for nested structure'.format(modality))

def _tf_core_yield_sorted_items(iterable):
    if False:
        i = 10
        return i + 15
    "Yield (key, value) pairs for `iterable` in a deterministic order.\n\n  For Sequences, the key will be an int, the array index of a value.\n  For Mappings, the key will be the dictionary key.\n  For objects (e.g. namedtuples), the key will be the attribute name.\n\n  In all cases, the keys will be iterated in sorted order.\n\n  Args:\n    iterable: an iterable.\n\n  Yields:\n    The iterable's (key, value) pairs, in order of sorted keys.\n  "
    if isinstance(iterable, list):
        for item in enumerate(iterable):
            yield item
    elif type(iterable) == tuple:
        for item in enumerate(iterable):
            yield item
    elif isinstance(iterable, (dict, _collections_abc.Mapping)):
        for key in _tf_core_sorted(iterable):
            yield (key, iterable[key])
    elif _is_attrs(iterable):
        for item in _get_attrs_items(iterable):
            yield item
    elif is_namedtuple(iterable):
        for field in iterable._fields:
            yield (field, getattr(iterable, field))
    elif _is_composite_tensor(iterable):
        type_spec = iterable._type_spec
        yield (type_spec.value_type.__name__, type_spec._to_components(iterable))
    elif _is_type_spec(iterable):
        yield (iterable.value_type.__name__, iterable._component_specs)
    elif isinstance(iterable, CustomNestProtocol):
        flat_component = iterable.__tf_flatten__()[1]
        assert isinstance(flat_component, tuple)
        yield from enumerate(flat_component)
    else:
        for item in enumerate(iterable):
            yield item

def _tf_data_yield_value(iterable):
    if False:
        return 10
    'Yield elements of `iterable` in a deterministic order.\n\n  Args:\n    iterable: an iterable.\n\n  Yields:\n    The iterable elements in a deterministic order.\n  '
    if isinstance(iterable, _collections_abc.Mapping):
        for key in _tf_data_sorted(iterable):
            yield iterable[key]
    elif iterable.__class__.__name__ == 'SparseTensorValue':
        yield iterable
    elif _is_attrs(iterable):
        for (_, attr) in _get_attrs_items(iterable):
            yield attr
    elif isinstance(iterable, CustomNestProtocol):
        flat_component = iterable.__tf_flatten__()[1]
        assert isinstance(flat_component, tuple)
        yield from flat_component
    else:
        for value in iterable:
            yield value

def assert_same_structure(modality, nest1, nest2, check_types=True, expand_composites=False):
    if False:
        i = 10
        return i + 15
    'Asserts that two structures are nested in the same way.\n\n  For Modality.CORE refer to\n  [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)\n  for the definition of a structure. Note the method does not check the types of\n  atoms inside the structures.\n\n  Examples:\n\n  * These atom vs. atom comparisons will pass:\n\n    >>> tf.nest.assert_same_structure(1.5, tf.Variable(1, tf.uint32))\n    >>> tf.nest.assert_same_structure("abc", np.array([1, 2]))\n\n  * These nested structure vs. nested structure comparisons will pass:\n\n    >>> structure1 = (((1, 2), 3), 4, (5, 6))\n    >>> structure2 = ((("foo1", "foo2"), "foo3"), "foo4", ("foo5", "foo6"))\n    >>> structure3 = [(("a", "b"), "c"), "d", ["e", "f"]]\n    >>> tf.nest.assert_same_structure(structure1, structure2)\n    >>> tf.nest.assert_same_structure(structure1, structure3, check_types=False)\n\n    >>> import collections\n    >>> tf.nest.assert_same_structure(\n    ...     collections.namedtuple("bar", "a b")(1, 2),\n    ...     collections.namedtuple("foo", "a b")(2, 3),\n    ...     check_types=False)\n\n    >>> tf.nest.assert_same_structure(\n    ...     collections.namedtuple("bar", "a b")(1, 2),\n    ...     { "a": 1, "b": 2 },\n    ...     check_types=False)\n\n    >>> tf.nest.assert_same_structure(\n    ...     { "a": 1, "b": 2, "c": 3 },\n    ...     { "c": 6, "b": 5, "a": 4 })\n\n    >>> ragged_tensor1 = tf.RaggedTensor.from_row_splits(\n    ...       values=[3, 1, 4, 1, 5, 9, 2, 6],\n    ...       row_splits=[0, 4, 4, 7, 8, 8])\n    >>> ragged_tensor2 = tf.RaggedTensor.from_row_splits(\n    ...       values=[3, 1, 4],\n    ...       row_splits=[0, 3])\n    >>> tf.nest.assert_same_structure(\n    ...       ragged_tensor1,\n    ...       ragged_tensor2,\n    ...       expand_composites=True)\n\n  * These examples will raise exceptions:\n\n    >>> tf.nest.assert_same_structure([0, 1], np.array([0, 1]))\n    Traceback (most recent call last):\n    ...\n    ValueError: The two structures don\'t have the same nested structure\n\n    >>> tf.nest.assert_same_structure(\n    ...       collections.namedtuple(\'bar\', \'a b\')(1, 2),\n    ...       collections.namedtuple(\'foo\', \'a b\')(2, 3))\n    Traceback (most recent call last):\n    ...\n    TypeError: The two structures don\'t have the same nested structure\n\n  For Modality.DATA, nested structures are treated differently than\n  Modality.CORE. Please refer to class Modality\'s documentation above to read up\n  on these differences.\n\n  Args:\n    modality: enum value of supported modality [Modality.CORE or Modality.DATA]\n    nest1: an atom or a nested structure.\n    nest2: an atom or a nested structure.\n    check_types: - For Modality.CORE: if `True` (default) types of structures\n      are checked as well, including the keys of dictionaries. If set to\n      `False`, for example a list and a tuple of objects will look the same if\n      they have the same size. Note that namedtuples with identical name and\n      fields are always considered to have the same shallow structure. Two types\n      will also be considered the same if they are both list subtypes (which\n      allows "list" and "_ListWrapper" from trackable dependency tracking to\n      compare equal). `check_types=True` only checks type of sub-structures. The\n      types of atoms are not checked. - For Modality.DATA: if `True` (default)\n      types of sequences should be same as well. For dictionary, "type" of\n      dictionary is considered to include its keys. In other words, two\n      dictionaries with different keys are considered to have a different\n      "type". If set to `False`, two iterables are considered same as long as\n      they yield the elements that have same structures.\n    expand_composites: Arg only valid for Modality.CORE. If true, then composite\n      tensors such as `tf.sparse.SparseTensor` and `tf.RaggedTensor` are\n      expanded into their component tensors.\n\n  Raises:\n    ValueError: If the two structures do not have the same number of atoms or\n      if the two structures are not nested in the same way.\n    TypeError: If the two structures differ in the type of sequence in any of\n      their substructures. Only possible if `check_types` is `True`.\n  '
    if modality == Modality.CORE:
        _tf_core_assert_same_structure(nest1, nest2, check_types, expand_composites)
    elif modality == Modality.DATA:
        _tf_data_assert_same_structure(nest1, nest2, check_types)
    else:
        raise ValueError('Unknown modality used {} for nested structure'.format(modality))

def _tf_core_assert_same_structure(nest1, nest2, check_types=True, expand_composites=False):
    if False:
        for i in range(10):
            print('nop')
    check_types = bool(check_types)
    expand_composites = bool(expand_composites)
    try:
        _pywrap_utils.AssertSameStructure(nest1, nest2, check_types, expand_composites)
    except (ValueError, TypeError) as e:
        str1 = str(_tf_core_map_structure(lambda _: _DOT, nest1))
        str2 = str(_tf_core_map_structure(lambda _: _DOT, nest2))
        raise type(e)('%s\nEntire first structure:\n%s\nEntire second structure:\n%s' % (str(e), str1, str2))

def _tf_data_assert_same_structure(nest1, nest2, check_types=True):
    if False:
        while True:
            i = 10
    _pywrap_utils.AssertSameStructureForData(nest1, nest2, check_types)

def _tf_core_packed_nest_with_indices(structure, flat, index, is_nested_fn, sequence_fn=None):
    if False:
        return 10
    'Helper function for pack_sequence_as.\n\n  Args:\n    structure: structure to mimic.\n    flat: Flattened values to output substructure for.\n    index: Index at which to start reading from flat.\n    is_nested_fn: Function used to test if a value should be treated as a nested\n      structure.\n    sequence_fn: Function used to generate a new strcuture instance.\n\n  Returns:\n    The tuple (new_index, child), where:\n      * new_index - the updated index into `flat` having processed `structure`.\n      * packed - the subset of `flat` corresponding to `structure`,\n                 having started at `index`, and packed into the same nested\n                 format.\n\n  Raises:\n    ValueError: if `structure` contains more atoms than `flat`\n      (assuming indexing starts from `index`).\n  '
    packed = []
    sequence_fn = sequence_fn or sequence_like
    for s in _tf_core_yield_value(structure):
        if is_nested_fn(s):
            (new_index, child) = _tf_core_packed_nest_with_indices(s, flat, index, is_nested_fn, sequence_fn)
            packed.append(sequence_fn(s, child))
            index = new_index
        else:
            packed.append(flat[index])
            index += 1
    return (index, packed)

def _tf_data_packed_nest_with_indices(structure, flat, index):
    if False:
        for i in range(10):
            print('nop')
    'Helper function for pack_nest_as.\n\n  Args:\n    structure: Substructure (tuple of elements and/or tuples) to mimic\n    flat: Flattened values to output substructure for.\n    index: Index at which to start reading from flat.\n\n  Returns:\n    The tuple (new_index, child), where:\n      * new_index - the updated index into `flat` having processed `structure`.\n      * packed - the subset of `flat` corresponding to `structure`,\n                 having started at `index`, and packed into the same nested\n                 format.\n\n  Raises:\n    ValueError: if `structure` contains more elements than `flat`\n      (assuming indexing starts from `index`).\n  '
    packed = []
    for s in _tf_data_yield_value(structure):
        if _tf_data_is_nested(s):
            (new_index, child) = _tf_data_packed_nest_with_indices(s, flat, index)
            packed.append(sequence_like(s, child))
            index = new_index
        else:
            packed.append(flat[index])
            index += 1
    return (index, packed)

def flatten(modality, structure, expand_composites=False):
    if False:
        while True:
            i = 10
    'Flattens a nested structure.\n\n  - For Modality.CORE: refer to\n  [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)\n  for the definition of a structure.\n\n  If the structure is an atom, then returns a single-item list: [structure].\n\n  This is the inverse of the `nest.pack_sequence_as` method that takes in a\n  flattened list and re-packs it into the nested structure.\n\n  In the case of dict instances, the sequence consists of the values, sorted by\n  key to ensure deterministic behavior. This is true also for OrderedDict\n  instances: their sequence order is ignored, the sorting order of keys is used\n  instead. The same convention is followed in `nest.pack_sequence_as`. This\n  correctly repacks dicts and OrderedDicts after they have been flattened, and\n  also allows flattening an OrderedDict and then repacking it back using a\n  corresponding plain dict, or vice-versa. Dictionaries with non-sortable keys\n  cannot be flattened.\n\n  Users must not modify any collections used in nest while this function is\n  running.\n\n  Examples:\n\n  1. Python dict (ordered by key):\n\n    >>> dict = { "key3": "value3", "key1": "value1", "key2": "value2" }\n    >>> tf.nest.flatten(dict)\n    [\'value1\', \'value2\', \'value3\']\n\n  2. For a nested python tuple:\n\n    >>> tuple = ((1.0, 2.0), (3.0, 4.0, 5.0), 6.0)\n    >>> tf.nest.flatten(tuple)\n        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]\n\n  3. For a nested dictionary of dictionaries:\n\n    >>> dict = { "key3": {"c": (1.0, 2.0), "a": (3.0)},\n    ... "key1": {"m": "val1", "g": "val2"} }\n    >>> tf.nest.flatten(dict)\n    [\'val2\', \'val1\', 3.0, 1.0, 2.0]\n\n  4. Numpy array (will not flatten):\n\n    >>> array = np.array([[1, 2], [3, 4]])\n    >>> tf.nest.flatten(array)\n        [array([[1, 2],\n                [3, 4]])]\n\n  5. `tf.Tensor` (will not flatten):\n\n    >>> tensor = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])\n    >>> tf.nest.flatten(tensor)\n        [<tf.Tensor: shape=(3, 3), dtype=float32, numpy=\n          array([[1., 2., 3.],\n                 [4., 5., 6.],\n                 [7., 8., 9.]], dtype=float32)>]\n\n  6. `tf.RaggedTensor`: This is a composite tensor thats representation consists\n  of a flattened list of \'values\' and a list of \'row_splits\' which indicate how\n  to chop up the flattened list into different rows. For more details on\n  `tf.RaggedTensor`, please visit\n  https://www.tensorflow.org/api_docs/python/tf/RaggedTensor.\n\n  with `expand_composites=False`, we just return the RaggedTensor as is.\n\n    >>> tensor = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2]])\n    >>> tf.nest.flatten(tensor, expand_composites=False)\n    [<tf.RaggedTensor [[3, 1, 4, 1], [], [5, 9, 2]]>]\n\n  with `expand_composites=True`, we return the component Tensors that make up\n  the RaggedTensor representation (the values and row_splits tensors)\n\n    >>> tensor = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2]])\n    >>> tf.nest.flatten(tensor, expand_composites=True)\n    [<tf.Tensor: shape=(7,), dtype=int32, numpy=array([3, 1, 4, 1, 5, 9, 2],\n                                                      dtype=int32)>,\n     <tf.Tensor: shape=(4,), dtype=int64, numpy=array([0, 4, 4, 7])>]\n\n  Args:\n    modality: enum value of supported modality [Modality.CORE or Modality.DATA]\n    structure: an atom or a nested structure. Note, numpy arrays are considered\n      atoms and are not flattened.\n    expand_composites: Arg valid for Modality.CORE only. If true, then composite\n      tensors such as `tf.sparse.SparseTensor` and `tf.RaggedTensor` are\n      expanded into their component tensors.\n\n  Returns:\n    A Python list, the flattened version of the input.\n\n  Raises:\n    TypeError: The nest is or contains a dict with non-sortable keys.\n  '
    if modality == Modality.CORE:
        return _tf_core_flatten(structure, expand_composites)
    elif modality == Modality.DATA:
        return _tf_data_flatten(structure)
    else:
        raise ValueError('Unknown modality used {} for nested structure'.format(modality))

def _tf_core_flatten(structure, expand_composites=False):
    if False:
        while True:
            i = 10
    'See comments for flatten() in tensorflow/python/util/nest.py.'
    if structure is None:
        return [None]
    expand_composites = bool(expand_composites)
    return _pywrap_utils.Flatten(structure, expand_composites)

def pack_sequence_as(modality, structure, flat_sequence, expand_composites, sequence_fn=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns a given flattened sequence packed into a given structure.\n\n  - For Modality.CORE: Refer to\n  [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)\n  for the definition of a structure.\n\n  If `structure` is an atom, `flat_sequence` must be a single-item list;\n  in this case the return value is `flat_sequence[0]`.\n\n  If `structure` is or contains a dict instance, the keys will be sorted to\n  pack the flat sequence in deterministic order. This is true also for\n  `OrderedDict` instances: their sequence order is ignored, the sorting order of\n  keys is used instead. The same convention is followed in `flatten`.\n  This correctly repacks dicts and `OrderedDict`s after they have been\n  flattened, and also allows flattening an `OrderedDict` and then repacking it\n  back using a corresponding plain dict, or vice-versa.\n  Dictionaries with non-sortable keys cannot be flattened.\n\n  Examples:\n\n  1. Python dict:\n\n    >>> structure = { "key3": "", "key1": "", "key2": "" }\n    >>> flat_sequence = ["value1", "value2", "value3"]\n    >>> tf.nest.pack_sequence_as(structure, flat_sequence)\n    {\'key3\': \'value3\', \'key1\': \'value1\', \'key2\': \'value2\'}\n\n  2. For a nested python tuple:\n\n    >>> structure = ((\'a\',\'b\'), (\'c\',\'d\',\'e\'), \'f\')\n    >>> flat_sequence = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]\n    >>> tf.nest.pack_sequence_as(structure, flat_sequence)\n    ((1.0, 2.0), (3.0, 4.0, 5.0), 6.0)\n\n  3. For a nested dictionary of dictionaries:\n\n    >>> structure = { "key3": {"c": (\'alpha\', \'beta\'), "a": (\'gamma\')},\n    ...               "key1": {"e": "val1", "d": "val2"} }\n    >>> flat_sequence = [\'val2\', \'val1\', 3.0, 1.0, 2.0]\n    >>> tf.nest.pack_sequence_as(structure, flat_sequence)\n    {\'key3\': {\'c\': (1.0, 2.0), \'a\': 3.0}, \'key1\': {\'e\': \'val1\', \'d\': \'val2\'}}\n\n  4. Numpy array (considered a scalar):\n\n    >>> structure = [\'a\']\n    >>> flat_sequence = [np.array([[1, 2], [3, 4]])]\n    >>> tf.nest.pack_sequence_as(structure, flat_sequence)\n    [array([[1, 2],\n           [3, 4]])]\n\n  5. tf.Tensor (considered a scalar):\n\n    >>> structure = [\'a\']\n    >>> flat_sequence = [tf.constant([[1., 2., 3.], [4., 5., 6.]])]\n    >>> tf.nest.pack_sequence_as(structure, flat_sequence)\n    [<tf.Tensor: shape=(2, 3), dtype=float32,\n     numpy= array([[1., 2., 3.], [4., 5., 6.]], dtype=float32)>]\n\n  6. `tf.RaggedTensor`: This is a composite tensor thats representation consists\n  of a flattened list of \'values\' and a list of \'row_splits\' which indicate how\n  to chop up the flattened list into different rows. For more details on\n  `tf.RaggedTensor`, please visit\n  https://www.tensorflow.org/api_docs/python/tf/RaggedTensor.\n\n  With `expand_composites=False`, we treat RaggedTensor as a scalar.\n\n    >>> structure = { "foo": tf.ragged.constant([[1, 2], [3]]),\n    ...               "bar": tf.constant([[5]]) }\n    >>> flat_sequence = [ "one", "two" ]\n    >>> tf.nest.pack_sequence_as(structure, flat_sequence,\n    ... expand_composites=False)\n    {\'foo\': \'two\', \'bar\': \'one\'}\n\n  With `expand_composites=True`, we expect that the flattened input contains\n  the tensors making up the ragged tensor i.e. the values and row_splits\n  tensors.\n\n    >>> structure = { "foo": tf.ragged.constant([[1., 2.], [3.]]),\n    ...               "bar": tf.constant([[5.]]) }\n    >>> tensors = tf.nest.flatten(structure, expand_composites=True)\n    >>> print(tensors)\n    [<tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[5.]],\n     dtype=float32)>,\n     <tf.Tensor: shape=(3,), dtype=float32, numpy=array([1., 2., 3.],\n     dtype=float32)>,\n     <tf.Tensor: shape=(3,), dtype=int64, numpy=array([0, 2, 3])>]\n    >>> verified_tensors = [tf.debugging.check_numerics(t, \'invalid tensor: \')\n    ...                     if t.dtype==tf.float32 else t\n    ...                     for t in tensors]\n    >>> tf.nest.pack_sequence_as(structure, verified_tensors,\n    ...                          expand_composites=True)\n    {\'foo\': <tf.RaggedTensor [[1.0, 2.0], [3.0]]>,\n     \'bar\': <tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[5.]],\n     dtype=float32)>}\n\n  - For Modality.DATA:  If `structure` is a scalar, `flat_sequence` must be a\n  single-element list;\n  in this case the return value is `flat_sequence[0]`.\n\n  Args:\n    modality: enum value of supported modality [Modality.CORE or Modality.DATA]\n    structure: - For Modality.CORE: Nested structure, whose structure is given\n      by nested lists, tuples, and dicts. Note: numpy arrays and strings are\n      considered scalars. - For Modality.DATA: tuple or list constructed of\n      scalars and/or other tuples/lists, or a scalar.  Note: numpy arrays are\n      considered scalars.\n    flat_sequence: flat sequence to pack.\n    expand_composites: Arg valid for Modality.CORE only. If true, then composite\n      tensors such as `tf.sparse.SparseTensor` and `tf.RaggedTensor` are\n      expanded into their component tensors.\n    sequence_fn: Arg valid for Modality.CORE only.\n\n  Returns:\n    packed: `flat_sequence` converted to have the same recursive structure as\n      `structure`.\n\n  Raises:\n    ValueError: If `flat_sequence` and `structure` have different\n      atom counts.\n    TypeError: For Modality.CORE only. `structure` is or contains a dict with\n    non-sortable keys.\n  '
    if modality == Modality.CORE:
        return _tf_core_pack_sequence_as(structure, flat_sequence, expand_composites, sequence_fn)
    elif modality == Modality.DATA:
        return _tf_data_pack_sequence_as(structure, flat_sequence)
    else:
        raise ValueError('Unknown modality used {} for nested structure'.format(modality))

def _tf_core_pack_sequence_as(structure, flat_sequence, expand_composites, sequence_fn=None):
    if False:
        i = 10
        return i + 15
    'Implements sequence packing, with the option to alter the structure.'
    is_nested_fn = _is_nested_or_composite if expand_composites else _tf_core_is_nested
    sequence_fn = sequence_fn or sequence_like

    def truncate(value, length):
        if False:
            for i in range(10):
                print('nop')
        value_str = str(value)
        return value_str[:length] + (value_str[length:] and '...')
    if not is_nested_fn(flat_sequence):
        raise TypeError('Attempted to pack value:\n  {}\ninto a structure, but found incompatible type `{}` instead.'.format(truncate(flat_sequence, 100), type(flat_sequence)))
    if not is_nested_fn(structure):
        if len(flat_sequence) != 1:
            raise ValueError('The target structure is of type `{}`\n  {}\nHowever the input is a sequence ({}) of length {}.\n  {}\nnest cannot guarantee that it is safe to map one to the other.'.format(type(structure), truncate(structure, 100), type(flat_sequence), len(flat_sequence), truncate(flat_sequence, 100)))
        return flat_sequence[0]
    try:
        (final_index, packed) = _tf_core_packed_nest_with_indices(structure, flat_sequence, 0, is_nested_fn, sequence_fn)
        if final_index < len(flat_sequence):
            raise IndexError
    except IndexError:
        flat_structure = _tf_core_flatten(structure, expand_composites=expand_composites)
        if len(flat_structure) != len(flat_sequence):
            raise ValueError('Could not pack sequence. Structure had %d atoms, but flat_sequence had %d items.  Structure: %s, flat_sequence: %s.' % (len(flat_structure), len(flat_sequence), structure, flat_sequence))
    return sequence_fn(structure, packed)

def _tf_data_pack_sequence_as(structure, flat_sequence):
    if False:
        while True:
            i = 10
    'Returns a given flattened sequence packed into a nest.\n\n  If `structure` is a scalar, `flat_sequence` must be a single-element list;\n  in this case the return value is `flat_sequence[0]`.\n\n  Args:\n    structure: tuple or list constructed of scalars and/or other tuples/lists,\n      or a scalar.  Note: numpy arrays are considered scalars.\n    flat_sequence: flat sequence to pack.\n\n  Returns:\n    packed: `flat_sequence` converted to have the same recursive structure as\n      `structure`.\n\n  Raises:\n    ValueError: If nest and structure have different element counts.\n  '
    if not (_tf_data_is_nested(flat_sequence) or isinstance(flat_sequence, list)):
        raise TypeError(f"Argument `flat_sequence` must be a sequence. Got '{type(flat_sequence).__name__}'.")
    if not _tf_data_is_nested(structure):
        if len(flat_sequence) != 1:
            raise ValueError(f'Argument `structure` is a scalar but `len(flat_sequence)`={len(flat_sequence)} > 1')
        return flat_sequence[0]
    flat_structure = _tf_data_flatten(structure)
    if len(flat_structure) != len(flat_sequence):
        raise ValueError(f'Could not pack sequence. Argument `structure` had {len(flat_structure)} elements, but argument `flat_sequence` had {len(flat_sequence)} elements. Received structure: {structure}, flat_sequence: {flat_sequence}.')
    (_, packed) = _tf_data_packed_nest_with_indices(structure, flat_sequence, 0)
    return sequence_like(structure, packed)

def map_structure(modality, func, *structure, **kwargs):
    if False:
        print('Hello World!')
    'Creates a new structure by applying `func` to each atom in `structure`.\n\n  - For Modality.CORE: Refer to\n  [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)\n  for the definition of a structure.\n\n  Applies `func(x[0], x[1], ...)` where x[i] enumerates all atoms in\n  `structure[i]`.  All items in `structure` must have the same arity,\n  and the return value will contain results with the same structure layout.\n\n  Examples:\n\n  * A single Python dict:\n\n  >>> a = {"hello": 24, "world": 76}\n  >>> tf.nest.map_structure(lambda p: p * 2, a)\n  {\'hello\': 48, \'world\': 152}\n\n  * Multiple Python dictionaries:\n\n  >>> d1 = {"hello": 24, "world": 76}\n  >>> d2 = {"hello": 36, "world": 14}\n  >>> tf.nest.map_structure(lambda p1, p2: p1 + p2, d1, d2)\n  {\'hello\': 60, \'world\': 90}\n\n  * A single Python list:\n\n  >>> a = [24, 76, "ab"]\n  >>> tf.nest.map_structure(lambda p: p * 2, a)\n  [48, 152, \'abab\']\n\n  * Scalars:\n\n  >>> tf.nest.map_structure(lambda x, y: x + y, 3, 4)\n  7\n\n  * Empty structures:\n\n  >>> tf.nest.map_structure(lambda x: x + 1, ())\n  ()\n\n  * Check the types of iterables:\n\n  >>> s1 = (((1, 2), 3), 4, (5, 6))\n  >>> s1_list = [[[1, 2], 3], 4, [5, 6]]\n  >>> tf.nest.map_structure(lambda x, y: None, s1, s1_list)\n  Traceback (most recent call last):\n  ...\n  TypeError: The two structures don\'t have the same nested structure\n\n  * Type check is set to False:\n\n  >>> s1 = (((1, 2), 3), 4, (5, 6))\n  >>> s1_list = [[[1, 2], 3], 4, [5, 6]]\n  >>> tf.nest.map_structure(lambda x, y: None, s1, s1_list, check_types=False)\n  (((None, None), None), None, (None, None))\n\n  - For Modality.DATA: Applies `func(x[0], x[1], ...)` where x[i] is an entry in\n  `structure[i]`.  All structures in `structure` must have the same arity,\n  and the return value will contain the results in the same structure.\n\n  Args:\n    modality: enum value of supported modality [Modality.CORE or Modality.DATA]\n    func: A callable that accepts as many arguments as there are structures.\n    *structure: - For Modality.CORE: atom or nested structure. - For\n      Modality.DATA: scalar, or tuple or list of constructed scalars and/or\n      other tuples/lists, or scalars.  Note: numpy arrays are considered\n      scalars.\n    **kwargs: Valid keyword args are: * `check_types`: - For Modality.CORE: If\n      set to `True` (default) the types of iterables within the structures have\n      to be same (e.g. `map_structure(func, [1], (1,))` raises a `TypeError`\n      exception). To allow this set this argument to `False`. Note that\n      namedtuples with identical name and fields are always considered to have\n      the same shallow structure. - For Modality.DATA: only valid keyword\n      argument is `check_types`. If set to `True` (default) the types of\n      iterables within the structures have to be same (e.g. `map_structure(func,\n      [1], (1,))` raises a `TypeError` exception). To allow this set this\n      argument to `False`. * `expand_composites`: Valid for Modality.CORE only.\n      If set to `True`, then composite tensors such as `tf.sparse.SparseTensor`\n      and `tf.RaggedTensor` are expanded into their component tensors.  If\n      `False` (the default), then composite tensors are not expanded.\n\n  Returns:\n    A new structure with the same arity as `structure[0]`, whose atoms\n    correspond to `func(x[0], x[1], ...)` where `x[i]` is the atom in the\n    corresponding location in `structure[i]`. If there are different structure\n    types and `check_types` is `False` the structure types of the first\n    structure will be used.\n\n  Raises:\n    TypeError: If `func` is not callable or if the structures do not match\n      each other by depth tree.\n    ValueError: If no structure is provided or if the structures do not match\n      each other by type.\n    ValueError: If wrong keyword arguments are provided.\n  '
    if modality == Modality.CORE:
        return _tf_core_map_structure(func, *structure, **kwargs)
    elif modality == Modality.DATA:
        return _tf_data_map_structure(func, *structure, **kwargs)
    else:
        raise ValueError('Unknown modality used {} for nested structure'.format(modality))

def _tf_core_map_structure(func, *structure, **kwargs):
    if False:
        while True:
            i = 10
    if not callable(func):
        raise TypeError('func must be callable, got: %s' % func)
    if not structure:
        raise ValueError('Must provide at least one structure')
    check_types = kwargs.pop('check_types', True)
    expand_composites = kwargs.pop('expand_composites', False)
    if kwargs:
        raise ValueError('Only valid keyword arguments are `check_types` and `expand_composites`, not: `%s`' % '`, `'.join(kwargs.keys()))
    for other in structure[1:]:
        _tf_core_assert_same_structure(structure[0], other, check_types=check_types, expand_composites=expand_composites)
    flat_structure = (_tf_core_flatten(s, expand_composites) for s in structure)
    entries = zip(*flat_structure)
    return _tf_core_pack_sequence_as(structure[0], [func(*x) for x in entries], expand_composites=expand_composites)

def _tf_data_map_structure(func, *structure, **check_types_dict):
    if False:
        i = 10
        return i + 15
    if not callable(func):
        raise TypeError(f'Argument `func` must be callable, got: {func}')
    if not structure:
        raise ValueError('Must provide at least one structure')
    if check_types_dict:
        if 'check_types' not in check_types_dict or len(check_types_dict) > 1:
            raise ValueError(f"Only valid keyword argument for `check_types_dict` is 'check_types'. Got {check_types_dict}.")
        check_types = check_types_dict['check_types']
    else:
        check_types = True
    for other in structure[1:]:
        _tf_data_assert_same_structure(structure[0], other, check_types=check_types)
    flat_structure = (_tf_data_flatten(s) for s in structure)
    entries = zip(*flat_structure)
    return _tf_data_pack_sequence_as(structure[0], [func(*x) for x in entries])

def yield_flat_up_to(modality, shallow_tree, input_tree, is_nested_fn, path=()):
    if False:
        return 10
    'Yields (path, value) pairs of input_tree flattened up to shallow_tree.\n\n  - For Modality.CORE: See comments for _tf_core_yield_flat_up_to() below\n  - For Modality.DATA: See comments for _tf_data_yield_flat_up_to() below\n\n  Args:\n    modality: enum value of supported modality [Modality.CORE or Modality.DATA]\n    shallow_tree: Nested structure. Traverse no further than its leaf nodes.\n    input_tree: Nested structure. Return the paths and values from this tree.\n      Must have the same upper structure as shallow_tree.\n    is_nested_fn: Arg valid for Modality.CORE only. Function used to test if a\n      value should be treated as a nested structure.\n    path: Arg valid for Modality.CORE only. Tuple. Optional argument, only used\n      when recursing. The path from the root of the original shallow_tree, down\n      to the root of the shallow_tree arg of this recursive call.\n\n  Yields:\n    Pairs of (path, value), where path the tuple path of a leaf node in\n    shallow_tree, and value is the value of the corresponding node in\n    input_tree.\n  '
    if modality == Modality.CORE:
        yield from _tf_core_yield_flat_up_to(shallow_tree, input_tree, is_nested_fn, path)
    elif modality == Modality.DATA:
        yield from _tf_data_yield_flat_up_to(shallow_tree, input_tree)
    else:
        raise ValueError('Unknown modality used {} for nested structure'.format(modality))

def _tf_core_yield_flat_up_to(shallow_tree, input_tree, is_nested_fn, path=()):
    if False:
        while True:
            i = 10
    'Yields (path, value) pairs of input_tree flattened up to shallow_tree.\n\n  Args:\n    shallow_tree: Nested structure. Traverse no further than its leaf nodes.\n    input_tree: Nested structure. Return the paths and values from this tree.\n      Must have the same upper structure as shallow_tree.\n    is_nested_fn: Function used to test if a value should be treated as a nested\n      structure.\n    path: Tuple. Optional argument, only used when recursing. The path from the\n      root of the original shallow_tree, down to the root of the shallow_tree\n      arg of this recursive call.\n\n  Yields:\n    Pairs of (path, value), where path the tuple path of a leaf node in\n    shallow_tree, and value is the value of the corresponding node in\n    input_tree.\n  '
    if not is_nested_fn(shallow_tree):
        yield (path, input_tree)
    else:
        input_tree = dict(_tf_core_yield_sorted_items(input_tree))
        for (shallow_key, shallow_subtree) in _tf_core_yield_sorted_items(shallow_tree):
            subpath = path + (shallow_key,)
            input_subtree = input_tree[shallow_key]
            for (leaf_path, leaf_value) in _tf_core_yield_flat_up_to(shallow_subtree, input_subtree, is_nested_fn, path=subpath):
                yield (leaf_path, leaf_value)

def _tf_data_yield_flat_up_to(shallow_tree, input_tree):
    if False:
        while True:
            i = 10
    'Yields elements `input_tree` partially flattened up to `shallow_tree`.'
    if _tf_data_is_nested(shallow_tree):
        for (shallow_branch, input_branch) in zip(_tf_data_yield_value(shallow_tree), _tf_data_yield_value(input_tree)):
            for input_leaf in _tf_data_yield_flat_up_to(shallow_branch, input_branch):
                yield input_leaf
    else:
        yield input_tree

def assert_shallow_structure(modality, shallow_tree, input_tree, check_types=True, expand_composites=False):
    if False:
        i = 10
        return i + 15
    'Asserts that `shallow_tree` is a shallow structure of `input_tree`.\n\n  This function tests if the `input_tree` structure can be created from\n  the `shallow_tree` structure by replacing its leaf nodes with deeper\n  tree structures.\n\n  Examples:\n\n  The following code will raise an exception:\n  ```python\n    shallow_tree = {"a": "A", "b": "B"}\n    input_tree = {"a": 1, "c": 2}\n    assert_shallow_structure(shallow_tree, input_tree)\n  ```\n\n  The following code will raise an exception:\n  ```python\n    shallow_tree = ["a", "b"]\n    input_tree = ["c", ["d", "e"], "f"]\n    assert_shallow_structure(shallow_tree, input_tree)\n  ```\n\n  Args:\n    modality: enum value of supported modality [Modality.CORE or Modality.DATA]\n    shallow_tree: an arbitrarily nested structure.\n    input_tree: an arbitrarily nested structure.\n    check_types: if `True` (default) the sequence types of `shallow_tree` and\n      `input_tree` have to be the same. Note that even with check_types==True,\n      this function will consider two different namedtuple classes with the same\n      name and _fields attribute to be the same class.\n    expand_composites: Valid for Modality.CORE only. If true, then composite\n      tensors such as `tf.sparse.SparseTensor` and `tf.RaggedTensor` are\n      expanded into their component tensors.\n\n  Raises:\n    TypeError: If `shallow_tree` is a sequence but `input_tree` is not.\n    TypeError: If the sequence types of `shallow_tree` are different from\n      `input_tree`. Only raised if `check_types` is `True`.\n    ValueError: If the sequence lengths of `shallow_tree` are different from\n      `input_tree`.\n  '
    if modality == Modality.CORE:
        _tf_core_assert_shallow_structure(shallow_tree, input_tree, check_types, expand_composites)
    elif modality == Modality.DATA:
        _tf_data_assert_shallow_structure(shallow_tree, input_tree, check_types)
    else:
        raise ValueError('Unknown modality used {} for nested structure'.format(modality))

def _tf_core_assert_shallow_structure(shallow_tree, input_tree, check_types=True, expand_composites=False):
    if False:
        while True:
            i = 10
    is_nested_fn = _is_nested_or_composite if expand_composites else _tf_core_is_nested
    if is_nested_fn(shallow_tree):
        if not is_nested_fn(input_tree):
            raise TypeError('If shallow structure is a sequence, input must also be a sequence. Input has type: %s.' % type(input_tree))
        if isinstance(shallow_tree, _wrapt.ObjectProxy):
            shallow_type = type(shallow_tree.__wrapped__)
        else:
            shallow_type = type(shallow_tree)
        if check_types and (not isinstance(input_tree, shallow_type)):
            shallow_is_namedtuple = is_namedtuple(shallow_tree, False)
            input_is_namedtuple = is_namedtuple(input_tree, False)
            if shallow_is_namedtuple and input_is_namedtuple:
                if not same_namedtuples(shallow_tree, input_tree):
                    raise TypeError(STRUCTURES_HAVE_MISMATCHING_TYPES.format(input_type=type(input_tree), shallow_type=type(shallow_tree)))
            elif isinstance(shallow_tree, list) and isinstance(input_tree, list):
                pass
            elif (_is_composite_tensor(shallow_tree) or _is_type_spec(shallow_tree)) and (_is_composite_tensor(input_tree) or _is_type_spec(input_tree)):
                pass
            elif not (isinstance(shallow_tree, _collections_abc.Mapping) and isinstance(input_tree, _collections_abc.Mapping)):
                raise TypeError(STRUCTURES_HAVE_MISMATCHING_TYPES.format(input_type=type(input_tree), shallow_type=type(shallow_tree)))
        if _is_composite_tensor(shallow_tree) or _is_composite_tensor(input_tree):
            if not ((_is_composite_tensor(input_tree) or _is_type_spec(input_tree)) and (_is_composite_tensor(shallow_tree) or _is_type_spec(shallow_tree))):
                raise TypeError(STRUCTURES_HAVE_MISMATCHING_TYPES.format(input_type=type(input_tree), shallow_type=type(shallow_tree)))
            type_spec_1 = (shallow_tree if _is_type_spec(shallow_tree) else shallow_tree._type_spec)._without_tensor_names()
            type_spec_2 = (input_tree if _is_type_spec(input_tree) else input_tree._type_spec)._without_tensor_names()
            if hasattr(type_spec_1, '_get_structure') and hasattr(type_spec_2, '_get_structure'):
                result = type_spec_1._get_structure() == type_spec_2._get_structure() or None
            else:
                result = type_spec_1.most_specific_common_supertype([type_spec_2])
            if result is None:
                raise ValueError('Incompatible CompositeTensor TypeSpecs: %s vs. %s' % (type_spec_1, type_spec_2))
        elif _is_type_spec(shallow_tree):
            if not _is_type_spec(input_tree):
                raise TypeError('If shallow structure is a TypeSpec, input must also be a TypeSpec.  Input has type: %s.' % type(input_tree))
        elif len(input_tree) != len(shallow_tree):
            raise ValueError(STRUCTURES_HAVE_MISMATCHING_LENGTHS.format(input_length=len(input_tree), shallow_length=len(shallow_tree)))
        elif len(input_tree) < len(shallow_tree):
            raise ValueError(INPUT_TREE_SMALLER_THAN_SHALLOW_TREE.format(input_size=len(input_tree), shallow_size=len(shallow_tree)))
        if isinstance(shallow_tree, _collections_abc.Mapping):
            absent_keys = set(shallow_tree) - set(input_tree)
            if absent_keys:
                raise ValueError(SHALLOW_TREE_HAS_INVALID_KEYS.format(sorted(absent_keys)))
        for (shallow_branch, input_branch) in zip(_tf_core_yield_value(shallow_tree), _tf_core_yield_value(input_tree)):
            _tf_core_assert_shallow_structure(shallow_branch, input_branch, check_types=check_types, expand_composites=expand_composites)

def _tf_data_assert_shallow_structure(shallow_tree, input_tree, check_types=True):
    if False:
        while True:
            i = 10
    if _tf_data_is_nested(shallow_tree):
        if not _tf_data_is_nested(input_tree):
            raise TypeError(f"If shallow structure is a sequence, input must also be a sequence. Input has type: '{type(input_tree).__name__}'.")
        if check_types and (not isinstance(input_tree, type(shallow_tree))):
            raise TypeError(f"The two structures don't have the same sequence type. Input structure has type '{type(input_tree).__name__}', while shallow structure has type '{type(shallow_tree).__name__}'.")
        if len(input_tree) != len(shallow_tree):
            raise ValueError(f"The two structures don't have the same sequence length. Input structure has length {len(input_tree)}, while shallow structure has length {len(shallow_tree)}.")
        if check_types and isinstance(shallow_tree, _collections_abc.Mapping):
            if set(input_tree) != set(shallow_tree):
                raise ValueError(f"The two structures don't have the same keys. Input structure has keys {list(input_tree)}, while shallow structure has keys {list(shallow_tree)}.")
            input_tree = sorted(input_tree.items())
            shallow_tree = sorted(shallow_tree.items())
        for (shallow_branch, input_branch) in zip(shallow_tree, input_tree):
            _tf_data_assert_shallow_structure(shallow_branch, input_branch, check_types=check_types)

def flatten_up_to(modality, shallow_tree, input_tree, check_types=True, expand_composites=False):
    if False:
        return 10
    "Flattens `input_tree` up to `shallow_tree`.\n\n  - For Modality.CORE: refer to\n  [tf.nest](https://www.tensorflow.org/api_docs/python/tf/nest)\n  for the definition of a structure.\n\n  Any further depth in structure in `input_tree` is retained as structures in\n  the partially flatten output.\n\n  If `shallow_tree` and `input_tree` are atoms, this returns a\n  single-item list: `[input_tree]`.\n\n  Use Case:\n\n  Sometimes we may wish to partially flatten a structure, retaining some\n  of the nested structure. We achieve this by specifying a shallow structure,\n  `shallow_tree`, we wish to flatten up to.\n\n  The input, `input_tree`, can be thought of as having the same structure layout\n  as `shallow_tree`, but with leaf nodes that are themselves tree structures.\n\n  Examples:\n\n  ```python\n  input_tree = [[[2, 2], [3, 3]], [[4, 9], [5, 5]]]\n  shallow_tree = [[True, True], [False, True]]\n\n  flattened_input_tree = flatten_up_to(shallow_tree, input_tree)\n  flattened_shallow_tree = flatten_up_to(shallow_tree, shallow_tree)\n\n  # Output is:\n  # [[2, 2], [3, 3], [4, 9], [5, 5]]\n  # [True, True, False, True]\n  ```\n\n  ```python\n  input_tree = [[('a', 1), [('b', 2), [('c', 3), [('d', 4)]]]]]\n  shallow_tree = [['level_1', ['level_2', ['level_3', ['level_4']]]]]\n\n  input_tree_flattened_as_shallow_tree = flatten_up_to(shallow_tree, input_tree)\n  input_tree_flattened = flatten(input_tree)\n\n  # Output is:\n  # [('a', 1), ('b', 2), ('c', 3), ('d', 4)]\n  # ['a', 1, 'b', 2, 'c', 3, 'd', 4]\n  ```\n\n  Edge Cases:\n\n  ```python\n  flatten_up_to(0, 0)  # Output: [0]\n  flatten_up_to(0, [0, 1, 2])  # Output: [[0, 1, 2]]\n  flatten_up_to([0, 1, 2], 0)  # Output: TypeError\n  flatten_up_to([0, 1, 2], [0, 1, 2])  # Output: [0, 1, 2]\n\n  ```\n\n  Args:\n    modality: enum value of supported modality [Modality.CORE or Modality.DATA]\n    shallow_tree: a possibly pruned structure of input_tree.\n    input_tree: an atom or a nested structure. Note, numpy arrays are considered\n      atoms.\n    check_types: bool. If True, check that each node in shallow_tree has the\n      same type as the corresponding node in input_tree.\n    expand_composites: Arg valid for Modality.CORE only. If true, then composite\n      tensors such as `tf.sparse.SparseTensor` and `tf.RaggedTensor` are\n      expanded into their component tensors.\n\n  Returns:\n    A Python list, the partially flattened version of `input_tree` according to\n    the structure of `shallow_tree`.\n\n  Raises:\n    TypeError: If `shallow_tree` is a nested structure but `input_tree` is not.\n    TypeError: If the structure types of `shallow_tree` are different from\n      `input_tree`.\n    ValueError: If the structure lengths of `shallow_tree` are different from\n      `input_tree`.\n  "
    if modality == Modality.CORE:
        return _tf_core_flatten_up_to(shallow_tree, input_tree, check_types, expand_composites)
    elif modality == Modality.DATA:
        return _tf_data_flatten_up_to(shallow_tree, input_tree)
    else:
        raise ValueError('Unknown modality used {} for nested structure'.format(modality))

def _tf_core_flatten_up_to(shallow_tree, input_tree, check_types=True, expand_composites=False):
    if False:
        while True:
            i = 10
    is_nested_fn = _is_nested_or_composite if expand_composites else _tf_core_is_nested
    _tf_core_assert_shallow_structure(shallow_tree, input_tree, check_types=check_types, expand_composites=expand_composites)
    return [v for (_, v) in _tf_core_yield_flat_up_to(shallow_tree, input_tree, is_nested_fn)]

def _tf_data_flatten_up_to(shallow_tree, input_tree):
    if False:
        return 10
    _tf_data_assert_shallow_structure(shallow_tree, input_tree)
    return list(_tf_data_yield_flat_up_to(shallow_tree, input_tree))

def map_structure_up_to(modality, shallow_tree, func, *inputs, **kwargs):
    if False:
        return 10
    'Applies a function or op to a number of partially flattened inputs.\n\n  The `inputs` are flattened up to `shallow_tree` before being mapped.\n\n  Use Case:\n\n  Sometimes we wish to apply a function to a partially flattened\n  structure (for example when the function itself takes structure inputs). We\n  achieve this by specifying a shallow structure, `shallow_tree` we wish to\n  flatten up to.\n\n  The `inputs`, can be thought of as having the same structure layout as\n  `shallow_tree`, but with leaf nodes that are themselves tree structures.\n\n  This function therefore will return something with the same base structure as\n  `shallow_tree`.\n\n  Examples:\n\n  ```python\n  shallow_tree = [None, None]\n  inp_val = [1, 2, 3]\n  out = map_structure_up_to(shallow_tree, lambda x: 2 * x, inp_val)\n\n  # Output is: [2, 4]\n  ```\n\n  ```python\n  ab_tuple = collections.namedtuple("ab_tuple", "a, b")\n  op_tuple = collections.namedtuple("op_tuple", "add, mul")\n  inp_val = ab_tuple(a=2, b=3)\n  inp_ops = ab_tuple(a=op_tuple(add=1, mul=2), b=op_tuple(add=2, mul=3))\n  out = map_structure_up_to(inp_val, lambda val, ops: (val + ops.add) * ops.mul,\n                            inp_val, inp_ops)\n\n  # Output is: ab_tuple(a=6, b=15)\n  ```\n\n  ```python\n  data_list = [[2, 4, 6, 8], [[1, 3, 5, 7, 9], [3, 5, 7]]]\n  name_list = [\'evens\', [\'odds\', \'primes\']]\n  out = map_structure_up_to(\n      name_list,\n      lambda name, sec: "first_{}_{}".format(len(sec), name),\n      name_list, data_list)\n\n  # Output is: [\'first_4_evens\', [\'first_5_odds\', \'first_3_primes\']]\n  ```\n\n  Args:\n    modality: enum value of supported modality [Modality.CORE or Modality.DATA]\n    shallow_tree: a shallow structure, common to all the inputs.\n    func: callable which will be applied to each input individually.\n    *inputs: structures that are compatible with shallow_tree. The function\n      `func` is applied to corresponding structures due to partial flattening of\n      each input, so the function must support arity of `len(inputs)`.\n    **kwargs: Arg valid for Modality.CORE only. kwargs to feed to func().\n      Special kwarg `check_types` is not passed to func, but instead determines\n      whether the types of iterables within the structures have to be same (e.g.\n      `map_structure(func, [1], (1,))` raises a `TypeError` exception). To allow\n      this set this argument to `False`.\n\n  Raises:\n    TypeError: If `shallow_tree` is a nested structure but `input_tree` is not.\n    TypeError: If the structure types of `shallow_tree` are different from\n      `input_tree`.\n    ValueError: If the structure lengths of `shallow_tree` are different from\n      `input_tree`.\n\n  Returns:\n    result of repeatedly applying `func`, with the same structure layout as\n    `shallow_tree`.\n  '
    if modality == Modality.CORE:
        return _tf_core_map_structure_with_tuple_paths_up_to(shallow_tree, func, *inputs, **kwargs)
    elif modality == Modality.DATA:
        return _tf_data_map_structure_up_to(shallow_tree, func, *inputs)
    else:
        raise ValueError('Unknown modality used {} for nested structure'.format(modality))

def _tf_core_map_structure_with_tuple_paths_up_to(shallow_tree, func, *inputs, **kwargs):
    if False:
        i = 10
        return i + 15
    'See comments for map_structure_with_tuple_paths_up_to() in tensorflow/python/util/nest.py.'
    if not inputs:
        raise ValueError('Cannot map over no sequences')
    check_types = kwargs.pop('check_types', True)
    expand_composites = kwargs.pop('expand_composites', False)
    is_nested_fn = _is_nested_or_composite if expand_composites else _tf_core_is_nested
    for input_tree in inputs:
        _tf_core_assert_shallow_structure(shallow_tree, input_tree, check_types=check_types, expand_composites=expand_composites)
    flat_value_gen = (_tf_core_flatten_up_to(shallow_tree, input_tree, check_types, expand_composites=expand_composites) for input_tree in inputs)
    flat_path_gen = (path for (path, _) in _tf_core_yield_flat_up_to(shallow_tree, inputs[0], is_nested_fn))
    results = [func(*args, **kwargs) for args in zip(flat_path_gen, *flat_value_gen)]
    return _tf_core_pack_sequence_as(structure=shallow_tree, flat_sequence=results, expand_composites=expand_composites)

def _tf_data_map_structure_up_to(shallow_tree, func, *inputs):
    if False:
        while True:
            i = 10
    if not inputs:
        raise ValueError('Argument `inputs` is empty. Cannot map over no sequences.')
    for input_tree in inputs:
        _tf_data_assert_shallow_structure(shallow_tree, input_tree)
    all_flattened_up_to = (_tf_data_flatten_up_to(shallow_tree, input_tree) for input_tree in inputs)
    results = [func(*tensors) for tensors in zip(*all_flattened_up_to)]
    return _tf_data_pack_sequence_as(structure=shallow_tree, flat_sequence=results)