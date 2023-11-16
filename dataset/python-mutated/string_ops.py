"""Operations for working with string Tensors.

API docstring: tensorflow.strings
"""
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.gen_string_ops import *
from tensorflow.python.util import compat as util_compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export

@tf_export('strings.regex_full_match')
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def regex_full_match(input, pattern, name=None):
    if False:
        return 10
    'Match elements of `input` with regex `pattern`.\n\n  Args:\n    input: string `Tensor`, the source strings to process.\n    pattern: string or scalar string `Tensor`, regular expression to use,\n      see more details at https://github.com/google/re2/wiki/Syntax\n    name: Name of the op.\n\n  Returns:\n    bool `Tensor` of the same shape as `input` with match results.\n  '
    if isinstance(pattern, util_compat.bytes_or_text_types):
        return gen_string_ops.static_regex_full_match(input=input, pattern=pattern, name=name)
    return gen_string_ops.regex_full_match(input=input, pattern=pattern, name=name)
regex_full_match.__doc__ = gen_string_ops.regex_full_match.__doc__

@tf_export('strings.regex_replace', v1=['strings.regex_replace', 'regex_replace'])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('regex_replace')
def regex_replace(input, pattern, rewrite, replace_global=True, name=None):
    if False:
        print('Hello World!')
    'Replace elements of `input` matching regex `pattern` with `rewrite`.\n\n  >>> tf.strings.regex_replace("Text with tags.<br /><b>contains html</b>",\n  ...                          "<[^>]+>", " ")\n  <tf.Tensor: shape=(), dtype=string, numpy=b\'Text with tags.  contains html \'>\n\n  Args:\n    input: string `Tensor`, the source strings to process.\n    pattern: string or scalar string `Tensor`, regular expression to use,\n      see more details at https://github.com/google/re2/wiki/Syntax\n    rewrite: string or scalar string `Tensor`, value to use in match\n      replacement, supports backslash-escaped digits (\\1 to \\9) can be to insert\n      text matching corresponding parenthesized group.\n    replace_global: `bool`, if `True` replace all non-overlapping matches,\n      else replace only the first match.\n    name: A name for the operation (optional).\n\n  Returns:\n    string `Tensor` of the same shape as `input` with specified replacements.\n  '
    if isinstance(pattern, util_compat.bytes_or_text_types) and isinstance(rewrite, util_compat.bytes_or_text_types):
        return gen_string_ops.static_regex_replace(input=input, pattern=pattern, rewrite=rewrite, replace_global=replace_global, name=name)
    return gen_string_ops.regex_replace(input=input, pattern=pattern, rewrite=rewrite, replace_global=replace_global, name=name)

@tf_export('strings.format')
@dispatch.add_dispatch_support
def string_format(template, inputs, placeholder='{}', summarize=3, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Formats a string template using a list of tensors.\n\n  Formats a string template using a list of tensors, abbreviating tensors by\n  only printing the first and last `summarize` elements of each dimension\n  (recursively). If formatting only one tensor into a template, the tensor does\n  not have to be wrapped in a list.\n\n  Example:\n    Formatting a single-tensor template:\n\n    >>> tensor = tf.range(5)\n    >>> tf.strings.format("tensor: {}, suffix", tensor)\n    <tf.Tensor: shape=(), dtype=string, numpy=b\'tensor: [0 1 2 3 4], suffix\'>\n\n    Formatting a multi-tensor template:\n\n    >>> tensor_a = tf.range(2)\n    >>> tensor_b = tf.range(1, 4, 2)\n    >>> tf.strings.format("a: {}, b: {}, suffix", (tensor_a, tensor_b))\n    <tf.Tensor: shape=(), dtype=string, numpy=b\'a: [0 1], b: [1 3], suffix\'>\n\n\n  Args:\n    template: A string template to format tensor values into.\n    inputs: A list of `Tensor` objects, or a single Tensor.\n      The list of tensors to format into the template string. If a solitary\n      tensor is passed in, the input tensor will automatically be wrapped as a\n      list.\n    placeholder: An optional `string`. Defaults to `{}`.\n      At each placeholder occurring in the template, a subsequent tensor\n      will be inserted.\n    summarize: An optional `int`. Defaults to `3`.\n      When formatting the tensors, show the first and last `summarize`\n      entries of each tensor dimension (recursively). If set to -1, all\n      elements of the tensor will be shown.\n    name: A name for the operation (optional).\n\n  Returns:\n    A scalar `Tensor` of type `string`.\n\n  Raises:\n    ValueError: if the number of placeholders does not match the number of\n      inputs.\n  '
    if tensor_util.is_tf_type(inputs):
        inputs = [inputs]
    if template.count(placeholder) != len(inputs):
        raise ValueError(f'The template expects {template.count(placeholder)} tensors, but the inputs only has {len(inputs)}. Please ensure the number of placeholders in template matches inputs length.')
    return gen_string_ops.string_format(inputs, template=template, placeholder=placeholder, summarize=summarize, name=name)

def string_split(source, sep=None, skip_empty=True, delimiter=None):
    if False:
        print('Hello World!')
    "Split elements of `source` based on `delimiter` into a `SparseTensor`.\n\n  Let N be the size of source (typically N will be the batch size). Split each\n  element of `source` based on `delimiter` and return a `SparseTensor`\n  containing the split tokens. Empty tokens are ignored.\n\n  If `sep` is an empty string, each element of the `source` is split\n  into individual strings, each containing one byte. (This includes splitting\n  multibyte sequences of UTF-8.) If delimiter contains multiple bytes, it is\n  treated as a set of delimiters with each considered a potential split point.\n\n  For example:\n  N = 2, source[0] is 'hello world' and source[1] is 'a b c', then the output\n  will be\n\n  st.indices = [0, 0;\n                0, 1;\n                1, 0;\n                1, 1;\n                1, 2]\n  st.shape = [2, 3]\n  st.values = ['hello', 'world', 'a', 'b', 'c']\n\n  Args:\n    source: `1-D` string `Tensor`, the strings to split.\n    sep: `0-D` string `Tensor`, the delimiter character, the string should\n      be length 0 or 1. Default is ' '.\n    skip_empty: A `bool`. If `True`, skip the empty strings from the result.\n    delimiter: deprecated alias for `sep`.\n\n  Raises:\n    ValueError: If delimiter is not a string.\n\n  Returns:\n    A `SparseTensor` of rank `2`, the strings split according to the delimiter.\n    The first column of the indices corresponds to the row in `source` and the\n    second column corresponds to the index of the split component in this row.\n  "
    delimiter = deprecation.deprecated_argument_lookup('sep', sep, 'delimiter', delimiter)
    if delimiter is None:
        delimiter = ' '
    delimiter = ops.convert_to_tensor(delimiter, dtype=dtypes.string)
    source = ops.convert_to_tensor(source, dtype=dtypes.string)
    (indices, values, shape) = gen_string_ops.string_split(source, delimiter=delimiter, skip_empty=skip_empty)
    indices.set_shape([None, 2])
    values.set_shape([None])
    shape.set_shape([2])
    return sparse_tensor.SparseTensor(indices, values, shape)

def string_split_v2(source, sep=None, maxsplit=-1):
    if False:
        for i in range(10):
            print('nop')
    'Split elements of `source` based on `sep` into a `SparseTensor`.\n\n  Let N be the size of source (typically N will be the batch size). Split each\n  element of `source` based on `sep` and return a `SparseTensor`\n  containing the split tokens. Empty tokens are ignored.\n\n  For example, N = 2, source[0] is \'hello world\' and source[1] is \'a b c\',\n  then the output will be\n\n  st.indices = [0, 0;\n                0, 1;\n                1, 0;\n                1, 1;\n                1, 2]\n  st.shape = [2, 3]\n  st.values = [\'hello\', \'world\', \'a\', \'b\', \'c\']\n\n  If `sep` is given, consecutive delimiters are not grouped together and are\n  deemed to delimit empty strings. For example, source of `"1<>2<><>3"` and\n  sep of `"<>"` returns `["1", "2", "", "3"]`. If `sep` is None or an empty\n  string, consecutive whitespace are regarded as a single separator, and the\n  result will contain no empty strings at the start or end if the string has\n  leading or trailing whitespace.\n\n  Note that the above mentioned behavior matches python\'s str.split.\n\n  Args:\n    source: `1-D` string `Tensor`, the strings to split.\n    sep: `0-D` string `Tensor`, the delimiter character.\n    maxsplit: An `int`. If `maxsplit > 0`, limit of the split of the result.\n\n  Raises:\n    ValueError: If sep is not a string.\n\n  Returns:\n    A `SparseTensor` of rank `2`, the strings split according to the delimiter.\n    The first column of the indices corresponds to the row in `source` and the\n    second column corresponds to the index of the split component in this row.\n  '
    if sep is None:
        sep = ''
    sep = ops.convert_to_tensor(sep, dtype=dtypes.string)
    source = ops.convert_to_tensor(source, dtype=dtypes.string)
    (indices, values, shape) = gen_string_ops.string_split_v2(source, sep=sep, maxsplit=maxsplit)
    indices.set_shape([None, 2])
    values.set_shape([None])
    shape.set_shape([2])
    return sparse_tensor.SparseTensor(indices, values, shape)

def _reduce_join_reduction_dims(x, axis):
    if False:
        return 10
    'Returns range(rank(x) - 1, 0, -1) if axis is None; or axis otherwise.'
    if axis is not None:
        return axis
    else:
        if x.get_shape().ndims is not None:
            return constant_op.constant(np.arange(x.get_shape().ndims - 1, -1, -1), dtype=dtypes.int32)
        return math_ops.range(array_ops.rank(x) - 1, -1, -1)

@tf_export(v1=['strings.reduce_join', 'reduce_join'])
@dispatch.add_dispatch_support
@deprecation.deprecated_args(None, 'keep_dims is deprecated, use keepdims instead', 'keep_dims')
@deprecation.deprecated_endpoints('reduce_join')
def reduce_join(inputs, axis=None, keep_dims=None, separator='', name=None, reduction_indices=None, keepdims=None):
    if False:
        print('Hello World!')
    keepdims = deprecation.deprecated_argument_lookup('keepdims', keepdims, 'keep_dims', keep_dims)
    if keep_dims is None:
        keep_dims = False
    axis = deprecation.deprecated_argument_lookup('axis', axis, 'reduction_indices', reduction_indices)
    return reduce_join_v2(inputs=inputs, axis=axis, keepdims=keepdims, separator=separator, name=name)

@tf_export('strings.reduce_join', v1=[])
@dispatch.add_dispatch_support
def reduce_join_v2(inputs, axis=None, keepdims=False, separator='', name=None):
    if False:
        i = 10
        return i + 15
    'Joins all strings into a single string, or joins along an axis.\n\n  This is the reduction operation for the elementwise `tf.strings.join` op.\n\n  >>> tf.strings.reduce_join([[\'abc\',\'123\'],\n  ...                         [\'def\',\'456\']]).numpy()\n  b\'abc123def456\'\n  >>> tf.strings.reduce_join([[\'abc\',\'123\'],\n  ...                         [\'def\',\'456\']], axis=-1).numpy()\n  array([b\'abc123\', b\'def456\'], dtype=object)\n  >>> tf.strings.reduce_join([[\'abc\',\'123\'],\n  ...                         [\'def\',\'456\']],\n  ...                        axis=-1,\n  ...                        separator=" ").numpy()\n  array([b\'abc 123\', b\'def 456\'], dtype=object)\n\n  Args:\n    inputs: A `tf.string` tensor.\n    axis: Which axis to join along. The default behavior is to join all\n      elements, producing a scalar.\n    keepdims: If true, retains reduced dimensions with length 1.\n    separator: a string added between each string being joined.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `tf.string` tensor.\n  '
    with ops.name_scope(None, 'ReduceJoin', [inputs, axis]):
        inputs_t = ops.convert_to_tensor(inputs)
        axis = _reduce_join_reduction_dims(inputs_t, axis)
        return gen_string_ops.reduce_join(inputs=inputs_t, reduction_indices=axis, keep_dims=keepdims, separator=separator, name=name)
reduce_join.__doc__ = reduce_join_v2.__doc__

@tf_export(v1=['strings.length'])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def string_length(input, name=None, unit='BYTE'):
    if False:
        for i in range(10):
            print('nop')
    'Computes the length of each string given in the input tensor.\n\n  >>> strings = tf.constant([\'Hello\',\'TensorFlow\', \'🙂\'])\n  >>> tf.strings.length(strings).numpy() # default counts bytes\n  array([ 5, 10, 4], dtype=int32)\n  >>> tf.strings.length(strings, unit="UTF8_CHAR").numpy()\n  array([ 5, 10, 1], dtype=int32)\n\n  Args:\n    input: A `Tensor` of type `string`. The strings for which to compute the\n      length for each element.\n    name: A name for the operation (optional).\n    unit: An optional `string` from: `"BYTE", "UTF8_CHAR"`. Defaults to\n      `"BYTE"`. The unit that is counted to compute string length.  One of:\n        `"BYTE"` (for the number of bytes in each string) or `"UTF8_CHAR"` (for\n        the number of UTF-8 encoded Unicode code points in each string). Results\n        are undefined if `unit=UTF8_CHAR` and the `input` strings do not contain\n        structurally valid UTF-8.\n\n  Returns:\n    A `Tensor` of type `int32`, containing the length of the input string in\n    the same element of the input tensor.\n  '
    return gen_string_ops.string_length(input, unit=unit, name=name)

@tf_export('strings.length', v1=[])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def string_length_v2(input, unit='BYTE', name=None):
    if False:
        while True:
            i = 10
    return gen_string_ops.string_length(input, unit=unit, name=name)
string_length_v2.__doc__ = gen_string_ops.string_length.__doc__

@tf_export(v1=['substr'])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
@deprecation.deprecated(None, 'Use `tf.strings.substr` instead of `tf.substr`.')
def substr_deprecated(input, pos, len, name=None, unit='BYTE'):
    if False:
        i = 10
        return i + 15
    return substr(input, pos, len, name=name, unit=unit)
substr_deprecated.__doc__ = gen_string_ops.substr.__doc__

@tf_export(v1=['strings.substr'])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def substr(input, pos, len, name=None, unit='BYTE'):
    if False:
        i = 10
        return i + 15
    return gen_string_ops.substr(input, pos, len, unit=unit, name=name)
substr.__doc__ = gen_string_ops.substr.__doc__

@tf_export('strings.substr', v1=[])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def substr_v2(input, pos, len, unit='BYTE', name=None):
    if False:
        print('Hello World!')
    return gen_string_ops.substr(input, pos, len, unit=unit, name=name)
substr_v2.__doc__ = gen_string_ops.substr.__doc__
ops.NotDifferentiable('RegexReplace')
ops.NotDifferentiable('StringToHashBucket')
ops.NotDifferentiable('StringToHashBucketFast')
ops.NotDifferentiable('StringToHashBucketStrong')
ops.NotDifferentiable('ReduceJoin')
ops.NotDifferentiable('StringJoin')
ops.NotDifferentiable('StringSplit')
ops.NotDifferentiable('AsString')
ops.NotDifferentiable('EncodeBase64')
ops.NotDifferentiable('DecodeBase64')

@tf_export('strings.to_number', v1=[])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def string_to_number(input, out_type=dtypes.float32, name=None):
    if False:
        return 10
    'Converts each string in the input Tensor to the specified numeric type.\n\n  (Note that int32 overflow results in an error while float overflow\n  results in a rounded value.)\n\n  Examples:\n\n  >>> tf.strings.to_number("1.55")\n  <tf.Tensor: shape=(), dtype=float32, numpy=1.55>\n  >>> tf.strings.to_number("3", tf.int32)\n  <tf.Tensor: shape=(), dtype=int32, numpy=3>\n\n  Args:\n    input: A `Tensor` of type `string`.\n    out_type: An optional `tf.DType` from: `tf.float32, tf.float64, tf.int32,\n      tf.int64`. Defaults to `tf.float32`.\n      The numeric type to interpret each string in `string_tensor` as.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` of type `out_type`.\n  '
    return gen_parsing_ops.string_to_number(input, out_type, name)

@tf_export(v1=['strings.to_number', 'string_to_number'])
@dispatch.add_dispatch_support
def string_to_number_v1(string_tensor=None, out_type=dtypes.float32, name=None, input=None):
    if False:
        print('Hello World!')
    string_tensor = deprecation.deprecated_argument_lookup('input', input, 'string_tensor', string_tensor)
    return gen_parsing_ops.string_to_number(string_tensor, out_type, name)
string_to_number_v1.__doc__ = gen_parsing_ops.string_to_number.__doc__

@tf_export('strings.to_hash_bucket', v1=[])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def string_to_hash_bucket(input, num_buckets, name=None):
    if False:
        for i in range(10):
            print('nop')
    'Converts each string in the input Tensor to its hash mod by a number of buckets.\n\n  The hash function is deterministic on the content of the string within the\n  process.\n\n  Note that the hash function may change from time to time.\n  This functionality will be deprecated and it\'s recommended to use\n  `tf.strings.to_hash_bucket_fast()` or `tf.strings.to_hash_bucket_strong()`.\n\n  Examples:\n\n  >>> tf.strings.to_hash_bucket(["Hello", "TensorFlow", "2.x"], 3)\n  <tf.Tensor: shape=(3,), dtype=int64, numpy=array([2, 0, 1])>\n\n  Args:\n    input: A `Tensor` of type `string`.\n    num_buckets: An `int` that is `>= 1`. The number of buckets.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor` of type `int64`.\n  '
    return gen_string_ops.string_to_hash_bucket(input, num_buckets, name)

@tf_export(v1=['strings.to_hash_bucket', 'string_to_hash_bucket'])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def string_to_hash_bucket_v1(string_tensor=None, num_buckets=None, name=None, input=None):
    if False:
        for i in range(10):
            print('nop')
    string_tensor = deprecation.deprecated_argument_lookup('input', input, 'string_tensor', string_tensor)
    return gen_string_ops.string_to_hash_bucket(string_tensor, num_buckets, name)
string_to_hash_bucket_v1.__doc__ = gen_string_ops.string_to_hash_bucket.__doc__

@tf_export('strings.join', v1=['strings.join', 'string_join'])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints('string_join')
def string_join(inputs, separator='', name=None):
    if False:
        for i in range(10):
            print('nop')
    'Perform element-wise concatenation of a list of string tensors.\n\n  Given a list of string tensors of same shape, performs element-wise\n  concatenation of the strings of the same index in all tensors.\n\n\n  >>> tf.strings.join([\'abc\',\'def\']).numpy()\n  b\'abcdef\'\n  >>> tf.strings.join([[\'abc\',\'123\'],\n  ...                  [\'def\',\'456\'],\n  ...                  [\'ghi\',\'789\']]).numpy()\n  array([b\'abcdefghi\', b\'123456789\'], dtype=object)\n  >>> tf.strings.join([[\'abc\',\'123\'],\n  ...                  [\'def\',\'456\']],\n  ...                  separator=" ").numpy()\n  array([b\'abc def\', b\'123 456\'], dtype=object)\n\n  The reduction version of this elementwise operation is\n  `tf.strings.reduce_join`\n\n  Args:\n    inputs: A list of `tf.Tensor` objects of same size and `tf.string` dtype.\n    separator: A string added between each string being joined.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `tf.string` tensor.\n  '
    return gen_string_ops.string_join(inputs, separator=separator, name=name)

@tf_export('strings.unsorted_segment_join')
@dispatch.add_dispatch_support
def unsorted_segment_join(inputs, segment_ids, num_segments, separator='', name=None):
    if False:
        while True:
            i = 10
    'Joins the elements of `inputs` based on `segment_ids`.\n\n  Computes the string join along segments of a tensor.\n\n  Given `segment_ids` with rank `N` and `data` with rank `N+M`:\n\n  ```\n  output[i, k1...kM] = strings.join([data[j1...jN, k1...kM])\n  ```\n\n  where the join is over all `[j1...jN]` such that `segment_ids[j1...jN] = i`.\n\n  Strings are joined in row-major order.\n\n  For example:\n\n  >>> inputs = [\'this\', \'a\', \'test\', \'is\']\n  >>> segment_ids = [0, 1, 1, 0]\n  >>> num_segments = 2\n  >>> separator = \' \'\n  >>> tf.strings.unsorted_segment_join(inputs, segment_ids, num_segments,\n  ...                                  separator).numpy()\n  array([b\'this is\', b\'a test\'], dtype=object)\n\n  >>> inputs = [[\'Y\', \'q\', \'c\'], [\'Y\', \'6\', \'6\'], [\'p\', \'G\', \'a\']]\n  >>> segment_ids = [1, 0, 1]\n  >>> num_segments = 2\n  >>> tf.strings.unsorted_segment_join(inputs, segment_ids, num_segments,\n  ...                                  separator=\':\').numpy()\n  array([[b\'Y\', b\'6\', b\'6\'],\n         [b\'Y:p\', b\'q:G\', b\'c:a\']], dtype=object)\n\n  Args:\n    inputs: A list of `tf.Tensor` objects of type `tf.string`.\n    segment_ids: A tensor whose shape is a prefix of `inputs.shape` and whose\n      type must be `tf.int32` or `tf.int64`. Negative segment ids are not\n      supported.\n    num_segments: A scalar of type `tf.int32` or `tf.int64`. Must be\n      non-negative and larger than any segment id.\n    separator: The separator to use when joining. Defaults to `""`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `tf.string` tensor representing the concatenated values, using the given\n    separator.\n  '
    return gen_string_ops.unsorted_segment_join(inputs, segment_ids, num_segments, separator=separator, name=name)
dispatch.register_unary_elementwise_api(gen_string_ops.as_string)
dispatch.register_unary_elementwise_api(gen_string_ops.decode_base64)
dispatch.register_unary_elementwise_api(gen_string_ops.encode_base64)
dispatch.register_unary_elementwise_api(gen_string_ops.string_lower)
dispatch.register_unary_elementwise_api(gen_string_ops.string_upper)
dispatch.register_unary_elementwise_api(gen_string_ops.unicode_transcode)
dispatch.register_unary_elementwise_api(gen_string_ops.string_strip)
dispatch.register_unary_elementwise_api(gen_string_ops.string_to_hash_bucket_fast)
dispatch.register_unary_elementwise_api(gen_string_ops.string_to_hash_bucket_strong)
dispatch.register_unary_elementwise_api(gen_string_ops.unicode_script)