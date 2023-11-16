"""Logging and Summary Operations.

API docstring: tensorflow.logging
"""
import collections as py_collections
import os
import pprint
import random
import sys
from absl import logging
from tensorflow.python import pywrap_tfe
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.gen_logging_ops import *
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export

def enable_interactive_logging():
    if False:
        while True:
            i = 10
    pywrap_tfe.TFE_Py_EnableInteractivePythonLogging()
try:
    get_ipython()
    enable_interactive_logging()
except NameError:
    pass

@deprecated('2018-08-20', 'Use tf.print instead of tf.Print. Note that tf.print returns a no-output operator that directly prints the output. Outside of defuns or eager mode, this operator will not be executed unless it is directly specified in session.run or used as a control dependency for other operators. This is only a concern in graph mode. Below is an example of how to ensure tf.print executes in graph mode:\n')
@tf_export(v1=['Print'])
@dispatch.add_dispatch_support
def Print(input_, data, message=None, first_n=None, summarize=None, name=None):
    if False:
        while True:
            i = 10
    "Prints a list of tensors.\n\n  This is an identity op (behaves like `tf.identity`) with the side effect\n  of printing `data` when evaluating.\n\n  Note: This op prints to the standard error. It is not currently compatible\n    with jupyter notebook (printing to the notebook *server's* output, not into\n    the notebook).\n\n  @compatibility(TF2)\n  This API is deprecated. Use `tf.print` instead. `tf.print` does not need the\n  `input_` argument.\n\n  `tf.print` works in TF2 when executing eagerly and inside a `tf.function`.\n\n  In TF1-styled sessions, an explicit control dependency declaration is needed\n  to execute the `tf.print` operation. Refer to the documentation of\n  `tf.print` for more details.\n  @end_compatibility\n\n  Args:\n    input_: A tensor passed through this op.\n    data: A list of tensors to print out when op is evaluated.\n    message: A string, prefix of the error message.\n    first_n: Only log `first_n` number of times. Negative numbers log always;\n      this is the default.\n    summarize: Only print this many entries of each tensor. If None, then a\n      maximum of 3 elements are printed per input tensor.\n    name: A name for the operation (optional).\n\n  Returns:\n    A `Tensor`. Has the same type and contents as `input_`.\n\n    ```python\n    sess = tf.compat.v1.Session()\n    with sess.as_default():\n        tensor = tf.range(10)\n        print_op = tf.print(tensor)\n        with tf.control_dependencies([print_op]):\n          out = tf.add(tensor, tensor)\n        sess.run(out)\n    ```\n  "
    return gen_logging_ops._print(input_, data, message, first_n, summarize, name)

def _generate_placeholder_string(x, default_placeholder='{}'):
    if False:
        print('Hello World!')
    'Generate and return a string that does not appear in `x`.'
    placeholder = default_placeholder
    rng = random.Random(5)
    while placeholder in x:
        placeholder = placeholder + str(rng.randint(0, 9))
    return placeholder

def _is_filepath(output_stream):
    if False:
        for i in range(10):
            print('nop')
    'Returns True if output_stream is a file path.'
    return isinstance(output_stream, str) and output_stream.startswith('file://')

@tf_export('print')
@dispatch.add_dispatch_support
def print_v2(*inputs, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Print the specified inputs.\n\n  A TensorFlow operator that prints the specified inputs to a desired\n  output stream or logging level. The inputs may be dense or sparse Tensors,\n  primitive python objects, data structures that contain tensors, and printable\n  Python objects. Printed tensors will recursively show the first and last\n  elements of each dimension to summarize.\n\n  Example:\n    Single-input usage:\n\n    ```python\n    tensor = tf.range(10)\n    tf.print(tensor, output_stream=sys.stderr)\n    ```\n\n    (This prints "[0 1 2 ... 7 8 9]" to sys.stderr)\n\n    Multi-input usage:\n\n    ```python\n    tensor = tf.range(10)\n    tf.print("tensors:", tensor, {2: tensor * 2}, output_stream=sys.stdout)\n    ```\n\n    (This prints "tensors: [0 1 2 ... 7 8 9] {2: [0 2 4 ... 14 16 18]}" to\n    sys.stdout)\n\n    Changing the input separator:\n    ```python\n    tensor_a = tf.range(2)\n    tensor_b = tensor_a * 2\n    tf.print(tensor_a, tensor_b, output_stream=sys.stderr, sep=\',\')\n    ```\n\n    (This prints "[0 1],[0 2]" to sys.stderr)\n\n    Usage in a `tf.function`:\n\n    ```python\n    @tf.function\n    def f():\n        tensor = tf.range(10)\n        tf.print(tensor, output_stream=sys.stderr)\n        return tensor\n\n    range_tensor = f()\n    ```\n\n    (This prints "[0 1 2 ... 7 8 9]" to sys.stderr)\n\n  *Compatibility usage in TF 1.x graphs*:\n\n    In graphs manually created outside of `tf.function`, this method returns\n    the created TF operator that prints the data. To make sure the\n    operator runs, users need to pass the produced op to\n    `tf.compat.v1.Session`\'s run method, or to use the op as a control\n    dependency for executed ops by specifying\n    `with tf.compat.v1.control_dependencies([print_op])`.\n\n    ```python\n    tf.compat.v1.disable_v2_behavior()  # for TF1 compatibility only\n\n    sess = tf.compat.v1.Session()\n    with sess.as_default():\n      tensor = tf.range(10)\n      print_op = tf.print("tensors:", tensor, {2: tensor * 2},\n                          output_stream=sys.stdout)\n      with tf.control_dependencies([print_op]):\n        tripled_tensor = tensor * 3\n\n      sess.run(tripled_tensor)\n    ```\n\n    (This prints "tensors: [0 1 2 ... 7 8 9] {2: [0 2 4 ... 14 16 18]}" to\n    sys.stdout)\n\n  Note: In Jupyter notebooks and colabs, `tf.print` prints to the notebook\n    cell outputs. It will not write to the notebook kernel\'s console logs.\n\n  Args:\n    *inputs: Positional arguments that are the inputs to print. Inputs in the\n      printed output will be separated by spaces. Inputs may be python\n      primitives, tensors, data structures such as dicts and lists that may\n      contain tensors (with the data structures possibly nested in arbitrary\n      ways), and printable python objects.\n    output_stream: The output stream, logging level, or file to print to.\n      Defaults to sys.stderr, but sys.stdout, tf.compat.v1.logging.info,\n      tf.compat.v1.logging.warning, tf.compat.v1.logging.error,\n      absl.logging.info, absl.logging.warning and absl.logging.error are also\n      supported. To print to a file, pass a string started with "file://"\n      followed by the file path, e.g., "file:///tmp/foo.out".\n    summarize: The first and last `summarize` elements within each dimension are\n      recursively printed per Tensor. If None, then the first 3 and last 3\n      elements of each dimension are printed for each tensor. If set to -1, it\n      will print all elements of every tensor.\n    sep: The string to use to separate the inputs. Defaults to " ".\n    end: End character that is appended at the end the printed string. Defaults\n      to the newline character.\n    name: A name for the operation (optional).\n\n  Returns:\n    None when executing eagerly. During graph tracing this returns\n    a TF operator that prints the specified inputs in the specified output\n    stream or logging level. This operator will be automatically executed\n    except inside of `tf.compat.v1` graphs and sessions.\n\n  Raises:\n    ValueError: If an unsupported output stream is specified.\n  '
    output_stream = kwargs.pop('output_stream', sys.stderr)
    name = kwargs.pop('name', None)
    summarize = kwargs.pop('summarize', 3)
    sep = kwargs.pop('sep', ' ')
    end = kwargs.pop('end', os.linesep)
    if kwargs:
        raise ValueError('Unrecognized keyword arguments for tf.print: %s' % kwargs)
    format_name = None
    if name:
        format_name = name + '_format'
    output_stream_to_constant = {sys.stdout: 'stdout', sys.stderr: 'stderr', tf_logging.INFO: 'log(info)', tf_logging.info: 'log(info)', tf_logging.WARN: 'log(warning)', tf_logging.warning: 'log(warning)', tf_logging.warn: 'log(warning)', tf_logging.ERROR: 'log(error)', tf_logging.error: 'log(error)', logging.INFO: 'log(info)', logging.info: 'log(info)', logging.INFO: 'log(info)', logging.WARNING: 'log(warning)', logging.WARN: 'log(warning)', logging.warning: 'log(warning)', logging.warn: 'log(warning)', logging.ERROR: 'log(error)', logging.error: 'log(error)'}
    if _is_filepath(output_stream):
        output_stream_string = output_stream
    else:
        output_stream_string = output_stream_to_constant.get(output_stream)
        if not output_stream_string:
            raise ValueError('Unsupported output stream, logging level, or file.' + str(output_stream) + '. Supported streams are sys.stdout, sys.stderr, tf.logging.info, tf.logging.warning, tf.logging.error. ' + "File needs to be in the form of 'file://<filepath>'.")
    if len(inputs) == 1 and tensor_util.is_tf_type(inputs[0]) and (not isinstance(inputs[0], sparse_tensor.SparseTensor)) and (inputs[0].shape.ndims == 0) and (inputs[0].dtype == dtypes.string):
        formatted_string = inputs[0]
    else:
        templates = []
        tensors = []
        inputs_ordered_dicts_sorted = []
        for input_ in inputs:
            if isinstance(input_, py_collections.OrderedDict):
                inputs_ordered_dicts_sorted.append(py_collections.OrderedDict(sorted(input_.items())))
            else:
                inputs_ordered_dicts_sorted.append(input_)
        tensor_free_structure = nest.map_structure(lambda x: '' if tensor_util.is_tf_type(x) else x, inputs_ordered_dicts_sorted)
        tensor_free_template = ' '.join((pprint.pformat(x) for x in tensor_free_structure))
        placeholder = _generate_placeholder_string(tensor_free_template)
        for input_ in inputs:
            placeholders = []
            for x in nest.flatten(input_):
                if isinstance(x, sparse_tensor.SparseTensor):
                    tensors.extend([x.indices, x.values, x.dense_shape])
                    placeholders.append('SparseTensor(indices={}, values={}, shape={})'.format(placeholder, placeholder, placeholder))
                elif tensor_util.is_tf_type(x):
                    tensors.append(x)
                    placeholders.append(placeholder)
                else:
                    placeholders.append(x)
            if isinstance(input_, str):
                cur_template = input_
            else:
                cur_template = pprint.pformat(nest.pack_sequence_as(input_, placeholders))
            templates.append(cur_template)
        template = sep.join(templates)
        template = template.replace("'" + placeholder + "'", placeholder)
        formatted_string = string_ops.string_format(inputs=tensors, template=template, placeholder=placeholder, summarize=summarize, name=format_name)
    return gen_logging_ops.print_v2(formatted_string, output_stream=output_stream_string, name=name, end=end)

@ops.RegisterGradient('Print')
def _PrintGrad(op, *grad):
    if False:
        print('Hello World!')
    return list(grad) + [None] * (len(op.inputs) - 1)

def _Collect(val, collections, default_collections):
    if False:
        print('Hello World!')
    if collections is None:
        collections = default_collections
    for key in collections:
        ops.add_to_collection(key, val)

@deprecated('2016-11-30', 'Please switch to tf.summary.histogram. Note that tf.summary.histogram uses the node name instead of the tag. This means that TensorFlow will automatically de-duplicate summary names based on the scope they are created in.')
def histogram_summary(tag, values, collections=None, name=None):
    if False:
        print('Hello World!')
    "Outputs a `Summary` protocol buffer with a histogram.\n\n  This ops is deprecated. Please switch to tf.summary.histogram.\n\n  For an explanation of why this op was deprecated, and information on how to\n  migrate, look\n  ['here'](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/deprecated/__init__.py)\n\n  The generated\n  [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)\n  has one summary value containing a histogram for `values`.\n\n  This op reports an `InvalidArgument` error if any value is not finite.\n\n  Args:\n    tag: A `string` `Tensor`. 0-D.  Tag to use for the summary value.\n    values: A real numeric `Tensor`. Any shape. Values to use to build the\n      histogram.\n    collections: Optional list of graph collections keys. The new summary op is\n      added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A scalar `Tensor` of type `string`. The serialized `Summary` protocol\n    buffer.\n  "
    with ops.name_scope(name, 'HistogramSummary', [tag, values]) as scope:
        val = gen_logging_ops.histogram_summary(tag=tag, values=values, name=scope)
        _Collect(val, collections, [ops.GraphKeys.SUMMARIES])
    return val

@deprecated('2016-11-30', 'Please switch to tf.summary.image. Note that tf.summary.image uses the node name instead of the tag. This means that TensorFlow will automatically de-duplicate summary names based on the scope they are created in. Also, the max_images argument was renamed to max_outputs.')
def image_summary(tag, tensor, max_images=3, collections=None, name=None):
    if False:
        while True:
            i = 10
    "Outputs a `Summary` protocol buffer with images.\n\n  For an explanation of why this op was deprecated, and information on how to\n  migrate, look\n  ['here'](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/deprecated/__init__.py)\n\n  The summary has up to `max_images` summary values containing images. The\n  images are built from `tensor` which must be 4-D with shape `[batch_size,\n  height, width, channels]` and where `channels` can be:\n\n  *  1: `tensor` is interpreted as Grayscale.\n  *  3: `tensor` is interpreted as RGB.\n  *  4: `tensor` is interpreted as RGBA.\n\n  The images have the same number of channels as the input tensor. For float\n  input, the values are normalized one image at a time to fit in the range\n  `[0, 255]`.  `uint8` values are unchanged.  The op uses two different\n  normalization algorithms:\n\n  *  If the input values are all positive, they are rescaled so the largest one\n     is 255.\n\n  *  If any input value is negative, the values are shifted so input value 0.0\n     is at 127.  They are then rescaled so that either the smallest value is 0,\n     or the largest one is 255.\n\n  The `tag` argument is a scalar `Tensor` of type `string`.  It is used to\n  build the `tag` of the summary values:\n\n  *  If `max_images` is 1, the summary value tag is '*tag*/image'.\n  *  If `max_images` is greater than 1, the summary value tags are\n     generated sequentially as '*tag*/image/0', '*tag*/image/1', etc.\n\n  Args:\n    tag: A scalar `Tensor` of type `string`. Used to build the `tag` of the\n      summary values.\n    tensor: A 4-D `uint8` or `float32` `Tensor` of shape `[batch_size, height,\n      width, channels]` where `channels` is 1, 3, or 4.\n    max_images: Max number of batch elements to generate images for.\n    collections: Optional list of ops.GraphKeys.  The collections to add the\n      summary to.  Defaults to [ops.GraphKeys.SUMMARIES]\n    name: A name for the operation (optional).\n\n  Returns:\n    A scalar `Tensor` of type `string`. The serialized `Summary` protocol\n    buffer.\n  "
    with ops.name_scope(name, 'ImageSummary', [tag, tensor]) as scope:
        val = gen_logging_ops.image_summary(tag=tag, tensor=tensor, max_images=max_images, name=scope)
        _Collect(val, collections, [ops.GraphKeys.SUMMARIES])
    return val

@deprecated('2016-11-30', 'Please switch to tf.summary.audio. Note that tf.summary.audio uses the node name instead of the tag. This means that TensorFlow will automatically de-duplicate summary names based on the scope they are created in.')
def audio_summary(tag, tensor, sample_rate, max_outputs=3, collections=None, name=None):
    if False:
        i = 10
        return i + 15
    "Outputs a `Summary` protocol buffer with audio.\n\n  This op is deprecated. Please switch to tf.summary.audio.\n  For an explanation of why this op was deprecated, and information on how to\n  migrate, look\n  ['here'](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/deprecated/__init__.py)\n\n  The summary has up to `max_outputs` summary values containing audio. The\n  audio is built from `tensor` which must be 3-D with shape `[batch_size,\n  frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are\n  assumed to be in the range of `[-1.0, 1.0]` with a sample rate of\n  `sample_rate`.\n\n  The `tag` argument is a scalar `Tensor` of type `string`.  It is used to\n  build the `tag` of the summary values:\n\n  *  If `max_outputs` is 1, the summary value tag is '*tag*/audio'.\n  *  If `max_outputs` is greater than 1, the summary value tags are\n     generated sequentially as '*tag*/audio/0', '*tag*/audio/1', etc.\n\n  Args:\n    tag: A scalar `Tensor` of type `string`. Used to build the `tag` of the\n      summary values.\n    tensor: A 3-D `float32` `Tensor` of shape `[batch_size, frames, channels]`\n      or a 2-D `float32` `Tensor` of shape `[batch_size, frames]`.\n    sample_rate: A Scalar `float32` `Tensor` indicating the sample rate of the\n      signal in hertz.\n    max_outputs: Max number of batch elements to generate audio for.\n    collections: Optional list of ops.GraphKeys.  The collections to add the\n      summary to.  Defaults to [ops.GraphKeys.SUMMARIES]\n    name: A name for the operation (optional).\n\n  Returns:\n    A scalar `Tensor` of type `string`. The serialized `Summary` protocol\n    buffer.\n  "
    with ops.name_scope(name, 'AudioSummary', [tag, tensor]) as scope:
        sample_rate = ops.convert_to_tensor(sample_rate, dtype=dtypes.float32, name='sample_rate')
        val = gen_logging_ops.audio_summary_v2(tag=tag, tensor=tensor, max_outputs=max_outputs, sample_rate=sample_rate, name=scope)
        _Collect(val, collections, [ops.GraphKeys.SUMMARIES])
    return val

@deprecated('2016-11-30', 'Please switch to tf.summary.merge.')
def merge_summary(inputs, collections=None, name=None):
    if False:
        print('Hello World!')
    'Merges summaries.\n\n  This op is deprecated. Please switch to tf.compat.v1.summary.merge, which has\n  identical\n  behavior.\n\n  This op creates a\n  [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)\n  protocol buffer that contains the union of all the values in the input\n  summaries.\n\n  When the Op is run, it reports an `InvalidArgument` error if multiple values\n  in the summaries to merge use the same tag.\n\n  Args:\n    inputs: A list of `string` `Tensor` objects containing serialized `Summary`\n      protocol buffers.\n    collections: Optional list of graph collections keys. The new summary op is\n      added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A scalar `Tensor` of type `string`. The serialized `Summary` protocol\n    buffer resulting from the merging.\n  '
    with ops.name_scope(name, 'MergeSummary', inputs):
        val = gen_logging_ops.merge_summary(inputs=inputs, name=name)
        _Collect(val, collections, [])
    return val

@deprecated('2016-11-30', 'Please switch to tf.summary.merge_all.')
def merge_all_summaries(key=ops.GraphKeys.SUMMARIES):
    if False:
        print('Hello World!')
    'Merges all summaries collected in the default graph.\n\n  This op is deprecated. Please switch to tf.compat.v1.summary.merge_all, which\n  has\n  identical behavior.\n\n  Args:\n    key: `GraphKey` used to collect the summaries.  Defaults to\n      `GraphKeys.SUMMARIES`.\n\n  Returns:\n    If no summaries were collected, returns None.  Otherwise returns a scalar\n    `Tensor` of type `string` containing the serialized `Summary` protocol\n    buffer resulting from the merging.\n  '
    summary_ops = ops.get_collection(key)
    if not summary_ops:
        return None
    else:
        return merge_summary(summary_ops)

def get_summary_op():
    if False:
        for i in range(10):
            print('nop')
    'Returns a single Summary op that would run all summaries.\n\n  Either existing one from `SUMMARY_OP` collection or merges all existing\n  summaries.\n\n  Returns:\n    If no summaries were collected, returns None. Otherwise returns a scalar\n    `Tensor` of type `string` containing the serialized `Summary` protocol\n    buffer resulting from the merging.\n  '
    summary_op = ops.get_collection(ops.GraphKeys.SUMMARY_OP)
    if summary_op is not None:
        if summary_op:
            summary_op = summary_op[0]
        else:
            summary_op = None
    if summary_op is None:
        summary_op = merge_all_summaries()
        if summary_op is not None:
            ops.add_to_collection(ops.GraphKeys.SUMMARY_OP, summary_op)
    return summary_op

@deprecated('2016-11-30', 'Please switch to tf.summary.scalar. Note that tf.summary.scalar uses the node name instead of the tag. This means that TensorFlow will automatically de-duplicate summary names based on the scope they are created in. Also, passing a tensor or list of tags to a scalar summary op is no longer supported.')
def scalar_summary(tags, values, collections=None, name=None):
    if False:
        i = 10
        return i + 15
    "Outputs a `Summary` protocol buffer with scalar values.\n\n  This ops is deprecated. Please switch to tf.summary.scalar.\n  For an explanation of why this op was deprecated, and information on how to\n  migrate, look\n  ['here'](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/deprecated/__init__.py)\n\n  The input `tags` and `values` must have the same shape.  The generated\n  summary has a summary value for each tag-value pair in `tags` and `values`.\n\n  Args:\n    tags: A `string` `Tensor`.  Tags for the summaries.\n    values: A real numeric Tensor.  Values for the summaries.\n    collections: Optional list of graph collections keys. The new summary op is\n      added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.\n    name: A name for the operation (optional).\n\n  Returns:\n    A scalar `Tensor` of type `string`. The serialized `Summary` protocol\n    buffer.\n  "
    with ops.name_scope(name, 'ScalarSummary', [tags, values]) as scope:
        val = gen_logging_ops.scalar_summary(tags=tags, values=values, name=scope)
        _Collect(val, collections, [ops.GraphKeys.SUMMARIES])
    return val
ops.NotDifferentiable('HistogramSummary')
ops.NotDifferentiable('ImageSummary')
ops.NotDifferentiable('AudioSummary')
ops.NotDifferentiable('AudioSummaryV2')
ops.NotDifferentiable('MergeSummary')
ops.NotDifferentiable('ScalarSummary')
ops.NotDifferentiable('TensorSummary')
ops.NotDifferentiable('TensorSummaryV2')
ops.NotDifferentiable('Timestamp')