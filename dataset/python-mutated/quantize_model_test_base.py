"""Base test class for quantize_model Tests."""
import os
import re
from typing import Collection, Iterable, Mapping, Sequence, Tuple, Optional, Union, List
from absl.testing import parameterized
import numpy as np
import tensorflow
from tensorflow.compiler.mlir.quantization.tensorflow.python import representative_dataset as repr_dataset
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.lib.io import file_io
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop as while_loop_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import save as saved_model_save
from tensorflow.python.saved_model import signature_def_utils_impl
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.types import core
_AttrValType = Union[List[int], bool, str, None]

class QuantizedModelTest(test.TestCase, parameterized.TestCase):
    """Base test class for TF-quant tests."""

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        super().setUp()
        self._input_saved_model_path = self.create_tempdir('input').full_path
        self._output_saved_model_path = self.create_tempdir('output').full_path
        self._output_saved_model_path_2 = self.create_tempdir('output2').full_path

    def _get_dir_size(self, path: str='.'):
        if False:
            for i in range(10):
                print('nop')
        'Get the total size of files and sub-directories under the path.\n\n    Args:\n      path: Path of a directory or a file to calculate the total size.\n\n    Returns:\n      Total size of the directory or a file.\n    '
        total = 0
        for (root, _, files) in os.walk(path):
            for filename in files:
                total += os.path.getsize(os.path.join(root, filename))
        return total

    def _any_log_contains(self, substring: str, log_record_list: List['logging.LogRecord']) -> bool:
        if False:
            return 10
        'Returns True if any of the log contains a given substring.\n\n    Args:\n      substring: A piece of string to check whether it exists in the log\n        message.\n      log_record_list: A list of `absl.logging.LogRecord`s.\n\n    Returns:\n      True if and only if the substring exists in any of the log in\n      `log_record_list`.\n    '
        return any(map(lambda log_record: substring in str(log_record.message), log_record_list))

    def assertSizeRatioGreaterThan(self, path_a: str, path_b: str, threshold: float):
        if False:
            return 10
        'Check if the size ratio of the given paths is greater than the threshold.\n\n    Args:\n      path_a: Path of a directory or a file to be the nominator of the ratio.\n      path_b: Path of a directory or a file to be the denominator of the ratio.\n      threshold: a number to compare with.\n\n    Returns:\n      True if the size ratio of path_a / path_b is greater than threshold.\n    '
        size_a = self._get_dir_size(path_a)
        size_b = self._get_dir_size(path_b)
        size_ratio = size_a / size_b
        return self.assertGreater(size_ratio, threshold)

    def assertSizeRatioLessThan(self, path_a: str, path_b: str, threshold: float):
        if False:
            while True:
                i = 10
        'Check if the size ratio of the given paths is less than the threshold.\n\n    Args:\n      path_a: Path of a directory or a file to be the nominator of the ratio.\n      path_b: Path of a directory or a file to be the denominator of the ratio.\n      threshold: a number to compare with.\n\n    Returns:\n      True if the size ratio of path_a / path_b is less than threshold.\n    '
        size_a = self._get_dir_size(path_a)
        size_b = self._get_dir_size(path_b)
        size_ratio = size_a / size_b
        return self.assertLess(size_ratio, threshold)

    def _is_quantized_function(self, func: function_pb2.FunctionDef) -> bool:
        if False:
            return 10
        'Determine whether a FunctionDef is quantized.\n\n    Args:\n      func: A FunctionDef object.\n\n    Returns:\n      True iff `func` is quantized.\n    '
        return func.signature.name.startswith('quantized_')

    def _is_composite_function(self, func: function_pb2.FunctionDef) -> bool:
        if False:
            return 10
        'Determine whether a FunctionDef is composite function.\n\n    Args:\n      func: A FunctionDef object.\n\n    Returns:\n      True iff `func` is composte function.\n    '
        return func.signature.name.startswith('composite_')

    def _contains_op_with_name_and_attribute(self, nodes: Iterable[node_def_pb2.NodeDef], op_name: str, attr_name: str, attr_val: _AttrValType, node_name: str='') -> bool:
        if False:
            return 10
        "Determine whether there is a node whose operation name matches `op_name`.\n\n    If `attr_name` is given, additionally check if the `attr_val` matches with\n    the attribute value of the op.\n\n    Args:\n      nodes: Iterable of NodeDefs.\n      op_name: Name of the op to match.\n      attr_name: Name of the attribute of the op to match.\n      attr_val: Value of the attr_name to check.\n      node_name: Name of the node to match. Accepts regex2 format.\n\n    Returns:\n      True if there exists a node whose name matches `op_name` and 'attr_val' if\n      'attr_name' is given.\n    "

        def match_node_name(name):
            if False:
                i = 10
                return i + 15
            if not node_name:
                return True
            compiled_regex = re.compile(node_name)
            match = re.fullmatch(compiled_regex, name)
            return match is not None
        return any((node.attr.get(attr_name) == attr_val for node in nodes if node.op == op_name and match_node_name(node.name)))

    def _contains_quantized_function_call(self, graphdef: graph_pb2.GraphDef) -> bool:
        if False:
            return 10
        'Determines if the graph def has quantized function call.\n\n    Args:\n      graphdef: A GraphDef object.\n\n    Returns:\n      True if and only if the graph def contains a quantized function call.\n    '
        return any(map(self._is_quantized_function, graphdef.library.function))

    def _contains_composite_function_call(self, graphdef: graph_pb2.GraphDef) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Determines if the graph def has composite function call.\n\n    Args:\n      graphdef: A GraphDef object.\n\n    Returns:\n      True if and only if the graph def contains a composite function call.\n    '
        return any(map(self._is_composite_function, graphdef.library.function))

    def _contains_op(self, graphdef: graph_pb2.GraphDef, op_name: str, attr_name: str='', attr_val: _AttrValType=None, node_name: str='') -> bool:
        if False:
            return 10
        'Determines if the graph def contains the given op.\n\n    Args:\n      graphdef: A GraphDef object.\n      op_name: Name of the operation to find within the graph.\n      attr_name: Name of the attribute of the op to match.\n      attr_val: Value of the attr_name to check.\n      node_name: Name of the node to match. Accepts regex2 format.\n\n    Returns:\n      True if and only if the graph def contains an op named `op_name`. If\n      `attr_name` is given, check if the `attr_val` matches with the attribute\n      value of the op.\n    '
        if self._contains_op_with_name_and_attribute(nodes=graphdef.node, op_name=op_name, attr_name=attr_name, attr_val=attr_val, node_name=node_name):
            return True
        for func in graphdef.library.function:
            if self._contains_op_with_name_and_attribute(nodes=func.node_def, op_name=op_name, attr_name=attr_name, attr_val=attr_val, node_name=node_name):
                return True
        return False

    def _count_ops(self, graphdef: graph_pb2.GraphDef, op_names: Collection[str], attr_name: str='', attr_val: _AttrValType=None, get_op_name: bool=False) -> int:
        if False:
            i = 10
            return i + 15
        "Returns the number of given ops in a graph def.\n\n    Args:\n      graphdef: A GraphDef object.\n      op_names: Names of the operations to find within the graph.\n      attr_name: Name of the attribute of the ops to match.\n      attr_val: Value of the attr_name to check.\n      get_op_name: If set True, checks node.name rather than node.op.\n\n    Returns:\n      The number of occurrences of the given ops in a graph. The ops will be\n      counted only if the ops are named 'op_name' and has 'attr_val' if\n      'attr_name' is specified.\n    "
        op_count = 0
        for op_name in op_names:
            op_count += self._count_op_with_name_and_attribute(nodes=graphdef.node, op_name=op_name, attr_name=attr_name, attr_val=attr_val, get_op_name=get_op_name)
            for func in graphdef.library.function:
                op_count += self._count_op_with_name_and_attribute(nodes=func.node_def, op_name=op_name, attr_name=attr_name, attr_val=attr_val, get_op_name=get_op_name)
        return op_count

    def _count_op_with_name_and_attribute(self, nodes: Iterable[node_def_pb2.NodeDef], op_name: str, attr_name: str, attr_val: _AttrValType, get_op_name: bool=False) -> int:
        if False:
            i = 10
            return i + 15
        "Determine the number of nodes whose operation name matches `op_name`.\n\n    If `attr_name` is given, additionally check if the `attr_val` matches with\n    the attribute value of the op.\n\n    Args:\n      nodes: Iterable of NodeDefs.\n      op_name: Name of the op to match.\n      attr_name: Name of the attribute of the op to match.\n      attr_val: Value of the attr_name to check.\n      get_op_name: If set True, checks node.name rather than node.op.\n\n    Returns:\n      The number of occurrences of nodes whose name match `op_name` and\n      'attr_val' if 'attr_name' is given.\n    "
        if get_op_name:
            return len([node.attr.get(attr_name) == attr_val for node in nodes if node.name == op_name])
        else:
            return len([node.attr.get(attr_name) == attr_val for node in nodes if node.op == op_name])

    def _create_simple_tf1_conv_model(self, input_shape: Sequence[int]=(1, 3, 4, 3), filter_shape: Sequence[int]=(2, 3, 3, 2), use_variable_for_filter=False) -> Tuple[core.Tensor, core.Tensor]:
        if False:
            while True:
                i = 10
        'Creates a basic convolution model.\n\n    This is intended to be used for TF1 (graph mode) tests.\n\n    Args:\n      input_shape: Shape of the input tensor.\n      filter_shape: Shape of the filter.\n      use_variable_for_filter: Setting this to `True` makes the filter for the\n        conv operation a `tf.Variable`.\n\n    Returns:\n      in_placeholder: Input tensor placeholder.\n      output_tensor: The resulting tensor of the convolution operation.\n    '
        in_placeholder = array_ops.placeholder(dtypes.float32, shape=input_shape)
        filters = random_ops.random_uniform(shape=filter_shape, minval=-1.0, maxval=1.0)
        if use_variable_for_filter:
            filters = variables.Variable(filters)
        output_tensor = nn_ops.conv2d(in_placeholder, filters, strides=[1, 1, 2, 1], dilations=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
        return (in_placeholder, output_tensor)

    def _create_simple_tf1_gather_model(self, input_type: dtypes.DType, use_variable_for_filter=False) -> Tuple[core.Tensor, core.Tensor]:
        if False:
            return 10
        'Creates a basic gather model.\n\n    This is intended to be used for TF1 (graph mode) tests.\n\n    Args:\n      input_type: type of the input index tensor for gather operation.\n      use_variable_for_filter: Setting this to `True` makes the filter for the\n        gather operation a `tf.Variable`.\n\n    Returns:\n      in_placeholder: Input tensor placeholder.\n      output_tensor: The resulting tensor of the gather operation.\n    '
        in_placeholder = array_ops.placeholder(input_type, shape=6)
        filters = np.random.randn(128, 32).astype(np.float32)
        if use_variable_for_filter:
            filters = variables.Variable(filters)
        output_tensor = array_ops.gather_v2(filters, in_placeholder)
        return (in_placeholder, output_tensor)

    def _create_and_save_vocab_table_lookup_model_tf1(self, output_path: str, tags: Collection[str], signature_def_key: str) -> Tuple[Mapping[str, core.Tensor], Mapping[str, core.Tensor]]:
        if False:
            print('Hello World!')
        'Creates and saves a simple model that uses a vocab table.\n\n    Args:\n      output_path: Path to the directory to save the created model.\n      tags: Set of strings that identifies the saved meta graph.\n      signature_def_key: Name of the SignatureDef. Used to identify the\n        SignatureDef within the meta graph.\n\n    Returns:\n      inputs: A mapping of input_key -> input_tensor (placeholder). The input\n        key is "input_vocabs".\n      outputs: A mapping of output_key -> output_tensor. The output keys are\n        "lookup" and "output".\n    '
        with session.Session(graph=ops.Graph()) as sess:
            (input_vocabs_placeholder, lookup_tensor, output_tensor) = self._create_vocab_table_lookup_model_tf1(sess)
            inputs = {'input_vocabs': input_vocabs_placeholder}
            outputs = {'lookup': lookup_tensor, 'output': output_tensor}
            self._save_tf1_model(sess, output_path, signature_def_key, tags, inputs=inputs, outputs=outputs, init_op=lookup_ops.tables_initializer(), assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS))
        return (inputs, outputs)

    def _create_and_save_file_init_hash_table_model_tf1(self, output_path: str, tags: Collection[str], signature_def_key: str) -> Tuple[Mapping[str, core.Tensor], Mapping[str, core.Tensor]]:
        if False:
            print('Hello World!')
        'Creates and saves a model that uses a file-initialized table.\n\n    The asset file "vocab_file.txt" is used to initialize a hash table.\n\n    Args:\n      output_path: Path to the directory to save the created model.\n      tags: Set of strings that identifies the saved meta graph.\n      signature_def_key: Name of the SignatureDef. Used to identify the\n        SignatureDef within the meta graph.\n\n    Returns:\n      inputs: A mapping of input_key -> input_tensor (placeholder). The input\n        key is "input_vocabs".\n      outputs: A mapping of output_key -> output_tensor. The output keys are\n        "lookup" and "output".\n    '
        with session.Session(graph=ops.Graph()) as sess:
            (input_vocabs_placeholder, lookup_tensor, output_tensor) = self._create_table_init_from_file_model_tf1(sess)
            inputs = {'input_vocabs': input_vocabs_placeholder}
            outputs = {'lookup': lookup_tensor, 'output': output_tensor}
            self._save_tf1_model(sess, output_path, signature_def_key, tags, inputs=inputs, outputs=outputs, init_op=lookup_ops.tables_initializer(), assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS))
        return (inputs, outputs)

    def _create_table_init_from_file_model_tf1(self, sess: session.Session) -> Tuple[core.Tensor, core.Tensor, core.Tensor]:
        if False:
            print('Hello World!')
        'Creates a simple model that initializes a table from an asset file.\n\n    This model creates an asset file at "vocab_file.txt" containing\n    comma-separated vocabularies and uses it to initialize a\n    `StaticVocabularyTable`. For inference, the model performs a lookup with a\n    1D string tensor input vocabs.\n\n    Args:\n      sess: Tensorflow Session to create the model in.\n\n    Returns:\n      (input_vocabs_placeholder, lookup_vals, output_tensor), where\n      * input_vocabs_placeholder is a placeholder tensor of 1D strings\n      * lookup_vals is an output tensor that is a direct result of table lookup\n      * output_tensor is a float 2x2 matrix\n    '
        asset_dir = self.create_tempdir('assets').full_path
        asset_file = os.path.join(asset_dir, 'vocab_file.txt')
        content = '\n'.join(['static', 'range', 'quantization'])
        file_io.write_string_to_file(filename=asset_file, file_content=content)
        init = lookup_ops.TextFileInitializer(filename=asset_file, key_dtype=dtypes.string, key_index=lookup_ops.TextFileIndex.WHOLE_LINE, value_dtype=dtypes.int64, value_index=lookup_ops.TextFileIndex.LINE_NUMBER)
        table = lookup_ops.StaticHashTable(init, default_value=-1)
        input_vocabs_placeholder = array_ops.placeholder(dtypes.string, shape=(None,), name='input_vocabs')
        lookup_vals = math_ops.cast(table.lookup(input_vocabs_placeholder), dtypes.float32)
        matmul_input = array_ops_stack.stack([lookup_vals, lookup_vals])
        weight_row = array_ops.ones(shape=array_ops.shape(input_vocabs_placeholder), dtype=dtypes.float32)
        weight = array_ops.transpose_v2(array_ops_stack.stack([weight_row, weight_row]))
        output_tensor = math_ops.matmul(matmul_input, weight)
        return (input_vocabs_placeholder, lookup_vals, output_tensor)

    def _create_vocab_table_lookup_model_tf1(self, sess: session.Session) -> Tuple[core.Tensor, core.Tensor, core.Tensor]:
        if False:
            print('Hello World!')
        'Creates a simple model that initializes and lookups a vocab table.\n\n    This model creates an asset file at "vocab_file.txt" containing\n    comma-separated vocabularies.  It also initializes a `StaticVocabularyTable`\n    and performs a lookup with the input vocabs, which is a 1D tensor of\n    strings.\n\n    Args:\n      sess: Tensorflow Session to create the model in.\n\n    Returns:\n      (input_vocabs_placeholder, lookup_vals, output_tensor), where\n      * input_vocabs_placeholder is a placeholder tensor of 1D strings\n      * lookup_vals is an output tensor that is a direct result of table lookup\n      * output_tensor is a float 2x2 matrix\n    '
        asset_dir = self.create_tempdir('assets').full_path
        asset_file = os.path.join(asset_dir, 'vocab_file.txt')
        file_io.write_string_to_file(filename=asset_file, file_content='hello,model,quantization\n')
        vocab_file = asset.Asset(asset_file)
        raw_vocab = io_ops.read_file(vocab_file)
        vocabs = ragged_string_ops.string_split_v2(string_ops.string_strip(raw_vocab), sep=',')
        kv_init = lookup_ops.KeyValueTensorInitializer(keys=vocabs, values=np.array([0, 1, 2]), value_dtype=dtypes.int64)
        table = lookup_ops.StaticVocabularyTable(kv_init, num_oov_buckets=5)
        input_vocabs_placeholder = array_ops.placeholder(dtypes.string, shape=(None,), name='input_vocabs')
        lookup_vals = math_ops.cast(table.lookup(input_vocabs_placeholder), dtypes.float32)
        matmul_input = array_ops_stack.stack([lookup_vals, lookup_vals])
        weight_row = array_ops.ones(shape=array_ops.shape(input_vocabs_placeholder), dtype=dtypes.float32)
        weight = array_ops.transpose_v2(array_ops_stack.stack([weight_row, weight_row]))
        output_tensor = math_ops.matmul(matmul_input, weight)
        return (input_vocabs_placeholder, lookup_vals, output_tensor)

    def _create_and_save_vocab_table_lookup_qat_model_tf1(self, output_path: str, tags: Collection[str], signature_def_key: str) -> Tuple[Mapping[str, core.Tensor], Mapping[str, core.Tensor]]:
        if False:
            i = 10
            return i + 15
        'Creates and saves a simple QAT model that uses a vocab table.\n\n    Args:\n      output_path: Path to the directory to save the created model.\n      tags: Set of strings that identifies the saved meta graph.\n      signature_def_key: Name of the SignatureDef. Used to identify the\n        SignatureDef within the meta graph.\n\n    Returns:\n      inputs: A mapping of input_key -> input_tensor (placeholder). The input\n        key is "input_vocabs".\n      outputs: A mapping of output_key -> output_tensor. The output keys are\n        "lookup" and "output".\n    '
        with session.Session(graph=ops.Graph()) as sess:
            (input_vocabs_placeholder, lookup_tensor, output_tensor) = self._create_vocab_table_lookup_qat_model_tf1(sess)
            inputs = {'input_vocabs': input_vocabs_placeholder}
            outputs = {'lookup': lookup_tensor, 'output': output_tensor}
            self._save_tf1_model(sess, output_path, signature_def_key, tags, inputs=inputs, outputs=outputs, init_op=lookup_ops.tables_initializer(), assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS))
        return (inputs, outputs)

    def _create_vocab_table_lookup_qat_model_tf1(self, sess: session.Session) -> Tuple[core.Tensor, core.Tensor, core.Tensor]:
        if False:
            while True:
                i = 10
        'Creates a simple QAT model that initializes and lookups a vocab table.\n\n    This model creates an asset file at "vocab_file.txt" containing\n    comma-separated vocabularies.  It also initializes a `StaticVocabularyTable`\n    and performs a lookup with the input vocabs, which is a 1D tensor of\n    strings.\n\n    Args:\n      sess: Tensorflow Session to create the model in.\n\n    Returns:\n      (input_vocabs_placeholder, lookup_vals, output_tensor), where\n      * input_vocabs_placeholder is a placeholder tensor of 1D strings\n      * lookup_vals is an output tensor that is a direct result of table lookup\n      * output_tensor is a float 2x2 matrix\n    '
        asset_dir = self.create_tempdir('assets').full_path
        asset_file = os.path.join(asset_dir, 'vocab_file.txt')
        file_io.write_string_to_file(filename=asset_file, file_content='hello,model,quantization\n')
        vocab_file = asset.Asset(asset_file)
        raw_vocab = io_ops.read_file(vocab_file)
        vocabs = ragged_string_ops.string_split_v2(string_ops.string_strip(raw_vocab), sep=',')
        kv_init = lookup_ops.KeyValueTensorInitializer(keys=vocabs, values=np.array([0, 1, 2]), value_dtype=dtypes.int64)
        table = lookup_ops.StaticVocabularyTable(kv_init, num_oov_buckets=5)
        input_vocabs_placeholder = array_ops.placeholder(dtypes.string, shape=(None,), name='input_vocabs')
        lookup_vals = math_ops.cast(table.lookup(input_vocabs_placeholder), dtypes.float32)
        matmul_input = array_ops_stack.stack([lookup_vals, lookup_vals])
        matmul_input = array_ops.fake_quant_with_min_max_args(matmul_input, min=-0.3, max=0.3, num_bits=8, narrow_range=False)
        weight_row = array_ops.ones(shape=array_ops.shape(input_vocabs_placeholder), dtype=dtypes.float32)
        weight = array_ops.transpose_v2(array_ops_stack.stack([weight_row, weight_row]))
        weight = array_ops.fake_quant_with_min_max_args(weight, min=-0.1, max=0.2, num_bits=8, narrow_range=False)
        output_tensor = math_ops.matmul(matmul_input, weight)
        output_tensor = array_ops.fake_quant_with_min_max_args(output_tensor, min=-0.2, max=0.2, num_bits=8, narrow_range=False)
        return (input_vocabs_placeholder, lookup_vals, output_tensor)

    def _create_table_init_from_file_qat_model_tf1(self, sess: session.Session) -> Tuple[core.Tensor, core.Tensor, core.Tensor]:
        if False:
            print('Hello World!')
        'Creates a simple QAT model that initializes a table from an asset file.\n\n    This model creates an asset file at "vocab_file.txt" containing\n    comma-separated vocabularies and uses it to initialize a\n    `StaticVocabularyTable`. For inference, the model performs a lookup with a\n    1D string tensor input vocabs.\n\n    Args:\n      sess: Tensorflow Session to create the model in.\n\n    Returns:\n      (input_vocabs_placeholder, lookup_vals, output_tensor), where\n      * input_vocabs_placeholder is a placeholder tensor of 1D strings\n      * lookup_vals is an output tensor that is a direct result of table lookup\n      * output_tensor is a float 2x2 matrix\n    '
        asset_dir = self.create_tempdir('assets').full_path
        asset_file = os.path.join(asset_dir, 'vocab_file.txt')
        content = '\n'.join(['static', 'range', 'quantization'])
        file_io.write_string_to_file(filename=asset_file, file_content=content)
        init = lookup_ops.TextFileInitializer(filename=asset_file, key_dtype=dtypes.string, key_index=lookup_ops.TextFileIndex.WHOLE_LINE, value_dtype=dtypes.int64, value_index=lookup_ops.TextFileIndex.LINE_NUMBER)
        table = lookup_ops.StaticHashTable(init, default_value=-1)
        input_vocabs_placeholder = array_ops.placeholder(dtypes.string, shape=(None,), name='input_vocabs')
        lookup_vals = math_ops.cast(table.lookup(input_vocabs_placeholder), dtypes.float32)
        matmul_input = array_ops_stack.stack([lookup_vals, lookup_vals])
        matmul_input = array_ops.fake_quant_with_min_max_args(matmul_input, min=-0.3, max=0.3, num_bits=8, narrow_range=False)
        weight_row = array_ops.ones(shape=array_ops.shape(input_vocabs_placeholder), dtype=dtypes.float32)
        weight = array_ops.transpose_v2(array_ops_stack.stack([weight_row, weight_row]))
        weight = array_ops.fake_quant_with_min_max_args(weight, min=-0.1, max=0.2, num_bits=8, narrow_range=False)
        output_tensor = math_ops.matmul(matmul_input, weight)
        output_tensor = array_ops.fake_quant_with_min_max_args(output_tensor, min=-0.2, max=0.2, num_bits=8, narrow_range=False)
        return (input_vocabs_placeholder, lookup_vals, output_tensor)

    def _create_and_save_file_init_hash_table_qat_model_tf1(self, output_path: str, tags: Collection[str], signature_def_key: str) -> Tuple[Mapping[str, core.Tensor], Mapping[str, core.Tensor]]:
        if False:
            print('Hello World!')
        'Creates and saves a QAT model that uses a file-initialized table.\n\n    The asset file "vocab_file.txt" is used to initialize a hash table.\n\n    Args:\n      output_path: Path to the directory to save the created model.\n      tags: Set of strings that identifies the saved meta graph.\n      signature_def_key: Name of the SignatureDef. Used to identify the\n        SignatureDef within the meta graph.\n\n    Returns:\n      inputs: A mapping of input_key -> input_tensor (placeholder). The input\n        key is "input_vocabs".\n      outputs: A mapping of output_key -> output_tensor. The output keys are\n        "lookup" and "output".\n    '
        with session.Session(graph=ops.Graph()) as sess:
            (input_vocabs_placeholder, lookup_tensor, output_tensor) = self._create_table_init_from_file_qat_model_tf1(sess)
            inputs = {'input_vocabs': input_vocabs_placeholder}
            outputs = {'lookup': lookup_tensor, 'output': output_tensor}
            self._save_tf1_model(sess, output_path, signature_def_key, tags, inputs=inputs, outputs=outputs, init_op=lookup_ops.tables_initializer(), assets_collection=ops.get_collection(ops.GraphKeys.ASSET_FILEPATHS))
        return (inputs, outputs)

    def _create_data_generator(self, input_key: str, shape: Sequence[int], minval: float=-1.0, maxval: float=1.0, dtype: dtypes.DType=dtypes.float32, num_examples: int=8) -> repr_dataset.RepresentativeDataset:
        if False:
            while True:
                i = 10
        'Creates a data generator to be used as representative dataset.\n\n    Supports generating random value input tensors mapped by the `input_key`.\n\n    Args:\n      input_key: The string key that identifies the created tensor as an input.\n      shape: Shape of the tensor data.\n      minval: The lower bound of the generated input\n      maxval: The upper bound of the generated input\n      dtype: The type of the generated input - usually dtypes.float32 for float\n        and dtypes.int64 for int\n      num_examples: Number of examples in the representative dataset.\n\n    Yields:\n      data_gen: A `quantize_model._RepresentativeSample` filled with random\n        values.\n    '
        for _ in range(num_examples):
            yield {input_key: random_ops.random_uniform(shape, minval, maxval, dtype)}

    def _save_tf1_model(self, sess: session.Session, saved_model_path: str, signature_key: str, tags: Collection[str], inputs: Mapping[str, core.Tensor], outputs: Mapping[str, core.Tensor], init_op: Optional[ops.Operation]=None, assets_collection: Optional[Sequence[core.Symbol]]=None) -> None:
        if False:
            print('Hello World!')
        'Saves a TF1 model.\n\n    Args:\n      sess: Current tf.Session object.\n      saved_model_path: Directory to save the model.\n      signature_key: The key to the SignatureDef that inputs & outputs\n        correspond to.\n      tags: Set of tags associated with the model.\n      inputs: Input name -> input tensor mapping.\n      outputs: Output name -> output tensor mapping.\n      init_op: Op for initialization.\n      assets_collection: Assets collection. This collection is a list of string\n        tensors. Each tensor contains the asset file names.\n    '
        v1_builder = builder.SavedModelBuilder(saved_model_path)
        sig_def = signature_def_utils_impl.predict_signature_def(inputs=inputs, outputs=outputs)
        v1_builder.add_meta_graph_and_variables(sess, tags, signature_def_map={signature_key: sig_def}, main_op=init_op, assets_collection=assets_collection)
        v1_builder.save()

    def _create_simple_gather_and_conv_model(self, input_type: dtypes.DType, filter_shape: Sequence[int], is_qat_model: bool=False) -> module.Module:
        if False:
            while True:
                i = 10

        class SimpleGatherAndConvModel(module.Module):
            """A simple model with a single gather and a conv2d."""

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                'Initializes a SimpleGatherAndConvModel.'
                self.embedding_w = np.random.randn(1024, 3, 4, 3).astype('f4')
                self.conv_filters = np.random.uniform(low=-10, high=10, size=filter_shape).astype('f4')
                second_conv_filter_shape = (3, 3, filter_shape[-1], 1)
                self.second_conv_filters = np.random.uniform(low=-10, high=10, size=second_conv_filter_shape).astype('f4')

            @def_function.function(input_signature=[tensor_spec.TensorSpec(shape=[1], dtype=input_type, name='input_tensor')])
            def model(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
                if False:
                    i = 10
                    return i + 15
                'Performs a gather and a 2D convolution operation.\n\n        Args:\n          input_tensor: Input tensor to perform operation on.\n\n        Returns:\n          A map of: output key -> output result.\n        '
                out = array_ops.gather_v2(self.embedding_w, input_tensor)
                out = nn_ops.conv2d(out, self.conv_filters, strides=(1, 1, 2, 1), dilations=(1, 1, 1, 1), padding='SAME', data_format='NHWC')
                if is_qat_model:
                    out = array_ops.fake_quant_with_min_max_args(out, min=-0.1, max=0.2, num_bits=8, narrow_range=False)
                    second_conv_filters = array_ops.fake_quant_with_min_max_args(self.second_conv_filters, min=-0.1, max=0.2, num_bits=8, narrow_range=True)
                else:
                    second_conv_filters = self.second_conv_filters
                out = nn_ops.conv2d(out, second_conv_filters, strides=(1, 1, 2, 1), dilations=(1, 1, 1, 1), padding='SAME', data_format='NHWC')
                if is_qat_model:
                    out = array_ops.fake_quant_with_min_max_args(out, min=-0.1, max=0.2, num_bits=8, narrow_range=False)
                return {'output': out}
        return SimpleGatherAndConvModel()

    def _create_and_save_tf1_gather_model(self, saved_model_path: str, signature_key: str, tags: Collection[str], input_key: str, output_key: str, input_type: dtypes.DType, use_variable=False) -> core.Tensor:
        if False:
            i = 10
            return i + 15
        'Creates and saves a simple gather model.\n\n    This is intended to be used for TF1 (graph mode) tests.\n\n    Args:\n      saved_model_path: Directory to save the model.\n      signature_key: The key to the SignatureDef that inputs & outputs\n        correspond to.\n      tags: Set of tags associated with the model.\n      input_key: The key to the input tensor.\n      output_key: The key to the output tensor.\n      input_type: type of the input index tensor for gather operation.\n      use_variable: Setting this to `True` makes the filter for the gather\n        operation a `tf.Variable`.\n\n    Returns:\n      in_placeholder: The placeholder tensor used as an input to the model.\n    '
        with ops.Graph().as_default(), session.Session() as sess:
            (in_placeholder, output_tensor) = self._create_simple_tf1_gather_model(input_type=input_type, use_variable_for_filter=use_variable)
            if use_variable:
                sess.run(variables.global_variables_initializer())
            self._save_tf1_model(sess, saved_model_path, signature_key, tags, inputs={input_key: in_placeholder}, outputs={output_key: output_tensor})
            return in_placeholder

    def _create_gather_model(self, input_type, use_variable):
        if False:
            i = 10
            return i + 15

        class GatherModel(autotrackable.AutoTrackable):
            """A simple model with a single gather."""

            def __init__(self, use_variable):
                if False:
                    print('Hello World!')
                'Initializes a GatherModel.\n\n        Args:\n          use_variable: If True, creates a variable for weight.\n        '
                super(GatherModel, self).__init__()
                w_val = np.random.randn(128, 32).astype('f4')
                if use_variable:
                    self.w = variables.Variable(w_val)
                else:
                    self.w = w_val

            @def_function.function(input_signature=[tensor_spec.TensorSpec(shape=[6], dtype=input_type, name='input_tensor')])
            def __call__(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
                if False:
                    for i in range(10):
                        print('nop')
                'Performs a gather operation.'
                out = array_ops.gather_v2(self.w, input_tensor)
                return {'output': out}
        return GatherModel(use_variable)

    def _create_depthwise_conv2d_model(self, input_shape: Sequence[int], filter_shape: Sequence[int], has_bias: bool=False, has_batch_norm: bool=False, activation_fn: Optional[ops.Operation]=None, strides: Sequence[int]=(1, 2, 2, 1), dilations: Sequence[int]=(1, 1, 1, 1), padding: str='SAME'):
        if False:
            return 10

        class DepthwiseConvModel(module.Module):
            """A simple model with a single depthwise conv2d, bias and relu."""

            def __init__(self):
                if False:
                    print('Hello World!')
                self.out_channel_size = filter_shape[2] * filter_shape[3]
                self.filters = np.stack([np.random.uniform(low=-(i + 1), high=i + 1, size=filter_shape[:-2]).astype('f4') for i in range(self.out_channel_size)], axis=-1)
                self.filters = self.filters.reshape(filter_shape)
                self.bias = np.random.uniform(low=0, high=10, size=self.out_channel_size).astype('f4')

            @def_function.function(input_signature=[tensor_spec.TensorSpec(shape=input_shape, dtype=dtypes.float32)])
            def depthwise_conv(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
                if False:
                    while True:
                        i = 10
                'Performs a 2D depthwise convolution operation.\n\n        Args:\n          input_tensor: Input tensor to perform convolution on.\n\n        Returns:\n          A map of: output key -> output result.\n        '
                scale = [1.0] * self.out_channel_size
                offset = [0.5] * self.out_channel_size
                (mean, variance) = (scale, offset)
                out = nn_ops.depthwise_conv2d_native(input_tensor, self.filters, strides=[1, 2, 2, 1], dilations=[1, 1, 1, 1], padding='SAME', data_format='NHWC')
                if has_bias:
                    out = nn_ops.bias_add(out, self.bias)
                if has_batch_norm:
                    (out, _, _, _, _, _) = nn_ops.fused_batch_norm_v3(out, scale, offset, mean, variance, is_training=False)
                if activation_fn is not None:
                    out = activation_fn(out)
                return {'output': out}
        return DepthwiseConvModel()

    def _create_conv2d_model(self, input_shape: Sequence[int], filter_shape: Sequence[int], has_bias: bool=False, has_batch_norm: bool=False, activation_fn: Optional[ops.Operation]=None, strides: Sequence[int]=(1, 2, 2, 1), dilations: Sequence[int]=(1, 1, 1, 1), padding: str='SAME'):
        if False:
            while True:
                i = 10

        class ConvModel(module.Module):
            """A simple model with a single conv2d, bias and relu."""

            def __init__(self):
                if False:
                    print('Hello World!')
                self.out_channel_size = filter_shape[-1]
                self.filters = np.stack([np.random.uniform(low=-(i + 1), high=i + 1, size=filter_shape[:-1]).astype('f4') for i in range(self.out_channel_size)], axis=-1)
                self.bias = np.random.uniform(low=0, high=10, size=self.out_channel_size).astype('f4')

            @def_function.function(input_signature=[tensor_spec.TensorSpec(shape=input_shape, dtype=dtypes.float32)])
            def conv(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
                if False:
                    while True:
                        i = 10
                'Performs a 2D convolution operation.\n\n        Args:\n          input_tensor: Input tensor to perform convolution on.\n\n        Returns:\n          A map of: output key -> output result.\n        '
                scale = [1.0] * self.out_channel_size
                offset = [0.5] * self.out_channel_size
                (mean, variance) = (scale, offset)
                out = nn_ops.conv2d(input_tensor, self.filters, strides=strides, dilations=dilations, padding=padding, data_format='NHWC')
                if has_bias:
                    out = nn_ops.bias_add(out, self.bias, data_format='NHWC')
                if has_batch_norm:
                    (out, _, _, _, _, _) = nn_ops.fused_batch_norm_v3(out, scale, offset, mean, variance, is_training=False)
                if activation_fn is not None:
                    out = activation_fn(out)
                return {'output': out}
        return ConvModel()

    def _create_matmul_model(self, input_shape: Sequence[int], weight_shape: Sequence[int], saved_model_path: str, has_bias: bool=False, activation_fn: Optional[ops.Operation]=None, bias_size: Optional[int]=None, use_biasadd: bool=True) -> module.Module:
        if False:
            return 10

        class MatmulModel(module.Module):
            """A simple model with a single matmul.

      Bias and activation function are optional.
      """

            def __init__(self, weight_shape: Sequence[int], bias_size: Optional[int]=None, activation_fn: Optional[ops.Operation]=None, use_biasadd: bool=True) -> None:
                if False:
                    print('Hello World!')
                'Initializes a MatmulModel.\n\n        Args:\n          weight_shape: Shape of the weight tensor.\n          bias_size: If None, do not use bias. Else, use given size as bias.\n          activation_fn: The activation function to be used. No activation\n            function if None.\n          use_biasadd: If True, use BiasAdd for adding bias, else use AddV2.\n        '
                self.bias_size = bias_size
                self.activation_fn = activation_fn
                self.use_biasadd = use_biasadd
                self.filters = np.random.uniform(low=-1.0, high=1.0, size=weight_shape)
                if bias_size is not None:
                    self.bias = np.random.uniform(low=-1.0, high=1.0, size=bias_size)

            def has_bias(self) -> bool:
                if False:
                    while True:
                        i = 10
                return self.bias_size is not None

            def has_reshape(self) -> bool:
                if False:
                    i = 10
                    return i + 15
                return self.has_bias() and self.bias_size != self.filters.shape[-1]

            @def_function.function
            def matmul(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
                if False:
                    i = 10
                    return i + 15
                'Performs a matrix multiplication.\n\n        Depending on self.has_bias and self.activation_fn, it may add a bias\n        term or\n        go through the activaction function.\n\n        Args:\n          input_tensor: Input tensor to matmul with the filter.\n\n        Returns:\n          A map of: output key -> output result.\n        '
                out = math_ops.matmul(input_tensor, self.filters, name='sample/matmul')
                if self.has_reshape():
                    input_shape = input_tensor.shape
                    if len(input_shape) == 3:
                        reshape_shape = (input_shape[0], -1, self.bias_size)
                    else:
                        reshape_shape = (-1, self.bias_size)
                    out = array_ops.reshape(out, reshape_shape)
                if self.has_bias():
                    if self.use_biasadd:
                        out = nn_ops.bias_add(out, self.bias)
                    else:
                        out = math_ops.add_v2(out, self.bias)
                if self.activation_fn is not None:
                    out = self.activation_fn(out)
                return {'output': out}
        if bias_size is None and has_bias:
            bias_size = weight_shape[-1]
        assert not (bias_size is not None) ^ has_bias
        if bias_size:
            input_height = input_shape[0] if len(input_shape) == 2 else input_shape[1]
            assert input_height * weight_shape[-1] % bias_size == 0
        model = MatmulModel(weight_shape, bias_size, activation_fn)
        saved_model_save.save(model, saved_model_path, signatures=model.matmul.get_concrete_function(tensor_spec.TensorSpec(shape=input_shape, dtype=dtypes.float32, name='input_tensor')))
        return model

    def _prepare_sample_einsum_datashapes(self, equation: str, generate_unknown_shape_signature: bool=False, use_bias: bool=False) -> Tuple[List[Optional[int]], List[Optional[int]], Optional[List[Optional[int]]], List[Optional[int]], List[Optional[int]]]:
        if False:
            while True:
                i = 10
        comma_pos = equation.find(',')
        arrow_pos = equation.find('->')
        x_labels = equation[0:comma_pos]
        y_labels = equation[comma_pos + 1:arrow_pos]
        out_labels = equation[arrow_pos + 1:]
        label_to_size = {'a': 4, 'b': 32, 'c': 64, 'd': 128, 'e': 8}
        x_shape = [label_to_size.get(x_label) for x_label in x_labels]
        y_shape = [label_to_size.get(y_label) for y_label in y_labels]
        bias_shape = None
        if use_bias:
            bias_shape = [label_to_size.get(out_label) for out_label in out_labels]
            bias_shape = bias_shape[-1:]
        contracting_dims = set()
        x_signature = list(x_shape)
        y_signature = list(y_shape)
        if generate_unknown_shape_signature:
            for c in x_labels:
                if c in y_labels:
                    contracting_dims.add(c)
            x_signature = [None if c not in contracting_dims else x_shape[cidx] for (cidx, c) in enumerate(x_labels)]
            y_signature = [None if c not in contracting_dims else y_shape[cidx] for (cidx, c) in enumerate(y_labels)]
        return (x_shape, y_shape, bias_shape, x_signature, y_signature)

    def _create_einsum_model(self, equation: str, y_shape: Sequence[int], x_signature: Sequence[Optional[int]], y_signature: Sequence[Optional[int]], bias_shape: Optional[Sequence[int]]=None, activation_fn: Optional[ops.Operation]=None, is_qat_model: bool=False) -> module.Module:
        if False:
            for i in range(10):
                print('nop')

        class EinsumModel(module.Module):
            """Einsum class."""

            def __init__(self):
                if False:
                    print('Hello World!')
                self._bias = None
                if bias_shape is not None:
                    self._bias = array_ops.constant(np.random.uniform(size=bias_shape), dtype=dtypes.float32)
                self._kernel = np.random.uniform(size=y_shape).astype('f4')
                self._min = (-0.8, -0.8, -0.9)
                self._max = (0.9, 0.9, 1.0)

            @def_function.function(input_signature=[tensor_spec.TensorSpec(name='x', shape=x_signature, dtype=dtypes.float32)])
            def einsum_with_kernel(self, x: core.Tensor) -> Mapping[str, core.Tensor]:
                if False:
                    i = 10
                    return i + 15
                return self._einsum(x, self._kernel)

            @def_function.function(input_signature=[tensor_spec.TensorSpec(name='x', shape=x_signature, dtype=dtypes.float32), tensor_spec.TensorSpec(name='y', shape=y_signature, dtype=dtypes.float32)])
            def einsum_without_kernel(self, x: core.Tensor, y: core.Tensor) -> Mapping[str, core.Tensor]:
                if False:
                    for i in range(10):
                        print('nop')
                return self._einsum(x, y)

            def _einsum(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                if is_qat_model:
                    x = array_ops.fake_quant_with_min_max_vars(x, min=ops.convert_to_tensor(self._min[0]), max=ops.convert_to_tensor(self._max[0]), num_bits=8, narrow_range=False)
                    y = array_ops.fake_quant_with_min_max_vars(y, min=ops.convert_to_tensor(self._min[1]), max=ops.convert_to_tensor(self._max[1]), num_bits=8, narrow_range=False)
                out = tensorflow.einsum(equation, x, y)
                if self._bias is not None:
                    out = nn_ops.bias_add(out, self._bias)
                if activation_fn is not None:
                    out = activation_fn(out)
                if is_qat_model:
                    out = array_ops.fake_quant_with_min_max_vars(out, min=ops.convert_to_tensor(self._min[2]), max=ops.convert_to_tensor(self._max[2]), num_bits=8, narrow_range=False)
                return {'output': out}
        return EinsumModel()

    def _create_and_save_tf1_conv_model(self, saved_model_path: str, signature_key: str, tags: Collection[str], input_key: str, output_key: str, *, input_shape: Sequence[int]=(1, 3, 4, 3), filter_shape: Sequence[int]=(2, 3, 3, 2), use_variable: bool=False) -> core.Tensor:
        if False:
            while True:
                i = 10
        'Creates and saves a simple convolution model.\n\n    This is intended to be used for TF1 (graph mode) tests.\n\n    Args:\n      saved_model_path: Directory to save the model.\n      signature_key: The key to the SignatureDef that inputs & outputs\n        correspond to.\n      tags: Set of tags associated with the model.\n      input_key: The key to the input tensor.\n      output_key: The key to the output tensor.\n      input_shape: Shape of the input tensor.\n      filter_shape: Shape of the filter.\n      use_variable: Setting this to `True` makes the filter for the conv\n        operation a `tf.Variable`.\n\n    Returns:\n      in_placeholder: The placeholder tensor used as an input to the model.\n    '
        with ops.Graph().as_default(), session.Session() as sess:
            (in_placeholder, output_tensor) = self._create_simple_tf1_conv_model(input_shape=input_shape, filter_shape=filter_shape, use_variable_for_filter=use_variable)
            if use_variable:
                sess.run(variables.global_variables_initializer())
            self._save_tf1_model(sess, saved_model_path, signature_key, tags, inputs={input_key: in_placeholder}, outputs={output_key: output_tensor})
        return in_placeholder

    def _create_while_model(self, input_shape: Sequence[int]=(1, 32, 32, 512)):
        if False:
            return 10

        class WhileModel(module.Module):
            """A model with a while op."""

            def __init__(self):
                if False:
                    while True:
                        i = 10
                w_shape = [3, 3] + [input_shape[-1], input_shape[-1]]
                self.w = np.random.uniform(low=-2, high=2, size=w_shape).astype('f4')

            @def_function.function
            def condition(self, x, w):
                if False:
                    for i in range(10):
                        print('nop')
                return math_ops.reduce_sum(x, keepdims=False) < 100

            @def_function.function
            def body(self, x, w):
                if False:
                    while True:
                        i = 10
                z = nn_ops.conv2d(x, w, padding='SAME')
                return (z, w)

            @def_function.function(input_signature=[tensor_spec.TensorSpec(shape=input_shape, dtype=dtypes.float32, name='input_tensor')])
            def main(self, x):
                if False:
                    return 10
                x1 = nn_ops.conv2d(x, self.w, padding='SAME')
                (x2, _) = while_loop_ops.while_loop(self.condition, self.body, [x, self.w])
                result = x1 + x2
                return {'output': result}
        return WhileModel()