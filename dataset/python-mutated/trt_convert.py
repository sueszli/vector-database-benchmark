"""Exposes the Python wrapper conversion to trt_graph."""
import collections
from functools import partial
import os
import platform
import sys
import tempfile
import numpy as np
import six as _six
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.compiler.tensorrt import utils as trt_utils
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import resource
from tensorflow.python.training import saver
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export
gen_trt_ops = LazyLoader('gen_trt_ops', globals(), 'tensorflow.compiler.tf2tensorrt.ops.gen_trt_ops')
_pywrap_py_utils = LazyLoader('_pywrap_py_utils', globals(), 'tensorflow.compiler.tf2tensorrt._pywrap_py_utils')
try:
    gen_trt_ops.trt_engine_op
except AttributeError:
    pass

def _to_bytes(s):
    if False:
        return 10
    'Encode s if it is a sequence of chars.'
    if isinstance(s, _six.text_type):
        return s.encode('utf-8', errors='surrogateescape')
    return s

def _to_string(s):
    if False:
        while True:
            i = 10
    'Decode s if it is a sequence of bytes.'
    if isinstance(s, _six.binary_type):
        return s.decode('utf-8')
    return s

class TrtPrecisionMode(object):
    FP32 = 'FP32'
    FP16 = 'FP16'
    INT8 = 'INT8'

    @staticmethod
    def supported_precision_modes():
        if False:
            while True:
                i = 10
        precisions = [TrtPrecisionMode.FP32, TrtPrecisionMode.FP16, TrtPrecisionMode.INT8]
        return precisions + [p.lower() for p in precisions]
if _pywrap_py_utils.is_tensorrt_enabled() and trt_utils.is_loaded_tensorrt_version_greater_equal(8, 4, 0):
    DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES = sys.maxsize - 512
else:
    DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES = 1 << 30
PROFILE_STRATEGY_RANGE = 'Range'
PROFILE_STRATEGY_OPTIMAL = 'Optimal'
PROFILE_STRATEGY_RANGE_OPTIMAL = 'Range+Optimal'
PROFILE_STRATEGY_IMPLICIT_BATCH_MODE_COMPATIBLE = 'ImplicitBatchModeCompatible'

def supported_profile_strategies():
    if False:
        print('Hello World!')
    return [PROFILE_STRATEGY_RANGE, PROFILE_STRATEGY_OPTIMAL, PROFILE_STRATEGY_RANGE_OPTIMAL, PROFILE_STRATEGY_IMPLICIT_BATCH_MODE_COMPATIBLE]

@tf_export('experimental.tensorrt.ConversionParams', v1=[])
class TrtConversionParams(collections.namedtuple('TrtConversionParams', ['max_workspace_size_bytes', 'precision_mode', 'minimum_segment_size', 'maximum_cached_engines', 'use_calibration', 'allow_build_at_runtime'])):
    """Parameters that are used for TF-TRT conversion.

  Fields:
    max_workspace_size_bytes: the maximum GPU temporary memory that the TRT
      engine can use at execution time. This corresponds to the
      'workspaceSize' parameter of nvinfer1::IBuilder::setMaxWorkspaceSize().
    precision_mode: one of the strings in
      TrtPrecisionMode.supported_precision_modes().
    minimum_segment_size: the minimum number of nodes required for a subgraph
      to be replaced by TRTEngineOp.
    maximum_cached_engines: max number of cached TRT engines for dynamic TRT
      ops. Created TRT engines for a dynamic dimension are cached. If the
      number of cached engines is already at max but none of them supports the
      input shapes, the TRTEngineOp will fall back to run the original TF
      subgraph that corresponds to the TRTEngineOp.
    use_calibration: this argument is ignored if precision_mode is not INT8.
      If set to True, a calibration graph will be created to calibrate the
      missing ranges. The calibration graph must be converted to an inference
      graph by running calibration with calibrate(). If set to False,
      quantization nodes will be expected for every tensor in the graph
      (excluding those which will be fused). If a range is missing, an error
      will occur. Please note that accuracy may be negatively affected if
      there is a mismatch between which tensors TRT quantizes and which
      tensors were trained with fake quantization.
    allow_build_at_runtime: whether to allow building TensorRT engines during
      runtime if no prebuilt TensorRT engine can be found that can handle the
      given inputs during runtime, then a new TensorRT engine is built at
      runtime if allow_build_at_runtime=True, and otherwise native TF is used.
  """

    def __new__(cls, max_workspace_size_bytes=DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES, precision_mode=TrtPrecisionMode.FP32, minimum_segment_size=3, maximum_cached_engines=1, use_calibration=True, allow_build_at_runtime=True):
        if False:
            i = 10
            return i + 15
        return super(TrtConversionParams, cls).__new__(cls, max_workspace_size_bytes, precision_mode, minimum_segment_size, maximum_cached_engines, use_calibration, allow_build_at_runtime)
DEFAULT_TRT_CONVERSION_PARAMS = TrtConversionParams()
_TRT_ENGINE_OP_NAME = 'TRTEngineOp'

def _check_conversion_params(conversion_params, is_v2=False):
    if False:
        print('Hello World!')
    "Validate the provided TrtConversionParams.\n\n  Args:\n    conversion_params: a TrtConversionParams instance.\n    is_v2: whether we're getting a RewriterConfig for TF 2.0.\n\n  Raises:\n    TypeError: if any of the parameters are of unexpected type.\n    ValueError: if any of the parameters are of unexpected value.\n  "
    supported_precision_modes = TrtPrecisionMode.supported_precision_modes()
    if conversion_params.precision_mode not in supported_precision_modes:
        raise ValueError("precision mode '{}' is not supported.It should be one of {}".format(conversion_params.precision_mode, supported_precision_modes))
    if conversion_params.minimum_segment_size <= 0 and conversion_params.minimum_segment_size != -1:
        raise ValueError('minimum segment size should be positive or -1 (to disable main graph conversion).')

def _check_trt_version_compatibility():
    if False:
        print('Hello World!')
    'Check compatibility of TensorRT version.\n\n  Raises:\n    RuntimeError: if the TensorRT library version is incompatible.\n  '
    if not _pywrap_py_utils.is_tensorrt_enabled():
        logging.error('Tensorflow needs to be built with TensorRT support enabled to allow TF-TRT to operate.')
        raise RuntimeError('Tensorflow has not been built with TensorRT support.')
    if platform.system() == 'Windows':
        logging.warn('Windows support is provided experimentally. No guarantee is made regarding functionality or engineering support. Use at your own risk.')
    linked_version = _pywrap_py_utils.get_linked_tensorrt_version()
    loaded_version = _pywrap_py_utils.get_loaded_tensorrt_version()
    logging.info('Linked TensorRT version: %s', str(linked_version))
    logging.info('Loaded TensorRT version: %s', str(loaded_version))

    def raise_trt_version_deprecated(version_type, trt_version):
        if False:
            print('Hello World!')
        assert version_type in ['linked', 'loaded'], "Incorrect value received for version_type: %s. Accepted: ['linked', 'loaded']" % version_type
        logging.error('The {version_type} version of TensorRT: `{trt_version}` has now been removed. Please upgrade to TensorRT 7 or more recent.'.format(version_type=version_type, trt_version=trt_utils.version_tuple_to_string(trt_version)))
        raise RuntimeError('Incompatible %s TensorRT versions' % version_type)
    if not trt_utils.is_linked_tensorrt_version_greater_equal(7, 0, 0):
        raise_trt_version_deprecated('linked', linked_version)
    if not trt_utils.is_loaded_tensorrt_version_greater_equal(7, 0, 0):
        raise_trt_version_deprecated('loaded', loaded_version)
    if loaded_version[0] != linked_version[0] or not trt_utils.is_loaded_tensorrt_version_greater_equal(*linked_version):
        logging.error('Loaded TensorRT %s but linked TensorFlow against TensorRT %s. A few requirements must be met:\n\t-It is required to use the same major version of TensorRT during compilation and runtime.\n\t-TensorRT does not support forward compatibility. The loaded version has to be equal or more recent than the linked version.', trt_utils.version_tuple_to_string(loaded_version), trt_utils.version_tuple_to_string(linked_version))
        raise RuntimeError('Incompatible TensorRT major version')
    elif loaded_version != linked_version:
        logging.info('Loaded TensorRT %s and linked TensorFlow against TensorRT %s. This is supported because TensorRT minor/patch upgrades are backward compatible.', trt_utils.version_tuple_to_string(loaded_version), trt_utils.version_tuple_to_string(linked_version))

def _get_tensorrt_rewriter_config(conversion_params, is_dynamic_op=None, max_batch_size=None, is_v2=False, disable_non_trt_optimizers=False, use_implicit_batch=True, profile_strategy=PROFILE_STRATEGY_RANGE):
    if False:
        return 10
    "Returns a RewriterConfig proto for TRT transformation.\n\n  Args:\n    conversion_params: a TrtConversionParams instance.\n    is_dynamic_op: whether to use dynamic engines.\n    max_batch_size: maximum batch size for static engines.\n    is_v2: whether we're getting a RewriterConfig for TF 2.0.\n    disable_non_trt_optimizers: Turn off all default Grappler optimizers.\n    use_implicit_batch: Whether to use implicit batch or explicit batch.\n    profile_strategy: dynamic shape optimization profile strategy.\n\n  Returns:\n    A RewriterConfig proto which sets a TensorRTOptimizer to run Grappler.\n\n  Raises:\n    TypeError: if any of the parameters are of unexpected type.\n    ValueError: if any of the parameters are of unexpected value.\n  "
    _check_conversion_params(conversion_params, is_v2=is_v2)
    if is_v2 and is_dynamic_op is not None and (not is_dynamic_op):
        raise ValueError('is_dynamic_op is either None or True for TF2')
    if not is_v2 and is_dynamic_op is None:
        raise ValueError("is_dynamic_op can't be None for TF1")
    if (is_dynamic_op is None or is_dynamic_op) and max_batch_size is not None:
        raise ValueError('max_batch_size has to be None for TF2 or when is_dynamic_op == True in TF1')
    if is_dynamic_op is not None and (not is_dynamic_op) and (not isinstance(max_batch_size, int)):
        raise ValueError('max_batch_size has to be an integer for is_dynamic_op==False in TF1')
    rewriter_config_with_trt = rewriter_config_pb2.RewriterConfig()
    rewriter_config_with_trt.remapping = False
    rewriter_config_with_trt.experimental_disable_folding_quantization_emulation = trt_utils.is_linked_tensorrt_version_greater_equal(8, 0, 0) or trt_utils.is_loaded_tensorrt_version_greater_equal(8, 0, 0)
    if not disable_non_trt_optimizers:
        rewriter_config_with_trt.optimizers.extend(['pruning', 'debug_stripper', 'layout', 'dependency', 'constfold', 'common_subgraph_elimination'])
    rewriter_config_with_trt.meta_optimizer_iterations = rewriter_config_pb2.RewriterConfig.ONE
    optimizer = rewriter_config_with_trt.custom_optimizers.add()
    if not disable_non_trt_optimizers:
        rewriter_config_with_trt.custom_optimizers.add().name = 'constfold'
    optimizer.name = 'TensorRTOptimizer'
    optimizer.parameter_map['minimum_segment_size'].i = conversion_params.minimum_segment_size
    optimizer.parameter_map['max_workspace_size_bytes'].i = conversion_params.max_workspace_size_bytes
    optimizer.parameter_map['precision_mode'].s = _to_bytes(conversion_params.precision_mode)
    optimizer.parameter_map['maximum_cached_engines'].i = conversion_params.maximum_cached_engines
    optimizer.parameter_map['use_calibration'].b = conversion_params.use_calibration
    optimizer.parameter_map['is_dynamic_op'].b = is_dynamic_op
    optimizer.parameter_map['allow_build_at_runtime'].b = conversion_params.allow_build_at_runtime
    if max_batch_size is not None:
        optimizer.parameter_map['max_batch_size'].i = max_batch_size
    optimizer.parameter_map['use_implicit_batch'].b = use_implicit_batch
    if not use_implicit_batch:
        optimizer.parameter_map['profile_strategy'].s = _to_bytes(profile_strategy.lower())
    if disable_non_trt_optimizers:
        trt_utils.disable_non_trt_optimizers_in_rewriter_config(rewriter_config_with_trt)
    return rewriter_config_with_trt

@deprecation.deprecated(None, "You shouldn't need a rewriter_config with the current TF-TRT APIs.")
def get_tensorrt_rewriter_config(conversion_params, is_dynamic_op=None, max_batch_size=None, is_v2=False, disable_non_trt_optimizers=False):
    if False:
        print('Hello World!')
    return _get_tensorrt_rewriter_config(conversion_params, is_dynamic_op, max_batch_size, is_v2, disable_non_trt_optimizers)

def _get_canonical_engine_name(name):
    if False:
        while True:
            i = 10
    return name.split('/')[-1]

class TrtGraphConverter(object):
    """A converter for TF-TRT transformation for TF 1.x GraphDef/SavedModels.

  To run the conversion without quantization calibration (e.g. for FP32/FP16
  precision modes):

  ```python
  converter = TrtGraphConverter(
      input_saved_model_dir="my_dir",
      precision_mode=TrtPrecisionMode.FP16)
  converted_graph_def = converter.convert()
  converter.save(output_saved_model_dir)
  ```

  To run the conversion with quantization calibration:

  ```python
  converter = TrtGraphConverter(
      input_saved_model_dir="my_dir",
      precision_mode=TrtPrecisionMode.INT8)
  converter.convert()

  # Run calibration 10 times.
  converted_graph_def = converter.calibrate(
      fetch_names=['output:0'],
      num_runs=10,
      feed_dict_fn=lambda: {'input:0': my_next_data()})

  converter.save(output_saved_model_dir)
  ```
  """

    def __init__(self, input_saved_model_dir=None, input_saved_model_tags=None, input_saved_model_signature_key=None, input_graph_def=None, nodes_denylist=None, max_batch_size=1, max_workspace_size_bytes=DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES, precision_mode=TrtPrecisionMode.FP32, minimum_segment_size=3, is_dynamic_op=False, maximum_cached_engines=1, use_calibration=True):
        if False:
            while True:
                i = 10
        "Initializes the converter.\n\n    Args:\n      input_saved_model_dir: the directory to load the SavedModel which contains\n        the input graph to transforms. Used only when input_graph_def is None.\n      input_saved_model_tags: list of tags to load the SavedModel.\n      input_saved_model_signature_key: the key of the signature to optimize the\n        graph for.\n      input_graph_def: a GraphDef object containing a model to be transformed.\n        If set to None, the graph will be read from the SavedModel loaded from\n        input_saved_model_dir.\n      nodes_denylist: list of node names to prevent the converter from touching.\n      max_batch_size: max size for the input batch.\n      max_workspace_size_bytes: the maximum GPU temporary memory which the TRT\n        engine can use at execution time. This corresponds to the\n        'workspaceSize' parameter of nvinfer1::IBuilder::setMaxWorkspaceSize().\n      precision_mode: one of TrtPrecisionMode.supported_precision_modes().\n      minimum_segment_size: the minimum number of nodes required for a subgraph\n        to be replaced by TRTEngineOp.\n      is_dynamic_op: whether to generate dynamic TRT ops which will build the\n        TRT network and engine at run time.\n      maximum_cached_engines: max number of cached TRT engines in dynamic TRT\n        ops. If the number of cached engines is already at max but none of them\n        can serve the input, the TRTEngineOp will fall back to run the TF\n        function based on which the TRTEngineOp is created.\n      use_calibration: this argument is ignored if precision_mode is not INT8.\n        If set to True, a calibration graph will be created to calibrate the\n        missing ranges. The calibration graph must be converted to an inference\n        graph by running calibration with calibrate(). If set to False,\n        quantization nodes will be expected for every tensor in the graph\n        (excluding those which will be fused). If a range is missing, an error\n        will occur. Please note that accuracy may be negatively affected if\n        there is a mismatch between which tensors TRT quantizes and which\n        tensors were trained with fake quantization.\n\n    Raises:\n      ValueError: if the combination of the parameters is invalid.\n      RuntimeError: if this class is used in TF 2.0.\n    "
        if context.executing_eagerly():
            raise RuntimeError('Please use tf.experimental.tensorrt.Converter in TF 2.0.')
        if input_graph_def and input_saved_model_dir:
            raise ValueError('Can only specify one of input_graph_def and input_saved_model_dir')
        if not input_graph_def and (not input_saved_model_dir):
            raise ValueError('Must specify one of input_graph_def and input_saved_model_dir')
        _check_trt_version_compatibility()
        self._input_graph_def = input_graph_def
        self._nodes_denylist = nodes_denylist
        self._input_saved_model_dir = input_saved_model_dir
        self._converted = False
        self._grappler_meta_graph_def = None
        self._input_saved_model_tags = input_saved_model_tags or [tag_constants.SERVING]
        self._input_saved_model_signature_key = input_saved_model_signature_key or signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        self._calibration_graph = None
        self._calibration_data_collected = False
        self._need_calibration = (precision_mode == TrtPrecisionMode.INT8 or precision_mode == TrtPrecisionMode.INT8.lower()) and use_calibration
        if self._need_calibration and (not is_dynamic_op):
            logging.warn('INT8 precision mode with calibration is supported with dynamic TRT ops only. Disregarding is_dynamic_op parameter.')
            is_dynamic_op = True
        self._is_dynamic_op = is_dynamic_op
        if is_dynamic_op:
            self._max_batch_size = None
            if max_batch_size is not None:
                logging.warn('When is_dynamic_op==True max_batch_size should be None')
        else:
            if not isinstance(max_batch_size, int):
                raise ValueError('When is_dynamic_op==False max_batch_size should be an integer')
            self._max_batch_size = max_batch_size
        self._conversion_params = TrtConversionParams(max_workspace_size_bytes=max_workspace_size_bytes, precision_mode=precision_mode, minimum_segment_size=minimum_segment_size, maximum_cached_engines=maximum_cached_engines, use_calibration=use_calibration, allow_build_at_runtime=True)
        _check_conversion_params(self._conversion_params)
        self._test_only_disable_non_trt_optimizers = False

    def _run_conversion(self):
        if False:
            while True:
                i = 10
        "Run Grappler's OptimizeGraph() tool to convert the graph."
        grappler_session_config = config_pb2.ConfigProto()
        custom_rewriter_config = _get_tensorrt_rewriter_config(conversion_params=self._conversion_params, is_dynamic_op=self._is_dynamic_op, max_batch_size=self._max_batch_size, disable_non_trt_optimizers=self._test_only_disable_non_trt_optimizers, use_implicit_batch=True)
        grappler_session_config.graph_options.rewrite_options.CopyFrom(custom_rewriter_config)
        self._converted_graph_def = tf_optimizer.OptimizeGraph(grappler_session_config, self._grappler_meta_graph_def, graph_id=b'tf_graph')
        self._converted = True

    def _add_nodes_denylist(self):
        if False:
            print('Hello World!')
        if self._nodes_denylist:
            collection_def = self._grappler_meta_graph_def.collection_def['train_op']
            denylist = collection_def.node_list.value
            for i in self._nodes_denylist:
                if isinstance(i, tensor.Tensor):
                    denylist.append(_to_bytes(i.name))
                else:
                    denylist.append(_to_bytes(i))

    def _convert_graph_def(self):
        if False:
            for i in range(10):
                print('nop')
        'Convert the input GraphDef.'
        graph = ops.Graph()
        with graph.as_default():
            importer.import_graph_def(self._input_graph_def, name='')
        self._grappler_meta_graph_def = saver.export_meta_graph(graph_def=graph.as_graph_def(add_shapes=True), graph=graph)
        self._add_nodes_denylist()
        self._run_conversion()

    def _collections_to_keep(self, collection_keys):
        if False:
            i = 10
            return i + 15
        collections_to_remove = ops.GraphKeys._VARIABLE_COLLECTIONS + [ops.GraphKeys.TRAIN_OP, ops.GraphKeys.WHILE_CONTEXT, ops.GraphKeys.COND_CONTEXT]
        return [key for key in collection_keys if key not in collections_to_remove]

    def _convert_saved_model(self):
        if False:
            while True:
                i = 10
        'Convert the input SavedModel.'
        graph = ops.Graph()
        with session.Session(graph=graph) as sess:
            input_meta_graph_def = loader.load(sess, self._input_saved_model_tags, self._input_saved_model_dir)
            input_signature_def = input_meta_graph_def.signature_def[self._input_saved_model_signature_key]

            def _gather_names(tensor_info):
                if False:
                    print('Hello World!')
                'Get the node names from a TensorInfo.'
                return {tensor_info[key].name.split(':')[0] for key in tensor_info}
            output_node_names = _gather_names(input_signature_def.inputs).union(_gather_names(input_signature_def.outputs))
            for collection_key in self._collections_to_keep(input_meta_graph_def.collection_def):
                for op in sess.graph.get_collection(collection_key):
                    if isinstance(op, ops.Operation):
                        output_node_names.add(op.name.split(':')[0])
            frozen_graph_def = convert_to_constants.convert_variables_to_constants(sess, sess.graph.as_graph_def(add_shapes=True), list(output_node_names))
            self._grappler_meta_graph_def = meta_graph_pb2.MetaGraphDef()
            self._grappler_meta_graph_def.graph_def.CopyFrom(frozen_graph_def)
            for collection_key in self._collections_to_keep(input_meta_graph_def.collection_def):
                self._grappler_meta_graph_def.collection_def[collection_key].CopyFrom(input_meta_graph_def.collection_def[collection_key])
            self._add_nodes_denylist()
            self._grappler_meta_graph_def.meta_info_def.CopyFrom(input_meta_graph_def.meta_info_def)
            self._grappler_meta_graph_def.signature_def[self._input_saved_model_signature_key].CopyFrom(input_signature_def)
        self._run_conversion()

    def convert(self):
        if False:
            while True:
                i = 10
        'Run the TF-TRT conversion.\n\n    Returns:\n      The converted GraphDef for TF 1.x.\n    '
        assert not self._converted
        if self._input_graph_def:
            self._convert_graph_def()
        else:
            self._convert_saved_model()
        return self._converted_graph_def

    def calibrate(self, fetch_names, num_runs, feed_dict_fn=None, input_map_fn=None):
        if False:
            i = 10
            return i + 15
        'Run the calibration and return the calibrated GraphDef.\n\n    Args:\n      fetch_names: a list of output tensor name to fetch during calibration.\n      num_runs: number of runs of the graph during calibration.\n      feed_dict_fn: a function that returns a dictionary mapping input names (as\n        strings) in the GraphDef to be calibrated to values (e.g. Python list,\n        numpy arrays, etc). One and only one of `feed_dict_fn` and\n        `input_map_fn` should be specified.\n      input_map_fn: a function that returns a dictionary mapping input names (as\n        strings) in the GraphDef to be calibrated to Tensor objects. The values\n        of the named input tensors in the GraphDef to be calibrated will be\n        re-mapped to the respective `Tensor` values during calibration. One and\n        only one of `feed_dict_fn` and `input_map_fn` should be specified.\n\n    Raises:\n      ValueError: if the input combination is invalid.\n      RuntimeError: if this method is called in eager mode.\n\n    Returns:\n      The GraphDef after the calibration.\n    '
        assert self._converted
        assert self._need_calibration
        assert not self._calibration_data_collected
        if feed_dict_fn and input_map_fn or (not feed_dict_fn and (not input_map_fn)):
            raise ValueError('Should specify one and only one of feed_dict_fn and input_map_fn.')
        if input_map_fn:
            for (k, v) in input_map_fn().items():
                if not isinstance(k, str):
                    raise ValueError('Keys of input_map_fn must be of type str')
                if not isinstance(v, tensor.Tensor):
                    raise ValueError('Values of input_map_fn must be of type tf.Tensor')
        self._calibration_graph = ops.Graph()
        with self._calibration_graph.as_default():
            fetches = importer.import_graph_def(self._converted_graph_def, input_map=input_map_fn() if input_map_fn else None, return_elements=fetch_names, name='')
        calibrate_rewriter_cfg = rewriter_config_pb2.RewriterConfig()
        if self._test_only_disable_non_trt_optimizers:
            trt_utils.disable_non_trt_optimizers_in_rewriter_config(calibrate_rewriter_cfg)
        calibrate_config = config_pb2.ConfigProto(allow_soft_placement=True, graph_options=config_pb2.GraphOptions(rewrite_options=calibrate_rewriter_cfg))
        with session.Session(graph=self._calibration_graph, config=calibrate_config) as calibration_sess:
            for _ in range(num_runs):
                calibration_sess.run(fetches, feed_dict=feed_dict_fn() if feed_dict_fn else None)
            device_to_get_resource_op_map = {}
            with self._calibration_graph.as_default():
                resource_name_input = array_ops.placeholder(dtypes.string)
                for node in self._converted_graph_def.node:
                    if node.op == _TRT_ENGINE_OP_NAME:
                        if node.device not in device_to_get_resource_op_map:
                            with self._calibration_graph.device(node.device):
                                serialized_resources_output = gen_trt_ops.get_calibration_data_op(resource_name_input)
                            device_to_get_resource_op_map[node.device] = serialized_resources_output
                        calibration_result = calibration_sess.run(device_to_get_resource_op_map[node.device], feed_dict={resource_name_input: _get_canonical_engine_name(node.name)})
                        node.attr['calibration_data'].s = calibration_result
            self._calibration_data_collected = True
        return self._converted_graph_def

    def save(self, output_saved_model_dir):
        if False:
            print('Hello World!')
        'Save the converted graph as a SavedModel.\n\n    Args:\n      output_saved_model_dir: construct a SavedModel using the converted\n        GraphDef and save it to the specified directory. This option only works\n        when the input graph is loaded from a SavedModel, i.e. when\n        input_saved_model_dir is specified and input_graph_def is None in\n        __init__().\n\n    Raises:\n      ValueError: if the input to the converter is a GraphDef instead of a\n      SavedModel.\n    '
        assert self._converted
        if self._need_calibration:
            assert self._calibration_data_collected
        if self._input_graph_def:
            raise ValueError('Not able to save to a SavedModel since input is a GraphDef')

        def _restore_collections(dest_graph, src_meta_graph_def, collection_keys):
            if False:
                while True:
                    i = 10
            'Restores collections that we need to keep.'
            scope = ''
            for key in collection_keys:
                collection_def = src_meta_graph_def.collection_def[key]
                kind = collection_def.WhichOneof('kind')
                if kind is None:
                    logging.error('Cannot identify data type for collection %s. Skipping.', key)
                    continue
                from_proto = ops.get_from_proto_function(key)
                if from_proto and kind == 'bytes_list':
                    proto_type = ops.get_collection_proto_type(key)
                    for value in collection_def.bytes_list.value:
                        proto = proto_type()
                        proto.ParseFromString(value)
                        try:
                            new_value = from_proto(proto, import_scope=scope)
                        except:
                            continue
                        dest_graph.add_to_collection(key, new_value)
                else:
                    field = getattr(collection_def, kind)
                    if kind == 'node_list':
                        for value in field.value:
                            name = ops.prepend_name_scope(value, scope)
                            try:
                                col_op = dest_graph.as_graph_element(name)
                            except (TypeError, ValueError, KeyError):
                                continue
                            dest_graph.add_to_collection(key, col_op)
                    elif kind == 'int64_list':
                        for value in field.value:
                            dest_graph.add_to_collection(key, int(value))
                    else:
                        for value in field.value:
                            dest_graph.add_to_collection(key, ops.prepend_name_scope(value, scope))
        saved_model_builder = builder.SavedModelBuilder(output_saved_model_dir)
        with ops.Graph().as_default():
            importer.import_graph_def(self._converted_graph_def, name='')
            _restore_collections(ops.get_default_graph(), self._grappler_meta_graph_def, self._collections_to_keep(self._grappler_meta_graph_def.collection_def))
            with session.Session() as sess:
                saved_model_builder.add_meta_graph_and_variables(sess, self._input_saved_model_tags, signature_def_map=self._grappler_meta_graph_def.signature_def)
        saved_model_builder.save()

def _get_resource_handle(name, device):
    if False:
        for i in range(10):
            print('nop')
    with ops.device(device):
        return gen_trt_ops.create_trt_resource_handle(resource_name=name)

def _remove_native_segments(input_func):
    if False:
        for i in range(10):
            print('nop')
    'Remove native segments from the input TF-TRT Converted Function.\n\n  Args:\n    input_func: provide the concrete function with native segment nodes. The\n      transformed output func will not contain any native segment nodes. All the\n      TRTEngineOp references will be deleted and reset to default empty func.\n  '
    input_graph_def = input_func.graph.as_graph_def()
    nodes_deleted = 0
    for func_id in reversed(range(len(input_graph_def.library.function))):
        f = input_graph_def.library.function[func_id]
        if 'native_segment' in f.signature.name:
            nodes_deleted += 1
            while context.context().has_function(f.signature.name):
                context.context().remove_function(f.signature.name)
            del input_graph_def.library.function[func_id]
    logging.info(f'Found and deleted native segments from {nodes_deleted} TRTEngineOp nodes.')
    for node in input_graph_def.node:
        if node.op == 'TRTEngineOp':
            del node.attr['segment_func']
    for func in input_graph_def.library.function:
        for node in func.node_def:
            if node.op == 'TRTEngineOp':
                del node.attr['segment_func']
    new_func = _construct_function_from_graph_def(input_func, input_graph_def)
    return new_func

class _TRTEngineResource(resource.TrackableResource):
    """Class to track the serialized engines resource."""

    def __init__(self, resource_name, filename, maximum_cached_engines, device='GPU'):
        if False:
            while True:
                i = 10
        super(_TRTEngineResource, self).__init__(device=device)
        self._resource_name = resource_name
        self._filename = self._track_trackable(asset.Asset(filename), '_serialized_trt_resource_filename')
        self._maximum_cached_engines = maximum_cached_engines

    def _create_resource(self):
        if False:
            print('Hello World!')
        return _get_resource_handle(self._resource_name, self._resource_device)

    def _initialize(self):
        if False:
            i = 10
            return i + 15
        gen_trt_ops.initialize_trt_resource(self.resource_handle, self._filename, max_cached_engines_count=self._maximum_cached_engines)

    def _destroy_resource(self):
        if False:
            return 10
        handle = _get_resource_handle(self._resource_name, self._resource_device)
        with ops.device(self._resource_device):
            gen_resource_variable_ops.destroy_resource_op(handle, ignore_lookup_error=True)

def _print_row(fields, positions, print_fn):
    if False:
        i = 10
        return i + 15
    'Prints a row.'
    line = ''
    for (i, field) in enumerate(fields):
        field = str(field)
        end_line_pos = positions[i]
        if i > 0:
            line = line + ' '
        line = '{0:{min_length}}'.format(line + field, min_length=end_line_pos)
        if len(line) > end_line_pos:
            line = line[:end_line_pos - 4] + ' ...'
    print_fn(line)

def _construct_function_from_graph_def(func, graph_def, frozen_func=None):
    if False:
        print('Hello World!')
    'Rebuild function from graph_def.'
    if frozen_func is None:
        frozen_func = func
    for f in graph_def.library.function:
        while context.context().has_function(f.signature.name):
            context.context().remove_function(f.signature.name)
    captures = {c[1].name.split(':')[0]: c[0] for c in frozen_func.graph.captures}
    new_func = wrap_function.function_from_graph_def(graph_def, [tensor.name for tensor in frozen_func.inputs], [tensor.name for tensor in frozen_func.outputs], captures)
    new_func.graph.structured_outputs = nest.pack_sequence_as(func.graph.structured_outputs, new_func.graph.structured_outputs)
    new_func._function_type = func.function_type
    new_func.graph.structured_input_signature = func.structured_input_signature
    return new_func

def _apply_inlining(func):
    if False:
        print('Hello World!')
    "Apply an inlining optimization to the function's graph definition."
    graph_def = func.graph.as_graph_def()
    for function in graph_def.library.function:
        if 'api_implements' in function.attr:
            del function.attr['api_implements']
    meta_graph = saver.export_meta_graph(graph_def=graph_def, graph=func.graph)
    for name in ['variables', 'model_variables', 'trainable_variables', 'local_variables']:
        raw_list = []
        for raw in meta_graph.collection_def['variables'].bytes_list.value:
            variable = variable_pb2.VariableDef()
            variable.ParseFromString(raw)
            variable.ClearField('initializer_name')
            raw_list.append(variable.SerializeToString())
        meta_graph.collection_def[name].bytes_list.value[:] = raw_list
    fetch_collection = meta_graph_pb2.CollectionDef()
    for array in func.inputs + func.outputs:
        fetch_collection.node_list.value.append(array.name)
    meta_graph.collection_def['train_op'].CopyFrom(fetch_collection)
    config = config_pb2.ConfigProto()
    rewrite_options = config.graph_options.rewrite_options
    rewrite_options.min_graph_nodes = -1
    rewrite_options.optimizers.append('function')
    new_graph_def = tf_optimizer.OptimizeGraph(config, meta_graph)
    return new_graph_def

def _annotate_variable_ops(func, graph_def):
    if False:
        for i in range(10):
            print('nop')
    'Annotates variable operations with custom `_shape` attribute.\n\n  This is required for the converters and shape inference. The graph\n  definition is modified in-place.\n\n  Args:\n    func: Function represented by the graph definition.\n    graph_def: Graph definition to be annotated in-place.\n\n  Raises:\n    RuntimeError: if some shapes cannot be annotated.\n  '
    ph_shape_map = {}
    for (ph, var) in zip(func.graph.internal_captures, func.variables):
        ph_shape_map[ph.name] = var.shape
    name_to_node = {node.name: node for node in graph_def.node}
    for node in graph_def.node:
        if node.op == 'ReadVariableOp' or node.op == 'ResourceGather':
            node_ = node
            while name_to_node[node_.input[0]].op == 'Identity':
                node_ = name_to_node[node_.input[0]]
            ph_name = node_.input[0] + ':0'
            if ph_name in ph_shape_map:
                shape = ph_shape_map[ph_name]
                node.attr['_shape'].shape.CopyFrom(shape.as_proto())
            else:
                raise RuntimeError('Not found in the function captures: {}'.format(ph_name))

def _save_calibration_table(node):
    if False:
        for i in range(10):
            print('nop')
    try:
        calibration_table = gen_trt_ops.get_calibration_data_op(_get_canonical_engine_name(node.name))
        node.attr['calibration_data'].s = calibration_table.numpy()
    except (errors.UnknownError, errors.NotFoundError):
        logging.warning('Warning calibration error for %s', node.name)

def _convert_to_tensor(inp):
    if False:
        i = 10
        return i + 15
    try:
        if isinstance(inp, dict):
            args = []
            kwargs = {k: ops.convert_to_tensor(v) for (k, v) in inp.items()}
        else:
            kwargs = {}
            if isinstance(inp, (list, tuple)):
                args = map(ops.convert_to_tensor, inp)
            else:
                args = [ops.convert_to_tensor(inp)]
    except:
        error_msg = 'Failed to convert input to tensor.'
        logging.error(error_msg + '\ninp = `{0}`\n'.format(inp))
        raise RuntimeError(error_msg)
    return (args, kwargs)

@tf_export('experimental.tensorrt.Converter', v1=[])
class TrtGraphConverterV2(object):
    """An offline converter for TF-TRT transformation for TF 2.0 SavedModels.

  Windows support is provided experimentally. No guarantee is made regarding
  functionality or engineering support. Use at your own risk.

  There are several ways to run the conversion:

  1. FP32/FP16 precision

     ```python
     params = tf.experimental.tensorrt.ConversionParams(
         precision_mode='FP16')
     converter = tf.experimental.tensorrt.Converter(
         input_saved_model_dir="my_dir", conversion_params=params)
     converter.convert()
     converter.save(output_saved_model_dir)
     ```

     In this case, no TRT engines will be built or saved in the converted
     SavedModel. But if input data is available during conversion, we can still
     build and save the TRT engines to reduce the cost during inference (see
     option 2 below).

  2. FP32/FP16 precision with pre-built engines

     ```python
     params = tf.experimental.tensorrt.ConversionParams(
         precision_mode='FP16',
         # Set this to a large enough number so it can cache all the engines.
         maximum_cached_engines=16)
     converter = tf.experimental.tensorrt.Converter(
         input_saved_model_dir="my_dir", conversion_params=params)
     converter.convert()

     # Define a generator function that yields input data, and use it to execute
     # the graph to build TRT engines.
     def my_input_fn():
       for _ in range(num_runs):
         inp1, inp2 = ...
         yield inp1, inp2

     converter.build(input_fn=my_input_fn)  # Generate corresponding TRT engines
     converter.save(output_saved_model_dir)  # Generated engines will be saved.
     ```

     In this way, one engine will be built/saved for each unique input shapes of
     the TRTEngineOp. This is good for applications that cannot afford building
     engines during inference but have access to input data that is similar to
     the one used in production (for example, that has the same input shapes).
     Also, the generated TRT engines is platform dependent, so we need to run
     `build()` in an environment that is similar to production (e.g. with
     same type of GPU).

  3. INT8 precision and calibration with pre-built engines

     ```python
     params = tf.experimental.tensorrt.ConversionParams(
         precision_mode='INT8',
         # Currently only one INT8 engine is supported in this mode.
         maximum_cached_engines=1,
         use_calibration=True)
     converter = tf.experimental.tensorrt.Converter(
         input_saved_model_dir="my_dir", conversion_params=params)

     # Define a generator function that yields input data, and run INT8
     # calibration with the data. All input data should have the same shape.
     # At the end of convert(), the calibration stats (e.g. range information)
     # will be saved and can be used to generate more TRT engines with different
     # shapes. Also, one TRT engine will be generated (with the same shape as
     # the calibration data) for save later.
     def my_calibration_input_fn():
       for _ in range(num_runs):
         inp1, inp2 = ...
         yield inp1, inp2

     converter.convert(calibration_input_fn=my_calibration_input_fn)

     # (Optional) Generate more TRT engines offline (same as the previous
     # option), to avoid the cost of generating them during inference.
     def my_input_fn():
       for _ in range(num_runs):
         inp1, inp2 = ...
         yield inp1, inp2
     converter.build(input_fn=my_input_fn)

     # Save the TRT engine and the engines.
     converter.save(output_saved_model_dir)
     ```
  4. To use dynamic shape, we need to call the build method with an input
     function to generate profiles. This step is similar to the INT8 calibration
     step described above. The converter also needs to be created with
     use_dynamic_shape=True and one of the following profile_strategies for
     creating profiles based on the inputs produced by the input function:
     * `Range`: create one profile that works for inputs with dimension values
       in the range of [min_dims, max_dims] where min_dims and max_dims are
       derived from the provided inputs.
     * `Optimal`: create one profile for each input. The profile only works for
       inputs with the same dimensions as the input it is created for. The GPU
       engine will be run with optimal performance with such inputs.
     * `Range+Optimal`: create the profiles for both `Range` and `Optimal`.
  """

    def _verify_profile_strategy(self, strategy):
        if False:
            i = 10
            return i + 15
        supported_strategies = [s.lower() for s in supported_profile_strategies()]
        if strategy.lower() not in supported_strategies:
            raise ValueError("profile_strategy '{}' is not supported. It should be one of {}".format(strategy, supported_profile_strategies()))
        if strategy == 'ImplicitBatchModeCompatible':
            logging.warn('ImplicitBatchModeCompatible strategy is deprecated, and using it may result in errors during engine building. Please consider using a different profile strategy.')

    @deprecation.deprecated_args(None, 'Use individual converter parameters instead', 'conversion_params')
    def __init__(self, input_saved_model_dir=None, input_saved_model_tags=None, input_saved_model_signature_key=None, use_dynamic_shape=None, dynamic_shape_profile_strategy=None, max_workspace_size_bytes=DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES, precision_mode=TrtPrecisionMode.FP32, minimum_segment_size=3, maximum_cached_engines=1, use_calibration=True, allow_build_at_runtime=True, conversion_params=None):
        if False:
            while True:
                i = 10
        "Initialize the converter.\n\n    Args:\n      input_saved_model_dir: the directory to load the SavedModel which contains\n        the input graph to transforms. Required.\n      input_saved_model_tags: list of tags to load the SavedModel.\n      input_saved_model_signature_key: the key of the signature to optimize the\n        graph for.\n      use_dynamic_shape: whether to enable dynamic shape support. None is\n        equivalent to False in the current implementation.\n      dynamic_shape_profile_strategy: one of the strings in\n        supported_profile_strategies(). None is equivalent to Range in the\n        current implementation.\n      max_workspace_size_bytes: the maximum GPU temporary memory that the TRT\n        engine can use at execution time. This corresponds to the\n        'workspaceSize' parameter of nvinfer1::IBuilder::setMaxWorkspaceSize().\n      precision_mode: one of the strings in\n        TrtPrecisionMode.supported_precision_modes().\n      minimum_segment_size: the minimum number of nodes required for a subgraph\n        to be replaced by TRTEngineOp.\n      maximum_cached_engines: max number of cached TRT engines for dynamic TRT\n        ops. Created TRT engines for a dynamic dimension are cached. If the\n        number of cached engines is already at max but none of them supports the\n        input shapes, the TRTEngineOp will fall back to run the original TF\n        subgraph that corresponds to the TRTEngineOp.\n      use_calibration: this argument is ignored if precision_mode is not INT8.\n        If set to True, a calibration graph will be created to calibrate the\n        missing ranges. The calibration graph must be converted to an inference\n        graph by running calibration with calibrate(). If set to False,\n        quantization nodes will be expected for every tensor in the graph\n        (excluding those which will be fused). If a range is missing, an error\n        will occur. Please note that accuracy may be negatively affected if\n        there is a mismatch between which tensors TRT quantizes and which\n        tensors were trained with fake quantization.\n      allow_build_at_runtime: whether to allow building TensorRT engines during\n        runtime if no prebuilt TensorRT engine can be found that can handle the\n        given inputs during runtime, then a new TensorRT engine is built at\n        runtime if allow_build_at_runtime=True, and otherwise native TF is used.\n      conversion_params: a TrtConversionParams instance (deprecated).\n\n    Raises:\n      ValueError: if the combination of the parameters is invalid.\n    "
        assert context.executing_eagerly()
        if conversion_params is None:
            conversion_params = TrtConversionParams(max_workspace_size_bytes=max_workspace_size_bytes, precision_mode=precision_mode, minimum_segment_size=minimum_segment_size, maximum_cached_engines=maximum_cached_engines, use_calibration=use_calibration, allow_build_at_runtime=allow_build_at_runtime)
        _check_trt_version_compatibility()
        _check_conversion_params(conversion_params, is_v2=True)
        self._conversion_params = conversion_params
        self._input_saved_model_dir = input_saved_model_dir
        self._input_saved_model_tags = input_saved_model_tags or [tag_constants.SERVING]
        self._input_saved_model_signature_key = input_saved_model_signature_key or signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        self.freeze = not trt_utils.is_experimental_feature_activated('disable_graph_freezing')
        self._need_calibration = (conversion_params.precision_mode == TrtPrecisionMode.INT8 or conversion_params.precision_mode == TrtPrecisionMode.INT8.lower()) and conversion_params.use_calibration
        self._calibration_input_fn = None
        self._converted = False
        self._device = None
        self._build_called_once = False
        self._calibrated = False
        if use_dynamic_shape is None:
            self._use_dynamic_shape = False
        else:
            self._use_dynamic_shape = use_dynamic_shape
        if not self.freeze and (not self._use_dynamic_shape):
            logging.warn('Disabling graph freezing is only possible in dynamic shape mode. The graph will be frozen.')
            self.freeze = True
        self._profile_strategy = 'Unknown'
        if self._use_dynamic_shape:
            if dynamic_shape_profile_strategy is None:
                self._profile_strategy = PROFILE_STRATEGY_RANGE
            else:
                self._verify_profile_strategy(dynamic_shape_profile_strategy)
                self._profile_strategy = dynamic_shape_profile_strategy
        self._test_only_disable_non_trt_optimizers = False

    def _need_trt_profiles(self):
        if False:
            print('Hello World!')
        return self._use_dynamic_shape

    def _run_conversion(self, meta_graph_def):
        if False:
            i = 10
            return i + 15
        "Run Grappler's OptimizeGraph() tool to convert the graph.\n\n    Args:\n      meta_graph_def: the MetaGraphDef instance to run the optimizations on.\n\n    Returns:\n      The optimized GraphDef.\n    "
        grappler_session_config = config_pb2.ConfigProto()
        custom_rewriter_config = _get_tensorrt_rewriter_config(conversion_params=self._conversion_params._replace(allow_build_at_runtime=True), is_dynamic_op=True, max_batch_size=None, disable_non_trt_optimizers=self._test_only_disable_non_trt_optimizers, use_implicit_batch=not self._use_dynamic_shape, profile_strategy=self._profile_strategy)
        grappler_session_config.graph_options.rewrite_options.CopyFrom(custom_rewriter_config)
        return tf_optimizer.OptimizeGraph(grappler_session_config, meta_graph_def, graph_id=b'tf_graph')

    def _for_each_trt_node(self, graph_def, fn):
        if False:
            print('Hello World!')
        'Helper method to manipulate all TRTEngineOps in a GraphDef.'
        for node in graph_def.node:
            if node.op == _TRT_ENGINE_OP_NAME:
                fn(node)
        for func in graph_def.library.function:
            for node in func.node_def:
                if node.op == _TRT_ENGINE_OP_NAME:
                    fn(node)

    def _execute_calibration(self, calibration_input_fn):
        if False:
            while True:
                i = 10
        'Run INT8 calibration with the provided input generator function.'
        for inp in calibration_input_fn():
            (args, kwargs) = _convert_to_tensor(inp)
            self._converted_func(*args, **kwargs)
        self._for_each_trt_node(self._converted_graph_def, _save_calibration_table)
        self._converted_func = _construct_function_from_graph_def(self._converted_func, self._converted_graph_def)
        self._calibrated = True

    def convert(self, calibration_input_fn=None):
        if False:
            for i in range(10):
                print('nop')
        'Convert the input SavedModel in 2.0 format.\n\n    Args:\n      calibration_input_fn: a generator function that yields input data as a\n        list or tuple or dict, which will be used to execute the converted\n        signature for calibration. All the returned input data should have the\n        same shape. Example: `def input_fn(): yield input1, input2, input3`\n\n        If dynamic_shape_mode==False, (or if the graph has static input shapes)\n        then we run calibration and build the calibrated engine during\n        conversion.\n\n        If dynamic_shape_mode==True (and the graph has any unknown input\n        shape), then the reference to calibration_input_fn is stored, and the\n        calibration is actually performed when we build the engine (see\n        build()).\n\n    Raises:\n      ValueError: if the input combination is invalid.\n\n    Returns:\n      The TF-TRT converted Function.\n    '
        assert not self._converted
        device_requested = array_ops.zeros([]).device
        if 'gpu' not in device_requested.lower():
            raise ValueError(f'Specified device is not a GPU: {device_requested}')
        if 'gpu:0' not in device_requested.lower():
            self._device = device_requested
            logging.info(f'Placing imported graph from `{self._input_saved_model_dir}` on device: {self._device}')
        if self._need_calibration and (not calibration_input_fn):
            raise ValueError('Should specify calibration_input_fn because INT8 calibration is needed')
        if not self._need_calibration and calibration_input_fn:
            raise ValueError('Should not specify calibration_input_fn because INT8 calibration is not needed')
        self._saved_model = load.load(self._input_saved_model_dir, self._input_saved_model_tags)
        func = self._saved_model.signatures[self._input_saved_model_signature_key]
        if self.freeze:
            frozen_func = convert_to_constants.convert_variables_to_constants_v2(func)
        else:
            inlined_graph_def = _apply_inlining(func)
            _annotate_variable_ops(func, inlined_graph_def)
            frozen_func = _construct_function_from_graph_def(func, inlined_graph_def)
        frozen_graph_def = frozen_func.graph.as_graph_def()
        logging.info('Clearing prior device assignments in loaded saved model')
        for node in frozen_graph_def.node:
            node.device = ''
        if self._device is None:
            grappler_meta_graph_def = saver.export_meta_graph(graph_def=frozen_graph_def, graph=frozen_func.graph)
        else:
            with ops.Graph().as_default() as graph, ops.device(self._device):
                importer.import_graph_def(frozen_graph_def, name='')
                grappler_meta_graph_def = saver.export_meta_graph(graph_def=graph.as_graph_def(), graph=graph)
        fetch_collection = meta_graph_pb2.CollectionDef()
        for array in frozen_func.inputs + frozen_func.outputs:
            fetch_collection.node_list.value.append(array.name)
        grappler_meta_graph_def.collection_def['train_op'].CopyFrom(fetch_collection)
        self._converted_graph_def = self._run_conversion(grappler_meta_graph_def)
        self._converted_func = _construct_function_from_graph_def(func, self._converted_graph_def, frozen_func)
        if self._need_calibration:
            if not self._need_trt_profiles():
                self._execute_calibration(calibration_input_fn)
            else:
                self._calibration_input_fn = calibration_input_fn
        self._converted = True
        graphviz_path = os.environ.get('TF_TRT_EXPORT_GRAPH_VIZ_PATH', default=None)
        if graphviz_path is not None:
            try:
                trt_utils.draw_graphdef_as_graphviz(graphdef=self._converted_func.graph.as_graph_def(add_shapes=True), dot_output_filename=graphviz_path)
            except Exception as e:
                logging.error(f'An Exception occurred during the export of the graph visualization: {e}')
        return self._converted_func

    def build(self, input_fn):
        if False:
            i = 10
            return i + 15
        "Run inference with converted graph in order to build TensorRT engines.\n\n    If the conversion requires INT8 calibration, then a reference to the\n    calibration function was stored during the call to convert(). Calibration\n    will be performed while we build the TensorRT engines.\n\n    Args:\n      input_fn: a generator function that provides the input data as a single\n        array, OR a list or tuple of the arrays OR a dict, which will be used\n        to execute the converted signature to generate TRT engines.\n        Example 1:\n        `def input_fn():\n             # Let's assume a network with 1 input tensor.\n             # We generate 2 sets of dummy input data:\n             input_shapes = [(1, 16),    # 1st shape\n                             (2, 32)]    # 2nd shape\n             for shapes in input_shapes:\n                 # return an input tensor\n                 yield np.zeros(shape).astype(np.float32)'\n\n        Example 2:\n        `def input_fn():\n             # Let's assume a network with 2 input tensors.\n             # We generate 3 sets of dummy input data:\n             input_shapes = [[(1, 16), (2, 16)], # 1st input list\n                             [(2, 32), (4, 32)], # 2nd list of two tensors\n                             [(4, 32), (8, 32)]] # 3rd input list\n             for shapes in input_shapes:\n                 # return a list of input tensors\n                 yield [np.zeros(x).astype(np.float32) for x in shapes]`\n\n    Raises:\n      NotImplementedError: build() is already called.\n      RuntimeError: the input_fx is None.\n    "
        if self._build_called_once:
            raise NotImplementedError('build() is already called. It is not supported to call build() more than once.')
        if not input_fn:
            raise RuntimeError('input_fn is None. Method build() needs input_fn to be specified in order to build TensorRT engines')
        if not self._converted:
            raise RuntimeError('Need to call convert() before build()')
        if self._need_calibration and (not self._calibrated) and (self._calibration_input_fn is None):
            raise RuntimeError('Need to provide the calibration_input_fn arg while calling convert().')

        def _set_profile_generation_mode(value, node):
            if False:
                for i in range(10):
                    print('nop')
            node.attr['_profile_generation_mode'].b = value
        if self._need_trt_profiles():
            self._for_each_trt_node(self._converted_graph_def, partial(_set_profile_generation_mode, True))
            func = _construct_function_from_graph_def(self._converted_func, self._converted_graph_def)
        else:
            func = self._converted_func
        first_input = None
        for inp in input_fn():
            if first_input is None:
                first_input = inp
            (args, kwargs) = _convert_to_tensor(inp)
            func(*args, **kwargs)
        if self._need_trt_profiles():
            self._for_each_trt_node(self._converted_graph_def, partial(_set_profile_generation_mode, False))
        if self._need_calibration and (not self._calibrated):
            self._execute_calibration(self._calibration_input_fn)
        else:
            (args, kwargs) = _convert_to_tensor(first_input)
            self._converted_func(*args, **kwargs)
        self._build_called_once = True

    def save(self, output_saved_model_dir, save_gpu_specific_engines=True, options=None):
        if False:
            for i in range(10):
                print('nop')
        "Save the converted SavedModel.\n\n    Args:\n      output_saved_model_dir: directory to saved the converted SavedModel.\n      save_gpu_specific_engines: whether to save TRT engines that have been\n        built. When True, all engines are saved and when False, the engines\n        are not saved and will be rebuilt at inference time. By using\n        save_gpu_specific_engines=False after doing INT8 calibration, inference\n        can be done on different GPUs than the GPU that the model was calibrated\n        and saved on.\n      options: `tf.saved_model.SaveOptions` object for configuring save options.\n    Raises:\n      RuntimeError: if the needed calibration hasn't been done.\n    "
        assert self._converted
        if trt_utils.is_experimental_feature_activated('remove_native_segments'):
            logging.info("'remove_native_segments' experimental feature is enabled during saving of converted SavedModel.")
            self._converted_func = _remove_native_segments(self._converted_func)
            self._converted_graph_def = self._converted_func.graph.as_graph_def()
        if self._need_calibration and (not self._calibrated):
            raise RuntimeError('A model that requires INT8 calibration has to be built before saving it. Call build() to build and calibrate the TensorRT engines.')
        engine_asset_dir = tempfile.mkdtemp()
        resource_map = {}

        def _serialize_and_track_engine(node):
            if False:
                print('Hello World!')
            'Serialize TRT engines in the cache and track them.'
            canonical_engine_name = _get_canonical_engine_name(node.name)
            if canonical_engine_name in resource_map:
                return
            filename = os.path.join(engine_asset_dir, 'trt-serialized-engine.' + canonical_engine_name)
            try:
                gen_trt_ops.serialize_trt_resource(resource_name=canonical_engine_name, filename=filename, delete_resource=True, save_gpu_specific_engines=save_gpu_specific_engines)
            except errors.NotFoundError:
                logging.info('Could not find %s in TF-TRT cache. This can happen if build() is not called, which means TensorRT engines will be built and cached at runtime.', canonical_engine_name)
                return
            resource_map[canonical_engine_name] = _TRTEngineResource(canonical_engine_name, filename, self._conversion_params.maximum_cached_engines)
        self._for_each_trt_node(self._converted_graph_def, _serialize_and_track_engine)
        trackable = autotrackable.AutoTrackable() if self.freeze else self._saved_model
        trackable.trt_engine_resources = resource_map
        if not self._conversion_params.allow_build_at_runtime:

            def _reset_allow_build_at_runtime(node):
                if False:
                    print('Hello World!')
                node.attr['_allow_build_at_runtime'].b = False
            self._for_each_trt_node(self._converted_graph_def, _reset_allow_build_at_runtime)
            reset_converted_func = wrap_function.function_from_graph_def(self._converted_graph_def, [tensor.name for tensor in self._converted_func.inputs], [tensor.name for tensor in self._converted_func.outputs])
            reset_converted_func.graph.structured_outputs = nest.pack_sequence_as(self._converted_func.graph.structured_outputs, reset_converted_func.graph.structured_outputs)
            reset_converted_func.graph.structured_input_signature = self._converted_func.structured_input_signature
            self._converted_func = reset_converted_func
        signatures = {self._input_saved_model_signature_key: self._converted_func}
        save.save(trackable, output_saved_model_dir, signatures, options=options)

    def summary(self, line_length=160, detailed=True, print_fn=None):
        if False:
            return 10
        'This method describes the results of the conversion by TF-TRT.\n\n    It includes information such as the name of the engine, the number of nodes\n    per engine, the input and output dtype, along with the input shape of each\n    TRTEngineOp.\n\n    Args:\n      line_length: Default line length when printing on the console. Minimum 160\n        characters long.\n      detailed: Whether or not to show the nodes inside each TRTEngineOp.\n      print_fn: Print function to use. Defaults to `print`. It will be called on\n        each line of the summary. You can set it to a custom function in order\n        to capture the string summary.\n\n    Raises:\n      RuntimeError: if the graph is not converted.\n    '
        if not self._converted:
            raise RuntimeError(f'Impossible to call `{self.__class__.__name__}.summary()` before calling {self.__class__.__name__}.convert()`.')
        if line_length < 160:
            raise ValueError(f'Invalid `line_length` value has been received: {line_length}. Minimum: 160.')
        if print_fn is None:
            print_fn = print
        columns = [('TRTEngineOP Name', 0.2), ('Device', 0.09), ('# Nodes', 0.05), ('# Inputs', 0.09), ('# Outputs', 0.09), ('Input DTypes', 0.12), ('Output Dtypes', 0.12), ('Input Shapes', 0.12), ('Output Shapes', 0.12)]
        positions = [int(line_length * p) for (_, p) in columns]
        positions = np.cumsum(positions).tolist()
        headers = [h for (h, _) in columns]
        _print_row(headers, positions, print_fn=print_fn)
        print_fn('=' * line_length)
        n_engines = 0
        n_ops_converted = 0
        n_ops_not_converted = 0
        graphdef = self._converted_func.graph.as_graph_def(add_shapes=True)
        trtengineops_dict = dict()
        for node in graphdef.node:
            if node.op != 'TRTEngineOp':
                n_ops_not_converted += 1
                continue
            else:
                trtengineops_dict[node.name] = node
                n_engines += 1
        for (name, node) in sorted(trtengineops_dict.items()):
            node_device = node.device.split('/')[-1]
            in_shapes = trt_utils.get_node_io_shapes(node, 'input_shapes')
            out_shapes = trt_utils.get_node_io_shapes(node, '_output_shapes')
            in_dtypes = trt_utils.get_trtengineop_io_dtypes(node, 'InT')
            out_dtypes = trt_utils.get_trtengineop_io_dtypes(node, 'OutT')
            in_nodes_count = trt_utils.get_trtengineop_io_nodes_count(node, 'InT')
            out_nodes_count = trt_utils.get_trtengineop_io_nodes_count(node, 'OutT')
            (node_count, converted_ops_dict) = trt_utils.get_trtengineop_node_op_count(graphdef, name)
            n_ops_converted += node_count
            if n_engines != 1:
                print_fn(f"\n{'-' * 40}\n")
            _print_row(fields=[name, node_device, node_count, in_nodes_count, out_nodes_count, in_dtypes, out_dtypes, in_shapes, out_shapes], positions=positions, print_fn=print_fn)
            if detailed:
                print_fn()
                for (key, value) in sorted(dict(converted_ops_dict).items()):
                    print_fn(f'\t- {key}: {value}x')
        print_fn(f"\n{'=' * line_length}")
        print_fn(f'[*] Total number of TensorRT engines: {n_engines}')
        total_ops = n_ops_not_converted + n_ops_converted
        conversion_ratio = n_ops_converted / total_ops * 100
        print_fn(f'[*] % of OPs Converted: {conversion_ratio:.2f}% [{n_ops_converted}/{total_ops}]\n')

def create_inference_graph(input_graph_def, outputs, max_batch_size=1, max_workspace_size_bytes=DEFAULT_TRT_MAX_WORKSPACE_SIZE_BYTES, precision_mode=TrtPrecisionMode.FP32, minimum_segment_size=3, is_dynamic_op=False, maximum_cached_engines=1, input_saved_model_dir=None, input_saved_model_tags=None, input_saved_model_signature_key=None, output_saved_model_dir=None):
    if False:
        print('Hello World!')
    "Python wrapper for the TRT transformation.\n\n  Args:\n    input_graph_def: a GraphDef object containing a model to be transformed. If\n      set to None, the graph will be read from the SavedModel loaded from\n      input_saved_model_dir.\n    outputs: list of tensors or node names for the model outputs. Only used when\n      input_graph_def is not None.\n    max_batch_size: max size for the input batch.\n    max_workspace_size_bytes: the maximum GPU temporary memory which the TRT\n      engine can use at execution time. This corresponds to the 'workspaceSize'\n      parameter of nvinfer1::IBuilder::setMaxWorkspaceSize().\n    precision_mode: one of TrtPrecisionMode.supported_precision_modes().\n    minimum_segment_size: the minimum number of nodes required for a subgraph to\n      be replaced by TRTEngineOp.\n    is_dynamic_op: whether to generate dynamic TRT ops which will build the TRT\n      network and engine at run time.\n    maximum_cached_engines: max number of cached TRT engines in dynamic TRT ops.\n      If the number of cached engines is already at max but none of them can\n      serve the input, the TRTEngineOp will fall back to run the TF function\n      based on which the TRTEngineOp is created.\n    input_saved_model_dir: the directory to load the SavedModel which contains\n      the input graph to transforms. Used only when input_graph_def is None.\n    input_saved_model_tags: list of tags to load the SavedModel.\n    input_saved_model_signature_key: the key of the signature to optimize the\n      graph for.\n    output_saved_model_dir: if not None, construct a SavedModel using the\n      returned GraphDef and save it to the specified directory. This option only\n      works when the input graph is loaded from a SavedModel, i.e. when\n      input_saved_model_dir is specified and input_graph_def is None.\n\n  Returns:\n    A GraphDef transformed from input_graph_def (or the SavedModel graph def\n    loaded from input_saved_model_dir, if input_graph_def is not present), where\n    all TRT compatible subgraphs are replaced with TRTEngineOps, and a TF\n    function is added for each of the subgraphs.\n\n    If is_dynamic_op is True, each TRTEngineOp will contain a serialized\n    subgraph GraphDef, which will be converted to a TRT engine at execution time\n    and the TRT engine will be cached for future usage. A new TRT engine will be\n    created each time when none of the cached engines match the input shapes. If\n    it fails to execute the TRT engine or the number of cached engines reaches\n    maximum_cached_engines, the op will fall back to call the corresponding TF\n    function.\n\n    If is_dynamic_op is False, each TRTEngineOp will contain a serialized TRT\n    engine created from the corresponding subgraph. No more engines will be\n    created on the fly, and the op will fall back to call the corresponding TF\n    function when it fails to execute the engine.\n\n  Raises:\n    ValueError: if the combination of the parameters is invalid.\n  "
    trt_converter = TrtGraphConverter(input_saved_model_dir=input_saved_model_dir, input_saved_model_tags=input_saved_model_tags, input_saved_model_signature_key=input_saved_model_signature_key, input_graph_def=input_graph_def, nodes_denylist=outputs, max_batch_size=max_batch_size, max_workspace_size_bytes=max_workspace_size_bytes, precision_mode=precision_mode, minimum_segment_size=minimum_segment_size, is_dynamic_op=is_dynamic_op, maximum_cached_engines=maximum_cached_engines, use_calibration=False)
    converted_graph_def = trt_converter.convert()
    if output_saved_model_dir:
        trt_converter.save(output_saved_model_dir)
    return converted_graph_def