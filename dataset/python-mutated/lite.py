"""TensorFlow Lite tooling helper functionality."""
import enum
import functools
import pprint
import shutil
import sys
import tempfile
import time
import warnings
from absl import logging
from google.protobuf import text_format as _text_format
from google.protobuf.message import DecodeError
from tensorflow.core.framework import graph_pb2 as _graph_pb2
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op
from tensorflow.lite.python import conversion_metadata_schema_py_generated as conversion_metdata_fb
from tensorflow.lite.python import lite_constants as constants
from tensorflow.lite.python.convert import convert_graphdef as _convert_graphdef
from tensorflow.lite.python.convert import convert_graphdef_with_arrays as _convert_graphdef_with_arrays
from tensorflow.lite.python.convert import convert_jax_hlo as _convert_jax_hlo
from tensorflow.lite.python.convert import convert_saved_model as _convert_saved_model
from tensorflow.lite.python.convert import ConverterError
from tensorflow.lite.python.convert import deduplicate_readonly_buffers as _deduplicate_readonly_buffers
from tensorflow.lite.python.convert import mlir_quantize as _mlir_quantize
from tensorflow.lite.python.convert import mlir_sparsify as _mlir_sparsify
from tensorflow.lite.python.convert import OpsSet
from tensorflow.lite.python.convert import toco_convert
from tensorflow.lite.python.convert_phase import Component
from tensorflow.lite.python.convert_phase import convert_phase
from tensorflow.lite.python.convert_phase import SubComponent
from tensorflow.lite.python.convert_saved_model import freeze_saved_model as _freeze_saved_model
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.lite.python.interpreter import load_delegate
from tensorflow.lite.python.interpreter import OpResolverType
from tensorflow.lite.python.metrics import metrics
from tensorflow.lite.python.op_hint import convert_op_hints_to_stubs
from tensorflow.lite.python.op_hint import is_ophint_converted as _is_ophint_converted
from tensorflow.lite.python.op_hint import OpHint
from tensorflow.lite.python.optimize import calibrator as _calibrator
from tensorflow.lite.python.util import _xla_computation
from tensorflow.lite.python.util import build_debug_info_func as _build_debug_info_func
from tensorflow.lite.python.util import convert_debug_info_func as _convert_debug_info_func
from tensorflow.lite.python.util import freeze_graph as _freeze_graph
from tensorflow.lite.python.util import get_debug_info as _get_debug_info
from tensorflow.lite.python.util import get_grappler_config as _get_grappler_config
from tensorflow.lite.python.util import get_sparsity_modes as _get_sparsity_modes
from tensorflow.lite.python.util import get_tensor_name as _get_tensor_name
from tensorflow.lite.python.util import get_tensors_from_tensor_names as _get_tensors_from_tensor_names
from tensorflow.lite.python.util import get_tf_type_name as _get_tf_type_name
from tensorflow.lite.python.util import is_frozen_graph as _is_frozen_graph
from tensorflow.lite.python.util import model_input_signature as _model_input_signature
from tensorflow.lite.python.util import modify_model_io_type as _modify_model_io_type
from tensorflow.lite.python.util import populate_conversion_metadata as _populate_conversion_metadata
from tensorflow.lite.python.util import run_graph_optimizations as _run_graph_optimizations
from tensorflow.lite.python.util import set_tensor_shapes as _set_tensor_shapes
from tensorflow.lite.python.util import trace_model_call as _trace_model_call
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.lite.tools.optimize.debugging.python.debugger import QuantizationDebugger
from tensorflow.lite.tools.optimize.debugging.python.debugger import QuantizationDebugOptions
from tensorflow.python.client import session as _session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function as _def_function
from tensorflow.python.eager import function as _function
from tensorflow.python.framework import byte_swap_tensor as bst
from tensorflow.python.framework import convert_to_constants as _convert_to_constants
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import versions
from tensorflow.python.framework.errors_impl import NotFoundError as _NotFoundError
from tensorflow.python.framework.importer import import_graph_def as _import_graph_def
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import loader_impl as _loader_impl
from tensorflow.python.saved_model import save as _save
from tensorflow.python.saved_model import save_options as _save_options
from tensorflow.python.saved_model import signature_constants as _signature_constants
from tensorflow.python.saved_model import tag_constants as _tag_constants
from tensorflow.python.saved_model.load import load as _load
from tensorflow.python.saved_model.loader_impl import parse_saved_model_with_debug_info as _parse_saved_model_with_debug_info
from tensorflow.python.util import deprecation as _deprecation
from tensorflow.python.util import keras_deps
from tensorflow.python.util.tf_export import tf_export as _tf_export

@_tf_export('lite.Optimize')
class Optimize(enum.Enum):
    """Enum defining the optimizations to apply when generating a tflite model.

  DEFAULT
      The default optimization strategy that enables post-training quantization.
      The type of post-training quantization that will be used is dependent on
      the other converter options supplied. Refer to the
      [documentation](/lite/performance/post_training_quantization) for further
      information on the types available and how to use them.

  OPTIMIZE_FOR_SIZE
      Deprecated. Does the same as DEFAULT.

  OPTIMIZE_FOR_LATENCY
      Deprecated. Does the same as DEFAULT.

  EXPERIMENTAL_SPARSITY
      Experimental flag, subject to change.

      Enable optimization by taking advantage of the sparse model weights
      trained with pruning.

      The converter will inspect the sparsity pattern of the model weights and
      do its best to improve size and latency.
      The flag can be used alone to optimize float32 models with sparse weights.
      It can also be used together with the DEFAULT optimization mode to
      optimize quantized models with sparse weights.
  """
    DEFAULT = 'DEFAULT'
    OPTIMIZE_FOR_SIZE = 'OPTIMIZE_FOR_SIZE'
    OPTIMIZE_FOR_LATENCY = 'OPTIMIZE_FOR_LATENCY'
    EXPERIMENTAL_SPARSITY = 'EXPERIMENTAL_SPARSITY'

    def __str__(self):
        if False:
            return 10
        return str(self.value)

@_tf_export('lite.RepresentativeDataset')
class RepresentativeDataset:
    """Representative dataset used to optimize the model.

  This is a generator function that provides a small dataset to calibrate or
  estimate the range, i.e, (min, max) of all floating-point arrays in the model
  (such as model input, activation outputs of intermediate layers, and model
  output) for quantization. Usually, this is a small subset of a few hundred
  samples randomly chosen, in no particular order, from the training or
  evaluation dataset.
  """

    def __init__(self, input_gen):
        if False:
            i = 10
            return i + 15
        'Creates a representative dataset.\n\n    Args:\n      input_gen: A generator function that generates input samples for the model\n        and has the same order, type and shape as the inputs to the model.\n        Usually, this is a small subset of a few hundred samples randomly\n        chosen, in no particular order, from the training or evaluation dataset.\n    '
        self.input_gen = input_gen

@_tf_export('lite.TargetSpec')
class TargetSpec:
    """Specification of target device used to optimize the model.

  Attributes:
    supported_ops: Experimental flag, subject to change. Set of `tf.lite.OpsSet`
      options, where each option represents a set of operators supported by the
      target device. (default {tf.lite.OpsSet.TFLITE_BUILTINS}))
    supported_types: Set of `tf.dtypes.DType` data types supported on the target
      device. If initialized, optimization might be driven by the smallest type
      in this set. (default set())
    experimental_select_user_tf_ops: Experimental flag, subject to change. Set
      of user's TensorFlow operators' names that are required in the TensorFlow
      Lite runtime. These ops will be exported as select TensorFlow ops in the
      model (in conjunction with the tf.lite.OpsSet.SELECT_TF_OPS flag). This is
      an advanced feature that should only be used if the client is using TF ops
      that may not be linked in by default with the TF ops that are provided
      when using the SELECT_TF_OPS path. The client is responsible for linking
      these ops into the target runtime.
    experimental_supported_backends: Experimental flag, subject to change. Set
      containing names of supported backends. Currently only "GPU" is supported,
      more options will be available later.
  """

    def __init__(self, supported_ops=None, supported_types=None, experimental_select_user_tf_ops=None, experimental_supported_backends=None):
        if False:
            i = 10
            return i + 15
        if supported_ops is None:
            supported_ops = {OpsSet.TFLITE_BUILTINS}
        self.supported_ops = supported_ops
        if supported_types is None:
            supported_types = set()
        self.supported_types = supported_types
        if experimental_select_user_tf_ops is None:
            experimental_select_user_tf_ops = set()
        self.experimental_select_user_tf_ops = experimental_select_user_tf_ops
        self.experimental_supported_backends = experimental_supported_backends
        self._experimental_custom_op_registerers = []
        self._experimental_supported_accumulation_type = None

class QuantizationMode:
    """QuantizationMode determines the quantization type from user options."""

    def __init__(self, optimizations, target_spec, representative_dataset, graph_def, disable_per_channel=False, experimental_new_dynamic_range_quantizer=False, experimental_low_bit_qat=False, full_integer_quantization_bias_type=None, experimental_mlir_variable_quantization=False):
        if False:
            i = 10
            return i + 15
        self._optimizations = optimizations
        for deprecated_optimization in [Optimize.OPTIMIZE_FOR_SIZE, Optimize.OPTIMIZE_FOR_LATENCY]:
            if deprecated_optimization in self._optimizations:
                logging.warning('Optimization option %s is deprecated, please use optimizations=[Optimize.DEFAULT] instead.', deprecated_optimization)
        self._target_spec = target_spec
        self._representative_dataset = representative_dataset
        self._graph_def = graph_def
        if self._is_int8_target_required():
            self._validate_int8_required()
        self.enable_mlir_variable_quantization = experimental_mlir_variable_quantization
        if self._is_float16_target_required():
            self._validate_float16_required()
        self._disable_per_channel = disable_per_channel
        self._enable_new_dynamic_range_quantizer = experimental_new_dynamic_range_quantizer
        self._experimental_low_bit_qat = experimental_low_bit_qat
        self._full_integer_quantization_bias_type = full_integer_quantization_bias_type
        self._validate_full_integer_quantization_bias_type()

    def is_post_training_int8_only_quantization(self):
        if False:
            print('Hello World!')
        return self.is_any_optimization_enabled() and self._representative_dataset is not None and (not self._is_int16x8_target_required()) and (not self.is_allow_float()) and self._is_int8_target_required()

    def is_post_training_int8_quantization_with_float_fallback(self):
        if False:
            for i in range(10):
                print('nop')
        return self.is_any_optimization_enabled() and self._representative_dataset is not None and (not self._is_int16x8_target_required()) and self.is_allow_float() and (self._smallest_supported_type() == _dtypes.int8)

    def is_post_training_int8_quantization(self):
        if False:
            print('Hello World!')
        return self.is_post_training_int8_only_quantization() or self.is_post_training_int8_quantization_with_float_fallback()

    def is_post_training_int16x8_only_quantization(self):
        if False:
            for i in range(10):
                print('nop')
        return self.is_any_optimization_enabled() and self._representative_dataset is not None and self._is_int16x8_target_required() and (not self.is_allow_float())

    def is_post_training_int16x8_quantization_with_float_fallback(self):
        if False:
            return 10
        return self.is_any_optimization_enabled() and self._representative_dataset is not None and self._is_int16x8_target_required() and self.is_allow_float()

    def is_post_training_int16x8_quantization(self):
        if False:
            while True:
                i = 10
        return self.is_post_training_int16x8_only_quantization() or self.is_post_training_int16x8_quantization_with_float_fallback()

    def is_post_training_integer_quantization(self):
        if False:
            for i in range(10):
                print('nop')
        return self.is_post_training_int8_quantization() or self.is_post_training_int16x8_quantization()

    def is_low_bit_quantize_aware_training(self):
        if False:
            return 10
        return self.is_any_optimization_enabled() and self.is_quantization_aware_trained_model() and self._experimental_low_bit_qat

    def is_quantization_aware_training(self):
        if False:
            print('Hello World!')
        return self.is_any_optimization_enabled() and self.is_quantization_aware_trained_model() and (not self.is_low_bit_quantize_aware_training())

    def is_integer_quantization(self):
        if False:
            print('Hello World!')
        return self.is_post_training_integer_quantization() or self.is_quantization_aware_training() or self.is_low_bit_quantize_aware_training()

    def is_post_training_dynamic_range_quantization(self):
        if False:
            for i in range(10):
                print('nop')
        return self.is_any_optimization_enabled() and self._representative_dataset is None and (not self.is_quantization_aware_trained_model()) and (self._smallest_supported_type() == _dtypes.int8)

    def is_post_training_float16_quantization(self):
        if False:
            i = 10
            return i + 15
        return self.is_any_optimization_enabled() and self._smallest_supported_type().size == 2 and (_dtypes.float16 in self._target_spec.supported_types)

    def is_bfloat16_quantization(self):
        if False:
            i = 10
            return i + 15
        return self.is_any_optimization_enabled() and self._smallest_supported_type().size == 2 and (_dtypes.bfloat16 in self._target_spec.supported_types)

    def activations_type(self):
        if False:
            i = 10
            return i + 15
        if self.is_integer_quantization():
            if self._is_int16x8_target_required():
                return _dtypes.int16
            else:
                return _dtypes.int8
        else:
            return _dtypes.float32

    def bias_type(self):
        if False:
            return 10
        if self._full_integer_quantization_bias_type:
            return self._full_integer_quantization_bias_type
        if self.activations_type() == _dtypes.int16:
            return _dtypes.int64
        elif self.activations_type() == _dtypes.int8:
            return _dtypes.int32
        else:
            return _dtypes.float32

    def converter_flags(self, inference_ty=None, inference_input_ty=None):
        if False:
            i = 10
            return i + 15
        'Flags to the converter.'
        if self.is_integer_quantization():
            is_low_bit_qat = self.is_low_bit_quantize_aware_training()
            return {'inference_type': inference_ty if inference_ty is not None else self.activations_type(), 'inference_input_type': _dtypes.float32, 'post_training_quantize': False, 'quantize_to_float16': False, 'disable_infer_tensor_range': is_low_bit_qat, 'use_fake_quant_num_bits': is_low_bit_qat, 'enable_mlir_variable_quantization': self.enable_mlir_variable_quantization}
        elif self.is_post_training_dynamic_range_quantization():
            return {'inference_type': _dtypes.float32, 'inference_input_type': _dtypes.float32, 'post_training_quantize': True, 'quantize_to_float16': False, 'disable_per_channel_quantization': self._disable_per_channel, 'enable_mlir_dynamic_range_quantizer': self._enable_new_dynamic_range_quantizer, 'enable_mlir_variable_quantization': self.enable_mlir_variable_quantization}
        elif self.is_post_training_float16_quantization():
            return {'inference_type': _dtypes.float32, 'inference_input_type': _dtypes.float32, 'post_training_quantize': True, 'quantize_to_float16': True, 'accumulation_type': self._target_spec._experimental_supported_accumulation_type, 'allow_bfloat16': self.is_bfloat16_quantization(), 'enable_mlir_dynamic_range_quantizer': self._enable_new_dynamic_range_quantizer, 'enable_mlir_variable_quantization': self.enable_mlir_variable_quantization}
        else:
            return {'inference_type': inference_ty if inference_ty is not None else _dtypes.float32, 'inference_input_type': inference_input_ty, 'post_training_quantize': False, 'quantize_to_float16': False, 'allow_bfloat16': self.is_bfloat16_quantization()}

    def _validate_int8_required(self):
        if False:
            i = 10
            return i + 15
        'Int8 mode requires certain parameters to exist and be compatible.'
        if set(self._target_spec.supported_ops) == {OpsSet.TFLITE_BUILTINS_INT8} and (not (set(self._target_spec.supported_types) == set() or set(self._target_spec.supported_types) == {_dtypes.int8})):
            raise ValueError('As full integer quantization has been enabled by setting `target_spec.supported_ops`={tf.lite.OpsSet.TFLITE_BUILTINS_INT8}, thus `target_spec.supported_types` should be left uninitizalized or set to {tf.int8}.')
        if set(self._target_spec.supported_types) == {_dtypes.int8}:
            self._target_spec.supported_ops = {OpsSet.TFLITE_BUILTINS_INT8}
        if not self._representative_dataset and (not self.is_quantization_aware_training()):
            raise ValueError('For full integer quantization, a `representative_dataset` must be specified.')
        if self._representative_dataset:
            if not isinstance(self._representative_dataset, RepresentativeDataset):
                self._representative_dataset = RepresentativeDataset(self._representative_dataset)

    def _validate_float16_required(self):
        if False:
            for i in range(10):
                print('nop')
        'Float16 mode requires certain parameters to exist and be compatible.'
        if self.enable_mlir_variable_quantization:
            raise ValueError('`_experimental_variable_quantization` is only supported for full integer quantization.')

    def _validate_full_integer_quantization_bias_type(self):
        if False:
            while True:
                i = 10
        'Validates bias type for full interger quantization.'
        bias_type = self._full_integer_quantization_bias_type
        if not bias_type:
            return
        if self.activations_type() == _dtypes.float32:
            raise ValueError('`full_integer_quantization_bias_type` is only supported for full integer quantization.')
        if self.activations_type() == _dtypes.int8 and bias_type != _dtypes.int32:
            raise ValueError(f'Expected bias type to be `dtypes.int32` for Int8Quant. Current setting bias type: {bias_type}')
        if self.activations_type() == _dtypes.int16 and bias_type != _dtypes.int32 and (bias_type != _dtypes.int64):
            raise ValueError(f'Expected bias type to be `dtypes.int32` or `dtypes.int64` for Int16Quant. Current setting bias type: {bias_type}')

    def _is_int8_target_required(self):
        if False:
            return 10
        return OpsSet.TFLITE_BUILTINS_INT8 in set(self._target_spec.supported_ops) or set(self._target_spec.supported_types) == set([_dtypes.int8])

    def _is_int16x8_target_required(self):
        if False:
            while True:
                i = 10
        return OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8 in set(self._target_spec.supported_ops)

    def is_allow_float(self):
        if False:
            while True:
                i = 10
        return OpsSet.TFLITE_BUILTINS in set(self._target_spec.supported_ops) or OpsSet.SELECT_TF_OPS in set(self._target_spec.supported_ops)

    def _is_float16_target_required(self):
        if False:
            i = 10
            return i + 15
        return _dtypes.float16 in self._target_spec.supported_types

    def is_any_optimization_enabled(self):
        if False:
            while True:
                i = 10
        return bool(set(self._optimizations).intersection([Optimize.OPTIMIZE_FOR_LATENCY, Optimize.OPTIMIZE_FOR_SIZE, Optimize.DEFAULT]))

    def _smallest_supported_type(self):
        if False:
            i = 10
            return i + 15
        if self._target_spec.supported_types:
            return min(self._target_spec.supported_types, key=lambda x: x.size)
        else:
            return _dtypes.int8

    def is_quantization_aware_trained_model(self):
        if False:
            for i in range(10):
                print('nop')
        'Checks if the graph contains any training-time quantization ops.'
        training_quant_ops = frozenset({'FakeQuantWithMinMaxVars', 'FakeQuantWithMinMaxVarsPerChannel', 'FakeQuantWithMinMaxArgs', 'QuantizeAndDequantizeV2', 'QuantizeAndDequantizeV3'})
        if self._graph_def:
            for node_def in self._graph_def.node:
                if node_def.op in training_quant_ops:
                    return True
            for function in self._graph_def.library.function:
                for node_def in function.node_def:
                    if node_def.op in training_quant_ops:
                        return True
        return False

class TFLiteConverterBase:
    """Converter superclass to share functionality between V1 and V2 converters."""
    _original_model_type = conversion_metdata_fb.ModelType.NONE

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.optimizations = set()
        self.representative_dataset = None
        self.target_spec = TargetSpec()
        self.allow_custom_ops = False
        self.experimental_new_converter = True
        self.experimental_new_quantizer = True
        self.experimental_enable_resource_variables = True
        self._experimental_calibrate_only = False
        self._experimental_sparsify_model = False
        self._experimental_disable_per_channel = False
        self._debug_info = None
        self.saved_model_dir = None
        self._saved_model_tags = None
        self._saved_model_version = 0
        self._saved_model_exported_names = []
        self._tflite_metrics = metrics.TFLiteConverterMetrics()
        self._collected_converter_params = {}
        self.unfold_batchmatmul = False
        self.legalize_custom_tensor_list_ops = False
        self._experimental_lower_tensor_list_ops = True
        self._experimental_default_to_single_batch_in_tensor_list_ops = False
        self._experimental_unfold_large_splat_constant = False
        self._experimental_tf_quantization_mode = None
        self._experimental_full_integer_quantization_bias_type = None
        self._experimental_quantization_options = None
        self.exclude_conversion_metadata = False
        self._metadata = conversion_metdata_fb.ConversionMetadataT()
        self._metadata.environment = conversion_metdata_fb.EnvironmentT()
        self._metadata.options = conversion_metdata_fb.ConversionOptionsT()
        self._metadata.environment.tensorflowVersion = versions.__version__
        self._metadata.environment.modelType = self._get_original_model_type()
        self._experimental_enable_dynamic_update_slice = False
        self._experimental_preserve_assert_op = False
        self._experimental_guarantee_all_funcs_one_use = False
        self.experimental_new_dynamic_range_quantizer = True
        self._experimental_low_bit_qat = False
        self._experimental_allow_all_select_tf_ops = False
        self._experimental_variable_quantization = False
        self._experimental_disable_fuse_mul_and_fc = False
        self._experimental_use_buffer_offset = False
        self._experimental_reduce_type_precision = False
        self.mlir_dump_dir = None
        self.mlir_dump_pass_regex = None
        self.mlir_dump_func_regex = None
        self.mlir_enable_timing = None
        self.mlir_print_ir_before = None
        self.mlir_print_ir_after = None
        self.mlir_print_ir_module_scope = None
        self.mlir_elide_elementsattrs_if_larger = None

    def _grappler_config(self, optimizers=None):
        if False:
            i = 10
            return i + 15
        'Creates a tf.compat.v1.ConfigProto for configuring Grappler.\n\n    Args:\n      optimizers: List of strings that represents the list of optimizers.\n\n    Returns:\n      tf.ConfigProto.\n    '
        if not optimizers:
            optimizers = []
        if not self.experimental_new_converter:
            optimizers.append('constfold')
        is_only_flex_enabled = set([OpsSet.SELECT_TF_OPS]) == set(self.target_spec.supported_ops)
        if is_only_flex_enabled:
            optimizers.append('layout')
        return _get_grappler_config(optimizers)

    def _quantize(self, result, input_type, output_type, activations_type, bias_type, allow_float, enable_variable_quantization):
        if False:
            while True:
                i = 10
        'Quantize the model.'
        custom_op_registerers_by_name = [x for x in self.target_spec._experimental_custom_op_registerers if isinstance(x, str)]
        custom_op_registerers_by_func = [x for x in self.target_spec._experimental_custom_op_registerers if not isinstance(x, str)]
        if not isinstance(self.representative_dataset, RepresentativeDataset):
            self.representative_dataset = RepresentativeDataset(self.representative_dataset)
        result = _calibrator.add_intermediate_tensors(result)
        calibrate_quantize = _calibrator.Calibrator(result, custom_op_registerers_by_name, custom_op_registerers_by_func)
        if self._experimental_calibrate_only or self.experimental_new_quantizer:
            calibrated = calibrate_quantize.calibrate(self.representative_dataset.input_gen)
        if self._experimental_calibrate_only:
            return calibrated
        elif self.experimental_new_quantizer and activations_type != _dtypes.int16:
            return _mlir_quantize(calibrated, self._experimental_disable_per_channel, input_data_type=input_type, output_data_type=output_type, enable_variable_quantization=enable_variable_quantization)
        else:
            return calibrate_quantize.calibrate_and_quantize(self.representative_dataset.input_gen, input_type, output_type, allow_float, activations_type, bias_type, disable_per_channel=self._experimental_disable_per_channel)

    def _is_unknown_shapes_allowed(self):
        if False:
            while True:
                i = 10
        return self.experimental_new_converter

    def _get_base_converter_args(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the base converter args.\n\n    Returns:\n      {key str: val}\n    '
        args = {'input_format': constants.TENSORFLOW_GRAPHDEF, 'allow_custom_ops': self.allow_custom_ops, 'debug_info': self._debug_info, 'target_ops': self.target_spec.supported_ops, 'enable_mlir_converter': self.experimental_new_converter, 'select_user_tf_ops': self.target_spec.experimental_select_user_tf_ops, 'supported_backends': self.target_spec.experimental_supported_backends, 'unfold_batchmatmul': self.unfold_batchmatmul, 'legalize_custom_tensor_list_ops': self.legalize_custom_tensor_list_ops, 'lower_tensor_list_ops': self._experimental_lower_tensor_list_ops, 'unfold_large_splat_constant': self._experimental_unfold_large_splat_constant, 'default_to_single_batch_in_tensor_list_ops': self._experimental_default_to_single_batch_in_tensor_list_ops, 'tf_quantization_mode': self._experimental_tf_quantization_mode, 'experimental_enable_resource_variables': self.experimental_enable_resource_variables, 'enable_dynamic_update_slice': self._experimental_enable_dynamic_update_slice, 'preserve_assert_op': self._experimental_preserve_assert_op, 'guarantee_all_funcs_one_use': self._experimental_guarantee_all_funcs_one_use, 'allow_all_select_tf_ops': self._experimental_allow_all_select_tf_ops, 'disable_fuse_mul_and_fc': self._experimental_disable_fuse_mul_and_fc, 'quantization_options': self._experimental_quantization_options, 'mlir_dump_dir': self.mlir_dump_dir, 'mlir_dump_pass_regex': self.mlir_dump_pass_regex, 'mlir_dump_func_regex': self.mlir_dump_func_regex, 'mlir_enable_timing': self.mlir_enable_timing, 'mlir_print_ir_before': self.mlir_print_ir_before, 'mlir_print_ir_after': self.mlir_print_ir_after, 'mlir_print_ir_module_scope': self.mlir_print_ir_module_scope, 'mlir_elide_elementsattrs_if_larger': self.mlir_elide_elementsattrs_if_larger, 'use_buffer_offset': self._experimental_use_buffer_offset, 'reduce_type_precision': self._experimental_reduce_type_precision}
        if self.saved_model_dir:
            args.update({'saved_model_dir': self.saved_model_dir, 'saved_model_version': self._saved_model_version, 'saved_model_tags': self._saved_model_tags, 'saved_model_exported_names': self._saved_model_exported_names})
        if self._experimental_quantization_options:
            logging.warning('Configs from custom methods in experimental_quantization_options may not produce a valid tflite model. Note that currently this option only supports StableHLO path. Setting this option in TFLite path will be a no-op.')
        return args

    def _contains_function_with_implements_attr(self, saved_model_proto):
        if False:
            while True:
                i = 10
        meta_graph = saved_model_proto.meta_graphs[0]
        for function in meta_graph.graph_def.library.function:
            if function.attr.get('_implements', None) or function.attr.get('api_implements', None):
                return True
        return False

    def _parse_saved_model_args(self, always_enable_saved_model_import=False):
        if False:
            print('Hello World!')
        'Parses SavedModel arguments from the given Keras/RNN SavedModel.\n\n    Args:\n      always_enable_saved_model_import: Bool. When the value is true, it enables\n        MLIR saved model import path regardless of checking the conditions.\n    '
        if not self.experimental_new_converter:
            self.saved_model_dir = None
            return
        if self.saved_model_dir:
            try:
                (saved_model_proto, _) = _parse_saved_model_with_debug_info(self.saved_model_dir)
            except OSError:
                self.saved_model_dir = None
                return
            if not always_enable_saved_model_import and (not self._contains_function_with_implements_attr(saved_model_proto)):
                self.saved_model_dir = None
                return
            if not self._saved_model_exported_names:
                self._saved_model_exported_names = []
            self._saved_model_version = saved_model_proto.saved_model_schema_version
            if self._saved_model_version == 0:
                self.saved_model_dir = None
                logging.warning('SavedModel schema version is zero.')
                return
            if self._saved_model_version not in [1, 2]:
                raise ValueError('SavedModel file format({0}) is not supported'.format(self._saved_model_version))

    def _sparsify_model(self):
        if False:
            for i in range(10):
                print('nop')
        return Optimize.EXPERIMENTAL_SPARSITY in self.optimizations

    def _increase_conversion_attempt_metric(self):
        if False:
            for i in range(10):
                print('nop')
        self._tflite_metrics.increase_counter_converter_attempt()

    def _increase_conversion_success_metric(self):
        if False:
            for i in range(10):
                print('nop')
        self._tflite_metrics.increase_counter_converter_success()

    @classmethod
    def _set_original_model_type(cls, model_type):
        if False:
            while True:
                i = 10
        'Stores the original model type.'
        if model_type == conversion_metdata_fb.ModelType.NONE:
            raise ValueError('The original model type should be specified.')
        cls._original_model_type = model_type

    def _get_original_model_type(self):
        if False:
            while True:
                i = 10
        'One-time getter to return original model type and set it to NONE.'
        model_type = TFLiteConverterBase._original_model_type
        TFLiteConverterBase._original_model_type = conversion_metdata_fb.ModelType.NONE
        return model_type

    def _save_conversion_params_metric(self, graph_def=None, inference_type=None, inference_input_type=None):
        if False:
            return 10
        'Set conversion parameter metrics.'
        converter_kwargs = self._collected_converter_params
        converter_kwargs.update(self._get_base_converter_args())
        quant_mode = QuantizationMode(self.optimizations, self.target_spec, self.representative_dataset, graph_def, self._experimental_disable_per_channel, self.experimental_new_dynamic_range_quantizer, self._experimental_low_bit_qat, self._experimental_full_integer_quantization_bias_type, self._experimental_variable_quantization)
        converter_kwargs.update({'tf_version': self._metadata.environment.tensorflowVersion, 'api_version': self._metadata.environment.apiVersion, 'original_model_format': self._metadata.environment.modelType, 'optimization_default': quant_mode.is_any_optimization_enabled(), 'optimization_post_training_dynamic_range': quant_mode.is_post_training_dynamic_range_quantization(), 'optimization_post_training_float16': quant_mode.is_post_training_float16_quantization(), 'optimization_post_training_integer_quantize': quant_mode.is_post_training_integer_quantization(), 'optimization_qat': quant_mode.is_quantization_aware_training(), 'optimization_low_bit_qat': quant_mode.is_low_bit_quantize_aware_training(), 'optimization_sparsify': self._sparsify_model(), 'activations_type': quant_mode.activations_type()})
        converter_kwargs.update(quant_mode.converter_flags(inference_type, inference_input_type))
        if self.target_spec._experimental_supported_accumulation_type:
            converter_kwargs.update({'accumulation_type': self.target_spec._experimental_supported_accumulation_type})

        def format_element(elem):
            if False:
                return 10
            if isinstance(elem, enum.Enum):
                return str(elem.value)
            return pprint.pformat(elem)

        def format_param(param):
            if False:
                print('Hello World!')
            if isinstance(param, (list, tuple, set)):
                if not param:
                    return 'None'
                string_list = [format_element(x) for x in param]
                return ','.join(sorted(string_list))
            return format_element(param)
        for (key, value) in converter_kwargs.items():
            self._tflite_metrics.set_converter_param(key, format_param(value))
        self._tflite_metrics.set_export_required()
        self._metadata.options.allowCustomOps = self.allow_custom_ops
        self._metadata.options.enableSelectTfOps = OpsSet.SELECT_TF_OPS in self.target_spec.supported_ops
        self._metadata.options.forceSelectTfOps = set([OpsSet.SELECT_TF_OPS]) == set(self.target_spec.supported_ops)
        self._metadata.options.modelOptimizationModes = []
        if quant_mode.is_post_training_float16_quantization():
            self._metadata.options.modelOptimizationModes.append(conversion_metdata_fb.ModelOptimizationMode.PTQ_FLOAT16)
        if quant_mode.is_post_training_dynamic_range_quantization():
            self._metadata.options.modelOptimizationModes.append(conversion_metdata_fb.ModelOptimizationMode.PTQ_DYNAMIC_RANGE)
        if quant_mode.is_post_training_int8_quantization():
            self._metadata.options.modelOptimizationModes.append(conversion_metdata_fb.ModelOptimizationMode.PTQ_FULL_INTEGER)
        if quant_mode.is_post_training_int16x8_quantization():
            self._metadata.options.modelOptimizationModes.append(conversion_metdata_fb.ModelOptimizationMode.PTQ_INT16)
        if quant_mode.is_quantization_aware_training():
            self._metadata.options.modelOptimizationModes.append(conversion_metdata_fb.ModelOptimizationMode.QUANTIZATION_AWARE_TRAINING)

    def _set_conversion_latency_metric(self, value):
        if False:
            while True:
                i = 10
        self._tflite_metrics.set_converter_latency(value)

    @convert_phase(Component.OPTIMIZE_TFLITE_MODEL)
    def _optimize_tflite_model(self, model, quant_mode, quant_io=True):
        if False:
            for i in range(10):
                print('nop')
        'Apply optimizations on a TFLite model.'
        if quant_mode.is_integer_quantization():
            (in_type, out_type) = (self.inference_input_type, self.inference_output_type)
            if quant_mode.is_post_training_integer_quantization():
                q_in_type = in_type if in_type and quant_io else _dtypes.float32
                q_out_type = out_type if out_type and quant_io else _dtypes.float32
                q_activations_type = quant_mode.activations_type()
                q_bias_type = quant_mode.bias_type()
                q_allow_float = quant_mode.is_allow_float()
                q_variable_quantization = quant_mode.enable_mlir_variable_quantization
                model = self._quantize(model, q_in_type, q_out_type, q_activations_type, q_bias_type, q_allow_float, q_variable_quantization)
            m_in_type = in_type if in_type else _dtypes.float32
            m_out_type = out_type if out_type else _dtypes.float32
            if not (quant_mode.is_post_training_integer_quantization() and self.experimental_new_quantizer and quant_io and (m_in_type in [_dtypes.int8, _dtypes.uint8, _dtypes.float32]) and (m_out_type in [_dtypes.int8, _dtypes.uint8, _dtypes.float32])):
                model = _modify_model_io_type(model, m_in_type, m_out_type)
        if self._sparsify_model():
            model = _mlir_sparsify(model)
        if not self._experimental_use_buffer_offset:
            try:
                model_object = flatbuffer_utils.convert_bytearray_to_object(model)
                if _check_model_use_buffer_offset(model_object):
                    return model
                model = _deduplicate_readonly_buffers(model)
            except Exception:
                logging.warning('Buffer deduplication procedure will be skipped when flatbuffer library is not properly loaded')
        return model

    def _convert_and_export_metrics(self, convert_func, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Wraps around convert function to export metrics.\n\n    Args:\n      convert_func: The convert function to wrap.\n      *args: Positional arguments of the convert function.\n      **kwargs: The keyword arguments of the convert function.\n\n    Returns:\n      The decorator to wrap the convert function.\n    '
        self._increase_conversion_attempt_metric()
        self._save_conversion_params_metric()
        start_time = time.process_time()
        result = convert_func(self, *args, **kwargs)
        elapsed_time_ms = (time.process_time() - start_time) * 1000
        if result:
            self._increase_conversion_success_metric()
        self._set_conversion_latency_metric(round(elapsed_time_ms))
        self._tflite_metrics.export_metrics()
        if self.exclude_conversion_metadata or self._experimental_use_buffer_offset:
            return result
        model_object = flatbuffer_utils.convert_bytearray_to_object(result)
        if _check_model_use_buffer_offset(model_object):
            return result
        sparsity_modes = _get_sparsity_modes(model_object)
        self._metadata.options.modelOptimizationModes.extend(sparsity_modes)
        model_object = _populate_conversion_metadata(model_object, self._metadata)
        return flatbuffer_utils.convert_object_to_bytearray(model_object)

def _check_model_use_buffer_offset(model_object):
    if False:
        return 10
    'Checks if a model object uses buffer offsets to store constant buffers.\n\n  Args:\n    model_object: tflite model, a python object\n\n  Returns:\n    True of the model_object has the metadata entry "buffer_location"\n    False otherwise\n  '
    if not model_object.metadata:
        return False
    for meta in model_object.metadata:
        if meta.name.decode('utf-8') == 'buffer_location':
            return True
    return False

def _export_metrics(convert_func):
    if False:
        for i in range(10):
            print('nop')
    'The decorator around convert function to export metrics.'

    @functools.wraps(convert_func)
    def wrapper(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        return self._convert_and_export_metrics(convert_func, *args, **kwargs)
    return wrapper

class TFLiteConverterBaseV2(TFLiteConverterBase):
    """Converter subclass to share functionality between V2 converters."""

    def __init__(self):
        if False:
            return 10
        'Constructor for TFLiteConverter.'
        super(TFLiteConverterBaseV2, self).__init__()
        self.inference_input_type = _dtypes.float32
        self.inference_output_type = _dtypes.float32
        self._metadata.environment.apiVersion = 2

    def _validate_inference_input_output_types(self, quant_mode):
        if False:
            for i in range(10):
                print('nop')
        'Validate inference_input_type and inference_output_type flags.'
        default_types = [_dtypes.float32]
        if quant_mode.is_integer_quantization():
            if quant_mode.is_post_training_int16x8_quantization():
                all_types = default_types + [_dtypes.int16]
            else:
                all_types = default_types + [_dtypes.int8, _dtypes.uint8, _dtypes.int16]
            if self.inference_input_type not in all_types or self.inference_output_type not in all_types:
                all_types_names = ['tf.' + t.name for t in all_types]
                raise ValueError('The inference_input_type and inference_output_type must be in {}.'.format(all_types_names))
        elif self.inference_input_type not in default_types or self.inference_output_type not in default_types:
            raise ValueError('The inference_input_type and inference_output_type must be tf.float32.')

    @convert_phase(Component.PREPARE_TF_MODEL, SubComponent.LOAD_SAVED_MODEL)
    def _load_saved_model(self, saved_model_dir, saved_model_tags):
        if False:
            for i in range(10):
                print('nop')
        'Load graph_def from saved model with the default serving signature key.\n\n    Args:\n      saved_model_dir: Directory of the SavedModel.\n      saved_model_tags: Set of tags identifying the MetaGraphDef within the\n        SavedModel to analyze.\n\n    Returns:\n      graph_def: The loaded GraphDef.\n      input_tensors: List of input tensors.\n      output_tensors: List of output tensors.\n    '
        graph = _ops.Graph()
        saved_model = _loader_impl.SavedModelLoader(saved_model_dir)
        saved_model.load_graph(graph, tags=saved_model_tags)
        meta_graph = saved_model.get_meta_graph_def_from_tags(saved_model_tags)
        graph_def = meta_graph.graph_def
        signature_def = meta_graph.signature_def[_signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        input_tensors = [graph.get_tensor_by_name(signature_def.inputs[key].name) for key in signature_def.inputs]
        output_tensors = [graph.get_tensor_by_name(signature_def.outputs[key].name) for key in signature_def.outputs]
        return (graph_def, input_tensors, output_tensors)

    @convert_phase(Component.PREPARE_TF_MODEL, SubComponent.VALIDATE_INPUTS)
    def _validate_inputs(self, graph_def, input_tensors):
        if False:
            return 10
        'Validate the input parameters.\n\n    Args:\n      graph_def: The TensorFlow GraphDef.\n      input_tensors: List of input tensors.\n\n    Raise:\n      ValueError: Input shape is not specified. Invalid quantization parameters.\n    '
        self._save_conversion_params_metric(graph_def)
        self._quant_mode = QuantizationMode(self.optimizations, self.target_spec, self.representative_dataset, graph_def, self._experimental_disable_per_channel, self.experimental_new_dynamic_range_quantizer, self._experimental_low_bit_qat, self._experimental_full_integer_quantization_bias_type, self._experimental_variable_quantization)
        self._validate_inference_input_output_types(self._quant_mode)
        if not self._is_unknown_shapes_allowed():
            for tensor in input_tensors:
                shape_list = tensor.shape.as_list()
                if None in shape_list[1:]:
                    raise ValueError("None is only supported in the 1st dimension. Tensor '{0}' has invalid shape '{1}'.".format(_get_tensor_name(tensor), shape_list))
                elif shape_list and shape_list[0] is None:
                    shape = tensor.shape.as_list()
                    shape[0] = 1
                    tensor.set_shape(shape)
        if self._trackable_obj is None or not hasattr(self._trackable_obj, 'graph_debug_info'):
            self._debug_info = _get_debug_info(_build_debug_info_func(self._funcs[0].graph), graph_def)
        else:
            self._debug_info = _get_debug_info(_convert_debug_info_func(self._trackable_obj.graph_debug_info), graph_def)

    @convert_phase(Component.PREPARE_TF_MODEL, SubComponent.OPTIMIZE_TF_MODEL)
    def _optimize_tf_model(self, graph_def, input_tensors, output_tensors, frozen_func):
        if False:
            for i in range(10):
                print('nop')
        'Run a Grappler pass to optimize the TensorFlow graph.\n\n    Args:\n      graph_def: Frozen GraphDef to be optimized.\n      input_tensors: List of input tensors.\n      output_tensors: List of output tensors.\n      frozen_func: TensorFlow Graph.\n\n    Returns:\n      The optimized TensorFlow graph.\n    '
        grappler_config = self._grappler_config()
        if grappler_config.graph_options.rewrite_options.optimizers:
            graph_def = _run_graph_optimizations(graph_def, input_tensors, output_tensors, config=grappler_config, graph=frozen_func.graph)
        return graph_def

    def _convert_from_saved_model(self, graph_def):
        if False:
            while True:
                i = 10
        'Helper method that converts saved model.\n\n    Args:\n      graph_def: GraphDef object for the model, used only for stats.\n\n    Returns:\n      The converted TFLite model.\n    '
        self._save_conversion_params_metric(graph_def)
        quant_mode = QuantizationMode(self.optimizations, self.target_spec, self.representative_dataset, graph_def, self._experimental_disable_per_channel, self.experimental_new_dynamic_range_quantizer, self._experimental_low_bit_qat, self._experimental_full_integer_quantization_bias_type, self._experimental_variable_quantization)
        self._validate_inference_input_output_types(quant_mode)
        converter_kwargs = {'enable_tflite_resource_variables': self.experimental_enable_resource_variables}
        converter_kwargs.update(self._get_base_converter_args())
        converter_kwargs.update(quant_mode.converter_flags())
        result = _convert_saved_model(**converter_kwargs)
        return self._optimize_tflite_model(result, quant_mode, quant_io=self.experimental_new_quantizer)

    def convert(self, graph_def, input_tensors, output_tensors):
        if False:
            return 10
        'Converts a TensorFlow GraphDef based on instance variables.\n\n    Args:\n      graph_def: Frozen TensorFlow GraphDef.\n      input_tensors: List of input tensors.\n      output_tensors: List of output tensors.\n\n    Returns:\n      The converted data in serialized format.\n\n    Raises:\n      ValueError:\n        No concrete functions is specified.\n        Multiple concrete functions are specified.\n        Input shape is not specified.\n        Invalid quantization parameters.\n    '
        self._validate_inputs(graph_def, input_tensors)
        converter_kwargs = self._get_base_converter_args()
        converter_kwargs.update(self._quant_mode.converter_flags())
        if not self.experimental_new_converter:
            logging.warning('Please consider switching to the new converter by setting experimental_new_converter=True. The old converter is deprecated.')
        else:
            logging.info('Using new converter: If you encounter a problem please file a bug. You can opt-out by setting experimental_new_converter=False')
        result = _convert_graphdef(input_data=graph_def, input_tensors=input_tensors, output_tensors=output_tensors, **converter_kwargs)
        return self._optimize_tflite_model(result, self._quant_mode, quant_io=self.experimental_new_quantizer)

class TFLiteSavedModelConverterV2(TFLiteConverterBaseV2):
    """Converts the given SavedModel into TensorFlow Lite model.

  Attributes:
      saved_model_dir: Directory of the SavedModel.
  """

    def __init__(self, saved_model_dir, saved_model_tags=None, saved_model_exported_names=None, trackable_obj=None):
        if False:
            return 10
        'Constructor for TFLiteConverter.\n\n    Args:\n      saved_model_dir: Directory of the SavedModel.\n      saved_model_tags: Set of tags identifying the MetaGraphDef within the\n        SavedModel to analyze. All tags in the tag set must be present. (default\n        {tf.saved_model.SERVING}).\n      saved_model_exported_names: Names to be exported when the saved model\n        import path is on.\n      trackable_obj: tf.AutoTrackable object associated with `funcs`. A\n        reference to this object needs to be maintained so that Variables do not\n        get garbage collected since functions have a weak reference to\n        Variables. This is only required when the tf.AutoTrackable object is not\n        maintained by the user (e.g. `from_saved_model`).\n    '
        super(TFLiteSavedModelConverterV2, self).__init__()
        self.saved_model_dir = saved_model_dir
        self._saved_model_tags = saved_model_tags
        self._saved_model_exported_names = saved_model_exported_names
        self._trackable_obj = trackable_obj
        self._parse_saved_model_args(always_enable_saved_model_import=True)

    @_export_metrics
    def convert(self):
        if False:
            i = 10
            return i + 15
        'Converts a TensorFlow GraphDef based on instance variables.\n\n    Returns:\n      The converted data in serialized format.\n\n    Raises:\n      ValueError:\n        No concrete functions is specified.\n        Multiple concrete functions are specified.\n        Input shape is not specified.\n        Invalid quantization parameters.\n    '
        (graph_def, input_tensors, output_tensors) = self._load_saved_model(self.saved_model_dir, self._saved_model_tags)
        if self.saved_model_dir is None or not self.experimental_new_converter:
            (graph_def, _, _, _) = _freeze_saved_model(self.saved_model_dir, None, None, None, self._saved_model_tags, _signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY)
            self.saved_model_dir = None
            return super(TFLiteSavedModelConverterV2, self).convert(graph_def, input_tensors, output_tensors)
        if self._trackable_obj is None:
            self._debug_info = _get_debug_info(_build_debug_info_func(self._funcs[0].graph), graph_def)
        else:
            self._debug_info = _get_debug_info(_convert_debug_info_func(self._trackable_obj.graph_debug_info), graph_def)
        return self._convert_from_saved_model(graph_def)

class TFLiteKerasModelConverterV2(TFLiteConverterBaseV2):
    """Converts the given Keras model into TensorFlow Lite model."""

    def __init__(self, keras_model, trackable_obj=None):
        if False:
            while True:
                i = 10
        'Constructor for TFLiteConverter.\n\n    Args:\n      keras_model: tf.Keras.Model.\n      trackable_obj: tf.AutoTrackable object associated with `funcs`. A\n        reference to this object needs to be maintained so that Variables do not\n        get garbage collected since functions have a weak reference to\n        Variables. This is only required when the tf.AutoTrackable object is not\n        maintained by the user (e.g. `from_saved_model`).\n    '
        super(TFLiteKerasModelConverterV2, self).__init__()
        self._keras_model = keras_model
        self._trackable_obj = trackable_obj
        self.experimental_lower_to_saved_model = True

    @convert_phase(Component.PREPARE_TF_MODEL, SubComponent.CONVERT_KERAS_TO_SAVED_MODEL)
    def _convert_keras_to_saved_model(self, output_dir):
        if False:
            print('Hello World!')
        'Save Keras model to the SavedModel format.\n\n    Args:\n      output_dir: The output directory to save the SavedModel.\n\n    Returns:\n      graph_def: The frozen GraphDef.\n      input_tensors: List of input tensors.\n      output_tensors: List of output tensors.\n    '
        try:
            _save.save(self._keras_model, output_dir, options=_save_options.SaveOptions(save_debug_info=True))
        except Exception:
            return (None, None, None)
        self.saved_model_dir = output_dir
        self._saved_model_tags = set([_tag_constants.SERVING])
        self._saved_model_exported_names = [_signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        self._parse_saved_model_args(always_enable_saved_model_import=self.experimental_lower_to_saved_model)
        if self.saved_model_dir:
            (graph_def, input_tensors, output_tensors) = self._load_saved_model(self.saved_model_dir, self._saved_model_tags)
            self._trackable_obj = _load(self.saved_model_dir, self._saved_model_tags)
            return (graph_def, input_tensors, output_tensors)
        return (None, None, None)

    @convert_phase(Component.PREPARE_TF_MODEL, SubComponent.FREEZE_KERAS_MODEL)
    def _freeze_keras_model(self):
        if False:
            while True:
                i = 10
        'Freeze Keras model to frozen graph.\n\n    Returns:\n      graph_def: The frozen GraphDef.\n      input_tensors: List of input tensors.\n      output_tensors: List of output tensors.\n      frozen_func: The frozen ConcreteFunction.\n    '
        input_signature = None
        if not isinstance(self._keras_model.call, _def_function.Function):
            input_signature = _model_input_signature(self._keras_model, keep_original_batch_size=True)
        func = _trace_model_call(self._keras_model, input_signature)
        concrete_func = func.get_concrete_function()
        self._funcs = [concrete_func]
        (frozen_func, graph_def) = _convert_to_constants.convert_variables_to_constants_v2_as_graph(self._funcs[0], lower_control_flow=False)
        input_tensors = [tensor for tensor in frozen_func.inputs if tensor.dtype != _dtypes.resource]
        output_tensors = frozen_func.outputs
        return (graph_def, input_tensors, output_tensors, frozen_func)

    def _convert_as_saved_model(self):
        if False:
            print('Hello World!')
        'Converts a Keras model as a saved model.\n\n    Returns:\n      The converted data in serialized format.\n    '
        temp_dir = tempfile.mkdtemp()
        try:
            (graph_def, input_tensors, output_tensors) = self._convert_keras_to_saved_model(temp_dir)
            if self.saved_model_dir:
                return super(TFLiteKerasModelConverterV2, self).convert(graph_def, input_tensors, output_tensors)
        finally:
            shutil.rmtree(temp_dir, True)

    @_export_metrics
    def convert(self):
        if False:
            i = 10
            return i + 15
        'Converts a keras model based on instance variables.\n\n    Returns:\n      The converted data in serialized format.\n\n    Raises:\n      ValueError:\n        Multiple concrete functions are specified.\n        Input shape is not specified.\n        Invalid quantization parameters.\n    '
        saved_model_convert_result = self._convert_as_saved_model()
        if saved_model_convert_result:
            return saved_model_convert_result
        (graph_def, input_tensors, output_tensors, frozen_func) = self._freeze_keras_model()
        graph_def = self._optimize_tf_model(graph_def, input_tensors, output_tensors, frozen_func)
        return super(TFLiteKerasModelConverterV2, self).convert(graph_def, input_tensors, output_tensors)

class TFLiteFrozenGraphConverterV2(TFLiteConverterBaseV2):
    """Converts the given frozen graph into TensorFlow Lite model."""

    def __init__(self, funcs, trackable_obj=None):
        if False:
            i = 10
            return i + 15
        'Constructor for TFLiteConverter.\n\n    Args:\n      funcs: List of TensorFlow ConcreteFunctions. The list should not contain\n        duplicate elements.\n      trackable_obj: tf.AutoTrackable object associated with `funcs`. A\n        reference to this object needs to be maintained so that Variables do not\n        get garbage collected since functions have a weak reference to\n        Variables. This is only required when the tf.AutoTrackable object is not\n        maintained by the user (e.g. `from_saved_model`).\n    '
        super(TFLiteFrozenGraphConverterV2, self).__init__()
        self._funcs = funcs
        self._trackable_obj = trackable_obj
        self.experimental_lower_to_saved_model = True

    @convert_phase(Component.PREPARE_TF_MODEL, SubComponent.FREEZE_CONCRETE_FUNCTION)
    def _freeze_concrete_function(self):
        if False:
            i = 10
            return i + 15
        'Convert the given ConcreteFunction to frozen graph.\n\n    Returns:\n      graph_def: The frozen GraphDef.\n      input_tensors: List of input tensors.\n      output_tensors: List of output tensors.\n      frozen_func: The frozen ConcreteFunction.\n\n    Raises:\n      ValueError: none or multiple ConcreteFunctions provided.\n    '
        if len(self._funcs) == 0:
            raise ValueError('No ConcreteFunction is specified.')
        if len(self._funcs) > 1:
            raise ValueError('This converter can only convert a single ConcreteFunction. Converting multiple functions is under development.')
        (frozen_func, graph_def) = _convert_to_constants.convert_variables_to_constants_v2_as_graph(self._funcs[0], lower_control_flow=False)
        input_tensors = [tensor for tensor in frozen_func.inputs if tensor.dtype != _dtypes.resource]
        output_tensors = frozen_func.outputs
        return (graph_def, input_tensors, output_tensors, frozen_func)

    @convert_phase(Component.PREPARE_TF_MODEL, SubComponent.CONVERT_CONCRETE_FUNCTIONS_TO_SAVED_MODEL)
    def _convert_concrete_functions_to_saved_model(self, output_dir):
        if False:
            for i in range(10):
                print('nop')
        'Save concrete functions to the SavedModel format.\n\n    Args:\n      output_dir: The output directory to save the SavedModel.\n\n    Returns:\n      graph_def: The frozen GraphDef.\n      input_tensors: List of input tensors.\n      output_tensors: List of output tensors.\n    '
        if len(self._funcs) == 0:
            raise ValueError('No ConcreteFunction is specified.')
        if not self.experimental_lower_to_saved_model:
            return (None, None, None)
        if not self._trackable_obj or isinstance(self._trackable_obj, (_function.ConcreteFunction, _def_function.Function)):
            return (None, None, None)
        signatures = {}
        signature_keys = []
        try:
            if len(self._funcs) == 1:
                signatures[_signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = self._funcs[0]
                signature_keys = [_signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
            else:
                for func in self._funcs:
                    signatures[func.graph.name] = func
                    signature_keys.append(func.graph.name)
            _save.save(self._trackable_obj, output_dir, signatures=signatures, options=_save_options.SaveOptions(save_debug_info=True))
        except Exception:
            return (None, None, None)
        self.saved_model_dir = output_dir
        self._saved_model_tags = set([_tag_constants.SERVING])
        self._saved_model_exported_names = signature_keys
        self._parse_saved_model_args(always_enable_saved_model_import=True)
        if self.saved_model_dir:
            (graph_def, input_tensors, output_tensors) = self._load_saved_model(self.saved_model_dir, self._saved_model_tags)
            self._trackable_obj = _load(self.saved_model_dir, self._saved_model_tags)
            return (graph_def, input_tensors, output_tensors)
        return (None, None, None)

    def _convert_as_saved_model(self):
        if False:
            for i in range(10):
                print('nop')
        'Converts the given concrete functions as a saved model format.\n\n    Returns:\n      The converted data in serialized format.\n    '
        temp_dir = tempfile.mkdtemp()
        try:
            (graph_def, input_tensors, _) = self._convert_concrete_functions_to_saved_model(temp_dir)
            if self.saved_model_dir:
                self._validate_inputs(graph_def, input_tensors)
                return self._convert_from_saved_model(graph_def)
        finally:
            shutil.rmtree(temp_dir, True)
        return None

    @_export_metrics
    def convert(self):
        if False:
            i = 10
            return i + 15
        'Converts a TensorFlow GraphDef based on instance variables.\n\n    Returns:\n      The converted data in serialized format.\n\n    Raises:\n      ValueError:\n        No concrete functions is specified.\n        Multiple concrete functions are specified.\n        Input shape is not specified.\n        Invalid quantization parameters.\n    '
        if self.experimental_lower_to_saved_model:
            saved_model_convert_result = self._convert_as_saved_model()
            if saved_model_convert_result:
                return saved_model_convert_result
        (graph_def, input_tensors, output_tensors, frozen_func) = self._freeze_concrete_function()
        graph_def = self._optimize_tf_model(graph_def, input_tensors, output_tensors, frozen_func)
        return super(TFLiteFrozenGraphConverterV2, self).convert(graph_def, input_tensors, output_tensors)

class TFLiteJaxConverterV2(TFLiteConverterBaseV2):
    """Converts the given jax model into TensorFlow Lite model."""

    def __init__(self, serving_funcs, inputs):
        if False:
            for i in range(10):
                print('nop')
        'Constructor for TFLiteConverter.\n\n    Args:\n      serving_funcs: A list functions of the serving func of the jax module, the\n        model params should already be inlined. (e.g., `serving_func =\n        functools.partial(model, params=params)`)\n      inputs: Array of input tensor placeholders tuple,s like `jnp.zeros`. For\n        example, wrapped in an array like "[(\'input1\', input1), (\'input2\',\n        input2)]]".\n\n    Jax functions are polymorphic, for example:\n\n    ```python\n    def add(a, b):\n      return a + b\n    ```\n\n    Will yield different computations if different input signatures are passed\n    in: Pass `add(10.0, 20.0)` will yield a scalar `add` while pass\n    `add(np.random((100, 1)), np.random(100, 100))` will yield a broadcasting\n    add.  We will need the input information to do tracing for the converter\n    to properly convert the model. So it\'s important to pass in the desired\n    `input placeholders` with the correct input shape/type.\n\n    In the converted tflite model, the function name will be default to "main",\n    the output names will be the traced outputs. The output ordering shall\n    match the serving function.\n    '
        super(TFLiteJaxConverterV2, self).__init__()
        self._serving_funcs = serving_funcs
        self._inputs = inputs

    @_export_metrics
    def convert(self):
        if False:
            i = 10
            return i + 15
        'Converts a Jax serving func based on instance variables.\n\n    Returns:\n      The converted data in serialized format.\n\n    Raises:\n      ImportError:\n        If cannot import the xla_computation from jax.\n      ValueError:\n        No serving function is specified.\n        Input tensors are not specified.\n        The truth value of an array with more than one element is ambiguous.\n        Failed to convert the given Jax function to hlo.\n    '
        if not _xla_computation:
            raise ImportError('Cannot import xla_computation from jax.')
        if not self._serving_funcs:
            raise ValueError('No serving func is specified.')
        if not self._inputs:
            raise ValueError('Input tensors are not specified.')
        if len(self._inputs) != len(self._serving_funcs):
            msg = 'Input tensor mapping len {} does not match serving func len {}.'.format(len(self._inputs), len(self._serving_funcs))
            raise ValueError(msg)
        if not isinstance(self._inputs, (tuple, list)):
            raise ValueError('Input tensors should be pass in a tuple list wrapped in an array.')
        if len(self._serving_funcs) > 1:
            raise ValueError('Currently only support single serving function.')
        if not isinstance(self._inputs[0], (tuple, list)):
            raise ValueError('The input placeholders are not a dictionary.')
        input_names = []
        ordered_inputs = []
        for (input_name, tensor) in self._inputs[0]:
            input_names.append(input_name)
            ordered_inputs.append(tensor)
        try:
            xla_compuation = _xla_computation(self._serving_funcs[0], backend='cpu')
            hlo_proto = xla_compuation(*ordered_inputs).as_serialized_hlo_module_proto()
        except Exception:
            raise ValueError('Failed to convert the given Jax function to hlo.')
        converter_kwargs = {'input_content': hlo_proto, 'input_names': input_names, 'is_proto_format': True}
        converter_kwargs.update(self._get_base_converter_args())
        quant_mode = QuantizationMode(self.optimizations, self.target_spec, self.representative_dataset, None)
        self._validate_inference_input_output_types(quant_mode)
        converter_kwargs.update(quant_mode.converter_flags())
        result = _convert_jax_hlo(**converter_kwargs)
        return self._optimize_tflite_model(result, quant_mode, quant_io=self.experimental_new_quantizer)

@_tf_export('lite.TFLiteConverter', v1=[])
class TFLiteConverterV2(TFLiteFrozenGraphConverterV2):
    """Converts a TensorFlow model into TensorFlow Lite model.

  Attributes:
    optimizations: Experimental flag, subject to change. Set of optimizations to
      apply. e.g {tf.lite.Optimize.DEFAULT}. (default None, must be None or a
      set of values of type `tf.lite.Optimize`)
    representative_dataset: A generator function used for integer quantization
      where each generated sample has the same order, type and shape as the
      inputs to the model. Usually, this is a small subset of a few hundred
      samples randomly chosen, in no particular order, from the training or
      evaluation dataset. This is an optional attribute, but required for full
      integer quantization, i.e, if `tf.int8` is the only supported type in
      `target_spec.supported_types`. Refer to `tf.lite.RepresentativeDataset`.
      (default None)
    target_spec: Experimental flag, subject to change. Specifications of target
      device, including supported ops set, supported types and a set of user's
      defined TensorFlow operators required in the TensorFlow Lite runtime.
      Refer to `tf.lite.TargetSpec`.
    inference_input_type: Data type of the input layer. Note that integer types
      (tf.int8 and tf.uint8) are currently only supported for post training
      integer quantization and quantization aware training. (default tf.float32,
      must be in {tf.float32, tf.int8, tf.uint8})
    inference_output_type: Data type of the output layer. Note that integer
      types (tf.int8 and tf.uint8) are currently only supported for post
      training integer quantization and quantization aware training. (default
      tf.float32, must be in {tf.float32, tf.int8, tf.uint8})
    allow_custom_ops: Boolean indicating whether to allow custom operations.
      When False, any unknown operation is an error. When True, custom ops are
      created for any op that is unknown. The developer needs to provide these
      to the TensorFlow Lite runtime with a custom resolver. (default False)
    exclude_conversion_metadata: Whether not to embed the conversion metadata
      into the converted model. (default False)
    experimental_new_converter: Experimental flag, subject to change. Enables
      MLIR-based conversion. (default True)
    experimental_new_quantizer: Experimental flag, subject to change. Enables
      MLIR-based quantization conversion instead of Flatbuffer-based conversion.
      (default True)
    experimental_enable_resource_variables: Experimental flag, subject to
      change. Enables [resource
      variables](https://tensorflow.org/guide/migrate/tf1_vs_tf2#resourcevariables_instead_of_referencevariables)
      to be converted by this converter. This is only allowed if the
      from_saved_model interface is used. (default True)

  Example usage:

  ```python
  # Converting a SavedModel to a TensorFlow Lite model.
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  tflite_model = converter.convert()

  # Converting a tf.Keras model to a TensorFlow Lite model.
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()

  # Converting ConcreteFunctions to a TensorFlow Lite model.
  converter = tf.lite.TFLiteConverter.from_concrete_functions([func], model)
  tflite_model = converter.convert()

  # Converting a Jax model to a TensorFlow Lite model.
  converter = tf.lite.TFLiteConverter.experimental_from_jax(
      [func], [[ ('input1', input1), ('input2', input2)]])
  tflite_model = converter.convert()
  ```
  """

    def __init__(self, funcs, trackable_obj=None):
        if False:
            i = 10
            return i + 15
        'Constructor for TFLiteConverter.\n\n    Args:\n      funcs: List of TensorFlow ConcreteFunctions. The list should not contain\n        duplicate elements.\n      trackable_obj: tf.AutoTrackable object associated with `funcs`. A\n        reference to this object needs to be maintained so that Variables do not\n        get garbage collected since functions have a weak reference to\n        Variables. This is only required when the tf.AutoTrackable object is not\n        maintained by the user (e.g. `from_saved_model`).\n    '
        super(TFLiteConverterV2, self).__init__(funcs, trackable_obj)

    @classmethod
    def from_concrete_functions(cls, funcs, trackable_obj=None):
        if False:
            print('Hello World!')
        'Creates a TFLiteConverter object from ConcreteFunctions.\n\n    Args:\n      funcs: List of TensorFlow ConcreteFunctions. The list should not contain\n        duplicate elements. Currently converter can only convert a single\n        ConcreteFunction. Converting multiple functions is under development.\n      trackable_obj:   An `AutoTrackable` object (typically `tf.module`)\n        associated with `funcs`. A reference to this object needs to be\n        maintained so that Variables do not get garbage collected since\n        functions have a weak reference to Variables.\n\n    Returns:\n      TFLiteConverter object.\n\n    Raises:\n      Invalid input type.\n    '
        TFLiteConverterBase._set_original_model_type(conversion_metdata_fb.ModelType.TF_CONCRETE_FUNCTIONS)
        if trackable_obj is None:
            logging.warning('Please consider providing the trackable_obj argument in the from_concrete_functions. Providing without the trackable_obj argument is deprecated and it will use the deprecated conversion path.')
        for func in funcs:
            if not isinstance(func, _function.ConcreteFunction):
                message = 'This function takes in a list of ConcreteFunction.'
                if isinstance(func, _def_function.Function):
                    message += ' To get the ConcreteFunction from a Function, call get_concrete_function.'
                raise ValueError(message)
        return cls(funcs, trackable_obj)

    @classmethod
    def from_saved_model(cls, saved_model_dir, signature_keys=None, tags=None):
        if False:
            return 10
        "Creates a TFLiteConverter object from a SavedModel directory.\n\n    Args:\n      saved_model_dir: SavedModel directory to convert.\n      signature_keys: List of keys identifying SignatureDef containing inputs\n        and outputs. Elements should not be duplicated. By default the\n        `signatures` attribute of the MetaGraphdef is used. (default\n        saved_model.signatures)\n      tags: Set of tags identifying the MetaGraphDef within the SavedModel to\n        analyze. All tags in the tag set must be present. (default\n        {tf.saved_model.SERVING} or {'serve'})\n\n    Returns:\n      TFLiteConverter object.\n\n    Raises:\n      Invalid signature keys.\n    "
        TFLiteConverterBase._set_original_model_type(conversion_metdata_fb.ModelType.TF_SAVED_MODEL)
        if not context.executing_eagerly():
            signature_key = None
            if signature_keys:
                if len(signature_keys) != 1:
                    raise ValueError('Only support a single signature key.')
                else:
                    signature_key = signature_keys[0]
            logging.warning('Invoking the TF1 implementation of TFLiteConverter because eager is disabled. Consider enabling eager.')
            return TFLiteConverter.from_saved_model(saved_model_dir, signature_key=signature_key, tag_set=tags)
        if tags is None:
            tags = set([_tag_constants.SERVING])
        with context.eager_mode():
            saved_model = _load(saved_model_dir, tags)
        if not signature_keys:
            signature_keys = saved_model.signatures
        if not signature_keys:
            raise ValueError('Only support at least one signature key.')
        if len(signature_keys) > 1 and hasattr(saved_model, 'serve') and (not hasattr(saved_model, '_default_save_signature')):
            saved_model.serving_default = saved_model.serve
            delattr(saved_model, 'serve')
            signature_keys = ['serving_default']
        funcs = []
        for key in signature_keys:
            if key not in saved_model.signatures:
                raise ValueError("Invalid signature key '{}' found. Valid keys are '{}'.".format(key, ','.join(saved_model.signatures)))
            funcs.append(saved_model.signatures[key])
        saved_model_converter = TFLiteSavedModelConverterV2(saved_model_dir, tags, signature_keys, saved_model)
        if saved_model_converter.saved_model_dir:
            return saved_model_converter
        return cls(funcs, saved_model)

    @classmethod
    def from_keras_model(cls, model):
        if False:
            for i in range(10):
                print('nop')
        'Creates a TFLiteConverter object from a Keras model.\n\n    Args:\n      model: tf.Keras.Model\n\n    Returns:\n      TFLiteConverter object.\n    '
        TFLiteConverterBase._set_original_model_type(conversion_metdata_fb.ModelType.KERAS_MODEL)
        return TFLiteKerasModelConverterV2(model)

    @classmethod
    @_deprecation.deprecated(None, 'Use `jax2tf.convert` and (`lite.TFLiteConverter.from_saved_model` or `lite.TFLiteConverter.from_concrete_functions`) instead.')
    def experimental_from_jax(cls, serving_funcs, inputs):
        if False:
            while True:
                i = 10
        'Creates a TFLiteConverter object from a Jax model with its inputs.\n\n    Args:\n      serving_funcs: A array of Jax functions with all the weights applied\n        already.\n      inputs: A array of Jax input placeholders tuples list, e.g.,\n        jnp.zeros(INPUT_SHAPE). Each tuple list should correspond with the\n        serving function.\n\n    Returns:\n      TFLiteConverter object.\n    '
        TFLiteConverterBase._set_original_model_type(conversion_metdata_fb.ModelType.JAX)
        return TFLiteJaxConverterV2(serving_funcs, inputs)

    def convert(self):
        if False:
            i = 10
            return i + 15
        'Converts a TensorFlow GraphDef based on instance variables.\n\n    Returns:\n      The converted data in serialized format.\n\n    Raises:\n      ValueError:\n        No concrete functions is specified.\n        Multiple concrete functions are specified.\n        Input shape is not specified.\n        Invalid quantization parameters.\n    '
        return super(TFLiteConverterV2, self).convert()

class TFLiteConverterBaseV1(TFLiteConverterBase):
    """Converter subclass to share functionality between V1 converters."""

    def __init__(self, experimental_debug_info_func):
        if False:
            while True:
                i = 10
        'Constructor for TFLiteConverter.\n\n    Args:\n      experimental_debug_info_func: An experimental function to retrieve the\n        graph debug info for a set of nodes from the `graph_def`.\n    '
        super(TFLiteConverterBaseV1, self).__init__()
        self.inference_type = _dtypes.float32
        self.inference_input_type = None
        self.inference_output_type = None
        self.output_format = constants.TFLITE
        self.quantized_input_stats = {}
        self.default_ranges_stats = None
        self.drop_control_dependency = True
        self.reorder_across_fake_quant = False
        self.change_concat_input_ranges = False
        self.dump_graphviz_dir = None
        self.dump_graphviz_video = False
        self.conversion_summary_dir = None
        self._debug_info_func = experimental_debug_info_func
        self._metadata.environment.apiVersion = 1

    def __setattr__(self, name, value):
        if False:
            while True:
                i = 10
        if name == 'post_training_quantize':
            warnings.warn('Property %s is deprecated, please use optimizations=[Optimize.DEFAULT] instead.' % name)
            if value:
                self.optimizations = [Optimize.DEFAULT]
            else:
                self.optimizations = []
            return
        if name == 'target_ops':
            warnings.warn('Property %s is deprecated, please use target_spec.supported_ops instead.' % name)
            self.target_spec.supported_ops = value
            return
        object.__setattr__(self, name, value)

    def __getattribute__(self, name):
        if False:
            while True:
                i = 10
        if name == 'post_training_quantize':
            warnings.warn('Property %s is deprecated, please use optimizations=[Optimize.DEFAULT] instead.' % name)
            return Optimize.DEFAULT in set(self.optimizations)
        if name == 'target_ops':
            warnings.warn('Property %s is deprecated, please use target_spec.supported_ops instead.' % name)
            return self.target_spec.supported_ops
        return object.__getattribute__(self, name)

    def _validate_quantized_input_stats(self, converter_kwargs, quant_mode):
        if False:
            print('Hello World!')
        'Ensure the `quantized_input_stats` flag is provided if required.'
        quantized_types = frozenset({_dtypes.int8, _dtypes.uint8})
        requires_quantized_input_stats = (converter_kwargs['inference_type'] in quantized_types or converter_kwargs['inference_input_type'] in quantized_types) and (not quant_mode.is_post_training_integer_quantization())
        if requires_quantized_input_stats and (not converter_kwargs['quantized_input_stats']):
            raise ValueError('The `quantized_input_stats` flag must be defined when either `inference_type` flag or `inference_input_type` flag is set to tf.int8 or tf.uint8. Currently, `inference_type={}` and `inference_input_type={}`.'.format(_get_tf_type_name(converter_kwargs['inference_type']), _get_tf_type_name(converter_kwargs['inference_input_type'])))

    @convert_phase(Component.PREPARE_TF_MODEL, SubComponent.VALIDATE_INPUTS)
    def _validate_inputs(self, input_tensors, quantized_input_stats):
        if False:
            for i in range(10):
                print('nop')
        'Validate input parameters.\n\n    Args:\n      input_tensors: List of input tensors.\n      quantized_input_stats: Map of input tensor names to a tuple of floats\n        representing the mean and standard deviation of the training data.\n\n    Raises:\n      ValueError:\n        Input shape is not specified.\n        Quantization input stats is required but not provided.\n    '
        if not self._is_unknown_shapes_allowed() and self._has_valid_tensors():
            for tensor in input_tensors:
                shape = tensor.shape
                if not shape:
                    raise ValueError("Provide an input shape for input array '{0}'.".format(_get_tensor_name(tensor)))
                shape_list = shape.as_list()
                if None in shape_list[1:]:
                    raise ValueError("None is only supported in the 1st dimension. Tensor '{0}' has invalid shape '{1}'.".format(_get_tensor_name(tensor), shape_list))
                elif shape_list and shape_list[0] is None:
                    self._set_batch_size(batch_size=1)
        if quantized_input_stats:
            self._quantized_stats = []
            invalid_stats = []
            for name in self.get_input_arrays():
                if name in quantized_input_stats:
                    self._quantized_stats.append(quantized_input_stats[name])
                else:
                    invalid_stats.append(name)
            if invalid_stats:
                raise ValueError("Quantization input stats are not available for input tensors '{0}'.".format(','.join(invalid_stats)))
        else:
            self._quantized_stats = None

    @convert_phase(Component.PREPARE_TF_MODEL, SubComponent.OPTIMIZE_TF_MODEL)
    def _optimize_tf_model(self, graph_def, input_tensors, output_tensors, quant_mode):
        if False:
            return 10
        'Run a Grappler pass to optimize the TensorFlow graph.\n\n    Args:\n      graph_def: Frozen GraphDef to be optimized.\n      input_tensors: List of input tensors.\n      output_tensors: List of output tensors.\n      quant_mode: the quantization mode.\n\n    Returns:\n      The optimized TensorFlow graph.\n    '
        if self.saved_model_dir or quant_mode.is_quantization_aware_trained_model():
            return graph_def
        try:
            graph = _convert_to_constants.disable_lower_using_switch_merge(graph_def)
            optimized_graph = _run_graph_optimizations(graph, input_tensors, output_tensors, config=self._grappler_config(['function']))
            return optimized_graph
        except Exception:
            return graph_def

    def convert(self):
        if False:
            return 10
        'Converts a TensorFlow GraphDef based on instance variables.\n\n    Returns:\n      The converted data in serialized format. Either a TFLite Flatbuffer or a\n      Graphviz graph depending on value in `output_format`.\n\n    Raises:\n      ValueError:\n        Input shape is not specified.\n        None value for dimension in input_tensor.\n    '
        self._validate_inputs(self._input_tensors, self.quantized_input_stats)
        quant_mode = QuantizationMode(self.optimizations, self.target_spec, self.representative_dataset, self._graph_def, self._experimental_disable_per_channel, self.experimental_new_dynamic_range_quantizer, self._experimental_low_bit_qat, self._experimental_full_integer_quantization_bias_type, self._experimental_variable_quantization)
        optimized_graph = self._optimize_tf_model(self._graph_def, self._input_tensors, self._output_tensors, quant_mode)
        self._debug_info = _get_debug_info(self._debug_info_func, optimized_graph)
        converter_kwargs = self._get_base_converter_args()
        converter_kwargs.update(quant_mode.converter_flags(self.inference_type, self.inference_input_type))
        converter_kwargs.update({'output_format': self.output_format, 'quantized_input_stats': self._quantized_stats, 'default_ranges_stats': self.default_ranges_stats, 'drop_control_dependency': self.drop_control_dependency, 'reorder_across_fake_quant': self.reorder_across_fake_quant, 'change_concat_input_ranges': self.change_concat_input_ranges, 'dump_graphviz_dir': self.dump_graphviz_dir, 'dump_graphviz_video': self.dump_graphviz_video, 'conversion_summary_dir': self.conversion_summary_dir})
        self._validate_quantized_input_stats(converter_kwargs, quant_mode)
        if not self.experimental_new_converter:
            logging.warning('Please consider switching to the new converter by setting experimental_new_converter=True. The old converter is deprecated.')
        else:
            logging.info('Using experimental converter: If you encountered a problem please file a bug. You can opt-out by setting experimental_new_converter=False')
        if self._has_valid_tensors():
            result = _convert_graphdef(input_data=optimized_graph, input_tensors=self._input_tensors, output_tensors=self._output_tensors, **converter_kwargs)
        else:
            result = _convert_graphdef_with_arrays(input_data=optimized_graph, input_arrays_with_shape=self._input_arrays_with_shape, output_arrays=self._output_arrays, control_output_arrays=self._control_output_arrays, **converter_kwargs)
        return self._optimize_tflite_model(result, quant_mode, quant_io=self.experimental_new_quantizer)

    def get_input_arrays(self):
        if False:
            i = 10
            return i + 15
        'Returns a list of the names of the input tensors.\n\n    Returns:\n      List of strings.\n    '
        if self._has_valid_tensors():
            return [_get_tensor_name(tensor) for tensor in self._input_tensors]
        else:
            return [name for (name, _) in self._input_arrays_with_shape]

    def _has_valid_tensors(self):
        if False:
            while True:
                i = 10
        'Checks if the input and output tensors have been initialized.\n\n    Returns:\n      Bool.\n    '
        return self._input_tensors is not None and self._output_tensors

    def _set_batch_size(self, batch_size):
        if False:
            i = 10
            return i + 15
        'Sets the first dimension of the input tensor to `batch_size`.\n\n    Args:\n      batch_size: Batch size for the model. Replaces the first dimension of an\n        input size array if undefined. (default 1)\n\n    Raises:\n      ValueError: input_tensor is not defined.\n    '
        if not self._has_valid_tensors():
            raise ValueError('The batch size cannot be set for this model. Please use input_shapes parameter.')
        for tensor in self._input_tensors:
            shape = tensor.shape.as_list()
            if shape[0] is None:
                shape[0] = batch_size
                tensor.set_shape(shape)

    def _is_unknown_shapes_allowed(self):
        if False:
            print('Hello World!')
        if _is_ophint_converted(self._graph_def):
            return False
        if not super(TFLiteConverterBaseV1, self)._is_unknown_shapes_allowed():
            return False
        if self.conversion_summary_dir:
            logging.warning('`conversion_summary_dir` does not work with unknown shapes. Graphs with unknown shapes might be different than when this flag is disabled.')
            return False
        return True

    def _save_conversion_params_metric(self):
        if False:
            return 10
        self._collected_converter_params.update({'output_format': self.output_format, 'default_ranges_stats': self.default_ranges_stats, 'drop_control_dependency': self.drop_control_dependency, 'reorder_across_fake_quant': self.reorder_across_fake_quant, 'change_concat_input_ranges': self.change_concat_input_ranges, 'dump_graphviz_dir': self.dump_graphviz_dir, 'dump_graphviz_video': self.dump_graphviz_video, 'conversion_summary_dir': self.conversion_summary_dir})
        super(TFLiteConverterBaseV1, self)._save_conversion_params_metric(self._graph_def, self.inference_type, self.inference_input_type)

class TFLiteSavedModelConverter(TFLiteConverterBaseV1):
    """Converts the given SavedModel into TensorFlow Lite model.

  Attributes:
      saved_model_dir: Directory of the SavedModel.
  """

    def __init__(self, saved_model_dir, saved_model_tags, saved_model_exported_names, experimental_debug_info_func=None):
        if False:
            print('Hello World!')
        'Constructor for TFLiteConverter.\n\n    Args:\n      saved_model_dir: Directory of the SavedModel.\n      saved_model_tags: Set of tags identifying the MetaGraphDef within the\n        SavedModel to analyze. All tags in the tag set must be present. (default\n        {tf.saved_model.SERVING}).\n      saved_model_exported_names: Names to be exported when the saved model\n        import path is on.\n      experimental_debug_info_func: An experimental function to retrieve the\n        graph debug info for a set of nodes from the `graph_def`.\n\n    Raises:\n      ValueError: Invalid arguments.\n    '
        super(TFLiteSavedModelConverter, self).__init__(experimental_debug_info_func)
        self.saved_model_dir = saved_model_dir
        self._saved_model_tags = saved_model_tags
        self._saved_model_exported_names = saved_model_exported_names
        signature_key = _signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        if len(self._saved_model_exported_names) != 1:
            raise ValueError('Only support a single signature key.')
        signature_key = self._saved_model_exported_names[0]
        result = _freeze_saved_model(self.saved_model_dir, None, None, None, self._saved_model_tags, signature_key)
        self._graph_def = result[0]
        self._input_tensors = result[1]
        self._output_tensors = result[2]
        self._parse_saved_model_args()

    @_export_metrics
    def convert(self):
        if False:
            i = 10
            return i + 15
        "Converts a TensorFlow GraphDef based on instance variables.\n\n    Note that in the converted TensorFlow Lite model, the input tensor's order\n    might be changed each time `convert` is called. To access input tensor\n    information, please consider using the `SignatureRunner` API\n    (`interpreter.get_signature_runner`).\n\n    Returns:\n      The converted data in serialized format. Either a TFLite Flatbuffer or a\n      Graphviz graph depending on value in `output_format`.\n\n    Raises:\n      ValueError:\n        Input shape is not specified.\n        None value for dimension in input_tensor.\n    "
        return super(TFLiteSavedModelConverter, self).convert()

class TFLiteKerasModelConverter(TFLiteConverterBaseV1):
    """Converts the given SavedModel into TensorFlow Lite model."""

    def __init__(self, model_file, input_arrays=None, input_shapes=None, output_arrays=None, custom_objects=None):
        if False:
            return 10
        'Constructor for TFLiteConverter.\n\n    Args:\n      model_file: Full filepath of HDF5 file containing the tf.keras model.\n      input_arrays: List of input tensors to freeze graph with. Uses input\n        arrays from SignatureDef when none are provided. (default None)\n      input_shapes: Dict of strings representing input tensor names to list of\n        integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).\n        Automatically determined when input shapes is None (e.g., {"foo" :\n        None}). (default None)\n      output_arrays: List of output tensors to freeze graph with. Uses output\n        arrays from SignatureDef when none are provided. (default None)\n      custom_objects: Dict mapping names (strings) to custom classes or\n        functions to be considered during model deserialization. (default None)\n\n    Raises:\n      ValueError: Invalid arguments.\n    '
        super(TFLiteKerasModelConverter, self).__init__(experimental_debug_info_func=None)
        if context.executing_eagerly():
            if input_arrays or output_arrays:
                raise ValueError('`input_arrays` and `output_arrays` are unsupported with Eager mode. If your model requires any of these parameters, please use disable_eager_execution().')
            keras_model = keras_deps.get_load_model_function()(model_file, custom_objects)
            function = _trace_model_call(keras_model)
            concrete_func = function.get_concrete_function()
            frozen_func = _convert_to_constants.convert_variables_to_constants_v2(concrete_func, lower_control_flow=False)
            _set_tensor_shapes(frozen_func.inputs, input_shapes)
            self._keras_model = keras_model
            self._graph_def = frozen_func.graph.as_graph_def()
            self._input_tensors = frozen_func.inputs
            self._output_tensors = frozen_func.outputs
            self._debug_info_func = _build_debug_info_func(frozen_func.graph)
            return
        keras_deps.get_clear_session_function()()
        keras_model = keras_deps.get_load_model_function()(model_file, custom_objects)
        sess = keras_deps.get_get_session_function()()
        if input_arrays:
            input_tensors = _get_tensors_from_tensor_names(sess.graph, input_arrays)
        else:
            input_tensors = keras_model.inputs
        if output_arrays:
            output_tensors = _get_tensors_from_tensor_names(sess.graph, output_arrays)
        else:
            output_tensors = keras_model.outputs
        _set_tensor_shapes(input_tensors, input_shapes)
        graph_def = _freeze_graph(sess, input_tensors, output_tensors)
        self._keras_model = keras_model
        self._graph_def = graph_def
        self._input_tensors = input_tensors
        self._output_tensors = output_tensors
        self._debug_info_func = _build_debug_info_func(sess.graph)

    @convert_phase(Component.PREPARE_TF_MODEL, SubComponent.FREEZE_KERAS_MODEL)
    def _freeze_keras_model(self, output_dir):
        if False:
            while True:
                i = 10
        'Save Keras model to Saved Model format.\n\n    Args:\n      output_dir: The output directory to save the SavedModel.\n    '
        try:
            self._keras_model.save(output_dir, save_format='tf')
        except Exception:
            return None
        tag_set = set([_tag_constants.SERVING])
        signature_key = _signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        (graph_def, input_tensors, output_tensors, sess_graph) = _freeze_saved_model(output_dir, None, None, None, tag_set, signature_key)
        self.saved_model_dir = output_dir
        self._saved_model_tags = tag_set
        self._saved_model_exported_names = [signature_key]
        self._parse_saved_model_args()
        if self.saved_model_dir:
            self._graph_def = graph_def
            self._input_tensors = input_tensors
            self._output_tensors = output_tensors
            self._debug_info_func = _build_debug_info_func(sess_graph)

    def _convert_as_saved_model(self):
        if False:
            print('Hello World!')
        'Converts a Keras model as a saved model.\n\n    Returns:\n      The converted data in serialized format.\n    '
        temp_dir = tempfile.mkdtemp()
        try:
            self._freeze_keras_model(temp_dir)
            if self.saved_model_dir:
                return super(TFLiteKerasModelConverter, self).convert()
        finally:
            shutil.rmtree(temp_dir, True)

    @_export_metrics
    def convert(self):
        if False:
            print('Hello World!')
        'Converts a Keras model based on instance variables.\n\n    Returns:\n      The converted data in serialized format. Either a TFLite Flatbuffer or a\n      Graphviz graph depending on value in `output_format`.\n\n    Raises:\n      ValueError:\n        Input shape is not specified.\n        None value for dimension in input_tensor.\n    '
        saved_model_convert_result = self._convert_as_saved_model()
        if saved_model_convert_result:
            return saved_model_convert_result
        return super(TFLiteKerasModelConverter, self).convert()

class TFLiteFrozenGraphConverter(TFLiteConverterBaseV1):
    """Converts the given frozen graph def into TensorFlow Lite model."""

    def __init__(self, graph_def, input_tensors, output_tensors, input_arrays_with_shape=None, output_arrays=None, experimental_debug_info_func=None):
        if False:
            return 10
        'Constructor for TFLiteConverter.\n\n    Args:\n      graph_def: Frozen TensorFlow GraphDef.\n      input_tensors: List of input tensors. Type and shape are computed using\n        `foo.shape` and `foo.dtype`.\n      output_tensors: List of output tensors (only .name is used from this).\n      input_arrays_with_shape: Tuple of strings representing input tensor names\n        and list of integers representing input shapes (e.g., [("foo", [1, 16,\n        16, 3])]). Use only when graph cannot be loaded into TensorFlow and when\n        `input_tensors` and `output_tensors` are None. (default None)\n      output_arrays: List of output tensors to freeze graph with. Use only when\n        graph cannot be loaded into TensorFlow and when `input_tensors` and\n        `output_tensors` are None. (default None)\n      experimental_debug_info_func: An experimental function to retrieve the\n        graph debug info for a set of nodes from the `graph_def`.\n\n    Raises:\n      ValueError: Invalid arguments.\n    '
        super(TFLiteFrozenGraphConverter, self).__init__(experimental_debug_info_func)
        self._graph_def = graph_def
        self._input_tensors = input_tensors
        self._output_tensors = output_tensors
        self._control_output_arrays = None
        if not self._has_valid_tensors():
            self._input_arrays_with_shape = input_arrays_with_shape
            self._output_arrays = output_arrays
        if input_tensors is not None and input_arrays_with_shape is not None:
            logging.warning('input_arrays_with_shape will be ignored when both the given input_tensors and input_arrays_with_shape are not None.')
        if output_tensors is not None and output_arrays is not None:
            logging.warning('output_arrays will be ignored when both the given output_tensors and output_arrays are not None.')

    @_export_metrics
    def convert(self):
        if False:
            while True:
                i = 10
        'Converts a TensorFlow GraphDef based on instance variables.\n\n    Returns:\n      The converted data in serialized format. Either a TFLite Flatbuffer or a\n      Graphviz graph depending on value in `output_format`.\n\n    Raises:\n      ValueError:\n        Input shape is not specified.\n        None value for dimension in input_tensor.\n    '
        if not self._has_valid_tensors():
            if not self._input_arrays_with_shape or not (self._output_arrays or self._control_output_arrays):
                raise ValueError('If input_tensors and output_tensors are None, both input_arrays_with_shape and output_arrays|control_output_arrays must be defined.')
        return super(TFLiteFrozenGraphConverter, self).convert()

@_tf_export(v1=['lite.TFLiteConverter'])
class TFLiteConverter(TFLiteFrozenGraphConverter):
    """Convert a TensorFlow model into `output_format`.

  This is used to convert from a TensorFlow GraphDef, SavedModel or tf.keras
  model into either a TFLite FlatBuffer or graph visualization.

  Attributes:
    optimizations: Experimental flag, subject to change. Set of optimizations to
      apply. e.g {tf.lite.Optimize.DEFAULT}. (default None, must be None or a
      set of values of type `tf.lite.Optimize`)
    representative_dataset: A generator function used for integer quantization
      where each generated sample has the same order, type and shape as the
      inputs to the model. Usually, this is a small subset of a few hundred
      samples randomly chosen, in no particular order, from the training or
      evaluation dataset. This is an optional attribute, but required for full
      integer quantization, i.e, if `tf.int8` is the only supported type in
      `target_spec.supported_types`. Refer to `tf.lite.RepresentativeDataset`.
      (default None)
    target_spec: Experimental flag, subject to change. Specifications of target
      device, including supported ops set, supported types and a set of user's
      defined TensorFlow operators required in the TensorFlow Lite runtime.
      Refer to `tf.lite.TargetSpec`.
    inference_type: Data type of numeric arrays, excluding the input layer.
      (default tf.float32, must be in {tf.float32, tf.int8, tf.uint8})
    inference_input_type: Data type of the numeric arrays in the input layer. If
      `inference_input_type` is in {tf.int8, tf.uint8}, then
      `quantized_input_stats` must be provided. (default is the value assigned
      to `inference_type`, must be in {tf.float32, tf.int8, tf.uint8})
    inference_output_type: Data type of the numeric arrays in the output layer.
      (default is the value assigned to `inference_type`, must be in
      {tf.float32, tf.int8, tf.uint8})
    quantized_input_stats: Map of input tensor names to a tuple of floats
      representing the mean and standard deviation of the training data. (e.g.,
      {"foo" : (0., 1.)}). Required if `inference_input_type` is tf.int8 or
      tf.uint8. (default None)
    default_ranges_stats: Tuple of integers (min, max) representing range values
      for all numeric arrays without a specified range. Intended for
      experimenting with quantization via "dummy quantization". (default None)
    allow_custom_ops: Boolean indicating whether to allow custom operations.
      When False any unknown operation is an error. When True, custom ops are
      created for any op that is unknown. The developer will need to provide
      these to the TensorFlow Lite runtime with a custom resolver. (default
      False)
    drop_control_dependency: Boolean indicating whether to drop control
      dependencies silently. This is due to TFLite not supporting control
      dependencies. (default True)
    reorder_across_fake_quant: Boolean indicating whether to reorder FakeQuant
      nodes in unexpected locations. Used when the location of the FakeQuant
      nodes is preventing graph transformations necessary to convert the graph.
      Results in a graph that differs from the quantized training graph,
      potentially causing differing arithmetic behavior. (default False)
    change_concat_input_ranges: Boolean to change behavior of min/max ranges for
      inputs and outputs of the concat operator for quantized models. Changes
      the ranges of concat operator overlap when true. (default False)
    output_format: Output file format. (default
      tf.compat.v1.lite.constants.TFLITE, must be in
      {tf.compat.v1.lite.constants.TFLITE,
      tf.compat.v1.lite.constants.GRAPHVIZ_DOT})
    dump_graphviz_dir: Full filepath of folder to dump the graphs at various
      stages of processing GraphViz .dot files. Preferred over
      `output_format=tf.compat.v1.lite.constants.GRAPHVIZ_DOT` in order to keep
      the requirements of the output file. (default None)
    dump_graphviz_video: Boolean indicating whether to dump the GraphViz .dot
      files after every graph transformation. Requires the `dump_graphviz_dir`
      flag to be specified. (default False)
    conversion_summary_dir: Full path of the directory to store conversion logs.
      (default None)
    exclude_conversion_metadata: Whether not to embed the conversion metadata
      into the converted model. (default False)
    target_ops: Deprecated. Please use `target_spec.supported_ops` instead.
    post_training_quantize: Deprecated. Please use `optimizations` instead and
      set it to `{tf.lite.Optimize.DEFAULT}`. (default False)
    experimental_new_converter: Experimental flag, subject to change. Enables
      MLIR-based conversion. (default True)
    experimental_new_quantizer: Experimental flag, subject to change. Enables
      MLIR-based quantization conversion instead of Flatbuffer-based conversion.
      (default True)  Example usage:  ```python # Converting a GraphDef from
      session. converter = tf.compat.v1.lite.TFLiteConverter.from_session( sess,
      in_tensors, out_tensors) tflite_model = converter.convert()
      open("converted_model.tflite", "wb").write(tflite_model)  # Converting a
      GraphDef from file. converter =
      tf.compat.v1.lite.TFLiteConverter.from_frozen_graph( graph_def_file,
      input_arrays, output_arrays) tflite_model = converter.convert()
      open("converted_model.tflite", "wb").write(tflite_model)  # Converting a
      SavedModel. converter =
      tf.compat.v1.lite.TFLiteConverter.from_saved_model( saved_model_dir)
      tflite_model = converter.convert() open("converted_model.tflite",
      "wb").write(tflite_model)  # Converting a tf.keras model. converter =
      tf.compat.v1.lite.TFLiteConverter.from_keras_model_file( keras_model)
      tflite_model = converter.convert() open("converted_model.tflite",
      "wb").write(tflite_model) ```
  """

    def __init__(self, graph_def, input_tensors, output_tensors, input_arrays_with_shape=None, output_arrays=None, experimental_debug_info_func=None):
        if False:
            for i in range(10):
                print('nop')
        'Constructor for TFLiteConverter.\n\n    Args:\n      graph_def: Frozen TensorFlow GraphDef.\n      input_tensors: List of input tensors. Type and shape are computed using\n        `foo.shape` and `foo.dtype`.\n      output_tensors: List of output tensors (only .name is used from this).\n      input_arrays_with_shape: Tuple of strings representing input tensor names\n        and list of integers representing input shapes (e.g., [("foo" : [1, 16,\n        16, 3])]). Use only when graph cannot be loaded into TensorFlow and when\n        `input_tensors` and `output_tensors` are None. (default None)\n      output_arrays: List of output tensors to freeze graph with. Use only when\n        graph cannot be loaded into TensorFlow and when `input_tensors` and\n        `output_tensors` are None. (default None)\n      experimental_debug_info_func: An experimental function to retrieve the\n        graph debug info for a set of nodes from the `graph_def`.\n\n    Raises:\n      ValueError: Invalid arguments.\n    '
        super(TFLiteConverter, self).__init__(graph_def, input_tensors, output_tensors, input_arrays_with_shape, output_arrays, experimental_debug_info_func)

    @classmethod
    def from_session(cls, sess, input_tensors, output_tensors):
        if False:
            while True:
                i = 10
        'Creates a TFLiteConverter class from a TensorFlow Session.\n\n    Args:\n      sess: TensorFlow Session.\n      input_tensors: List of input tensors. Type and shape are computed using\n        `foo.shape` and `foo.dtype`.\n      output_tensors: List of output tensors (only .name is used from this).\n\n    Returns:\n      TFLiteConverter class.\n    '
        TFLiteConverterBase._set_original_model_type(conversion_metdata_fb.ModelType.TF_SESSION)
        graph_def = _freeze_graph(sess, input_tensors, output_tensors)
        return cls(graph_def, input_tensors, output_tensors, experimental_debug_info_func=_build_debug_info_func(sess.graph))

    @classmethod
    def from_frozen_graph(cls, graph_def_file, input_arrays, output_arrays, input_shapes=None):
        if False:
            i = 10
            return i + 15
        'Creates a TFLiteConverter class from a file containing a frozen GraphDef.\n\n    Args:\n      graph_def_file: Full filepath of file containing frozen GraphDef.\n      input_arrays: List of input tensors to freeze graph with.\n      output_arrays: List of output tensors to freeze graph with.\n      input_shapes: Dict of strings representing input tensor names to list of\n        integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).\n        Automatically determined when input shapes is None (e.g., {"foo" :\n        None}). (default None)\n\n    Returns:\n      TFLiteConverter class.\n\n    Raises:\n      IOError:\n        File not found.\n        Unable to parse input file.\n      ValueError:\n        The graph is not frozen.\n        input_arrays or output_arrays contains an invalid tensor name.\n        input_shapes is not correctly defined when required\n    '
        TFLiteConverterBase._set_original_model_type(conversion_metdata_fb.ModelType.TF_GRAPH_DEF)
        with _ops.Graph().as_default():
            with _session.Session() as sess:
                if not gfile.Exists(graph_def_file):
                    raise IOError("File '{0}' does not exist.".format(graph_def_file))
                with gfile.GFile(graph_def_file, 'rb') as f:
                    file_content = f.read()
                try:
                    graph_def = _graph_pb2.GraphDef()
                    graph_def.ParseFromString(file_content)
                except (_text_format.ParseError, DecodeError):
                    try:
                        print("Ignore 'tcmalloc: large alloc' warnings.")
                        if not isinstance(file_content, str):
                            file_content = file_content.decode('utf-8')
                        graph_def = _graph_pb2.GraphDef()
                        _text_format.Merge(file_content, graph_def)
                    except (_text_format.ParseError, DecodeError):
                        raise IOError("Unable to parse input file '{}'.".format(graph_def_file))
                if sys.byteorder == 'big':
                    bst.swap_tensor_content_in_graph_node(graph_def, 'little', 'big')
                load_model_in_session = True
                try:
                    _import_graph_def(graph_def, name='')
                except _NotFoundError:
                    load_model_in_session = False
                if load_model_in_session:
                    if not _is_frozen_graph(sess):
                        raise ValueError('Please freeze the graph using freeze_graph.py.')
                    input_tensors = _get_tensors_from_tensor_names(sess.graph, input_arrays)
                    output_tensors = _get_tensors_from_tensor_names(sess.graph, output_arrays)
                    _set_tensor_shapes(input_tensors, input_shapes)
                    return cls(sess.graph_def, input_tensors, output_tensors)
                else:
                    if not input_shapes:
                        raise ValueError('input_shapes must be defined for this model.')
                    if set(input_arrays) != set(input_shapes.keys()):
                        raise ValueError('input_shapes must contain a value for each item in input_array.')
                    input_arrays_with_shape = [(name, input_shapes[name]) for name in input_arrays]
                    return cls(graph_def, input_tensors=None, output_tensors=None, input_arrays_with_shape=input_arrays_with_shape, output_arrays=output_arrays)

    @classmethod
    def from_saved_model(cls, saved_model_dir, input_arrays=None, input_shapes=None, output_arrays=None, tag_set=None, signature_key=None):
        if False:
            i = 10
            return i + 15
        'Creates a TFLiteConverter class from a SavedModel.\n\n    Args:\n      saved_model_dir: SavedModel directory to convert.\n      input_arrays: List of input tensors to freeze graph with. Uses input\n        arrays from SignatureDef when none are provided. (default None)\n      input_shapes: Dict of strings representing input tensor names to list of\n        integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).\n        Automatically determined when input shapes is None (e.g., {"foo" :\n        None}). (default None)\n      output_arrays: List of output tensors to freeze graph with. Uses output\n        arrays from SignatureDef when none are provided. (default None)\n      tag_set: Set of tags identifying the MetaGraphDef within the SavedModel to\n        analyze. All tags in the tag set must be present. (default\n        {tf.saved_model.SERVING})\n      signature_key: Key identifying SignatureDef containing inputs and outputs.\n        (default tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY)\n\n    Returns:\n      TFLiteConverter class.\n    '
        TFLiteConverterBase._set_original_model_type(conversion_metdata_fb.ModelType.TF_SAVED_MODEL)
        if tag_set is None:
            tag_set = set([_tag_constants.SERVING])
        if signature_key is None:
            signature_key = _signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        saved_model_converter = TFLiteSavedModelConverter(saved_model_dir, tag_set, [signature_key])
        if saved_model_converter.saved_model_dir:
            return saved_model_converter
        result = _freeze_saved_model(saved_model_dir, input_arrays, input_shapes, output_arrays, tag_set, signature_key)
        return cls(graph_def=result[0], input_tensors=result[1], output_tensors=result[2], experimental_debug_info_func=_build_debug_info_func(result[3]))

    @classmethod
    def from_keras_model_file(cls, model_file, input_arrays=None, input_shapes=None, output_arrays=None, custom_objects=None):
        if False:
            for i in range(10):
                print('nop')
        'Creates a TFLiteConverter class from a tf.keras model file.\n\n    Args:\n      model_file: Full filepath of HDF5 file containing the tf.keras model.\n      input_arrays: List of input tensors to freeze graph with. Uses input\n        arrays from SignatureDef when none are provided. (default None)\n      input_shapes: Dict of strings representing input tensor names to list of\n        integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).\n        Automatically determined when input shapes is None (e.g., {"foo" :\n        None}). (default None)\n      output_arrays: List of output tensors to freeze graph with. Uses output\n        arrays from SignatureDef when none are provided. (default None)\n      custom_objects: Dict mapping names (strings) to custom classes or\n        functions to be considered during model deserialization. (default None)\n\n    Returns:\n      TFLiteConverter class.\n    '
        TFLiteConverterBase._set_original_model_type(conversion_metdata_fb.ModelType.KERAS_MODEL)
        return TFLiteKerasModelConverter(model_file, input_arrays, input_shapes, output_arrays, custom_objects)

    def convert(self):
        if False:
            while True:
                i = 10
        'Converts a TensorFlow GraphDef based on instance variables.\n\n    Returns:\n      The converted data in serialized format. Either a TFLite Flatbuffer or a\n      Graphviz graph depending on value in `output_format`.\n\n    Raises:\n      ValueError:\n        Input shape is not specified.\n        None value for dimension in input_tensor.\n    '
        return super(TFLiteConverter, self).convert()

@_tf_export(v1=['lite.TocoConverter'])
class TocoConverter:
    """Convert a TensorFlow model into `output_format`.

  This class has been deprecated. Please use `lite.TFLiteConverter` instead.
  """

    @classmethod
    @_deprecation.deprecated(None, 'Use `lite.TFLiteConverter.from_session` instead.')
    def from_session(cls, sess, input_tensors, output_tensors):
        if False:
            print('Hello World!')
        'Creates a TocoConverter class from a TensorFlow Session.'
        return TFLiteConverter.from_session(sess, input_tensors, output_tensors)

    @classmethod
    @_deprecation.deprecated(None, 'Use `lite.TFLiteConverter.from_frozen_graph` instead.')
    def from_frozen_graph(cls, graph_def_file, input_arrays, output_arrays, input_shapes=None):
        if False:
            i = 10
            return i + 15
        'Creates a TocoConverter class from a file containing a frozen graph.'
        return TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays, input_shapes)

    @classmethod
    @_deprecation.deprecated(None, 'Use `lite.TFLiteConverter.from_saved_model` instead.')
    def from_saved_model(cls, saved_model_dir, input_arrays=None, input_shapes=None, output_arrays=None, tag_set=None, signature_key=None):
        if False:
            return 10
        'Creates a TocoConverter class from a SavedModel.'
        return TFLiteConverter.from_saved_model(saved_model_dir, input_arrays, input_shapes, output_arrays, tag_set, signature_key)

    @classmethod
    @_deprecation.deprecated(None, 'Use `lite.TFLiteConverter.from_keras_model_file` instead.')
    def from_keras_model_file(cls, model_file, input_arrays=None, input_shapes=None, output_arrays=None):
        if False:
            i = 10
            return i + 15
        'Creates a TocoConverter class from a tf.keras model file.'
        return TFLiteConverter.from_keras_model_file(model_file, input_arrays, input_shapes, output_arrays)