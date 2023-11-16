"""Defines a wrapper class for overridden python method definitions."""
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Optional
import uuid
from absl import logging
from tensorflow.compiler.mlir.quantization.tensorflow import exported_model_pb2
from tensorflow.compiler.mlir.quantization.tensorflow import quantization_options_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import calibration_algorithm
from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import calibration_statistics_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import pywrap_calibration
from tensorflow.compiler.mlir.quantization.tensorflow.python import pywrap_function_lib
from tensorflow.compiler.mlir.quantization.tensorflow.python import representative_dataset as rd
from tensorflow.compiler.mlir.quantization.tensorflow.python import save_model
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.trackable import autotrackable
from tensorflow.python.types import core
_ASSETS_DIR = 'assets'
_ASSETS_EXTRA_DIR = 'assets.extra'

def _get_saver_def_or_none(exported_model: exported_model_pb2.ExportedModel) -> Optional[saver_pb2.SaverDef]:
    if False:
        for i in range(10):
            print('nop')
    'Returns the SaverDef from ExportedModel, None otherwise.\n\n  Args:\n    exported_model: ExportedModel to take the SaverDef from.\n\n  Returns:\n    SaverDef instance if the field `saver_def` is set. None otherwise.\n  '
    if exported_model.HasField('saver_def'):
        return exported_model.saver_def
    return None

def _copy_assets(src_path: str, dst_path: str) -> None:
    if False:
        while True:
            i = 10
    'Copies the assets directory of the saved model.\n\n  Clones the contents of the assets/ directory from the source saved model\n  directory to the destination saved model directory. Nothing will be copied if\n  there are no assets directory in the source directory.\n\n  Args:\n    src_path: Source saved model directory.\n    dst_path: Destination saved model directory. This directory must exist.\n  '
    for assets_dir_name in [_ASSETS_DIR, _ASSETS_EXTRA_DIR]:
        src_assets_path = file_io.join(src_path, assets_dir_name)
        if not file_io.file_exists_v2(src_assets_path):
            continue
        dst_assets_path = file_io.join(dst_path, assets_dir_name)
        file_io.create_dir_v2(dst_assets_path)
        for (curr_dir, _, files) in file_io.walk_v2(src_assets_path):
            for asset_file_name in files:
                src_asset_file = file_io.join(curr_dir, asset_file_name)
                curr_dst_dir = curr_dir.replace(src_assets_path, dst_assets_path)
                dst_asset_file = file_io.join(curr_dst_dir, asset_file_name)
                file_io.copy_v2(src_asset_file, dst_asset_file)
                logging.info('Copied asset file: %s -> %s', src_asset_file, dst_asset_file)

def _validate_representative_dataset(representative_dataset: rd.RepresentativeDatasetOrMapping, signature_keys: Collection[str]) -> None:
    if False:
        i = 10
        return i + 15
    'Validates the representative dataset, based on the signature keys.\n\n  Representative dataset can be provided in two different forms: a single\n  instance of `RepresentativeDataset` or a map of signature key to the\n  corresponding `RepresentativeDataset`. These have a relationship with\n  `signature_keys`.\n\n  This function validates the following conditions:\n  * If `len(signature_keys) > 1`, then `representative_dataset` should be a\n    mapping where the keys exactly match the elements in `signature_keys`.\n  * If `len(signature_keys) == 1`, then both a mapping and a single instance of\n    `RepresentativeDataset` are allowed.\n  * This function also assumes `len(signature_keys) > 0`.\n\n  Args:\n    representative_dataset: A `RepresentativeDataset` or a map of string to\n      `RepresentativeDataset` to be validated.\n    signature_keys: A collection of strings that contains the signature keys,\n      each identifying a `SignatureDef`.\n\n  Raises:\n    ValueError: Iff `representative_dataset` does not satisfy the conditions\n      above.\n  '
    if isinstance(representative_dataset, Mapping):
        if set(signature_keys) != set(representative_dataset.keys()):
            raise ValueError(f'The signature keys and the keys of representative dataset map do not match. Signature keys: {set(signature_keys)}, representative dataset map: {set(representative_dataset.keys())}.')
    elif len(signature_keys) > 1:
        raise ValueError(f'Representative dataset is not a mapping (got: {type(representative_dataset)}), but there is more than one signature key provided. Please provide a map of {{signature_key -> dataset}} with more than one signature key.')

def _replace_tensors_by_numpy_ndarrays(repr_ds_map: rd.RepresentativeDatasetMapping) -> None:
    if False:
        i = 10
        return i + 15
    'Replaces tf.Tensors by their evaluated numpy arrays.\n\n  This assumes that tf.Tensors in representative samples are created in the\n  default Graph. It will raise an error if tensors are created in a different\n  graph.\n\n  Args:\n    repr_ds_map: SignatureDef key -> RepresentativeDataset mapping.\n  '
    with session.Session() as sess:
        for signature_def_key in repr_ds_map:
            ds = repr_ds_map[signature_def_key]
            repr_ds_map[signature_def_key] = rd.replace_tensors_by_numpy_ndarrays(ds, sess)

def _create_sample_validator(expected_input_keys: Collection[str]) -> Callable[[rd.RepresentativeSample], rd.RepresentativeSample]:
    if False:
        while True:
            i = 10
    'Creates a validator function for a representative sample.\n\n  Args:\n    expected_input_keys: Input keys (keyword argument names) that the function\n      the sample will be used for is expecting to receive.\n\n  Returns:\n    A callable that validates a `RepresentativeSample`.\n  '

    def validator(sample: rd.RepresentativeSample) -> rd.RepresentativeSample:
        if False:
            while True:
                i = 10
        "Validates a single instance of representative sample.\n\n    This provides a simple check for `sample` that this is a mapping of\n    {input_key: input_value}.\n\n    Args:\n      sample: A `RepresentativeSample` to validate.\n\n    Returns:\n      `sample` iff it is valid.\n\n    Raises:\n      ValueError: iff the sample isn't an instance of `Mapping`.\n      KeyError: iff the sample does not have the set of input keys that match\n        the input keys of the function.\n    "
        if not isinstance(sample, Mapping):
            raise ValueError(f'Invalid representative sample type. Provide a mapping (usually a dict) of {{input_key: input_value}}. Got type: {type(sample)} instead.')
        if set(sample.keys()) != expected_input_keys:
            raise KeyError(f'Invalid input keys for representative sample. The function expects input keys of: {set(expected_input_keys)}. Got: {set(sample.keys())}. Please provide correct input keys for representative samples.')
        return sample
    return validator

def _log_sample_num_for_calibration(representative_dataset: rd.RepresentativeDataset) -> rd.RepresentativeDataset:
    if False:
        return 10
    'Logs the sample number for calibration.\n\n  If in debug logging level, the "sample number / total num samples" is logged\n  for every 5 iterations.\n\n  This is often useful when tracking the progress of the calibration step which\n  is often slow and may look stale if there\'s no logs being printed.\n\n  Args:\n    representative_dataset: The representative dataset.\n\n  Yields:\n    The representative samples from `representative_dataset` without any\n    modification.\n  '
    num_samples: Optional[int] = rd.get_num_samples(representative_dataset)
    if num_samples is None:
        total_num_samples = '?'
        logging.info('Representative dataset size unknown.')
    else:
        total_num_samples = str(num_samples)
        logging.info('Using representative dataset of size: %s', total_num_samples)
    sample_num = 0
    for sample in representative_dataset:
        sample_num += 1
        logging.log_every_n(logging.DEBUG, 'Running representative sample for calibration: %d / %s', 5, sample_num, total_num_samples)
        yield sample
    logging.info('Running representative samples complete: %d / %s', sample_num, total_num_samples)

def _run_function_for_calibration_graph_mode(sess: session.Session, signature_def: meta_graph_pb2.SignatureDef, representative_dataset: rd.RepresentativeDataset) -> None:
    if False:
        while True:
            i = 10
    'Runs the representative dataset through a function for calibration.\n\n  NOTE: This is intended to be run in graph mode (TF1).\n\n  The function is identified by the SignatureDef.\n\n  Args:\n    sess: The Session object to run the function in.\n    signature_def: A SignatureDef that identifies a function by specifying the\n      inputs and outputs.\n    representative_dataset: The representative dataset to run through the\n      function.\n  '
    output_tensor_names = [output_tensor_info.name for output_tensor_info in signature_def.outputs.values()]
    sample_validator = _create_sample_validator(expected_input_keys=signature_def.inputs.keys())
    for sample in map(sample_validator, _log_sample_num_for_calibration(representative_dataset)):
        feed_dict = rd.create_feed_dict_from_input_data(sample, signature_def)
        sess.run(output_tensor_names, feed_dict=feed_dict)

def _run_graph_for_calibration_graph_mode(model_dir: str, tags: Collection[str], representative_dataset_map: rd.RepresentativeDatasetMapping) -> None:
    if False:
        return 10
    'Runs the graph for calibration in graph mode.\n\n  This function assumes _graph mode_ (used when legacy TF1 is used or when eager\n  mode is explicitly disabled) when running the graph. This step is used in\n  order to collect the statistics in CustomAggregatorOp for quantization using\n  the representative dataset for the actual data provided for inference.\n\n  Args:\n    model_dir: Path to SavedModel directory.\n    tags: Collection of tags identifying the MetaGraphDef within the SavedModel.\n    representative_dataset_map: A map where signature keys are mapped to\n      corresponding representative datasets.\n\n  Raises:\n    ValueError: When running the function with the representative dataset fails.\n  '
    _replace_tensors_by_numpy_ndarrays(representative_dataset_map)
    with ops.Graph().as_default(), session.Session() as sess:
        meta_graph: meta_graph_pb2.MetaGraphDef = loader_impl.load(sess, tags, export_dir=model_dir)
        for (signature_key, repr_ds) in representative_dataset_map.items():
            sig_def = meta_graph.signature_def[signature_key]
            try:
                _run_function_for_calibration_graph_mode(sess, signature_def=sig_def, representative_dataset=repr_ds)
            except Exception as ex:
                raise ValueError(f'Failed to run representative dataset through the function with the signature key: {signature_key}.') from ex

def _convert_values_to_tf_tensors(sample: rd.RepresentativeSample) -> Mapping[str, core.Tensor]:
    if False:
        print('Hello World!')
    'Converts TensorLike values of `sample` to Tensors.\n\n  Creates a copy of `sample`, where each value is converted to Tensors\n  unless it is already a Tensor.\n  The values are not converted in-place (i.e. `sample` is not mutated).\n\n  Args:\n    sample: A representative sample, which is a map of {name -> tensorlike\n      value}.\n\n  Returns:\n    Converted map of {name -> tensor}.\n  '
    tensor_mapping = {}
    for (name, tensorlike_value) in sample.items():
        if isinstance(tensorlike_value, core.Tensor):
            tensor_value = tensorlike_value
        else:
            tensor_value = tensor_conversion.convert_to_tensor_v2_with_dispatch(tensorlike_value)
        tensor_mapping[name] = tensor_value
    return tensor_mapping

def _run_function_for_calibration_eager_mode(func: wrap_function.WrappedFunction, representative_dataset: rd.RepresentativeDataset) -> None:
    if False:
        print('Hello World!')
    'Runs the representative dataset through a function for calibration.\n\n  NOTE: This is intended to be run in eager mode (TF2).\n\n  Args:\n    func: The function to run the representative samples through.\n    representative_dataset: Representative dataset used for calibration. The\n      input keys and input values of the representative samples should match the\n      keyword arguments of `func`.\n  '
    (_, keyword_args) = func.structured_input_signature
    sample_validator = _create_sample_validator(expected_input_keys=keyword_args.keys())
    for sample in map(sample_validator, _log_sample_num_for_calibration(representative_dataset)):
        func_kwargs = _convert_values_to_tf_tensors(sample)
        func(**func_kwargs)

def _run_graph_for_calibration_eager_mode(model_dir: str, tags: Collection[str], representative_dataset_map: rd.RepresentativeDatasetMapping) -> None:
    if False:
        return 10
    'Runs the graph for calibration in eager mode.\n\n  This function assumes _eager mode_ (enabled in TF2 by default) when running\n  the graph. This step is used in order to collect the statistics in\n  CustomAggregatorOp for quantization using the representative dataset for the\n  actual data provided for inference.\n\n  Args:\n    model_dir: Path to SavedModel directory.\n    tags: Collection of tags identifying the MetaGraphDef within the SavedModel.\n    representative_dataset_map: A map where signature keys are mapped to\n      corresponding representative datasets.\n\n  Raises:\n    ValueError: When running the function with the representative dataset fails.\n  '
    root: autotrackable.AutoTrackable = load.load(model_dir, tags)
    for (signature_key, repr_ds) in representative_dataset_map.items():
        try:
            _run_function_for_calibration_eager_mode(func=root.signatures[signature_key], representative_dataset=repr_ds)
        except Exception as ex:
            raise ValueError(f'Failed to run representative dataset through the function with the signature key: {signature_key}.') from ex

def _run_graph_for_calibration(float_model_dir: str, signature_keys: Sequence[str], tags: Collection[str], representative_dataset: rd.RepresentativeDatasetOrMapping, force_graph_mode_calibration: bool) -> None:
    if False:
        while True:
            i = 10
    'Runs the graph for calibration using representative datasets.\n\n  Args:\n    float_model_dir: Path to the model to calibrate.\n    signature_keys: Sequence of keys identifying SignatureDef containing inputs\n      and outputs.\n    tags: Collection of tags identifying the MetaGraphDef within the SavedModel\n      to analyze.\n    representative_dataset: An iterator that returns a dictionary of {input_key:\n      input_value} or a mapping from signature keys to such iterators. When\n      `signature_keys` contains more than one signature key,\n      `representative_datsaet` should be a mapping that maps each signature keys\n      to the corresponding representative dataset.\n    force_graph_mode_calibration: If set to true, it forces calibration in graph\n      model instead of eager mode when the context is in eager mode.\n\n  Raises:\n    ValueError iff:\n      * The representative dataset format is invalid.\n      * It fails to run the functions using the representative datasets.\n  '
    try:
        _validate_representative_dataset(representative_dataset, signature_keys)
    except Exception as ex:
        raise ValueError('Invalid representative dataset.') from ex
    representative_dataset_map = representative_dataset
    if not isinstance(representative_dataset, Mapping):
        representative_dataset_map = {signature_keys[0]: representative_dataset}
    try:
        if context.executing_eagerly() and (not force_graph_mode_calibration):
            logging.info('Calibration step is executed in eager mode.')
            _run_graph_for_calibration_eager_mode(float_model_dir, tags, representative_dataset_map)
        else:
            logging.info('Calibration step is executed in graph mode.')
            _run_graph_for_calibration_graph_mode(float_model_dir, tags, representative_dataset_map)
    except Exception as ex:
        raise ValueError('Failed to run graph for post-training quantization calibration.') from ex
    logging.info('Calibration step complete.')

def _get_min_max_from_calibrator(node_id: bytes, calib_opts: quantization_options_pb2.CalibrationOptions) -> tuple[float, float]:
    if False:
        print('Hello World!')
    'Calculate min and max from statistics using calibration options.\n\n  Args:\n    node_id: bytes of node id.\n    calib_opts: Calibration options used for calculating min and max.\n\n  Returns:\n    (min_value, max_value): Min and max calculated using calib_opts.\n\n  Raises:\n    ValueError: Unsupported calibration method is given.\n  '
    statistics: calibration_statistics_pb2.CalibrationStatistics = pywrap_calibration.get_statistics_from_calibrator(node_id)
    (min_value, max_value) = calibration_algorithm.get_min_max_value(statistics, calib_opts)
    return (min_value, max_value)

def _add_calibration_statistics(graph_def: graph_pb2.GraphDef, calib_opts: quantization_options_pb2.CalibrationOptions) -> None:
    if False:
        i = 10
        return i + 15
    'Adds calibration statistics to the graph def.\n\n  This function must be run after running the graph with a representative\n  dataset. Retrieves calibration statistics from the global calibrator and adds\n  them to the corresponding nodes as attributes.\n\n  Args:\n    graph_def: GraphDef to add calibration statistics to.\n    calib_opts: Calibration options to calculate min and max.\n  '
    for function_def in graph_def.library.function:
        for node_def in function_def.node_def:
            if node_def.op != 'CustomAggregator':
                continue
            node_id = node_def.attr['id'].s
            try:
                (min_value, max_value) = _get_min_max_from_calibrator(node_id, calib_opts)
                pywrap_calibration.clear_data_from_calibrator(node_id)
                node_def.attr['min'].f = min_value
                node_def.attr['max'].f = max_value
            except ValueError:
                logging.warning('CustomAggregator id "%s" from FunctionDef "%s" does not have min or max values. Parts of this function are not quantized.', node_id.decode('utf-8'), function_def.signature.name)

class PyFunctionLibrary(pywrap_function_lib.PyFunctionLibrary):
    """Wrapper class for overridden python method definitions.

  This class contains python methods that overrides C++ virtual functions
  declared in `pywrap_function_lib.PyFunctionLibrary`.
  """

    def assign_ids_to_custom_aggregator_ops(self, exported_model_serialized: bytes) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        'Assigns UUIDs to each CustomAggregator op find in the graph def.\n\n    Args:\n      exported_model_serialized: Serialized `ExportedModel` instance.\n\n    Returns:\n      Serialized `ExportedModel` whose CustomAggregator ops are assigned UUIDs\n      to their `id` attributes.\n    '
        exported_model = exported_model_pb2.ExportedModel.FromString(exported_model_serialized)
        graph_def = exported_model.graph_def
        for function_def in graph_def.library.function:
            for node_def in function_def.node_def:
                if node_def.op == 'CustomAggregator':
                    node_def.attr['id'].s = uuid.uuid4().hex.encode('ascii')
        return exported_model.SerializeToString()

    def save_exported_model(self, dst_saved_model_path: str, exported_model_serialized: bytes, src_saved_model_path: str, tags: set[str], serialized_signature_def_map: dict[str, bytes]) -> None:
        if False:
            i = 10
            return i + 15
        'Saves `ExportedModel` to `dst_saved_model_path` as a SavedModel.\n\n    Args:\n      dst_saved_model_path: Destination path to save the exported model.\n      exported_model_serialized: Exported model to export as SavedModel.\n      src_saved_model_path: Path to the source SavedModel. This will be used to\n        copy the asset files to `dst_saved_model_path`.\n      tags: Tags to attach to the saved MetaGraphDef.\n      serialized_signature_def_map: Signature key -> serialized SignatureDef.\n    '
        exported_model = exported_model_pb2.ExportedModel.FromString(exported_model_serialized)
        signature_def_map = {}
        for (key, serialized_signature_def) in serialized_signature_def_map.items():
            signature_def_map[key] = meta_graph_pb2.SignatureDef.FromString(serialized_signature_def)
        save_model.save_model_v1(exported_model.graph_def, dst_saved_model_path, signature_def_map, tags, init_op_name=exported_model.init_node_name, saver_def=_get_saver_def_or_none(exported_model), checkpoint_dir=exported_model.checkpoint_dir, function_aliases=exported_model.function_aliases, asset_file_defs=exported_model.asset_file_defs)
        _copy_assets(src_saved_model_path, dst_saved_model_path)

    def run_calibration(self, saved_model_path: str, exported_model_serialized: bytes, quantization_options_serialized: bytes, representative_dataset: rd.RepresentativeDatasetOrMapping) -> bytes:
        if False:
            return 10
        'Runs calibration and adds calibration statistics to exported model.\n\n    Args:\n      saved_model_path: Path to the SavedModel to run calibration.\n      exported_model_serialized: Serialized `ExportedModel` that corresponds to\n        the SavedModel at `saved_model_path`.\n      quantization_options_serialized: Serialized `QuantizationOptions`.\n      representative_dataset: Representative dataset to run calibration.\n\n    Returns:\n      Updated exported model (serialized) where the collected calibration\n      statistics are added to `CustomerAggregator` nodes at the `min` and `max`\n      attributes.\n    '
        quantization_options = quantization_options_pb2.QuantizationOptions.FromString(quantization_options_serialized)
        _run_graph_for_calibration(saved_model_path, quantization_options.signature_keys, quantization_options.tags, representative_dataset, quantization_options.force_graph_mode_calibration)
        exported_model = exported_model_pb2.ExportedModel.FromString(exported_model_serialized)
        _add_calibration_statistics(exported_model.graph_def, quantization_options.calibration_options)
        return exported_model.SerializeToString()