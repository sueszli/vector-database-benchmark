"""Defines types required for representative datasets for quantization."""
import collections.abc
import os
from typing import Iterable, Mapping, Optional, Union
import numpy as np
from tensorflow.compiler.mlir.quantization.tensorflow import quantization_options_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import session
from tensorflow.python.data.ops import readers
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import python_io
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
from tensorflow.python.util import tf_export
RepresentativeSample = Mapping[str, core.TensorLike]
RepresentativeDataset = Iterable[RepresentativeSample]
RepresentativeDatasetMapping = Mapping[str, RepresentativeDataset]
RepresentativeDatasetOrMapping = Union[RepresentativeDataset, RepresentativeDatasetMapping]
_RepresentativeDataSample = quantization_options_pb2.RepresentativeDataSample
_RepresentativeDatasetFile = quantization_options_pb2.RepresentativeDatasetFile

class RepresentativeDatasetSaver:
    """Representative dataset saver.

  Exposes a single method `save` that saves the provided representative dataset
  into files.

  This is useful when you would like to keep a snapshot of your representative
  dataset at a file system or when you need to pass the representative dataset
  as files.
  """

    def save(self, representative_dataset: RepresentativeDatasetMapping) -> Mapping[str, _RepresentativeDatasetFile]:
        if False:
            while True:
                i = 10
        'Saves the representative dataset.\n\n    Args:\n      representative_dataset: RepresentativeDatasetMapping which is a\n        signature_def_key -> representative dataset mapping.\n    '
        raise NotImplementedError('Method "save" is not implemented.')

@tf_export.tf_export('quantization.experimental.TfRecordRepresentativeDatasetSaver')
class TfRecordRepresentativeDatasetSaver(RepresentativeDatasetSaver):
    """Representative dataset saver in TFRecord format.

  Saves representative datasets for quantization calibration in TFRecord format.
  The samples are serialized as `RepresentativeDataSample`.

  The `save` method return a signature key to `RepresentativeDatasetFile` map,
  which can be used for QuantizationOptions.

  Example usage:

  ```python
  # Creating the representative dataset.
  representative_dataset = [{"input": tf.random.uniform(shape=(3, 3))}
                        for _ in range(256)]

  # Saving to a TFRecord file.
  dataset_file_map = (
    tf.quantization.experimental.TfRecordRepresentativeDatasetSaver(
          path_map={'serving_default': '/tmp/representative_dataset_path'}
      ).save({'serving_default': representative_dataset})
  )

  # Using in QuantizationOptions.
  quantization_options = tf.quantization.experimental.QuantizationOptions(
      signature_keys=['serving_default'],
      representative_datasets=dataset_file_map,
  )
  tf.quantization.experimental.quantize_saved_model(
      '/tmp/input_model',
      '/tmp/output_model',
      quantization_options=quantization_options,
  )
  ```
  """

    def __init__(self, path_map: Mapping[str, os.PathLike[str]]):
        if False:
            while True:
                i = 10
        'Initializes TFRecord represenatative dataset saver.\n\n    Args:\n      path_map: Signature def key -> path mapping. Each path is a TFRecord file\n        to which a `RepresentativeDataset` is saved. The signature def keys\n        should be a subset of the `SignatureDef` keys of the\n        `representative_dataset` argument of the `save()` call.\n    '
        self.path_map: Mapping[str, os.PathLike[str]] = path_map

    def _save_tf_record_dataset(self, repr_ds: RepresentativeDataset, signature_def_key: str) -> _RepresentativeDatasetFile:
        if False:
            for i in range(10):
                print('nop')
        'Saves `repr_ds` to a TFRecord file.\n\n    Each sample in `repr_ds` is serialized as `RepresentativeDataSample`.\n\n    Args:\n      repr_ds: `RepresentativeDataset` to save.\n      signature_def_key: The signature def key associated with `repr_ds`.\n\n    Returns:\n      a RepresentativeDatasetFile instance contains the path to the saved file.\n    '
        tfrecord_file_path = self.path_map[signature_def_key]
        with python_io.TFRecordWriter(tfrecord_file_path) as writer:
            for repr_sample in repr_ds:
                sample = _RepresentativeDataSample()
                for (input_name, input_value) in repr_sample.items():
                    sample.tensor_proto_inputs[input_name].CopyFrom(tensor_util.make_tensor_proto(input_value))
                writer.write(sample.SerializeToString())
        logging.info('Saved representative dataset for signature def: %s to: %s', signature_def_key, tfrecord_file_path)
        return _RepresentativeDatasetFile(tfrecord_file_path=str(tfrecord_file_path))

    def save(self, representative_dataset: RepresentativeDatasetMapping) -> Mapping[str, _RepresentativeDatasetFile]:
        if False:
            while True:
                i = 10
        'Saves the representative dataset.\n\n    Args:\n      representative_dataset: Signature def key -> representative dataset\n        mapping. Each dataset is saved in a separate TFRecord file whose path\n        matches the signature def key of `path_map`.\n\n    Raises:\n      ValueError: When the signature def key in `representative_dataset` is not\n      present in the `path_map`.\n\n    Returns:\n      A map from signature key to the RepresentativeDatasetFile instance\n      contains the path to the saved file.\n    '
        dataset_file_map = {}
        for (signature_def_key, repr_ds) in representative_dataset.items():
            if signature_def_key not in self.path_map:
                raise ValueError(f'SignatureDef key does not exist in the provided path_map: {signature_def_key}')
            dataset_file_map[signature_def_key] = self._save_tf_record_dataset(repr_ds, signature_def_key)
        return dataset_file_map

class RepresentativeDatasetLoader:
    """Representative dataset loader.

  Exposes the `load` method that loads the representative dataset from files.
  """

    def __init__(self, dataset_file_map: Mapping[str, _RepresentativeDatasetFile]) -> None:
        if False:
            print('Hello World!')
        'Initializes TFRecord represenatative dataset loader.\n\n    Args:\n      dataset_file_map: Signature key -> `RepresentativeDatasetFile` mapping.\n\n    Raises:\n      DecodeError: If the sample is not RepresentativeDataSample.\n    '
        self.dataset_file_map = dataset_file_map

    def _load_tf_record(self, tf_record_path: str) -> RepresentativeDataset:
        if False:
            i = 10
            return i + 15
        'Loads TFRecord containing samples of type`RepresentativeDataSample`.'
        samples = []
        with context.eager_mode():
            for sample_bytes in readers.TFRecordDatasetV2(filenames=[tf_record_path]):
                sample_proto = _RepresentativeDataSample.FromString(sample_bytes.numpy())
                sample = {}
                for (input_key, tensor_proto) in sample_proto.tensor_proto_inputs.items():
                    sample[input_key] = tensor_util.MakeNdarray(tensor_proto)
                samples.append(sample)
        return samples

    def load(self) -> RepresentativeDatasetMapping:
        if False:
            while True:
                i = 10
        'Loads the representative datasets.\n\n    Returns:\n      representative dataset mapping: A signature def key -> representative\n      mapping. The loader loads `RepresentativeDataset` for each path in\n      `self.dataset_file_map` and associates the loaded dataset to the\n      corresponding signature def key.\n    '
        repr_dataset_map = {}
        for (signature_def_key, dataset_file) in self.dataset_file_map.items():
            if dataset_file.HasField('tfrecord_file_path'):
                repr_dataset_map[signature_def_key] = self._load_tf_record(dataset_file.tfrecord_file_path)
            else:
                raise ValueError('Unsupported Representative Dataset filetype')
        return repr_dataset_map

def replace_tensors_by_numpy_ndarrays(repr_ds: RepresentativeDataset, sess: session.Session) -> RepresentativeDataset:
    if False:
        while True:
            i = 10
    'Replaces tf.Tensors in samples by their evaluated numpy arrays.\n\n  Note: This should be run in graph mode (default in TF1) only.\n\n  Args:\n    repr_ds: Representative dataset to replace the tf.Tensors with their\n      evaluated values. `repr_ds` is iterated through, so it may not be reusable\n      (e.g. if it is a generator object).\n    sess: Session instance used to evaluate tf.Tensors.\n\n  Returns:\n    The new representative dataset where each tf.Tensor is replaced by its\n    evaluated numpy ndarrays.\n  '
    new_repr_ds = []
    for sample in repr_ds:
        new_sample = {}
        for (input_key, input_data) in sample.items():
            if isinstance(input_data, core.Tensor):
                input_data = input_data.eval(session=sess)
            new_sample[input_key] = input_data
        new_repr_ds.append(new_sample)
    return new_repr_ds

def get_num_samples(repr_ds: RepresentativeDataset) -> Optional[int]:
    if False:
        while True:
            i = 10
    'Returns the number of samples if known.\n\n  Args:\n    repr_ds: Representative dataset.\n\n  Returns:\n    Returns the total number of samples in `repr_ds` if it can be determined\n    without iterating the entier dataset. Returns None iff otherwise. When it\n    returns None it does not mean the representative dataset is infinite or it\n    is malformed; it simply means the size cannot be determined without\n    iterating the whole dataset.\n  '
    if isinstance(repr_ds, collections.abc.Sized):
        try:
            return len(repr_ds)
        except Exception as ex:
            logging.info('Cannot determine the size of the dataset (%s).', ex)
            return None
    else:
        return None

def create_feed_dict_from_input_data(input_data: RepresentativeSample, signature_def: meta_graph_pb2.SignatureDef) -> Mapping[str, np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    "Constructs a feed_dict from input data.\n\n  Note: This function should only be used in graph mode.\n\n  This is a helper function that converts an 'input key -> input value' mapping\n  to a feed dict. A feed dict is an 'input tensor name -> input value' mapping\n  and can be directly passed to the `feed_dict` argument of `sess.run()`.\n\n  Args:\n    input_data: Input key -> input value mapping. The input keys should match\n      the input keys of `signature_def`.\n    signature_def: A SignatureDef representing the function that `input_data` is\n      an input to.\n\n  Returns:\n    Feed dict, which is intended to be used as input for `sess.run`. It is\n    essentially a mapping: input tensor name -> input value. Note that the input\n    value in the feed dict is not a `Tensor`.\n  "
    feed_dict = {}
    for (input_key, input_value) in input_data.items():
        input_tensor_name = signature_def.inputs[input_key].name
        value = input_value
        if isinstance(input_value, core.Tensor):
            value = input_value.eval()
        feed_dict[input_tensor_name] = value
    return feed_dict