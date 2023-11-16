"""Switching v2 features on and off."""
from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import counter
from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.experimental.ops import random_ops
from tensorflow.python.data.experimental.ops import readers as exp_readers
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.ops import resource_variables_toggle
from tensorflow.python.util.tf_export import tf_export
_v2_behavior_usage_gauge = monitoring.BoolGauge('/tensorflow/version/v2_behavior', 'whether v2_behavior is enabled or disabled', 'status')

@tf_export(v1=['enable_v2_behavior'])
def enable_v2_behavior():
    if False:
        print('Hello World!')
    'Enables TensorFlow 2.x behaviors.\n\n  This function can be called at the beginning of the program (before `Tensors`,\n  `Graphs` or other structures have been created, and before devices have been\n  initialized. It switches all global behaviors that are different between\n  TensorFlow 1.x and 2.x to behave as intended for 2.x.\n\n  This function is called in the main TensorFlow `__init__.py` file, user should\n  not need to call it, except during complex migrations.\n\n  @compatibility(TF2)\n  This function is not necessary if you are using TF2. V2 behavior is enabled by\n  default.\n  @end_compatibility\n  '
    _v2_behavior_usage_gauge.get_cell('enable').set(True)
    tf2.enable()
    ops.enable_eager_execution()
    tensor_shape.enable_v2_tensorshape()
    resource_variables_toggle.enable_resource_variables()
    tensor.enable_tensor_equality()
    control_flow_v2_toggles.enable_control_flow_v2()
    dataset_ops.Dataset = dataset_ops.DatasetV2
    readers.FixedLengthRecordDataset = readers.FixedLengthRecordDatasetV2
    readers.TFRecordDataset = readers.TFRecordDatasetV2
    readers.TextLineDataset = readers.TextLineDatasetV2
    counter.Counter = counter.CounterV2
    interleave_ops.choose_from_datasets = interleave_ops.choose_from_datasets_v2
    interleave_ops.sample_from_datasets = interleave_ops.sample_from_datasets_v2
    random_ops.RandomDataset = random_ops.RandomDatasetV2
    exp_readers.CsvDataset = exp_readers.CsvDatasetV2
    exp_readers.SqlDataset = exp_readers.SqlDatasetV2
    exp_readers.make_batched_features_dataset = exp_readers.make_batched_features_dataset_v2
    exp_readers.make_csv_dataset = exp_readers.make_csv_dataset_v2

@tf_export(v1=['disable_v2_behavior'])
def disable_v2_behavior():
    if False:
        return 10
    'Disables TensorFlow 2.x behaviors.\n\n  This function can be called at the beginning of the program (before `Tensors`,\n  `Graphs` or other structures have been created, and before devices have been\n  initialized. It switches all global behaviors that are different between\n  TensorFlow 1.x and 2.x to behave as intended for 1.x.\n\n  User can call this function to disable 2.x behavior during complex migrations.\n\n  @compatibility(TF2)\n  Using this function indicates that your software is not compatible\n  with eager execution and `tf.function` in TF2.\n\n  To migrate to TF2, rewrite your code to be compatible with eager execution.\n  Please refer to the [migration guide]\n  (https://www.tensorflow.org/guide/migrate) for additional resource on the\n  topic.\n  @end_compatibility\n  '
    _v2_behavior_usage_gauge.get_cell('disable').set(True)
    tf2.disable()
    ops.disable_eager_execution()
    tensor_shape.disable_v2_tensorshape()
    resource_variables_toggle.disable_resource_variables()
    tensor.disable_tensor_equality()
    control_flow_v2_toggles.disable_control_flow_v2()
    dataset_ops.Dataset = dataset_ops.DatasetV1
    readers.FixedLengthRecordDataset = readers.FixedLengthRecordDatasetV1
    readers.TFRecordDataset = readers.TFRecordDatasetV1
    readers.TextLineDataset = readers.TextLineDatasetV1
    counter.Counter = counter.CounterV1
    interleave_ops.choose_from_datasets = interleave_ops.choose_from_datasets_v1
    interleave_ops.sample_from_datasets = interleave_ops.sample_from_datasets_v1
    random_ops.RandomDataset = random_ops.RandomDatasetV1
    exp_readers.CsvDataset = exp_readers.CsvDatasetV1
    exp_readers.SqlDataset = exp_readers.SqlDatasetV1
    exp_readers.make_batched_features_dataset = exp_readers.make_batched_features_dataset_v1
    exp_readers.make_csv_dataset = exp_readers.make_csv_dataset_v1