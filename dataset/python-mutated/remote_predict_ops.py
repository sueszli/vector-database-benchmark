"""Operations for RemotePredict."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import tensorflow.compat.v1 as tf
from tensorflow_serving.experimental.tensorflow.ops.remote_predict.ops import gen_remote_predict_op
from tensorflow_serving.experimental.tensorflow.ops.remote_predict.ops.gen_remote_predict_op import *
_remote_predict_op_module = tf.load_op_library(os.path.join(tf.compat.v1.resource_loader.get_data_files_path(), '_remote_predict_op.so'))

def run(input_tensor_alias, input_tensors, output_tensor_alias, target_address, model_name, model_version=-1, max_rpc_deadline_millis=3000, output_types=None, name=None, signature_name='serving_default'):
    if False:
        for i in range(10):
            print('nop')
    'Runs a predict in remote process through rpc.\n\n  Args:\n    input_tensor_alias: input tensor alias for Predict\n    input_tensors: input tensors for Predict\n    output_tensor_alias: output tensor alias for Predict\n    target_address: target_address where the rpc is sent to\n    model_name: model_name that the Predict is running on\n    model_version: the model version for the Predict call. If unset, the highest\n      version available for serving will be targeted.\n    max_rpc_deadline_millis: rpc deadline in millis\n    output_types: output types for Predict\n    name: name for the op in the graph\n    signature_name: the signature def for remote graph inference\n\n  Returns:\n    output_tensors as a result of the Predict.\n\n  Raises ValueError if model_name value is missing.\n  '
    if model_name is None:
        raise ValueError('model_name must be specified.')
    return gen_remote_predict_op.tf_serving_remote_predict(input_tensor_alias, input_tensors, output_tensor_alias, target_address=target_address, model_name=model_name, model_version=model_version, fail_op_on_rpc_error=True, max_rpc_deadline_millis=max_rpc_deadline_millis, signature_name=signature_name, output_types=output_types, name=name)[2]

def run_returning_status(input_tensor_alias, input_tensors, output_tensor_alias, target_address, model_name, model_version=-1, max_rpc_deadline_millis=3000, output_types=None, name=None, signature_name='serving_default'):
    if False:
        i = 10
        return i + 15
    'Runs a predict in remote process through rpc.\n\n  Args:\n    input_tensor_alias: input tensor alias for Predict\n    input_tensors: input tensors for Predict\n    output_tensor_alias: output tensor alias for Predict\n    target_address: target_address where the rpc is sent to\n    model_name: model_name that the Predict is running on\n    model_version: the model version for the Predict call. If unset, the highest\n      version available for serving will be targeted.\n    max_rpc_deadline_millis: rpc deadline in millis\n    output_types: output types for Predict\n    name: name for the op in the graph\n    signature_name: the signature def for remote graph inference\n\n  Returns:\n    status_code, status_error_message and output_tensors.\n\n  Raises ValueError if model_name value is missing.\n  '
    if model_name is None:
        raise ValueError('model_name must be specified.')
    return gen_remote_predict_op.tf_serving_remote_predict(input_tensor_alias, input_tensors, output_tensor_alias, target_address=target_address, model_name=model_name, model_version=model_version, fail_op_on_rpc_error=False, max_rpc_deadline_millis=max_rpc_deadline_millis, signature_name=signature_name, output_types=output_types, name=name)