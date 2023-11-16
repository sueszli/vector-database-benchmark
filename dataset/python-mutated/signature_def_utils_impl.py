"""SignatureDef utility functions implementation."""
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import utils_impl as utils
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

@tf_export(v1=['saved_model.build_signature_def', 'saved_model.signature_def_utils.build_signature_def'])
@deprecation.deprecated_endpoints('saved_model.signature_def_utils.build_signature_def')
def build_signature_def(inputs=None, outputs=None, method_name=None, defaults=None):
    if False:
        print('Hello World!')
    'Utility function to build a SignatureDef protocol buffer.\n\n  Args:\n    inputs: Inputs of the SignatureDef defined as a proto map of string to\n      tensor info.\n    outputs: Outputs of the SignatureDef defined as a proto map of string to\n      tensor info.\n    method_name: Method name of the SignatureDef as a string.\n    defaults: Defaults of the SignatureDef defined as a proto map of string to\n      TensorProto.\n\n  Returns:\n    A SignatureDef protocol buffer constructed based on the supplied arguments.\n  '
    signature_def = meta_graph_pb2.SignatureDef()
    if inputs is not None:
        for item in inputs:
            signature_def.inputs[item].CopyFrom(inputs[item])
    if outputs is not None:
        for item in outputs:
            signature_def.outputs[item].CopyFrom(outputs[item])
    if method_name is not None:
        signature_def.method_name = method_name
    if defaults is not None:
        for (arg_name, default) in defaults.items():
            if isinstance(default, ops.EagerTensor):
                signature_def.defaults[arg_name].CopyFrom(tensor_util.make_tensor_proto(default.numpy()))
            elif default.op.type == 'Const':
                signature_def.defaults[arg_name].CopyFrom(default.op.get_attr('value'))
            else:
                raise ValueError(f'Unable to convert object {str(default)} of type {type(default)} to TensorProto.')
    return signature_def

@tf_export(v1=['saved_model.regression_signature_def', 'saved_model.signature_def_utils.regression_signature_def'])
@deprecation.deprecated_endpoints('saved_model.signature_def_utils.regression_signature_def')
def regression_signature_def(examples, predictions):
    if False:
        while True:
            i = 10
    'Creates regression signature from given examples and predictions.\n\n  This function produces signatures intended for use with the TensorFlow Serving\n  Regress API (tensorflow_serving/apis/prediction_service.proto), and so\n  constrains the input and output types to those allowed by TensorFlow Serving.\n\n  Args:\n    examples: A string `Tensor`, expected to accept serialized tf.Examples.\n    predictions: A float `Tensor`.\n\n  Returns:\n    A regression-flavored signature_def.\n\n  Raises:\n    ValueError: If examples is `None`.\n  '
    if examples is None:
        raise ValueError('Regression `examples` cannot be None.')
    if not isinstance(examples, tensor_lib.Tensor):
        raise ValueError(f'Expected regression `examples` to be of type Tensor. Found `examples` of type {type(examples)}.')
    if predictions is None:
        raise ValueError('Regression `predictions` cannot be None.')
    input_tensor_info = utils.build_tensor_info(examples)
    if input_tensor_info.dtype != types_pb2.DT_STRING:
        raise ValueError(f'Regression input tensors must be of type string. Found tensors with type {input_tensor_info.dtype}.')
    signature_inputs = {signature_constants.REGRESS_INPUTS: input_tensor_info}
    output_tensor_info = utils.build_tensor_info(predictions)
    if output_tensor_info.dtype != types_pb2.DT_FLOAT:
        raise ValueError(f'Regression output tensors must be of type float. Found tensors with type {output_tensor_info.dtype}.')
    signature_outputs = {signature_constants.REGRESS_OUTPUTS: output_tensor_info}
    signature_def = build_signature_def(signature_inputs, signature_outputs, signature_constants.REGRESS_METHOD_NAME)
    return signature_def

@tf_export(v1=['saved_model.classification_signature_def', 'saved_model.signature_def_utils.classification_signature_def'])
@deprecation.deprecated_endpoints('saved_model.signature_def_utils.classification_signature_def')
def classification_signature_def(examples, classes, scores):
    if False:
        return 10
    'Creates classification signature from given examples and predictions.\n\n  This function produces signatures intended for use with the TensorFlow Serving\n  Classify API (tensorflow_serving/apis/prediction_service.proto), and so\n  constrains the input and output types to those allowed by TensorFlow Serving.\n\n  Args:\n    examples: A string `Tensor`, expected to accept serialized tf.Examples.\n    classes: A string `Tensor`.  Note that the ClassificationResponse message\n      requires that class labels are strings, not integers or anything else.\n    scores: a float `Tensor`.\n\n  Returns:\n    A classification-flavored signature_def.\n\n  Raises:\n    ValueError: If examples is `None`.\n  '
    if examples is None:
        raise ValueError('Classification `examples` cannot be None.')
    if not isinstance(examples, tensor_lib.Tensor):
        raise ValueError(f'Classification `examples` must be a string Tensor. Found `examples` of type {type(examples)}.')
    if classes is None and scores is None:
        raise ValueError('Classification `classes` and `scores` cannot both be None.')
    input_tensor_info = utils.build_tensor_info(examples)
    if input_tensor_info.dtype != types_pb2.DT_STRING:
        raise ValueError(f'Classification input tensors must be of type string. Found tensors of type {input_tensor_info.dtype}')
    signature_inputs = {signature_constants.CLASSIFY_INPUTS: input_tensor_info}
    signature_outputs = {}
    if classes is not None:
        classes_tensor_info = utils.build_tensor_info(classes)
        if classes_tensor_info.dtype != types_pb2.DT_STRING:
            raise ValueError(f'Classification classes must be of type string Tensor. Found tensors of type {classes_tensor_info.dtype}.`')
        signature_outputs[signature_constants.CLASSIFY_OUTPUT_CLASSES] = classes_tensor_info
    if scores is not None:
        scores_tensor_info = utils.build_tensor_info(scores)
        if scores_tensor_info.dtype != types_pb2.DT_FLOAT:
            raise ValueError('Classification scores must be a float Tensor.')
        signature_outputs[signature_constants.CLASSIFY_OUTPUT_SCORES] = scores_tensor_info
    signature_def = build_signature_def(signature_inputs, signature_outputs, signature_constants.CLASSIFY_METHOD_NAME)
    return signature_def

@tf_export(v1=['saved_model.predict_signature_def', 'saved_model.signature_def_utils.predict_signature_def'])
@deprecation.deprecated_endpoints('saved_model.signature_def_utils.predict_signature_def')
def predict_signature_def(inputs, outputs):
    if False:
        i = 10
        return i + 15
    'Creates prediction signature from given inputs and outputs.\n\n  This function produces signatures intended for use with the TensorFlow Serving\n  Predict API (tensorflow_serving/apis/prediction_service.proto). This API\n  imposes no constraints on the input and output types.\n\n  Args:\n    inputs: dict of string to `Tensor`.\n    outputs: dict of string to `Tensor`.\n\n  Returns:\n    A prediction-flavored signature_def.\n\n  Raises:\n    ValueError: If inputs or outputs is `None`.\n  '
    if inputs is None or not inputs:
        raise ValueError('Prediction `inputs` cannot be None or empty.')
    if outputs is None or not outputs:
        raise ValueError('Prediction `outputs` cannot be None or empty.')
    signature_inputs = {key: utils.build_tensor_info(tensor) for (key, tensor) in inputs.items()}
    signature_outputs = {key: utils.build_tensor_info(tensor) for (key, tensor) in outputs.items()}
    signature_def = build_signature_def(signature_inputs, signature_outputs, signature_constants.PREDICT_METHOD_NAME)
    return signature_def

def supervised_train_signature_def(inputs, loss, predictions=None, metrics=None):
    if False:
        i = 10
        return i + 15
    return _supervised_signature_def(signature_constants.SUPERVISED_TRAIN_METHOD_NAME, inputs, loss=loss, predictions=predictions, metrics=metrics)

def supervised_eval_signature_def(inputs, loss, predictions=None, metrics=None):
    if False:
        for i in range(10):
            print('nop')
    return _supervised_signature_def(signature_constants.SUPERVISED_EVAL_METHOD_NAME, inputs, loss=loss, predictions=predictions, metrics=metrics)

def _supervised_signature_def(method_name, inputs, loss=None, predictions=None, metrics=None):
    if False:
        return 10
    'Creates a signature for training and eval data.\n\n  This function produces signatures that describe the inputs and outputs\n  of a supervised process, such as training or evaluation, that\n  results in loss, metrics, and the like. Note that this function only requires\n  inputs to be not None.\n\n  Args:\n    method_name: Method name of the SignatureDef as a string.\n    inputs: dict of string to `Tensor`.\n    loss: dict of string to `Tensor` representing computed loss.\n    predictions: dict of string to `Tensor` representing the output predictions.\n    metrics: dict of string to `Tensor` representing metric ops.\n\n  Returns:\n    A train- or eval-flavored signature_def.\n\n  Raises:\n    ValueError: If inputs or outputs is `None`.\n  '
    if inputs is None or not inputs:
        raise ValueError(f'{method_name} `inputs` cannot be None or empty.')
    signature_inputs = {key: utils.build_tensor_info(tensor) for (key, tensor) in inputs.items()}
    signature_outputs = {}
    for output_set in (loss, predictions, metrics):
        if output_set is not None:
            sig_out = {key: utils.build_tensor_info(tensor) for (key, tensor) in output_set.items()}
            signature_outputs.update(sig_out)
    signature_def = build_signature_def(signature_inputs, signature_outputs, method_name)
    return signature_def

@tf_export(v1=['saved_model.is_valid_signature', 'saved_model.signature_def_utils.is_valid_signature'])
@deprecation.deprecated_endpoints('saved_model.signature_def_utils.is_valid_signature')
def is_valid_signature(signature_def):
    if False:
        for i in range(10):
            print('nop')
    'Determine whether a SignatureDef can be served by TensorFlow Serving.'
    if signature_def is None:
        return False
    return _is_valid_classification_signature(signature_def) or _is_valid_regression_signature(signature_def) or _is_valid_predict_signature(signature_def)

def _is_valid_predict_signature(signature_def):
    if False:
        while True:
            i = 10
    "Determine whether the argument is a servable 'predict' SignatureDef."
    if signature_def.method_name != signature_constants.PREDICT_METHOD_NAME:
        return False
    if not signature_def.inputs.keys():
        return False
    if not signature_def.outputs.keys():
        return False
    return True

def _is_valid_regression_signature(signature_def):
    if False:
        i = 10
        return i + 15
    "Determine whether the argument is a servable 'regress' SignatureDef."
    if signature_def.method_name != signature_constants.REGRESS_METHOD_NAME:
        return False
    if set(signature_def.inputs.keys()) != set([signature_constants.REGRESS_INPUTS]):
        return False
    if signature_def.inputs[signature_constants.REGRESS_INPUTS].dtype != types_pb2.DT_STRING:
        return False
    if set(signature_def.outputs.keys()) != set([signature_constants.REGRESS_OUTPUTS]):
        return False
    if signature_def.outputs[signature_constants.REGRESS_OUTPUTS].dtype != types_pb2.DT_FLOAT:
        return False
    return True

def _is_valid_classification_signature(signature_def):
    if False:
        for i in range(10):
            print('nop')
    "Determine whether the argument is a servable 'classify' SignatureDef."
    if signature_def.method_name != signature_constants.CLASSIFY_METHOD_NAME:
        return False
    if set(signature_def.inputs.keys()) != set([signature_constants.CLASSIFY_INPUTS]):
        return False
    if signature_def.inputs[signature_constants.CLASSIFY_INPUTS].dtype != types_pb2.DT_STRING:
        return False
    allowed_outputs = set([signature_constants.CLASSIFY_OUTPUT_CLASSES, signature_constants.CLASSIFY_OUTPUT_SCORES])
    if not signature_def.outputs.keys():
        return False
    if set(signature_def.outputs.keys()) - allowed_outputs:
        return False
    if signature_constants.CLASSIFY_OUTPUT_CLASSES in signature_def.outputs and signature_def.outputs[signature_constants.CLASSIFY_OUTPUT_CLASSES].dtype != types_pb2.DT_STRING:
        return False
    if signature_constants.CLASSIFY_OUTPUT_SCORES in signature_def.outputs and signature_def.outputs[signature_constants.CLASSIFY_OUTPUT_SCORES].dtype != types_pb2.DT_FLOAT:
        return False
    return True

def op_signature_def(op, key):
    if False:
        for i in range(10):
            print('nop')
    "Creates a signature def with the output pointing to an op.\n\n  Note that op isn't strictly enforced to be an Op object, and may be a Tensor.\n  It is recommended to use the build_signature_def() function for Tensors.\n\n  Args:\n    op: An Op (or possibly Tensor).\n    key: Key to graph element in the SignatureDef outputs.\n\n  Returns:\n    A SignatureDef with a single output pointing to the op.\n  "
    return build_signature_def(outputs={key: utils.build_tensor_info_from_op(op)})

def load_op_from_signature_def(signature_def, key, import_scope=None):
    if False:
        for i in range(10):
            print('nop')
    'Load an Op from a SignatureDef created by op_signature_def().\n\n  Args:\n    signature_def: a SignatureDef proto\n    key: string key to op in the SignatureDef outputs.\n    import_scope: Scope used to import the op\n\n  Returns:\n    Op (or possibly Tensor) in the graph with the same name as saved in the\n      SignatureDef.\n\n  Raises:\n    NotFoundError: If the op could not be found in the graph.\n  '
    tensor_info = signature_def.outputs[key]
    try:
        return utils.get_element_from_tensor_info(tensor_info, import_scope=import_scope)
    except KeyError:
        raise errors.NotFoundError(None, None, f'The key "{key}" could not be found in the graph. Please make sure the SavedModel was created by the internal _SavedModelBuilder. If you are using the public API, please make sure the SignatureDef in the SavedModel does not contain the key "{key}".')