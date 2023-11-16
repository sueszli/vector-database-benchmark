"""Toggle to enable/disable resource variables."""
from tensorflow.python import tf2
from tensorflow.python.eager import monitoring
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
_api_usage_gauge = monitoring.BoolGauge('/tensorflow/api/resource_variables', 'Whether resource_variables_toggle.enable_resource_variables() is called.')
_DEFAULT_USE_RESOURCE = tf2.enabled()

@tf_export(v1=['enable_resource_variables'])
def enable_resource_variables() -> None:
    if False:
        print('Hello World!')
    'Creates resource variables by default.\n\n  Resource variables are improved versions of TensorFlow variables with a\n  well-defined memory model. Accessing a resource variable reads its value, and\n  all ops which access a specific read value of the variable are guaranteed to\n  see the same value for that tensor. Writes which happen after a read (by\n  having a control or data dependency on the read) are guaranteed not to affect\n  the value of the read tensor, and similarly writes which happen before a read\n  are guaranteed to affect the value. No guarantees are made about unordered\n  read/write pairs.\n\n  Calling tf.enable_resource_variables() lets you opt-in to this TensorFlow 2.0\n  feature.\n  '
    global _DEFAULT_USE_RESOURCE
    _DEFAULT_USE_RESOURCE = True
    logging.vlog(1, 'Enabling resource variables')
    _api_usage_gauge.get_cell().set(True)

@deprecation.deprecated(None, 'non-resource variables are not supported in the long term')
@tf_export(v1=['disable_resource_variables'])
def disable_resource_variables() -> None:
    if False:
        return 10
    'Opts out of resource variables.\n\n  If your code needs tf.disable_resource_variables() to be called to work\n  properly please file a bug.\n  '
    global _DEFAULT_USE_RESOURCE
    _DEFAULT_USE_RESOURCE = False
    logging.vlog(1, 'Disabling resource variables')
    _api_usage_gauge.get_cell().set(False)

@tf_export(v1=['resource_variables_enabled'])
def resource_variables_enabled() -> bool:
    if False:
        return 10
    'Returns `True` if resource variables are enabled.\n\n  Resource variables are improved versions of TensorFlow variables with a\n  well-defined memory model. Accessing a resource variable reads its value, and\n  all ops which access a specific read value of the variable are guaranteed to\n  see the same value for that tensor. Writes which happen after a read (by\n  having a control or data dependency on the read) are guaranteed not to affect\n  the value of the read tensor, and similarly writes which happen before a read\n  are guaranteed to affect the value. No guarantees are made about unordered\n  read/write pairs.\n\n  Calling tf.enable_resource_variables() lets you opt-in to this TensorFlow 2.0\n  feature.\n  '
    return _DEFAULT_USE_RESOURCE