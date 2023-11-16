"""Options for saving SavedModels."""
import enum
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export
is_oss = True

@tf_export('saved_model.experimental.VariablePolicy')
class VariablePolicy(enum.Enum):
    """Enum defining options for variable handling when saving.

  NONE
    No policy applied: Distributed variables are saved as one variable, with no
    device attached.

  SAVE_VARIABLE_DEVICES
    When saving variables, also save their device assignment.
    This is useful if one wants to hardcode devices in saved models, but it also
    makes them non-portable if soft device placement is disabled (more details
    in `tf.config.set_soft_device_placement`). This is currently not
    fully supported by `saved_model.load`, and is mainly intended to be used
    when one will be reading the saved model at a lower API level. In the
    example below, the graph saved by the call to `saved_model.save` will have
    the variable devices correctly specified:
    ```python
    exported = tf.train.Checkpoint()
    with tf.device('/GPU:0'):
      exported.x_gpu = tf.Variable(1.0)
    with tf.device('/CPU:0'):
      exported.x_cpu = tf.Variable(1.0)
    tf.saved_model.save(exported, export_dir,
        options = tf.saved_model.SaveOptions(
            experimental_variable_policy=
              tf.saved_model.experimental.VariablePolicy.SAVE_VARIABLE_DEVICES))
    ```
    Distributed variables are still saved as one variable under this policy.

  EXPAND_DISTRIBUTED_VARIABLES
    Distributed variables will be saved with information about their components,
    allowing for their restoration on load. Also, the saved graph will contain
    references to those variables. This is useful when one wants to use the
    model for training in environments where the original distribution strategy
    is not available.
  """
    NONE = None
    SAVE_VARIABLE_DEVICES = 'save_variable_devices'
    EXPAND_DISTRIBUTED_VARIABLES = 'expand_distributed_variables'

    def _save_variable_devices(self):
        if False:
            for i in range(10):
                print('nop')
        'Checks whether variable devices should be saved.'
        return self != VariablePolicy.NONE

    def _expand_distributed_variables(self):
        if False:
            return 10
        'Checks whether distributed variables should be expanded.'
        return self == VariablePolicy.EXPAND_DISTRIBUTED_VARIABLES

    @staticmethod
    def from_obj(obj):
        if False:
            while True:
                i = 10
        'Tries to convert `obj` to a VariablePolicy instance.'
        if obj is None:
            return VariablePolicy.NONE
        if isinstance(obj, VariablePolicy):
            return obj
        key = str(obj).lower()
        for policy in VariablePolicy:
            if key == policy.value:
                return policy
        raise ValueError(f'Received invalid VariablePolicy value: {obj}.')

@tf_export('saved_model.SaveOptions')
class SaveOptions:
    """Options for saving to SavedModel.

  This function may be used in the `options` argument in functions that
  save a SavedModel (`tf.saved_model.save`, `tf.keras.models.save_model`).
  """
    __slots__ = ('namespace_whitelist', 'save_debug_info', 'function_aliases', 'experimental_io_device', 'experimental_variable_policy', 'experimental_custom_gradients', 'experimental_image_format', 'experimental_skip_saver')

    def __init__(self, namespace_whitelist=None, save_debug_info=False, function_aliases=None, experimental_io_device=None, experimental_variable_policy=None, experimental_custom_gradients=True, experimental_image_format=False, experimental_skip_saver=False):
        if False:
            i = 10
            return i + 15
        'Creates an object that stores options for SavedModel saving.\n\n    Args:\n      namespace_whitelist: List of strings containing op namespaces to whitelist\n        when saving a model. Saving an object that uses namespaced ops must\n        explicitly add all namespaces to the whitelist. The namespaced ops must\n        be registered into the framework when loading the SavedModel. If no\n        whitelist is provided, all namespaced ops will be allowed.\n      save_debug_info: Boolean indicating whether debug information is saved. If\n        True, then a debug/saved_model_debug_info.pb file will be written with\n        the contents of a GraphDebugInfo binary protocol buffer containing stack\n        trace information for all ops and functions that are saved.\n      function_aliases: Python dict. Mapping from string to object returned by\n        @tf.function. A single tf.function can generate many ConcreteFunctions.\n        If a downstream tool wants to refer to all concrete functions generated\n        by a single tf.function you can use the `function_aliases` argument to\n        store a map from the alias name to all concrete function names. E.g. >>>\n        class Adder(tf.Module): ...   @tf.function ...   def double(self, x):\n        ...     return x + x  >>> model = Adder() >>>\n        model.double.get_concrete_function( ...   tf.TensorSpec(shape=[],\n        dtype=tf.float32, name="float_input")) >>>\n        model.double.get_concrete_function( ...   tf.TensorSpec(shape=[],\n        dtype=tf.string, name="string_input"))  >>> options =\n        tf.saved_model.SaveOptions( ...   function_aliases={\'double\':\n        model.double}) >>> tf.saved_model.save(model, \'/tmp/adder\',\n        options=options)\n      experimental_io_device: string. Applies in a distributed setting.\n        Tensorflow device to use to access the filesystem. If `None` (default)\n        then for each variable the filesystem is accessed from the CPU:0 device\n        of the host where that variable is assigned. If specified, the\n        filesystem is instead accessed from that device for all variables.  This\n        is for example useful if you want to save to a local directory, such as\n        "/tmp" when running in a distributed setting. In that case pass a device\n        for the host where the "/tmp" directory is accessible.\n      experimental_variable_policy: The policy to apply to variables when\n        saving. This is either a `saved_model.experimental.VariablePolicy` enum\n        instance or one of its value strings (case is not important). See that\n        enum documentation for details. A value of `None` corresponds to the\n        default policy.\n      experimental_custom_gradients: Boolean. When True, will save traced\n        gradient functions for the functions decorated by `tf.custom_gradient`.\n        Defaults to `True`.\n      experimental_image_format: New (highly) experimental format that is\n        capable of saving models larger than the 2GB protobuf limit. Enabling\n        this option will likely break compatibility with downstream consumers.\n        This option is currently disabled in OSS.\n      experimental_skip_saver: If True, will prevent SavedModel from creating\n        its native checkpointing ops - this is for models that do not use\n        SavedModel\'s native checkpointing functionality to avoid the costs\n        associated with creating and serializing those ops.\n    '
        self.namespace_whitelist = _validate_namespace_whitelist(namespace_whitelist)
        self.save_debug_info = save_debug_info
        self.function_aliases = function_aliases if function_aliases else dict()
        self.experimental_custom_gradients = experimental_custom_gradients
        self.experimental_io_device = experimental_io_device
        self.experimental_variable_policy = VariablePolicy.from_obj(experimental_variable_policy)
        self.experimental_skip_saver = experimental_skip_saver
        if experimental_image_format and is_oss:
            raise ValueError('The option `experimental_image_format` is disabled in OSS.')
        self.experimental_image_format = experimental_image_format

def _validate_namespace_whitelist(namespace_whitelist):
    if False:
        while True:
            i = 10
    'Validates namespace whitelist argument.'
    if namespace_whitelist is None:
        return None
    if not isinstance(namespace_whitelist, list):
        raise TypeError(f'`namespace_whitelist` must be a list of strings. Got: {namespace_whitelist} with type {type(namespace_whitelist)}.')
    processed = []
    for namespace in namespace_whitelist:
        if not isinstance(namespace, str):
            raise ValueError(f'Whitelisted namespace must be a string. Got: {namespace} of type {type(namespace)}.')
        processed.append(compat.as_str(namespace))
    return processed