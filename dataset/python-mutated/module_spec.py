"""The ModuleSpec interface, for the deprecated hub.Module class."""
import abc

class ModuleSpec(object):
    """Represents the contents of a hub.Module before it has been instantiated.

  Warning: Deprecated. This belongs to the hub.Module API and TF1 Hub format.
  For TF2, switch to plain SavedModels and hub.load().

  A ModuleSpec is the blueprint used by `Module` to create one or more instances
  of a specific module in one or more graphs. The details on how to construct
  the Module are internal to the library implementation but methods to inspect
  a Module interface are public.

  Note: Do not instantiate this class directly. Use `hub.load_module_spec` or
  `hub.create_module_spec`.

  THIS FUNCTION IS DEPRECATED.
  """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        if False:
            print('Hello World!')
        'Do not instantiate directly.'
        pass

    def export(self, path, _sentinel=None, checkpoint_path=None, name_transform_fn=None):
        if False:
            print('Hello World!')
        'Exports a ModuleSpec with weights taken from a checkpoint.\n\n    This is an helper to export modules directly from a ModuleSpec\n    without having to create a session and set the variables to the\n    intended values.\n\n    Example usage:\n\n    ```python\n    spec = hub.create_module_spec(module_fn)\n    spec.export("/path/to/export_module",\n                checkpoint_path="/path/to/training_model")\n    ```\n\n    In some cases, the variable name in the checkpoint does not match\n    the variable name in the module. It is possible to work around that\n    by providing a checkpoint_map_fn that performs the variable mapping.\n    For example with: `name_transform_fn = lambda x: "extra_scope/" + x`.\n\n    Args:\n      path: path where to export the module to.\n      _sentinel: used to prevent positional arguments besides `path`.\n      checkpoint_path: path where to load the weights for the module.\n        Mandatory parameter and must be passed by name.\n      name_transform_fn: optional function to provide mapping between\n        variable name in the module and the variable name in the checkpoint.\n\n    Raises:\n      ValueError: if missing mandatory `checkpoint_path` parameter.\n    '
        from tensorflow_hub.module import export_module_spec
        if not checkpoint_path:
            raise ValueError('Missing mandatory `checkpoint_path` parameter')
        name_transform_fn = name_transform_fn or (lambda x: x)
        export_module_spec(self, path, checkpoint_path, name_transform_fn)

    @abc.abstractmethod
    def get_signature_names(self, tags=None):
        if False:
            print('Hello World!')
        "Returns the module's signature names as an iterable of strings."
        pass

    @abc.abstractmethod
    def get_tags(self):
        if False:
            for i in range(10):
                print('nop')
        'Lists the graph variants as an iterable of set of tags.'
        return [set()]

    @abc.abstractmethod
    def get_input_info_dict(self, signature=None, tags=None):
        if False:
            print('Hello World!')
        'Describes the inputs required by a signature.\n\n    Args:\n      signature: A string with the signature to get inputs information for.\n        If None, the default signature is used if defined.\n      tags: Optional set of strings, specifying the graph variant to query.\n\n    Returns:\n      A dict from input names to objects that provide (1) a property `dtype`,\n      (2) a method `get_shape()`, (3) a read-only boolean property `is_sparse`,\n      (4) a read-only boolean property `is_composite`; and (5) a read-only\n      property `type_spec`. The first two are compatible with the common API of\n      Tensor, SparseTensor, and RaggedTensor objects.\n\n    Raises:\n      KeyError: if there is no such signature or graph variant.\n    '
        raise NotImplementedError()

    @abc.abstractmethod
    def get_output_info_dict(self, signature=None, tags=None):
        if False:
            while True:
                i = 10
        'Describes the outputs provided by a signature.\n\n    Args:\n      signature: A string with the signature to get ouputs information for.\n        If None, the default signature is used if defined.\n      tags: Optional set of strings, specifying the graph variant to query.\n\n    Returns: A dict from input names to objects that provide (1) a property\n      `dtype`, (2) a method `get_shape()`,(3) a read-only boolean property\n      `is_sparse`, (4) a read-only boolean property `is_composite`; and (5) a\n      read-only property `type_spec`. The first two are compatible with the\n      common API of Tensor, SparseTensor, and RaggedTensor objects.\n\n    Raises:\n      KeyError: if there is no such signature or graph variant.\n    '
        raise NotImplementedError()

    def get_attached_message(self, key, message_type, tags=None, required=False):
        if False:
            i = 10
            return i + 15
        'Returns the message attached to the module under the given key, or None.\n\n    Module publishers can attach protocol messages to modules at creation time\n    to provide module consumers with additional information, e.g., on module\n    usage or provenance (see see hub.attach_message()). A typical use would be\n    to store a small set of named values with modules of a certain type so\n    that a support library for consumers of such modules can be parametric\n    in those values.\n\n    This method can also be called on a Module instantiated from a ModuleSpec,\n    then `tags` are set to those used in module instatiation.\n\n    Args:\n      key: A string with the key of an attached message.\n      message_type: A concrete protocol message class (*not* object) used\n        to parse the attached message from its serialized representation.\n        The message type for a particular key must be advertised with the key.\n      tags: Optional set of strings, specifying the graph variant from which\n        to read the attached message.\n      required: An optional boolean. Setting it true changes the effect of\n        an unknown `key` from returning None to raising a KeyError with text\n        about attached messages.\n\n    Returns:\n      An instance of `message_type` with the message contents attached to the\n      module, or `None` if `key` is unknown and `required` is False.\n\n    Raises:\n      KeyError: if `key` is unknown and `required` is True.\n    '
        attached_bytes = self._get_attached_bytes(key, tags)
        if attached_bytes is None:
            if required:
                raise KeyError("No attached message for key '%s' in graph version %s of Hub Module" % (key, sorted(tags or [])))
            else:
                return None
        message = message_type()
        message.ParseFromString(attached_bytes)
        return message

    @abc.abstractmethod
    def _get_attached_bytes(self, key, tags):
        if False:
            return 10
        'Internal implementation of the storage of attached messages.\n\n    Args:\n      key: The `key` argument to get_attached_message().\n      tags: The `tags` argument to get_attached_message().\n\n    Returns:\n      The serialized message attached under `key` to the graph version\n      identified by `tags`, or None if absent.\n    '
        raise NotImplementedError()

    @abc.abstractmethod
    def _create_impl(self, name, trainable, tags):
        if False:
            for i in range(10):
                print('nop')
        'Internal.\n\n    Args:\n      name: A string with the an unused name scope.\n      trainable: A boolean, whether the Module is to be instantiated as\n        trainable.\n      tags: A set of strings specifying the graph variant to use.\n\n    Returns:\n      A ModuleImpl.\n    '
        raise NotImplementedError()