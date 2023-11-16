"""SignatureDef method name utility functions.

Utility functions for manipulating signature_def.method_names.
"""
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader_impl as loader
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export

@tf_export(v1=['saved_model.signature_def_utils.MethodNameUpdater'])
class MethodNameUpdater(object):
    """Updates the method name(s) of the SavedModel stored in the given path.

  The `MethodNameUpdater` class provides the functionality to update the method
  name field in the signature_defs of the given SavedModel. For example, it
  can be used to replace the `predict` `method_name` to `regress`.

  Typical usages of the `MethodNameUpdater`
  ```python
  ...
  updater = tf.compat.v1.saved_model.signature_def_utils.MethodNameUpdater(
      export_dir)
  # Update all signature_defs with key "foo" in all meta graph defs.
  updater.replace_method_name(signature_key="foo", method_name="regress")
  # Update a single signature_def with key "bar" in the meta graph def with
  # tags ["serve"]
  updater.replace_method_name(signature_key="bar", method_name="classify",
                              tags="serve")
  updater.save(new_export_dir)
  ```

  Note: This function will only be available through the v1 compatibility
  library as tf.compat.v1.saved_model.builder.MethodNameUpdater.
  """

    def __init__(self, export_dir):
        if False:
            print('Hello World!')
        'Creates an MethodNameUpdater object.\n\n    Args:\n      export_dir: Directory containing the SavedModel files.\n\n    Raises:\n      IOError: If the saved model file does not exist, or cannot be successfully\n      parsed.\n    '
        self._export_dir = export_dir
        self._saved_model = loader.parse_saved_model(export_dir)

    def replace_method_name(self, signature_key, method_name, tags=None):
        if False:
            for i in range(10):
                print('nop')
        'Replaces the method_name in the specified signature_def.\n\n    This will match and replace multiple sig defs iff tags is None (i.e when\n    multiple `MetaGraph`s have a signature_def with the same key).\n    If tags is not None, this will only replace a single signature_def in the\n    `MetaGraph` with matching tags.\n\n    Args:\n      signature_key: Key of the signature_def to be updated.\n      method_name: new method_name to replace the existing one.\n      tags: A tag or sequence of tags identifying the `MetaGraph` to update. If\n          None, all meta graphs will be updated.\n    Raises:\n      ValueError: if signature_key or method_name are not defined or\n          if no metagraphs were found with the associated tags or\n          if no meta graph has a signature_def that matches signature_key.\n    '
        if not signature_key:
            raise ValueError('`signature_key` must be defined.')
        if not method_name:
            raise ValueError('`method_name` must be defined.')
        if tags is not None and (not isinstance(tags, list)):
            tags = [tags]
        found_match = False
        for meta_graph_def in self._saved_model.meta_graphs:
            if tags is None or set(tags) == set(meta_graph_def.meta_info_def.tags):
                if signature_key not in meta_graph_def.signature_def:
                    raise ValueError(f"MetaGraphDef associated with tags {tags} does not have a signature_def with key: '{signature_key}'. This means either you specified the wrong signature key or forgot to put the signature_def with the corresponding key in your SavedModel.")
                meta_graph_def.signature_def[signature_key].method_name = method_name
                found_match = True
        if not found_match:
            raise ValueError(f'MetaGraphDef associated with tags {tags} could not be found in SavedModel. This means either you specified invalid tags or your SavedModel does not have a MetaGraphDef with the specified tags.')

    def save(self, new_export_dir=None):
        if False:
            for i in range(10):
                print('nop')
        'Saves the updated `SavedModel`.\n\n    Args:\n      new_export_dir: Path where the updated `SavedModel` will be saved. If\n          None, the input `SavedModel` will be overriden with the updates.\n\n    Raises:\n      errors.OpError: If there are errors during the file save operation.\n    '
        is_input_text_proto = file_io.file_exists(file_io.join(compat.as_bytes(self._export_dir), compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT)))
        if not new_export_dir:
            new_export_dir = self._export_dir
        if is_input_text_proto:
            path = file_io.join(compat.as_bytes(new_export_dir), compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT))
            file_io.write_string_to_file(path, str(self._saved_model))
        else:
            path = file_io.join(compat.as_bytes(new_export_dir), compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))
            file_io.write_string_to_file(path, self._saved_model.SerializeToString(deterministic=True))
        tf_logging.info('SavedModel written to: %s', compat.as_text(path))