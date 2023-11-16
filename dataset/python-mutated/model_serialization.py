"""Classes and functions implementing to Model SavedModel serialization."""
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import layer_serialization
from tensorflow.python.keras.saving.saved_model import save_impl

class ModelSavedModelSaver(layer_serialization.LayerSavedModelSaver):
    """Model SavedModel serialization."""

    @property
    def object_identifier(self):
        if False:
            print('Hello World!')
        return constants.MODEL_IDENTIFIER

    def _python_properties_internal(self):
        if False:
            i = 10
            return i + 15
        metadata = super(ModelSavedModelSaver, self)._python_properties_internal()
        metadata.pop('stateful')
        metadata['is_graph_network'] = self.obj._is_graph_network
        metadata['save_spec'] = self.obj._get_save_spec(dynamic_batch=False)
        metadata.update(saving_utils.model_metadata(self.obj, include_optimizer=True, require_config=False))
        return metadata

    def _get_serialized_attributes_internal(self, serialization_cache):
        if False:
            return 10
        default_signature = None
        if len(serialization_cache[constants.KERAS_CACHE_KEY]) == 1:
            default_signature = save_impl.default_save_signature(self.obj)
        (objects, functions) = super(ModelSavedModelSaver, self)._get_serialized_attributes_internal(serialization_cache)
        functions['_default_save_signature'] = default_signature
        return (objects, functions)

class SequentialSavedModelSaver(ModelSavedModelSaver):

    @property
    def object_identifier(self):
        if False:
            for i in range(10):
                print('nop')
        return constants.SEQUENTIAL_IDENTIFIER