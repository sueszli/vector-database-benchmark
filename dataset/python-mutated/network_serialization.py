"""Classes and functions implementing to Network SavedModel serialization."""
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import model_serialization

class NetworkSavedModelSaver(model_serialization.ModelSavedModelSaver):
    """Network serialization."""

    @property
    def object_identifier(self):
        if False:
            i = 10
            return i + 15
        return constants.NETWORK_IDENTIFIER