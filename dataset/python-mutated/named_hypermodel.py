import keras_tuner
from tensorflow import keras
from autokeras.engine import serializable
from autokeras.utils import utils

class NamedHyperModel(keras_tuner.HyperModel, serializable.Serializable):
    """

    # Arguments
        name: String. The name of the HyperModel. If unspecified, it will be set
            automatically with the class name.
    """

    def __init__(self, name: str=None, **kwargs):
        if False:
            while True:
                i = 10
        if not name:
            prefix = self.__class__.__name__
            name = prefix + '_' + str(keras.backend.get_uid(prefix))
            name = utils.to_snake_case(name)
        super().__init__(name=name, **kwargs)

    def get_config(self):
        if False:
            i = 10
            return i + 15
        'Get the configuration of the preprocessor.\n\n        # Returns\n            A dictionary of configurations of the preprocessor.\n        '
        return {'name': self.name, 'tunable': self.tunable}