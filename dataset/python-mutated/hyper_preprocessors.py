from autokeras import preprocessors
from autokeras.engine import hyper_preprocessor
from autokeras.utils import utils

def serialize(encoder):
    if False:
        return 10
    return utils.serialize_keras_object(encoder)

def deserialize(config, custom_objects=None):
    if False:
        while True:
            i = 10
    return utils.deserialize_keras_object(config, module_objects=globals(), custom_objects=custom_objects, printable_module_name='preprocessors')

class DefaultHyperPreprocessor(hyper_preprocessor.HyperPreprocessor):
    """HyperPreprocessor without Hyperparameters to tune.

    It would always return the same preprocessor. No hyperparameters to be
    tuned.

    # Arguments
        preprocessor: The Preprocessor to return when calling build.
    """

    def __init__(self, preprocessor, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self.preprocessor = preprocessor

    def build(self, hp, dataset):
        if False:
            i = 10
            return i + 15
        return self.preprocessor

    def get_config(self):
        if False:
            return 10
        config = super().get_config()
        config.update({'preprocessor': preprocessors.serialize(self.preprocessor)})
        return config

    @classmethod
    def from_config(cls, config):
        if False:
            return 10
        config['preprocessor'] = preprocessors.deserialize(config['preprocessor'])
        return super().from_config(config)