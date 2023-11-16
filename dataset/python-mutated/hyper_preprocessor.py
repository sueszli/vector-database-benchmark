from autokeras.engine import named_hypermodel

class HyperPreprocessor(named_hypermodel.NamedHyperModel):
    """Input data preprocessor search space.

    This class defines the search space for a Preprocessor.
    """

    def build(self, hp, dataset):
        if False:
            while True:
                i = 10
        'Build the `tf.data` input preprocessor.\n\n        # Arguments\n            hp: `HyperParameters` instance. The hyperparameters for building the\n                a Preprocessor.\n            dataset: tf.data.Dataset.\n\n        # Returns\n            an instance of Preprocessor.\n        '
        raise NotImplementedError