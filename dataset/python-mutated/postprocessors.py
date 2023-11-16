import numpy as np
from autokeras.engine import preprocessor

class PostProcessor(preprocessor.TargetPreprocessor):

    def transform(self, dataset):
        if False:
            print('Hello World!')
        return dataset

class SigmoidPostprocessor(PostProcessor):
    """Postprocessor for sigmoid outputs."""

    def postprocess(self, data):
        if False:
            i = 10
            return i + 15
        'Transform probabilities to zeros and ones.\n\n        # Arguments\n            data: numpy.ndarray. The output probabilities of the classification\n                head.\n\n        # Returns\n            numpy.ndarray. The zeros and ones predictions.\n        '
        data[data < 0.5] = 0
        data[data > 0.5] = 1
        return data

class SoftmaxPostprocessor(PostProcessor):
    """Postprocessor for softmax outputs."""

    def postprocess(self, data):
        if False:
            for i in range(10):
                print('nop')
        'Transform probabilities to zeros and ones.\n\n        # Arguments\n            data: numpy.ndarray. The output probabilities of the classification\n                head.\n\n        # Returns\n            numpy.ndarray. The zeros and ones predictions.\n        '
        idx = np.argmax(data, axis=-1)
        data = np.zeros(data.shape)
        data[np.arange(data.shape[0]), idx] = 1
        return data