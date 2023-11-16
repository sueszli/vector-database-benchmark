import os
import pickle
import numpy as np
from sklearn.externals import joblib

class MyPredictor(object):
    """An example Predictor for an AI Platform custom prediction routine."""

    def __init__(self, model, preprocessor):
        if False:
            return 10
        'Stores artifacts for prediction. Only initialized via `from_path`.'
        self._model = model
        self._preprocessor = preprocessor

    def predict(self, instances, **kwargs):
        if False:
            print('Hello World!')
        'Performs custom prediction.\n\n        Preprocesses inputs, then performs prediction using the trained\n        scikit-learn model.\n\n        Args:\n            instances: A list of prediction input instances.\n            **kwargs: A dictionary of keyword args provided as additional\n                fields on the predict request body.\n\n        Returns:\n            A list of outputs containing the prediction results.\n        '
        inputs = np.asarray(instances)
        preprocessed_inputs = self._preprocessor.preprocess(inputs)
        outputs = self._model.predict(preprocessed_inputs)
        return outputs.tolist()

    @classmethod
    def from_path(cls, model_dir):
        if False:
            while True:
                i = 10
        'Creates an instance of MyPredictor using the given path.\n\n        This loads artifacts that have been copied from your model directory in\n        Cloud Storage. MyPredictor uses them during prediction.\n\n        Args:\n            model_dir: The local directory that contains the trained\n                scikit-learn model and the pickled preprocessor instance. These\n                are copied from the Cloud Storage model directory you provide\n                when you deploy a version resource.\n\n        Returns:\n            An instance of `MyPredictor`.\n        '
        model_path = os.path.join(model_dir, 'model.joblib')
        model = joblib.load(model_path)
        preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        return cls(model, preprocessor)