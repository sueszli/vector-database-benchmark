class Predictor(object):
    """Interface for constructing custom predictors."""

    def predict(self, instances, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Performs custom prediction.\n\n        Instances are the decoded values from the request. They have already\n        been deserialized from JSON.\n\n        Args:\n            instances: A list of prediction input instances.\n            **kwargs: A dictionary of keyword args provided as additional\n                fields on the predict request body.\n\n        Returns:\n            A list of outputs containing the prediction results. This list must\n            be JSON serializable.\n        '
        raise NotImplementedError()

    @classmethod
    def from_path(cls, model_dir):
        if False:
            while True:
                i = 10
        'Creates an instance of Predictor using the given path.\n\n        Loading of the predictor should be done in this method.\n\n        Args:\n            model_dir: The local directory that contains the exported model\n                file along with any additional files uploaded when creating the\n                version resource.\n\n        Returns:\n            An instance implementing this Predictor class.\n        '
        raise NotImplementedError()