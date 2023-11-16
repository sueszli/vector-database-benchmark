from abc import abstractmethod
from bigdl.orca.learn.base_estimator import BaseEstimator

class Estimator(BaseEstimator):

    @abstractmethod
    def fit(self, data, epochs, batch_size):
        if False:
            for i in range(10):
                print('nop')
        '\n        Train the model with train data.\n\n        :param data: train data.\n        :param epochs: number of epochs to train.\n        :param batch_size: total batch size for each iteration.\n        '
        pass

    @abstractmethod
    def predict(self, data, batch_size):
        if False:
            for i in range(10):
                print('nop')
        '\n        Predict input data.\n\n        :param data: data to be predicted.\n        :param batch_size: batch size per thread. Default: 4.\n        :return: predicted result.\n        '
        pass

    @abstractmethod
    def evaluate(self, data, batch_size, num_steps=None):
        if False:
            return 10
        "\n        Evaluate model.\n\n        :param data: evaluation data.\n        :param batch_size: batch size per thread.\n        :param num_steps: Number of batches to compute update steps on. This corresponds also to\n        the number of times TorchRunner.validate_batch is called.\n        :return: evaluation result as a dictionary of {'metric name': metric value}\n        "
        pass

    @abstractmethod
    def get_model(self):
        if False:
            return 10
        '\n        Get the trained model.\n\n        :return: Trained model\n        '
        pass

    @abstractmethod
    def save(self, model_path):
        if False:
            i = 10
            return i + 15
        '\n        Save model to model_path.\n\n        :param model_path: path to save the trained model.\n        :return:\n        '
        pass

    @abstractmethod
    def load(self, model_path):
        if False:
            for i in range(10):
                print('nop')
        '\n        Load existing model from model_path\n\n        :param model_path: Path to the existing model.\n        :return:\n        '
        pass

    @abstractmethod
    def shutdown(self):
        if False:
            print('Hello World!')
        '\n        Shut down workers and releases resources.\n\n        :return:\n        '
        pass