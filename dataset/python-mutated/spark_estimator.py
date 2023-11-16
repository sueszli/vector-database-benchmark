from abc import abstractmethod
from bigdl.orca.learn.base_estimator import BaseEstimator
from bigdl.dllib.utils.log4Error import invalidInputError

class Estimator(BaseEstimator):

    @abstractmethod
    def fit(self, data, epochs, batch_size=32, feature_cols=None, label_cols=None, validation_data=None, checkpoint_trigger=None):
        if False:
            i = 10
            return i + 15
        '\n        Train the model with train data.\n\n        :param data: train data.\n        :param epochs: number of epochs to train.\n        :param batch_size: total batch size for each iteration. Default: 32.\n        :param feature_cols: feature column names if train data is Spark DataFrame.\n        :param label_cols: label column names if train data is Spark DataFrame.\n        :param validation_data: validation data. Validation data type should be the same\n        as train data.\n        :param checkpoint_trigger: when to trigger checkpoint during training.\n        Should be a bigdl.orca.learn.trigger, like EveryEpoch(),\n        SeveralIteration(num_iterations),etc.\n        '
        invalidInputError(False, 'not implemented')

    @abstractmethod
    def predict(self, data, batch_size=4, feature_cols=None):
        if False:
            i = 10
            return i + 15
        "\n        Predict input data\n\n        :param data: data to be predicted.\n        :param batch_size: batch size per thread. Default: 4.\n        :param feature_cols: list of feature column names if input data is Spark DataFrame.\n        :return: predicted result.\n         If input data is XShards or tf.data.Dataset, the predict result is a XShards,\n         and the schema for each result is: {'prediction': predicted numpy array or\n          list of predicted numpy arrays}.\n        "
        invalidInputError(False, 'not implemented')

    @abstractmethod
    def evaluate(self, data, batch_size=32, feature_cols=None, label_cols=None):
        if False:
            while True:
                i = 10
        "\n        Evaluate model.\n\n        :param data: evaluation data.\n        :param batch_size: batch size per thread. Default: 32.\n        :param feature_cols: feature column names if train data is Spark DataFrame.\n        :param label_cols: label column names if train data is Spark DataFrame.\n        :return: evaluation result as a dictionary of {'metric name': metric value}\n        "
        invalidInputError(False, 'not implemented')

    @abstractmethod
    def get_model(self):
        if False:
            while True:
                i = 10
        '\n        Get the trained model\n\n        :return: Trained model\n        '
        invalidInputError(False, 'not implemented')

    @abstractmethod
    def save(self, model_path):
        if False:
            while True:
                i = 10
        '\n        Save model to model_path\n\n        :param model_path: path to save the trained model.\n        :return:\n        '
        invalidInputError(False, 'not implemented')

    @abstractmethod
    def load(self, model_path):
        if False:
            i = 10
            return i + 15
        '\n        Load existing model from model_path\n\n        :param model_path: Path to the existing model.\n        :return:\n        '
        invalidInputError(False, 'not implemented')

    def set_tensorboard(self, log_dir, app_name):
        if False:
            print('Hello World!')
        "\n        Set summary information during the training process for visualization purposes.\n        Saved summary can be viewed via TensorBoard.\n        In order to take effect, it needs to be called before fit.\n\n        Training summary will be saved to 'log_dir/app_name/train'\n        and validation summary (if any) will be saved to 'log_dir/app_name/validation'.\n\n        :param log_dir: The base directory path to store training and validation logs.\n        :param app_name: The name of the application.\n        "
        self.log_dir = log_dir
        self.app_name = app_name

    @abstractmethod
    def clear_gradient_clipping(self):
        if False:
            while True:
                i = 10
        '\n        Clear gradient clipping parameters. In this case, gradient clipping will not be applied.\n        In order to take effect, it needs to be called before fit.\n\n        :return:\n        '
        invalidInputError(False, 'not implemented')

    @abstractmethod
    def set_constant_gradient_clipping(self, min, max):
        if False:
            return 10
        '\n        Set constant gradient clipping during the training process.\n        In order to take effect, it needs to be called before fit.\n\n        :param min: The minimum value to clip by.\n        :param max: The maximum value to clip by.\n        :return:\n        '
        invalidInputError(False, 'not implemented')

    @abstractmethod
    def set_l2_norm_gradient_clipping(self, clip_norm):
        if False:
            i = 10
            return i + 15
        '\n        Clip gradient to a maximum L2-Norm during the training process.\n        In order to take effect, it needs to be called before fit.\n\n        :param clip_norm: Gradient L2-Norm threshold.\n        :return:\n        '
        invalidInputError(False, 'not implemented')

    @abstractmethod
    def get_train_summary(self, tag=None):
        if False:
            i = 10
            return i + 15
        '\n        Get the scalar from model train summary\n        Return list of summary data of [iteration_number, scalar_value, timestamp]\n\n        :param tag: The string variable represents the scalar wanted\n        '
        invalidInputError(False, 'not implemented')

    @abstractmethod
    def get_validation_summary(self, tag=None):
        if False:
            print('Hello World!')
        '\n        Get the scalar from model validation summary\n        Return list of summary data of [iteration_number, scalar_value, timestamp]\n\n        Note: The metric and tag may not be consistent\n        Please look up following form to pass tag parameter\n        Left side is your metric during compile\n        Right side is the tag you should pass\n        \'Accuracy\'                  |   \'Top1Accuracy\'\n        \'BinaryAccuracy\'            |   \'Top1Accuracy\'\n        \'CategoricalAccuracy\'       |   \'Top1Accuracy\'\n        \'SparseCategoricalAccuracy\' |   \'Top1Accuracy\'\n        \'AUC\'                       |   \'AucScore\'\n        \'HitRatio\'                  |   \'HitRate@k\' (k is Top-k)\n        \'Loss\'                      |   \'Loss\'\n        \'MAE\'                       |   \'MAE\'\n        \'NDCG\'                      |   \'NDCG\'\n        \'TFValidationMethod\'        |   \'${name + " " + valMethod.toString()}\'\n        \'Top5Accuracy\'              |   \'Top5Accuracy\'\n        \'TreeNNAccuracy\'            |   \'TreeNNAccuracy()\'\n        \'MeanAveragePrecision\'      |   \'MAP@k\' (k is Top-k) (BigDL)\n        \'MeanAveragePrecision\'      |   \'PascalMeanAveragePrecision\' (Zoo)\n        \'StatelessMetric\'           |   \'${name}\'\n\n        :param tag: The string variable represents the scalar wanted\n        '
        invalidInputError(False, 'not implemented')

    @abstractmethod
    def load_orca_checkpoint(self, path, version):
        if False:
            return 10
        '\n        Load specified Orca checkpoint.\n\n        :param path: checkpoint directory which contains model.* and\n        optimMethod-TFParkTraining.* files.\n        :param version: checkpoint version, which is the suffix of model.* file,\n        i.e., for modle.4 file, the version is 4.\n        '
        invalidInputError(False, 'not implemented')

    def shutdown(self):
        if False:
            while True:
                i = 10
        '\n        Releases resources.\n\n        :return:\n        '
        pass