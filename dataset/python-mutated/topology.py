import bigdl.dllib.keras.autograd as autograd
from bigdl.dllib.feature.image import ImageSet
from bigdl.dllib.feature.text import TextSet
from bigdl.dllib.feature.common import FeatureSet
from bigdl.dllib.keras.base import ZooKerasLayer
from bigdl.dllib.keras.utils import *
from bigdl.dllib.nn.layer import Layer
from bigdl.dllib.utils.file_utils import callZooFunc
if sys.version >= '3':
    long = int
    unicode = str

class KerasNet(ZooKerasLayer):

    def save(self, path, over_write=False):
        if False:
            return 10
        invalidInputError(False, 'This is a deprecated method. Please use saveModel instead.')

    def saveModel(self, modelPath, weightPath=None, over_write=False):
        if False:
            print('Hello World!')
        '\n        Save this module to path with protobuf format.\n        :param modelPath: The path to save module, local file system,\n                          HDFS and Amazon S3 is supported.\n                          HDFS path should be like "hdfs://[host]:[port]/xxx"\n                          Amazon S3 path should be like "s3a://bucket/xxx"\n        :param weightPath: The Path for the parameters\n        :param over_write: override the existing model on modelPath or not.\n        '
        super(KerasNet, self).saveModel(modelPath=modelPath, weightPath=weightPath, over_write=over_write)

    def compile(self, optimizer, loss, metrics=None):
        if False:
            while True:
                i = 10
        "\n        Configure the learning process. It MUST be called before fit or evaluate.\n\n        # Arguments\n        optimizer: Optimization method to be used. One can alternatively pass in the corresponding\n                   string representation, such as 'sgd'.\n        loss: Criterion to be used. One can alternatively pass in the corresponding string\n              representation, such as 'mse'.\n        metrics: List of validation methods to be used. Default is None if no validation is needed.\n                 For convenience, string representations are supported: 'accuracy' (or 'acc'),\n                 'top5accuracy' (or 'top5acc'), 'mae', 'auc', 'treennaccuracy' and 'loss'.\n                 For example, you can either use [Accuracy()] or ['accuracy'].\n        "
        if isinstance(optimizer, six.string_types):
            optimizer = to_bigdl_optim_method(optimizer)
        criterion = loss
        if isinstance(loss, six.string_types):
            criterion = to_bigdl_criterion(loss)
        if callable(loss):
            from bigdl.dllib.keras.autograd import CustomLoss
            criterion = CustomLoss(loss, self.get_output_shape()[1:])
        if metrics and all((isinstance(metric, six.string_types) for metric in metrics)):
            metrics = to_bigdl_metrics(metrics, loss)
        callZooFunc(self.bigdl_type, 'zooCompile', self.value, optimizer, criterion, metrics)

    def set_tensorboard(self, log_dir, app_name):
        if False:
            i = 10
            return i + 15
        "\n        Set summary information during the training process for visualization purposes.\n        Saved summary can be viewed via TensorBoard.\n        In order to take effect, it needs to be called before fit.\n\n        Training summary will be saved to 'log_dir/app_name/train'\n        and validation summary (if any) will be saved to 'log_dir/app_name/validation'.\n\n        # Arguments\n        log_dir: The base directory path to store training and validation logs.\n        app_name: The name of the application.\n        "
        callZooFunc(self.bigdl_type, 'zooSetTensorBoard', self.value, log_dir, app_name)

    def get_train_summary(self, tag=None):
        if False:
            print('Hello World!')
        '\n        Get the scalar from model train summary\n        Return 2-D array like object which could be converted\n        by nd.array()\n        # Arguments\n        tag: The string variable represents the scalar wanted\n        '
        if tag != 'Loss' and tag != 'LearningRate' and (tag != 'Throughput'):
            invalidInputError(False, 'Only "Loss", "LearningRate", "Throughput"' + 'are supported in train summary')
        return callZooFunc(self.bigdl_type, 'zooGetScalarFromSummary', self.value, tag, 'Train')

    def get_validation_summary(self, tag=None):
        if False:
            i = 10
            return i + 15
        '\n        Get the scalar from model validation summary\n        Return 2-D array like object which could be converted\n        by np.array()\n\n        Note: The metric and tag may not be consistent\n        Please look up following form to pass tag parameter\n        Left side is your metric during compile\n        Right side is the tag you should pass\n        \'Accuracy\'                  |   \'Top1Accuracy\'\n        \'BinaryAccuracy\'            |   \'Top1Accuracy\'\n        \'CategoricalAccuracy\'       |   \'Top1Accuracy\'\n        \'SparseCategoricalAccuracy\' |   \'Top1Accuracy\'\n        \'AUC\'                       |   \'AucScore\'\n        \'HitRatio\'                  |   \'HitRate@k\' (k is Top-k)\n        \'Loss\'                      |   \'Loss\'\n        \'MAE\'                       |   \'MAE\'\n        \'NDCG\'                      |   \'NDCG\'\n        \'TFValidationMethod\'        |   \'${name + " " + valMethod.toString()}\'\n        \'Top5Accuracy\'              |   \'Top5Accuracy\'\n        \'TreeNNAccuracy\'            |   \'TreeNNAccuracy()\'\n        \'MeanAveragePrecision\'      |   \'MAP@k\' (k is Top-k) (BigDL)\n        \'MeanAveragePrecision\'      |   \'PascalMeanAveragePrecision\' (Zoo)\n        \'StatelessMetric\'           |   \'${name}\'\n        # Arguments\n        tag: The string variable represents the scalar wanted\n        '
        return callZooFunc(self.bigdl_type, 'zooGetScalarFromSummary', self.value, tag, 'Validation')

    def set_checkpoint(self, path, over_write=True):
        if False:
            print('Hello World!')
        '\n        Configure checkpoint settings to write snapshots every epoch during the training process.\n        In order to take effect, it needs to be called before fit.\n\n        # Arguments\n        path: The path to save snapshots. Make sure this path exists beforehand.\n        over_write: Whether to overwrite existing snapshots in the given path. Default is True.\n        '
        callZooFunc(self.bigdl_type, 'zooSetCheckpoint', self.value, path, over_write)

    def freeze(self, names=None):
        if False:
            print('Hello World!')
        '\n        Config layers that needed to be freeze\n\n        # Arguments\n        names: Layers to freeze.\n        '
        freeze_names = names if names else None
        if isinstance(freeze_names, six.string_types):
            freeze_names = [freeze_names]
        callZooFunc(self.bigdl_type, 'zooFreeze', self.value, freeze_names)

    def unfreeze(self, names=None):
        if False:
            i = 10
            return i + 15
        '\n        Config layers that needed to be unfreeze\n\n        # Arguments\n        names: Layers to unfreeze.\n        '
        unfreeze_names = names if names else None
        if isinstance(unfreeze_names, six.string_types):
            unfreeze_names = [unfreeze_names]
        callZooFunc(self.bigdl_type, 'zoounFreeze', self.value, unfreeze_names)

    def clear_gradient_clipping(self):
        if False:
            print('Hello World!')
        '\n        Clear gradient clipping parameters. In this case, gradient clipping will not be applied.\n        In order to take effect, it needs to be called before fit.\n        '
        callZooFunc(self.bigdl_type, 'zooClearGradientClipping', self.value)

    def set_constant_gradient_clipping(self, min, max):
        if False:
            print('Hello World!')
        '\n        Set constant gradient clipping during the training process.\n        In order to take effect, it needs to be called before fit.\n\n        # Arguments\n        min: The minimum value to clip by. Float.\n        max: The maximum value to clip by. Float.\n        '
        callZooFunc(self.bigdl_type, 'zooSetConstantGradientClipping', self.value, float(min), float(max))

    def set_gradient_clipping_by_l2_norm(self, clip_norm):
        if False:
            i = 10
            return i + 15
        '\n        Clip gradient to a maximum L2-Norm during the training process.\n        In order to take effect, it needs to be called before fit.\n\n        # Arguments\n        clip_norm: Gradient L2-Norm threshold. Float.\n        '
        callZooFunc(self.bigdl_type, 'zooSetGradientClippingByL2Norm', self.value, float(clip_norm))

    def set_evaluate_status(self):
        if False:
            print('Hello World!')
        '\n        Set the model to be in evaluate status, i.e. remove the effect of Dropout, etc.\n        '
        callZooFunc(self.bigdl_type, 'zooSetEvaluateStatus', self.value)
        return self

    def fit(self, x, y=None, batch_size=32, nb_epoch=10, validation_split=0, validation_data=None, distributed=True, feature_cols=None, label_cols=None, transform=None):
        if False:
            print('Hello World!')
        '\n        Train a model for a fixed number of epochs on a DataSet.\n\n        # Arguments\n        x: Input data. A Numpy array or RDD of Sample, ImageSet or TextSet or Spark DataFrame.\n        y: Labels. A Numpy array. Default is None if x is already Sample RDD or ImageSet or TextSet.\n        batch_size: Number of samples per gradient update. Default is 32.\n        nb_epoch: Number of epochs to train.\n        validation_data: Tuple (x_val, y_val) where x_val and y_val are both Numpy arrays.\n                         Can also be RDD of Sample or ImageSet or TextSet.\n                         Default is None if no validation is involved.\n        distributed: Boolean. Whether to train the model in distributed mode or local mode.\n                     Default is True. In local mode, x and y must both be Numpy arrays.\n        feature_cols: List of String, must be set if x is Spark DataFrame\n        label_cols: List of String, must be set if x is Spark DataFrame\n        '
        if distributed:
            if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                if validation_data:
                    validation_data = to_sample_rdd(*validation_data)
                elif validation_split != 0:
                    if validation_split > 1 or validation_split < 0:
                        invalidInputError(False, 'validation split must in range [0, 1]')
                    split_index = int(len(x) * (1 - validation_split))
                    validation_data = (x[split_index:], y[split_index:])
                    (x, y) = (x[:split_index], y[:split_index])
                    validation_data = to_sample_rdd(*validation_data)
                training_data = to_sample_rdd(x, y)
            elif (isinstance(x, RDD) or isinstance(x, ImageSet) or isinstance(x, TextSet)) or (isinstance(x, FeatureSet) and (not y)):
                training_data = x
            elif isinstance(x, DataFrame):
                if not label_cols:
                    invalidInputError(False, 'Please set label_cols')
                if 'image' not in x.columns:
                    if not feature_cols:
                        invalidInputError(False, 'Please set feature_cols')
                    callBigDlFunc(self.bigdl_type, 'zooFit', self.value, x, batch_size, nb_epoch, feature_cols, label_cols, validation_data)
                    return
                else:
                    callBigDlFunc(self.bigdl_type, 'zooFitImage', self.value, x, batch_size, nb_epoch, label_cols, transform, validation_data)
                    return
            else:
                y_error = ', y: %s. Excepted: x, y are ndarrays.' % type(y) if y else ''
                invalidInputError(False, 'Unsupported training data type x: %s %s' % (type(x), y_error))
            callZooFunc(self.bigdl_type, 'zooFit', self.value, training_data, batch_size, nb_epoch, validation_data)
        else:
            if validation_data:
                val_x = [JTensor.from_ndarray(x) for x in to_list(validation_data[0])]
                val_y = JTensor.from_ndarray(validation_data[1])
            else:
                (val_x, val_y) = (None, None)
            callZooFunc(self.bigdl_type, 'zooFit', self.value, [JTensor.from_ndarray(x) for x in to_list(x)], JTensor.from_ndarray(y), batch_size, nb_epoch, val_x, val_y)

    def evaluate(self, x, y=None, batch_size=32, feature_cols=None, label_cols=None, transform=None):
        if False:
            i = 10
            return i + 15
        '\n        Evaluate a model on a given dataset in distributed mode.\n\n        # Arguments\n        x: Evaluation data. A Numpy array or RDD of Sample or ImageSet or TextSet.\n        y: Labels. A Numpy array.\n           Default is None if x is already Sample RDD or ImageSet or TextSet.\n        batch_size: Number of samples per batch. Default is 32.\n        feature_cols: List of String, must be set if x is Spark DataFrame\n        label_cols: List of String, must be set if x is Spark DataFrame\n        '
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            data = to_sample_rdd(x, y)
        elif (isinstance(x, RDD) or isinstance(x, ImageSet) or isinstance(x, TextSet)) and (not y):
            data = x
        elif isinstance(x, DataFrame):
            if not label_cols:
                invalidInputError(False, 'Please set label_cols')
            if 'image' not in x.columns:
                if not feature_cols:
                    invalidInputError(False, 'Please set feature_cols')
                return callBigDlFunc(self.bigdl_type, 'zooEvaluate', self.value, x, batch_size, feature_cols, label_cols)
            else:
                return callBigDlFunc(self.bigdl_type, 'zooEvaluateImage', self.value, x, label_cols, transform, batch_size)
        else:
            invalidInputError(False, 'Unsupported evaluation data type: %s' % type(x))
        return callZooFunc(self.bigdl_type, 'zooEvaluate', self.value, data, batch_size)

    def forward(self, input):
        if False:
            for i in range(10):
                print('nop')
        "\n        NB: It's for debug only, please use optimizer.optimize() in production.\n        Takes an input object, and computes the corresponding output of the module\n        :param input: ndarray or list of ndarray\n        :param input: ndarray or list of ndarray or JTensor or list of JTensor.\n        :return: ndarray or list of ndarray\n        "
        (jinput, input_is_table) = self.check_input(input)
        output = callZooFunc(self.bigdl_type, 'zooForward', self.value, jinput, input_is_table)
        return self.convert_output(output)

    @staticmethod
    def convert_output(output):
        if False:
            i = 10
            return i + 15
        if type(output) is JTensor:
            return output.to_ndarray()
        elif len(output) == 1:
            return KerasNet.convert_output(output[0])
        else:
            return [KerasNet.convert_output(x) for x in output]

    def predict(self, x, batch_per_thread=4, distributed=True, feature_cols=None, prediction_col=None, transform=None):
        if False:
            return 10
        '\n        Use a model to do prediction.\n\n        # Arguments\n        x: Prediction data. A Numpy array or RDD of Sample or ImageSet.\n        batch_per_thread:\n          The default value is 4.\n          When distributed is True,the total batch size is batch_per_thread * rdd.getNumPartitions.\n          When distributed is False the total batch size is batch_per_thread * numOfCores.\n        distributed: Boolean. Whether to do prediction in distributed mode or local mode.\n                     Default is True. In local mode, x must be a Numpy array.\n        feature_cols: List of String, must be set if x is Spark DataFrame\n        prediction_col: String, must be set if x is Spark DataFrame\n        '
        if isinstance(x, ImageSet) or isinstance(x, TextSet):
            results = callZooFunc(self.bigdl_type, 'zooPredict', self.value, x, batch_per_thread)
            return ImageSet(results) if isinstance(x, ImageSet) else TextSet(results)
        if isinstance(x, DataFrame):
            if not prediction_col:
                invalidInputError(False, 'Please set prediction_col')
            if 'image' not in x.columns:
                results = callZooFunc(self.bigdl_type, 'zooPredict', self.value, x, feature_cols, prediction_col, batch_per_thread)
            else:
                results = callZooFunc(self.bigdl_type, 'zooPredictImage', self.value, x, prediction_col, transform, batch_per_thread)
            return results
        if distributed:
            if isinstance(x, np.ndarray):
                invalidInputError(len(x.shape) >= 2, 'x should be a batch of ndarray, ' + 'dim should >= 2, but got %s' % str(x.shape))
                x = np.resize(x, x.shape)
                data_rdd = to_sample_rdd(x, np.zeros([x.shape[0]]))
            elif isinstance(x, RDD):
                data_rdd = x
            else:
                invalidInputError(False, 'Unsupported prediction data type: %s' % type(x))
            results = callZooFunc(self.bigdl_type, 'zooPredict', self.value, data_rdd, batch_per_thread)
            return results.map(lambda result: Layer.convert_output(result))
        elif isinstance(x, np.ndarray) or isinstance(x, list):
            results = callZooFunc(self.bigdl_type, 'zooPredict', self.value, self._to_jtensors(x), batch_per_thread)
            return [Layer.convert_output(result) for result in results]
        else:
            invalidInputError(False, 'Unsupported prediction data type: %s' % type(x))

    def predict_classes(self, x, batch_per_thread=4, zero_based_label=True):
        if False:
            print('Hello World!')
        '\n        Use a model to predict for classes. By default, label predictions start from 0.\n\n        # Arguments\n        x: Prediction data. A Numpy array or RDD of Sample.\n        batch_per_partition:\n          The default value is 4.\n          When distributed is True, the total batch size is batch_per_thread * rdd.getNumPartitions.\n          When distributed is False the total batch size is batch_per_thread * numOfCores.\n        zero_based_label: Boolean. Whether result labels start from 0.\n                          Default is True. If False, result labels start from 1.\n        '
        if isinstance(x, np.ndarray):
            data_rdd = to_sample_rdd(x, np.zeros([x.shape[0]]))
        elif isinstance(x, RDD):
            data_rdd = x
        else:
            invalidInputError(False, 'Unsupported prediction data type: %s' % type(x))
        return callZooFunc(self.bigdl_type, 'zooPredictClasses', self.value, data_rdd, batch_per_thread, zero_based_label)

    def get_layer(self, name):
        if False:
            return 10
        layer = [l for l in self.layers if l.name() == name]
        if len(layer) == 0:
            invalidInputError(False, 'Could not find a layer named: %s' + name)
        elif len(layer) > 1:
            invalidInputError(False, 'There are multiple layers named: %s' + name)
        else:
            return layer[0]

    def summary(self, line_length=120, positions=[0.33, 0.55, 0.67, 1.0]):
        if False:
            i = 10
            return i + 15
        "\n        Print out the summary information of a BigDL Keras Model.\n\n        For each layer in the model, there will be a separate row containing four columns:\n        ________________________________________________________________________________\n        Layer (type)          Output Shape          Param #     Connected to\n        ================================================================================\n\n        In addition, total number of parameters of this model, separated into trainable and\n        non-trainable counts, will be printed out after the table.\n\n        # Arguments\n        line_length The total length of one row. Default is 120.\n        positions: The maximum absolute length proportion(%) of each field.\n                   List of Float of length 4.\n                   Usually you don't need to adjust this parameter.\n                   Default is [.33, .55, .67, 1.], meaning that\n                   the first field will occupy up to 33% of line_length,\n                   the second field will occupy up to (55-33)% of line_length,\n                   the third field will occupy up to (67-55)% of line_length,\n                   the fourth field will occupy the remaining line (100-67)%.\n                   If the field has a larger length, the remaining part will be trimmed.\n                   If the field has a smaller length, the remaining part will be white spaces.\n        "
        res = callZooFunc(self.bigdl_type, 'zooKerasNetSummary', self.value, line_length, [float(p) for p in positions])
        print(res)
        return res

    def to_model(self):
        if False:
            i = 10
            return i + 15
        from bigdl.dllib.keras.models import Model
        return Model.from_jvalue(callZooFunc(self.bigdl_type, 'kerasNetToModel', self.value))

    @property
    def layers(self):
        if False:
            while True:
                i = 10
        jlayers = callZooFunc(self.bigdl_type, 'getSubModules', self)
        layers = [Layer.of(jlayer) for jlayer in jlayers]
        return layers

    def flattened_layers(self, include_container=False):
        if False:
            return 10
        jlayers = callZooFunc(self.bigdl_type, 'getFlattenSubModules', self, include_container)
        layers = [Layer.of(jlayer) for jlayer in jlayers]
        return layers

class Input(autograd.Variable):
    """
    Used to instantiate an input node.

    # Arguments
    shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> input = Input(name="input1", shape=(3, 5))
    creating: createZooKerasInput
    """

    def __init__(self, shape=None, name=None, bigdl_type='float'):
        if False:
            while True:
                i = 10
        super(Input, self).__init__(input_shape=list(shape) if shape else None, node=None, jvalue=None, name=name)

class InputLayer(ZooKerasLayer):
    """
    Used as an entry point into a model.

    # Arguments
    input_shape: A shape tuple, not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> inputlayer = InputLayer(input_shape=(3, 5), name="input1")
    creating: createZooKerasInputLayer
    """

    def __init__(self, input_shape=None, **kwargs):
        if False:
            print('Hello World!')
        super(InputLayer, self).__init__(None, list(input_shape) if input_shape else None, **kwargs)

class Merge(ZooKerasLayer):
    """
    Used to merge a list of inputs into a single output, following some merge mode.
    Merge must have at least two input layers.

    When using this layer as the first layer in a model, you need to provide the argument
    input_shape for input layers (a list of shape tuples, does not include the batch dimension).

    # Arguments
    layers: A list of layer instances. Must be more than one layer.
    mode: Merge mode. String, must be one of: 'sum', 'mul', 'concat', 'ave', 'cos',
          'dot', 'max', 'sub', 'div', 'min'. Default is 'sum'.
    concat_axis: Int, axis to use when concatenating layers.
                 Only specify this when merge mode is 'concat'.
                 Default is -1, meaning the last axis of the input.
    input_shape: A list of shape tuples, each not including batch.
    name: String to set the name of the layer.
          If not specified, its name will by default to be a generated string.

    >>> l1 = InputLayer(input_shape=(3, 5))
    creating: createZooKerasInputLayer
    >>> l2 = InputLayer(input_shape=(3, 5))
    creating: createZooKerasInputLayer
    >>> merge = Merge(layers=[l1, l2], mode='sum', name="merge1")
    creating: createZooKerasMerge
    """

    def __init__(self, layers=None, mode='sum', concat_axis=-1, input_shape=None, **kwargs):
        if False:
            return 10
        super(Merge, self).__init__(None, list(layers) if layers else None, mode, concat_axis, input_shape, **kwargs)

def merge(inputs, mode='sum', concat_axis=-1, name=None):
    if False:
        return 10
    "\n    Functional merge. Only use this method if you are defining a graph model.\n    Used to merge a list of input nodes into a single output node (NOT layers!),\n    following some merge mode.\n\n    # Arguments\n    inputs: A list of node instances. Must be more than one node.\n    mode: Merge mode. String, must be one of: 'sum', 'mul', 'concat', 'ave', 'cos',\n          'dot', 'max', 'sub', 'div', 'min'. Default is 'sum'.\n    concat_axis: Int, axis to use when concatenating nodes.\n                 Only specify this when merge mode is 'concat'.\n                 Default is -1, meaning the last axis of the input.\n    name: String to set the name of the functional merge.\n          If not specified, its name will by default to be a generated string.\n    "
    return Merge(mode=mode, concat_axis=concat_axis, name=name)(list(inputs))