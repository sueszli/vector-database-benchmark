import sys
from bigdl.dllib.net.utils import find_placeholders, _check_the_same
from bigdl.orca.tfpark.tfnet import TFNet
from bigdl.orca.tfpark.tf_dataset import TFNdarrayDataset, check_data_compatible
from bigdl.orca.tfpark.tf_dataset import _standarize_feature_dataset
from bigdl.dllib.utils.log4Error import invalidInputError
if sys.version >= '3':
    long = int
    unicode = str

class TFPredictor:

    def __init__(self, sess, outputs, inputs=None, dataset=None):
        if False:
            while True:
                i = 10
        '\n        TFPredictor takes a list of TensorFlow tensors as the model outputs and\n        feed all the elements in TFDatasets to produce those outputs and returns\n        a Spark RDD with each of its elements representing the model prediction\n        for the corresponding input elements.\n\n        :param sess: the current TensorFlow Session, you should first use this session\n        to load the trained variables then pass into TFPredictor\n        :param outputs: the output tensors of the TensorFlow model\n        '
        if inputs is None:
            (dataset, inputs) = TFPredictor._get_datasets_and_inputs(outputs)
        self.sess = sess
        self.dataset = dataset
        self.inputs = inputs
        self.tfnet = TFNet.from_session(sess, self.inputs, outputs)
        if self.dataset.batch_per_thread <= 0:
            invalidInputError(False, 'You should set batch_per_thread on TFDataset ' + 'instead of batch_size for prediction')

    @staticmethod
    def _get_datasets_and_inputs(outputs):
        if False:
            for i in range(10):
                print('nop')
        import tensorflow as tf
        all_required_inputs = find_placeholders(outputs)
        dataset = tf.get_collection(all_required_inputs[0].name)[0]
        inputs = dataset.tensors
        _check_the_same(all_required_inputs, inputs)
        return (dataset, inputs)

    @classmethod
    def from_outputs(cls, sess, outputs):
        if False:
            return 10
        (dataset, inputs) = TFPredictor._get_datasets_and_inputs(outputs)
        return cls(sess, outputs, inputs, dataset)

    @classmethod
    def from_keras(cls, keras_model, dataset):
        if False:
            print('Hello World!')
        import tensorflow.keras.backend as K
        sess = K.get_session()
        outputs = keras_model.outputs
        inputs = keras_model.inputs
        check_data_compatible(dataset, keras_model, mode='inference')
        if isinstance(dataset, TFNdarrayDataset):
            dataset = _standarize_feature_dataset(dataset, keras_model)
        return cls(sess, outputs, inputs, dataset)

    def predict(self):
        if False:
            print('Hello World!')
        return self.tfnet.predict(self.dataset.get_prediction_data(), mini_batch=True)