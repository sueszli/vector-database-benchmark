import unittest
from coremltools._deps import _HAS_KERAS_TF
from coremltools.proto import FeatureTypes_pb2
import pytest
import six
if _HAS_KERAS_TF:
    import tensorflow as tf
    from keras.models import Sequential, Model
    from coremltools.converters import keras

@unittest.skipIf(not _HAS_KERAS_TF, 'Missing keras. Skipping tests.')
@pytest.mark.keras1
class KerasSingleLayerTest(unittest.TestCase):
    """
    Unit test class for testing scikit-learn converter.
    """

    @classmethod
    def setUpClass(self):
        if False:
            i = 10
            return i + 15
        '\n        Set up the unit test by loading common utilities.\n        '

    def test_dense(self):
        if False:
            return 10
        '\n        Test the conversion of Dense layer.\n        '
        from keras.layers import Dense
        model = Sequential()
        model.add(Dense(32, input_dim=16))
        input_names = ['input']
        output_names = ['output']
        spec = keras.convert(model, input_names, output_names).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetwork'))
        self.assertEquals(len(spec.description.input), len(input_names))
        six.assertCountEqual(self, input_names, [x.name for x in spec.description.input])
        self.assertEquals(len(spec.description.output), len(output_names))
        six.assertCountEqual(self, output_names, [x.name for x in spec.description.output])
        layers = spec.neuralNetwork.layers
        layer_0 = layers[0]
        self.assertIsNotNone(layer_0.innerProduct)

    def test_activations(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test the conversion for a Dense + Activation('something')\n        "
        from keras.layers import Dense, Activation
        keras_activation_options = ['tanh', 'softplus', 'softsign', 'relu', 'sigmoid', 'hard_sigmoid', 'linear']
        coreml_activation_options = ['tanh', 'softplus', 'softsign', 'ReLU', 'sigmoid', 'sigmoidHard', 'linear']
        for (i, k_act) in enumerate(keras_activation_options):
            c_act = coreml_activation_options[i]
            model = Sequential()
            model.add(Dense(32, input_dim=16))
            model.add(Activation(k_act))
            input_names = ['input']
            output_names = ['output']
            spec = keras.convert(model, input_names, output_names).get_spec()
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.description)
            self.assertTrue(spec.HasField('neuralNetwork'))
            self.assertEquals(len(spec.description.input), len(input_names))
            six.assertCountEqual(self, input_names, [x.name for x in spec.description.input])
            self.assertEquals(len(spec.description.output), len(output_names))
            six.assertCountEqual(self, output_names, [x.name for x in spec.description.output])
            layers = spec.neuralNetwork.layers
            self.assertIsNotNone(layers[0].innerProduct)
            self.assertIsNotNone(layers[1].activation)
            self.assertTrue(layers[1].activation.HasField(c_act))

    def test_activation_softmax(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test the conversion for a Dense + Activation('softmax')\n        "
        from keras.layers import Dense, Activation
        model = Sequential()
        model.add(Dense(32, input_dim=16))
        model.add(Activation('softmax'))
        input_names = ['input']
        output_names = ['output']
        spec = keras.convert(model, input_names, output_names).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetwork'))
        self.assertEquals(len(spec.description.input), len(input_names))
        six.assertCountEqual(self, input_names, [x.name for x in spec.description.input])
        self.assertEquals(len(spec.description.output), len(output_names))
        six.assertCountEqual(self, output_names, [x.name for x in spec.description.output])
        layers = spec.neuralNetwork.layers
        layer_0 = layers[0]
        self.assertIsNotNone(layer_0.innerProduct)
        layer_1 = layers[1]
        self.assertIsNotNone(layer_1.softmax)

    def test_dropout(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the conversion for a Dense + Dropout\n        '
        from keras.layers import Dense, Dropout
        model = Sequential()
        model.add(Dense(32, input_dim=16))
        model.add(Dropout(0.5))
        model.add(Dense(32, input_dim=16))
        input_names = ['input']
        output_names = ['output']
        spec = keras.convert(model, input_names, output_names).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetwork'))
        self.assertEquals(len(spec.description.input), len(input_names))
        six.assertCountEqual(self, input_names, [x.name for x in spec.description.input])
        self.assertEquals(len(spec.description.output), len(output_names))
        six.assertCountEqual(self, output_names, [x.name for x in spec.description.output])
        layers = spec.neuralNetwork.layers
        layer_0 = layers[0]
        self.assertIsNotNone(layer_0.innerProduct)
        self.assertEquals(len(layers), 2)

    def test_convolution(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the conversion of 2D convolutional layer.\n        '
        from keras.layers import Convolution2D
        model = Sequential()
        model.add(Convolution2D(input_shape=(64, 64, 3), nb_filter=32, nb_row=5, nb_col=5, init='glorot_uniform', activation=None, weights=None, border_mode='valid', subsample=(1, 1), bias=True))
        input_names = ['input']
        output_names = ['output']
        spec = keras.convert(model, input_names, output_names).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetwork'))
        self.assertEquals(len(spec.description.input), len(input_names))
        six.assertCountEqual(self, input_names, [x.name for x in spec.description.input])
        self.assertEquals(len(spec.description.output), len(output_names))
        six.assertCountEqual(self, output_names, [x.name for x in spec.description.output])
        layers = spec.neuralNetwork.layers
        layer_0 = layers[0]
        self.assertIsNotNone(layer_0.convolution)

    def test_upsample(self):
        if False:
            print('Hello World!')
        '\n        Test the conversion of 2D convolutional layer + upsample\n        '
        from keras.layers import Convolution2D, UpSampling2D
        model = Sequential()
        model.add(Convolution2D(input_shape=(64, 64, 3), nb_filter=32, nb_row=5, nb_col=5))
        model.add(UpSampling2D(size=(2, 2)))
        input_names = ['input']
        output_names = ['output']
        spec = keras.convert(model, input_names, output_names).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetwork'))
        self.assertEquals(len(spec.description.input), len(input_names))
        six.assertCountEqual(self, input_names, [x.name for x in spec.description.input])
        self.assertEquals(len(spec.description.output), len(output_names))
        six.assertCountEqual(self, output_names, [x.name for x in spec.description.output])
        layers = spec.neuralNetwork.layers
        layer_0 = layers[0]
        self.assertIsNotNone(layer_0.convolution)
        layer_1 = layers[1]
        self.assertIsNotNone(layer_1.upsample)

    def test_pooling(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the conversion of pooling layer.\n        '
        from keras.layers import Convolution2D, MaxPooling2D
        model = Sequential()
        model.add(Convolution2D(input_shape=(64, 64, 3), nb_filter=32, nb_row=5, nb_col=5, init='glorot_uniform', activation=None, weights=None, border_mode='valid', subsample=(1, 1), bias=True))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        input_names = ['input']
        output_names = ['output']
        spec = keras.convert(model, input_names, output_names).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetwork'))
        self.assertEquals(len(spec.description.input), len(input_names))
        six.assertCountEqual(self, input_names, [x.name for x in spec.description.input])
        self.assertEquals(len(spec.description.output), len(output_names))
        six.assertCountEqual(self, output_names, [x.name for x in spec.description.output])
        layers = spec.neuralNetwork.layers
        layer_0 = layers[0]
        self.assertIsNotNone(layer_0.convolution)

    def test_permute(self):
        if False:
            return 10
        '\n        Test the conversion of pooling layer.\n        '
        from keras.layers.core import Permute
        model = Sequential()
        model.add(Permute((3, 2, 1), input_shape=(10, 64, 3)))
        input_names = ['input']
        output_names = ['output']
        spec = keras.convert(model, input_names, output_names).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetwork'))
        self.assertEquals(len(spec.description.input), len(input_names))
        six.assertCountEqual(self, input_names, [x.name for x in spec.description.input])
        self.assertEquals(len(spec.description.output), len(output_names))
        six.assertCountEqual(self, output_names, [x.name for x in spec.description.output])
        layers = spec.neuralNetwork.layers
        layer_0 = layers[0]
        self.assertIsNotNone(layer_0.permute)

    def test_lstm(self):
        if False:
            while True:
                i = 10
        '\n        Test the conversion of an LSTM layer.\n        '
        from keras.layers import LSTM
        model = Sequential()
        model.add(LSTM(32, input_dim=24, input_length=10))
        input_names = ['input']
        output_names = ['output']
        spec = keras.convert(model, input_names, output_names).get_spec()
        print(spec)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetwork'))
        self.assertEquals(len(spec.description.input), len(input_names) + 2)
        self.assertEquals(32, spec.description.input[1].type.multiArrayType.shape[0])
        self.assertEquals(32, spec.description.input[2].type.multiArrayType.shape[0])
        self.assertEquals(len(spec.description.output), len(output_names) + 2)
        self.assertEquals(output_names[0], spec.description.output[0].name)
        self.assertEquals(32, spec.description.output[0].type.multiArrayType.shape[0])
        self.assertEquals(32, spec.description.output[1].type.multiArrayType.shape[0])
        self.assertEquals(32, spec.description.output[2].type.multiArrayType.shape[0])
        layers = spec.neuralNetwork.layers
        layer_0 = layers[0]
        self.assertIsNotNone(layer_0.uniDirectionalLSTM)
        self.assertEquals(len(layer_0.input), 3)
        self.assertEquals(len(layer_0.output), 3)

    def test_simple_rnn(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test the conversion of a simple RNN layer.\n        '
        from keras.layers import SimpleRNN
        model = Sequential()
        model.add(SimpleRNN(32, input_dim=32, input_length=10))
        input_names = ['input']
        output_names = ['output']
        spec = keras.convert(model, input_names, output_names).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetwork'))
        self.assertEquals(len(spec.description.input), len(input_names) + 1)
        self.assertEquals(input_names[0], spec.description.input[0].name)
        self.assertEquals(32, spec.description.input[1].type.multiArrayType.shape[0])
        self.assertEquals(len(spec.description.output), len(output_names) + 1)
        self.assertEquals(output_names[0], spec.description.output[0].name)
        self.assertEquals(32, spec.description.output[0].type.multiArrayType.shape[0])
        self.assertEquals(32, spec.description.output[1].type.multiArrayType.shape[0])
        layers = spec.neuralNetwork.layers
        layer_0 = layers[0]
        self.assertIsNotNone(layer_0.simpleRecurrent)
        self.assertEquals(len(layer_0.input), 2)
        self.assertEquals(len(layer_0.output), 2)

    def test_gru(self):
        if False:
            while True:
                i = 10
        '\n        Test the conversion of a GRU layer.\n        '
        from keras.layers import GRU
        model = Sequential()
        model.add(GRU(32, input_dim=32, input_length=10))
        input_names = ['input']
        output_names = ['output']
        spec = keras.convert(model, input_names, output_names).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetwork'))
        self.assertEquals(len(spec.description.input), len(input_names) + 1)
        self.assertEquals(input_names[0], spec.description.input[0].name)
        self.assertEquals(32, spec.description.input[1].type.multiArrayType.shape[0])
        self.assertEquals(len(spec.description.output), len(output_names) + 1)
        self.assertEquals(output_names[0], spec.description.output[0].name)
        self.assertEquals(32, spec.description.output[0].type.multiArrayType.shape[0])
        self.assertEquals(32, spec.description.output[1].type.multiArrayType.shape[0])
        layers = spec.neuralNetwork.layers
        layer_0 = layers[0]
        self.assertIsNotNone(layer_0.gru)
        self.assertEquals(len(layer_0.input), 2)
        self.assertEquals(len(layer_0.output), 2)

    def test_bidir(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the conversion of a bidirectional layer\n        '
        from keras.layers import LSTM
        from keras.layers.wrappers import Bidirectional
        model = Sequential()
        model.add(Bidirectional(LSTM(32, input_dim=32, input_length=10), input_shape=(10, 32)))
        input_names = ['input']
        output_names = ['output']
        spec = keras.convert(model, input_names, output_names).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetwork'))
        self.assertEquals(len(spec.description.input), len(input_names) + 4)
        self.assertEquals(input_names[0], spec.description.input[0].name)
        self.assertEquals(32, spec.description.input[1].type.multiArrayType.shape[0])
        self.assertEquals(32, spec.description.input[2].type.multiArrayType.shape[0])
        self.assertEquals(32, spec.description.input[3].type.multiArrayType.shape[0])
        self.assertEquals(32, spec.description.input[4].type.multiArrayType.shape[0])
        self.assertEquals(len(spec.description.output), len(output_names) + 4)
        self.assertEquals(output_names[0], spec.description.output[0].name)
        self.assertEquals(64, spec.description.output[0].type.multiArrayType.shape[0])
        self.assertEquals(32, spec.description.output[1].type.multiArrayType.shape[0])
        self.assertEquals(32, spec.description.output[2].type.multiArrayType.shape[0])
        self.assertEquals(32, spec.description.output[3].type.multiArrayType.shape[0])
        self.assertEquals(32, spec.description.output[4].type.multiArrayType.shape[0])
        layers = spec.neuralNetwork.layers
        layer_0 = layers[0]
        self.assertIsNotNone(layer_0.biDirectionalLSTM)
        self.assertEquals(len(layer_0.input), 5)
        self.assertEquals(len(layer_0.output), 5)

    def test_embedding(self):
        if False:
            i = 10
            return i + 15
        from keras.layers import Embedding
        model = Sequential()
        num_inputs = 10
        num_outputs = 3
        model.add(Embedding(num_inputs, num_outputs, input_length=5))
        input_names = ['input']
        output_names = ['output']
        spec = keras.convert(model, input_names, output_names).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetwork'))
        self.assertEquals(len(spec.description.input), len(input_names))
        layers = spec.neuralNetwork.layers
        layer_0 = layers[0]
        self.assertIsNotNone(layer_0.embedding)
        self.assertEquals(layer_0.embedding.inputDim, num_inputs)
        self.assertEquals(layer_0.embedding.outputChannels, num_outputs)
        self.assertEquals(len(layer_0.embedding.weights.floatValue), num_inputs * num_outputs)

    @unittest.skip
    def test_sentiment_analysis(self):
        if False:
            print('Hello World!')
        '\n        Test the conversion for a Embedding + LSTM + Dense layer\n        '
        from keras.layers import Dense, Embedding, LSTM
        max_features = 50
        embedded_dim = 32
        sequence_length = 10
        model = Sequential()
        model.add(Embedding(max_features, embedded_dim, input_length=sequence_length))
        model.add(LSTM(32))
        model.add(Dense(1, activation='sigmoid'))
        input_names = ['input']
        output_names = ['output']
        spec = keras.convert(model, input_names, output_names).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetwork'))
        self.assertEquals(len(spec.description.input), len(input_names))
        six.assertCountEqual(self, input_names, [x.name for x in spec.description.input])
        self.assertEquals(len(spec.description.output), len(output_names))
        six.assertCountEqual(self, output_names, [x.name for x in spec.description.output])
        layers = spec.neuralNetwork.layers
        self.assertIsNotNone(layers[0].innerProduct)
        self.assertIsNotNone(layers[1].recurrent)
        self.assertIsNotNone(layers[2].innerProduct)

    @unittest.skip
    def test_conv1d_lstm(self):
        if False:
            return 10
        from keras.layers import Convolution1D, LSTM, Dense
        model = Sequential()
        model.add(Convolution1D(32, 3, border_mode='same', input_shape=(10, 8)))
        model.add(LSTM(24))
        model.add(Dense(1, activation='sigmoid'))
        print('model.layers[1].output_shape=', model.layers[1].output_shape)
        input_names = ['input']
        output_names = ['output']
        spec = keras.convert(model, input_names, output_names).get_spec()
        self.assertIsNotNone(spec)
        self.assertTrue(spec.HasField('neuralNetwork'))
        self.assertEquals(len(spec.description.input), len(input_names))
        six.assertCountEqual(self, input_names, [x.name for x in spec.description.input])
        self.assertEquals(len(spec.description.output), len(output_names))
        six.assertCountEqual(self, output_names, [x.name for x in spec.description.output])
        layers = spec.neuralNetwork.layers
        self.assertIsNotNone(layers[0].convolution)
        self.assertIsNotNone(layers[1].simpleRecurrent)
        self.assertIsNotNone(layers[2].innerProduct)

    def test_batchnorm(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the conversion for a Convoultion2D + Batchnorm layer\n        '
        from keras.layers import Convolution2D
        from keras.layers.normalization import BatchNormalization
        model = Sequential()
        model.add(Convolution2D(input_shape=(64, 64, 3), nb_filter=32, nb_row=5, nb_col=5, init='glorot_uniform', activation=None, weights=None, border_mode='valid', subsample=(1, 1), bias=True))
        model.add(BatchNormalization(epsilon=1e-05))
        input_names = ['input']
        output_names = ['output']
        spec = keras.convert(model, input_names, output_names).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetwork'))
        self.assertEquals(len(spec.description.input), len(input_names))
        six.assertCountEqual(self, input_names, [x.name for x in spec.description.input])
        self.assertEquals(len(spec.description.output), len(output_names))
        six.assertCountEqual(self, output_names, [x.name for x in spec.description.output])
        layers = spec.neuralNetwork.layers
        self.assertIsNotNone(layers[0].convolution)
        self.assertIsNotNone(layers[1].batchnorm)

    def test_repeat_vector(self):
        if False:
            for i in range(10):
                print('nop')
        from keras.layers import RepeatVector
        model = Sequential()
        model.add(RepeatVector(3, input_shape=(5,)))
        input_names = ['input']
        output_names = ['output']
        spec = keras.convert(model, input_names, output_names).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetwork'))
        self.assertEquals(len(spec.description.input), len(input_names))
        six.assertCountEqual(self, input_names, [x.name for x in spec.description.input])
        self.assertEquals(len(spec.description.output), len(output_names))
        six.assertCountEqual(self, output_names, [x.name for x in spec.description.output])
        layers = spec.neuralNetwork.layers
        self.assertIsNotNone(layers[0].sequenceRepeat)

    @pytest.mark.xfail(raises=ValueError)
    def test_unsupported_variational_deconv(self):
        if False:
            while True:
                i = 10
        from keras.layers import Input, Lambda, Convolution2D, Flatten, Dense
        x = Input(shape=(8, 8, 3))
        conv_1 = Convolution2D(4, 2, 2, border_mode='same', activation='relu')(x)
        flat = Flatten()(conv_1)
        hidden = Dense(10, activation='relu')(flat)
        z_mean = Dense(10)(hidden)
        z_log_var = Dense(10)(hidden)

        def sampling(args):
            if False:
                return 10
            (z_mean, z_log_var) = args
            return z_mean + z_log_var
        z = Lambda(sampling, output_shape=(10,))([z_mean, z_log_var])
        model = Model([x], [z])
        spec = keras.convert(model, ['input'], ['output']).get_spec()

    def test_image_processing(self):
        if False:
            while True:
                i = 10
        '\n        Test the image-processing parameters.\n        '
        from keras.layers import Convolution2D
        model = Sequential()
        model.add(Convolution2D(input_shape=(64, 64, 3), nb_filter=32, nb_row=5, nb_col=5, init='glorot_uniform', activation=None, weights=None, border_mode='valid', subsample=(1, 1), bias=True))
        input_names = ['input']
        output_names = ['output']
        spec = keras.convert(model, input_names, output_names, image_input_names=['input'], red_bias=110.0, blue_bias=117.0, green_bias=120.0, is_bgr=True, image_scale=1.0).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetwork'))
        self.assertEquals(spec.description.input[0].type.WhichOneof('Type'), 'imageType')
        self.assertEquals(spec.description.input[0].type.imageType.colorSpace, FeatureTypes_pb2.ImageFeatureType.ColorSpace.Value('BGR'))
        preprocessing = spec.neuralNetwork.preprocessing[0]
        self.assertTrue(preprocessing.HasField('scaler'))
        pr_0 = preprocessing.scaler
        print('pr_0.channelScale = ', pr_0.channelScale)
        print('pr_0.redBias = ', pr_0.redBias)
        print('pr_0.blueBias = ', pr_0.blueBias)
        print('pr_0.greenBias = ', pr_0.greenBias)
        self.assertIsNotNone(pr_0.redBias)
        self.assertIsNotNone(pr_0.greenBias)
        self.assertIsNotNone(pr_0.blueBias)
        self.assertIsNotNone(pr_0.channelScale)
        self.assertEqual(pr_0.channelScale, 1.0)
        self.assertEqual(pr_0.redBias, 110.0)
        self.assertEqual(pr_0.blueBias, 117.0)
        self.assertEqual(pr_0.greenBias, 120.0)
        spec = keras.convert(model, input_names, output_names, image_input_names=['input'], red_bias=110.0, blue_bias=117.0, green_bias=120.0, is_bgr=False, image_scale=1.0).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetwork'))
        self.assertEquals(spec.description.input[0].type.WhichOneof('Type'), 'imageType')
        self.assertEquals(spec.description.input[0].type.imageType.colorSpace, FeatureTypes_pb2.ImageFeatureType.ColorSpace.Value('RGB'))
        preprocessing = spec.neuralNetwork.preprocessing[0]
        self.assertTrue(preprocessing.HasField('scaler'))
        pr_0 = preprocessing.scaler
        self.assertIsNotNone(pr_0.redBias)
        self.assertIsNotNone(pr_0.greenBias)
        self.assertIsNotNone(pr_0.blueBias)
        self.assertIsNotNone(pr_0.channelScale)
        self.assertEqual(pr_0.channelScale, 1.0)
        self.assertEqual(pr_0.redBias, 110.0)
        self.assertEqual(pr_0.blueBias, 117.0)
        self.assertEqual(pr_0.greenBias, 120.0)
        spec = keras.convert(model, input_names, output_names, image_input_names=['input'], is_bgr=False, image_scale=1.0).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetwork'))
        self.assertEquals(spec.description.input[0].type.WhichOneof('Type'), 'imageType')
        self.assertEquals(spec.description.input[0].type.imageType.colorSpace, FeatureTypes_pb2.ImageFeatureType.ColorSpace.Value('RGB'))
        preprocessing = spec.neuralNetwork.preprocessing[0]
        self.assertTrue(preprocessing.HasField('scaler'))
        pr_0 = preprocessing.scaler
        self.assertIsNotNone(pr_0.redBias)
        self.assertIsNotNone(pr_0.greenBias)
        self.assertIsNotNone(pr_0.blueBias)
        self.assertIsNotNone(pr_0.channelScale)
        self.assertEqual(pr_0.channelScale, 1.0)
        self.assertEqual(pr_0.redBias, 0.0)
        self.assertEqual(pr_0.blueBias, 0.0)
        self.assertEqual(pr_0.greenBias, 0.0)

    def test_classifier_string_classes(self):
        if False:
            print('Hello World!')
        from keras.layers import Dense
        from keras.layers import Activation
        model = Sequential()
        model.add(Dense(32, input_dim=16))
        model.add(Activation('softmax'))
        classes = ['c%s' % i for i in range(32)]
        input_names = ['input']
        output_names = ['prob_output']
        expected_output_names = ['prob_output', 'classLabel']
        spec = keras.convert(model, input_names, output_names, class_labels=classes).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetworkClassifier'))
        self.assertFalse(spec.HasField('neuralNetwork'))
        self.assertEquals(len(spec.description.input), len(input_names))
        six.assertCountEqual(self, input_names, [x.name for x in spec.description.input])
        self.assertEquals(len(spec.description.output), len(expected_output_names))
        self.assertEquals(expected_output_names, [x.name for x in spec.description.output])
        self.assertEquals(spec.description.output[0].type.WhichOneof('Type'), 'dictionaryType')
        self.assertEquals(spec.description.output[0].type.dictionaryType.WhichOneof('KeyType'), 'stringKeyType')
        self.assertEquals(spec.description.output[1].type.WhichOneof('Type'), 'stringType')
        self.assertTrue(spec.description.predictedFeatureName, 'classLabel')
        self.assertTrue(spec.description.predictedProbabilitiesName, 'prob_output')
        self.assertEqual(spec.WhichOneof('Type'), 'neuralNetworkClassifier', 'Expected a NN classifier model')
        self.assertEqual(spec.neuralNetworkClassifier.WhichOneof('ClassLabels'), 'stringClassLabels')
        class_from_proto = list(spec.neuralNetworkClassifier.stringClassLabels.vector)
        six.assertCountEqual(self, classes, class_from_proto)

    def test_classifier_file(self):
        if False:
            return 10
        from keras.layers import Dense
        from keras.layers import Activation
        import os
        import tempfile
        model = Sequential()
        model.add(Dense(32, input_dim=16))
        model.add(Activation('softmax'))
        classes = ['c%s' % i for i in range(32)]
        classes_file = tempfile.mktemp()
        with open(classes_file, 'w') as f:
            f.write('\n'.join(classes))
        input_names = ['input']
        output_names = ['prob_output']
        expected_output_names = ['prob_output', 'classLabel']
        spec = keras.convert(model, input_names, output_names, class_labels=classes).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetworkClassifier'))
        self.assertFalse(spec.HasField('neuralNetwork'))
        self.assertEquals(len(spec.description.input), len(input_names))
        six.assertCountEqual(self, input_names, [x.name for x in spec.description.input])
        self.assertEquals(len(spec.description.output), len(expected_output_names))
        self.assertEquals(expected_output_names, [x.name for x in spec.description.output])
        self.assertEquals(spec.description.output[0].type.WhichOneof('Type'), 'dictionaryType')
        self.assertEquals(spec.description.output[0].type.dictionaryType.WhichOneof('KeyType'), 'stringKeyType')
        self.assertEquals(spec.description.output[1].type.WhichOneof('Type'), 'stringType')
        self.assertTrue(spec.description.predictedFeatureName, 'classLabel')
        self.assertTrue(spec.description.predictedProbabilitiesName, 'prob_output')
        os.remove(classes_file)

    def test_classifier_integer_classes(self):
        if False:
            return 10
        from keras.layers import Dense
        from keras.layers import Activation
        model = Sequential()
        model.add(Dense(32, input_dim=16))
        model.add(Activation('softmax'))
        classes = list(range(32))
        input_names = ['input']
        output_names = ['prob_output']
        expected_output_names = ['prob_output', 'classLabel']
        spec = keras.convert(model, input_names, output_names, class_labels=classes).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetworkClassifier'))
        self.assertFalse(spec.HasField('neuralNetwork'))
        self.assertEquals(len(spec.description.input), len(input_names))
        six.assertCountEqual(self, input_names, [x.name for x in spec.description.input])
        self.assertEquals(len(spec.description.output), len(expected_output_names))
        self.assertEquals(expected_output_names, [x.name for x in spec.description.output])
        self.assertEquals(spec.description.output[0].type.WhichOneof('Type'), 'dictionaryType')
        self.assertEquals(spec.description.output[0].type.dictionaryType.WhichOneof('KeyType'), 'int64KeyType')
        self.assertEquals(spec.description.output[1].type.WhichOneof('Type'), 'int64Type')
        self.assertTrue(spec.description.predictedFeatureName, 'classLabel')
        self.assertTrue(spec.description.predictedProbabilitiesName, 'prob_output')
        self.assertEqual(spec.WhichOneof('Type'), 'neuralNetworkClassifier', 'Expected a NN classifier model')
        self.assertEqual(spec.neuralNetworkClassifier.WhichOneof('ClassLabels'), 'int64ClassLabels')
        class_from_proto = list(spec.neuralNetworkClassifier.int64ClassLabels.vector)
        six.assertCountEqual(self, classes, class_from_proto)

    def test_classifier_custom_class_name(self):
        if False:
            return 10
        from keras.layers import Dense
        from keras.layers import Activation
        model = Sequential()
        model.add(Dense(32, input_dim=16))
        model.add(Activation('softmax'))
        classes = ['c%s' % i for i in range(32)]
        input_names = ['input']
        output_names = ['prob_output']
        expected_output_names = ['prob_output', 'my_foo_bar_class_output']
        spec = keras.convert(model, input_names, output_names, class_labels=classes, predicted_feature_name='my_foo_bar_class_output').get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetworkClassifier'))
        self.assertFalse(spec.HasField('neuralNetwork'))
        self.assertEquals(len(spec.description.input), len(input_names))
        six.assertCountEqual(self, input_names, [x.name for x in spec.description.input])
        self.assertEquals(len(spec.description.output), len(expected_output_names))
        self.assertEquals(expected_output_names, [x.name for x in spec.description.output])
        self.assertEquals(spec.description.output[0].type.WhichOneof('Type'), 'dictionaryType')
        self.assertEquals(spec.description.output[0].type.dictionaryType.WhichOneof('KeyType'), 'stringKeyType')
        self.assertEquals(spec.description.output[1].type.WhichOneof('Type'), 'stringType')
        self.assertTrue(spec.description.predictedFeatureName, 'my_foo_bar_class_output')
        self.assertTrue(spec.description.predictedProbabilitiesName, 'prob_output')
        self.assertEqual(spec.WhichOneof('Type'), 'neuralNetworkClassifier', 'Expected a NN classifier model')
        self.assertEqual(spec.neuralNetworkClassifier.WhichOneof('ClassLabels'), 'stringClassLabels')
        class_from_proto = list(spec.neuralNetworkClassifier.stringClassLabels.vector)
        six.assertCountEqual(self, classes, class_from_proto)

    def test_default_interface_names(self):
        if False:
            i = 10
            return i + 15
        from keras.layers import Dense
        from keras.layers import Activation
        model = Sequential()
        model.add(Dense(32, input_dim=16))
        model.add(Activation('softmax'))
        expected_input_names = ['input1']
        expected_output_names = ['output1']
        spec = keras.convert(model).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetwork'))
        self.assertEquals(len(spec.description.input), len(expected_input_names))
        six.assertCountEqual(self, expected_input_names, [x.name for x in spec.description.input])
        self.assertEquals(len(spec.description.output), len(expected_output_names))
        self.assertEquals(expected_output_names, [x.name for x in spec.description.output])