from os import getcwd, chdir
from shutil import rmtree
from os.path import exists
from tempfile import mkdtemp
import pytest
import numpy as np
import coremltools as ct
import os
from coremltools._deps import _HAS_TF_1, _HAS_TF_2, _HAS_TORCH, MSG_TF1_NOT_FOUND, MSG_TF2_NOT_FOUND, MSG_TORCH_NOT_FOUND

@pytest.mark.skipif(not _HAS_TF_1, reason=MSG_TF1_NOT_FOUND)
@pytest.mark.skipif(ct.utils._macos_version() < (10, 15), reason='Model produces specification 4.')
class TestTensorFlow1ConverterExamples:

    @staticmethod
    def test_convert_from_frozen_graph(tmpdir):
        if False:
            while True:
                i = 10
        import tensorflow as tf
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=(1, 2, 3), name='input')
            y = tf.nn.relu(x, name='output')
        mlmodel = ct.convert(graph)
        test_input = np.random.rand(1, 2, 3) - 0.5
        with tf.compat.v1.Session(graph=graph) as sess:
            expected_val = sess.run(y, feed_dict={x: test_input})
        results = mlmodel.predict({'input': test_input})
        np.testing.assert_allclose(results['output'], expected_val)

    @staticmethod
    def test_convert_from_frozen_graph_file(tmpdir):
        if False:
            return 10
        import tensorflow as tf
        with tf.Graph().as_default() as graph:
            x = tf.placeholder(tf.float32, shape=(1, 2, 3), name='input')
            y = tf.nn.relu(x, name='output')
        save_path = str(tmpdir)
        tf.io.write_graph(graph, save_path, 'frozen_graph.pb', as_text=False)
        test_input = np.random.rand(1, 2, 3) - 0.5
        with tf.compat.v1.Session(graph=graph) as sess:
            expected_val = sess.run(y, feed_dict={x: test_input})
        pb_path = os.path.join(save_path, 'frozen_graph.pb')
        mlmodel = ct.convert(pb_path, inputs=[ct.TensorType(name='input', shape=(1, 2, 3))], outputs=['output'])
        mlmodel = ct.convert(pb_path, inputs=[ct.TensorType(shape=(1, 2, 3))], outputs=['output'])
        mlmodel = ct.convert(pb_path, outputs=['output'])
        results = mlmodel.predict({'input': test_input})
        np.testing.assert_allclose(results['output'], expected_val)
        mlmodel_path = os.path.join(save_path, 'model.mlmodel')
        mlmodel.save(mlmodel_path)
        results = mlmodel.predict({'input': test_input})
        np.testing.assert_allclose(results['output'], expected_val)

    @staticmethod
    def test_convert_from_saved_model_dir(tmpdir):
        if False:
            while True:
                i = 10
        test_input = np.random.rand(1, 3, 5) - 0.5
        import tensorflow as tf
        with tf.compat.v1.Session() as sess:
            x = tf.placeholder(shape=(1, 3, 5), dtype=tf.float32)
            y = tf.nn.relu(x)
            expected_val = sess.run(y, feed_dict={x: test_input})
        inputs = {'x': x}
        outputs = {'y': y}
        save_path = str(tmpdir)
        tf.compat.v1.saved_model.simple_save(sess, save_path, inputs, outputs)
        mlmodel = ct.convert(save_path)
        input_name = x.name.split(':')[0]
        results = mlmodel.predict({input_name: test_input})
        output_name = y.name.split(':')[0]
        np.testing.assert_allclose(results[output_name], expected_val)

    @staticmethod
    def test_freeze_and_convert_matmul_graph():
        if False:
            i = 10
            return i + 15
        import tensorflow as tf
        graph = tf.Graph()
        with graph.as_default():
            x = tf.placeholder(tf.float32, shape=[None, 20], name='input')
            W = tf.Variable(tf.truncated_normal([20, 10], stddev=0.1))
            b = tf.Variable(tf.ones([10]))
            y = tf.matmul(x, W) + b
            output_names = [y.op.name]
        import tempfile
        import os
        from tensorflow.python.tools.freeze_graph import freeze_graph
        model_dir = tempfile.mkdtemp()
        graph_def_file = os.path.join(model_dir, 'tf_graph.pb')
        checkpoint_file = os.path.join(model_dir, 'tf_model.ckpt')
        frozen_graph_file = os.path.join(model_dir, 'tf_frozen.pb')
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            tf.train.write_graph(sess.graph, model_dir, graph_def_file, as_text=False)
            saver = tf.train.Saver()
            saver.save(sess, checkpoint_file)
            freeze_graph(input_graph=graph_def_file, input_saver='', input_binary=True, input_checkpoint=checkpoint_file, output_node_names=','.join(output_names), restore_op_name='save/restore_all', filename_tensor_name='save/Const:0', output_graph=frozen_graph_file, clear_devices=True, initializer_nodes='')
        print('Tensorflow frozen graph saved at {}'.format(frozen_graph_file))
        mlmodel = ct.convert(frozen_graph_file)
        import shutil
        try:
            shutil.rmtree(model_dir)
        except:
            pass

@pytest.mark.skipif(not _HAS_TF_2, reason=MSG_TF2_NOT_FOUND)
class TestTensorFlow2ConverterExamples:

    def setup_class(self):
        if False:
            i = 10
            return i + 15
        self._cwd = getcwd()
        self._temp_dir = mkdtemp()
        chdir(self._temp_dir)
        import tensorflow as tf
        tf_keras_model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(128, activation=tf.nn.relu), tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
        tf_keras_model.save('./tf_keras_model.h5')
        tf_keras_model.save('./saved_model', save_format='tf')

    def teardown_class(self):
        if False:
            return 10
        chdir(self._cwd)
        if exists(self._temp_dir):
            rmtree(self._temp_dir)

    @staticmethod
    def test_convert_tf_keras_h5_file(tmpdir):
        if False:
            i = 10
            return i + 15
        import tensorflow as tf
        x = tf.keras.Input(shape=(32,), name='input')
        y = tf.keras.layers.Dense(16, activation='softmax')(x)
        keras_model = tf.keras.Model(x, y)
        save_dir = str(tmpdir)
        h5_path = os.path.join(save_dir, 'tf_keras_model.h5')
        keras_model.save(h5_path)
        mlmodel = ct.convert(h5_path)
        test_input = np.random.rand(2, 32)
        expected_val = keras_model(test_input)
        results = mlmodel.predict({'input': test_input})
        np.testing.assert_allclose(results['Identity'], expected_val, rtol=0.0001)

    @staticmethod
    def test_convert_tf_keras_model():
        if False:
            print('Hello World!')
        import tensorflow as tf
        x = tf.keras.Input(shape=(32,), name='input')
        y = tf.keras.layers.Dense(16, activation='softmax')(x)
        keras_model = tf.keras.Model(x, y)
        mlmodel = ct.convert(keras_model)
        test_input = np.random.rand(2, 32)
        expected_val = keras_model(test_input)
        results = mlmodel.predict({'input': test_input})
        np.testing.assert_allclose(results['Identity'], expected_val, rtol=0.0001)

    @staticmethod
    def test_convert_tf_keras_applications_model():
        if False:
            print('Hello World!')
        import tensorflow as tf
        tf_keras_model = tf.keras.applications.MobileNet(weights='imagenet', input_shape=(224, 224, 3))
        input_name = tf_keras_model.inputs[0].name.split(':')[0]
        output_name = tf_keras_model.outputs[0].name.split(':')[0]
        tf_graph_output_name = output_name.split('/')[-1]
        mlmodel = ct.convert(tf_keras_model, inputs=[ct.TensorType(shape=(1, 224, 224, 3))], outputs=[tf_graph_output_name])
        mlmodel.save('./mobilenet.mlmodel')

    @staticmethod
    def test_convert_from_saved_model_dir():
        if False:
            print('Hello World!')
        mlmodel = ct.convert('./saved_model')
        mlmodel.save('./model.mlmodel')

    @staticmethod
    def test_keras_custom_layer_model():
        if False:
            for i in range(10):
                print('nop')
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        class CustomDense(layers.Layer):

            def __init__(self, units=32):
                if False:
                    print('Hello World!')
                super(CustomDense, self).__init__()
                self.units = units

            def build(self, input_shape):
                if False:
                    return 10
                self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
                self.b = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True)

            def call(self, inputs):
                if False:
                    for i in range(10):
                        print('nop')
                return tf.matmul(inputs, self.w) + self.b
        inputs = keras.Input((4,))
        outputs = CustomDense(10)(inputs)
        model = keras.Model(inputs, outputs)
        ct.convert(model)

    @staticmethod
    def test_concrete_function_conversion():
        if False:
            i = 10
            return i + 15
        import tensorflow as tf

        @tf.function(input_signature=[tf.TensorSpec(shape=(6,), dtype=tf.float32)])
        def gelu_tanh_activation(x):
            if False:
                print('Hello World!')
            a = np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))
            y = 0.5 * (1.0 + tf.tanh(a))
            return x * y
        conc_func = gelu_tanh_activation.get_concrete_function()
        ct.convert([conc_func])

    @staticmethod
    def test_quickstart_example():
        if False:
            i = 10
            return i + 15
        import tensorflow as tf
        keras_model = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=(224, 224, 3), classes=1000)
        import urllib
        label_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
        class_labels = urllib.request.urlopen(label_url).read().splitlines()
        class_labels = class_labels[1:]
        assert len(class_labels) == 1000
        for (i, label) in enumerate(class_labels):
            if isinstance(label, bytes):
                class_labels[i] = label.decode('utf8')
        image_input = ct.ImageType(shape=(1, 224, 224, 3), bias=[-1, -1, -1], scale=1 / 127)
        classifier_config = ct.ClassifierConfig(class_labels)
        model = ct.convert(keras_model, inputs=[image_input], classifier_config=classifier_config)
        model.input_description['input_1'] = 'Input image to be classified'
        model.output_description['classLabel'] = 'Most likely image category'
        model.author = 'Original Paper: Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen'
        model.license = 'Please see https://github.com/tensorflow/tensorflow for license information, and https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenetfor the original source of the model.'
        model.short_description = 'Detects the dominant objects present in animage from a set of 1001 categories such as trees, animals,food, vehicles, person etc. The top-1 accuracy from the original publication is 74.7%.'
        model.version = '2.0'
        from PIL import Image
        import requests
        from io import BytesIO
        img_url = 'https://files.readme.io/02e3586-daisy.jpg'
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
        example_image = img.resize((224, 224))
        out_dict = model.predict({'input_1': example_image})
        assert out_dict['classLabel'] == 'daisy'

@pytest.mark.skipif(not _HAS_TORCH, reason=MSG_TORCH_NOT_FOUND)
class TestPyTorchConverterExamples:

    @staticmethod
    def test_convert_torch_vision_mobilenet_v2(tmpdir):
        if False:
            print('Hello World!')
        import torch
        import torchvision
        "\n        In this example, we'll instantiate a PyTorch classification model and convert\n        it to Core ML.\n        "
        '\n        Here we instantiate our model. In a real use case this would be your trained\n        model.\n        '
        model = torchvision.models.mobilenet_v2()
        '\n        The next thing we need to do is generate TorchScript for the model. The easiest\n        way to do this is by tracing it.\n        '
        "\n        It's important that a model be in evaluation mode (not training mode) when it's\n        traced. This makes sure things like dropout are disabled.\n        "
        model.eval()
        "\n        Tracing takes an example input and traces its flow through the model. Here we\n        are creating an example image input.\n\n        The rank and shape of the tensor will depend on your model use case. If your\n        model expects a fixed size input, use that size here. If it can accept a\n        variety of input sizes, it's generally best to keep the example input small to\n        shorten how long it takes to run a forward pass of your model. In all cases,\n        the rank of the tensor must be fixed.\n        "
        example_input = torch.rand(1, 3, 256, 256)
        '\n        Now we actually trace the model. This will produce the TorchScript that the\n        CoreML converter needs.\n        '
        traced_model = torch.jit.trace(model, example_input)
        '\n        Now with a TorchScript representation of the model, we can call the CoreML\n        converter. The converter also needs a description of the input to the model,\n        where we can give it a convenient name.\n        '
        mlmodel = ct.convert(traced_model, inputs=[ct.TensorType(name='input', shape=example_input.shape)])
        '\n        Now with a conversion complete, we can save the MLModel and run inference.\n        '
        save_path = os.path.join(str(tmpdir), 'mobilenet_v2.mlmodel')
        mlmodel.save(save_path)
        '\n        Running predict() is only supported on macOS.\n        '
        if ct.utils._is_macos():
            results = mlmodel.predict({'input': example_input.numpy()})
            expected = model(example_input)
            np.testing.assert_allclose(list(results.values())[0], expected.detach().numpy(), rtol=0.01)

    @staticmethod
    def test_int64_inputs():
        if False:
            for i in range(10):
                print('nop')
        import torch
        num_tokens = 3
        embedding_size = 5

        class TestModule(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super(TestModule, self).__init__()
                self.embedding = torch.nn.Embedding(num_tokens, embedding_size)

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.embedding(x)
        model = TestModule()
        model.eval()
        example_input = torch.randint(high=num_tokens, size=(2,), dtype=torch.int64)
        traced_model = torch.jit.trace(model, example_input)
        mlmodel = ct.convert(traced_model, inputs=[ct.TensorType(name='input', shape=example_input.shape, dtype=example_input.numpy().dtype)])
        if ct.utils._is_macos():
            result = mlmodel.predict({'input': example_input.detach().numpy().astype(np.float32)})
            expected = model(example_input)
            np.testing.assert_allclose(result['5'], expected.detach().numpy())
        with pytest.raises(ValueError, match='Duplicated inputs'):
            mlmodel = ct.convert(traced_model, inputs=[ct.TensorType(name='input', shape=example_input.shape, dtype=example_input.numpy().dtype), ct.TensorType(name='input', shape=example_input.shape, dtype=example_input.numpy().dtype)])
        with pytest.raises(ValueError, match='outputs must not be specified'):
            mlmodel = ct.convert(traced_model, inputs=[ct.TensorType(name='input', shape=example_input.shape, dtype=example_input.numpy().dtype)], outputs=['output'])

class TestMILExamples:

    @staticmethod
    def test_tutorial():
        if False:
            return 10
        from coremltools.converters.mil import Builder as mb

        @mb.program(input_specs=[mb.TensorSpec(shape=(1, 100, 100, 3))])
        def prog(x):
            if False:
                for i in range(10):
                    print('nop')
            x = mb.relu(x=x, name='relu')
            x = mb.transpose(x=x, perm=[0, 3, 1, 2], name='transpose')
            x = mb.reduce_mean(x=x, axes=[2, 3], keep_dims=False, name='reduce')
            x = mb.log(x=x, name='log')
            y = mb.add(x=1, y=2)
            return x
        print('prog:\n', prog)
        from coremltools.converters.mil.converter import _convert
        from coremltools import models
        proto = _convert(prog, convert_from='mil')
        model = models.MLModel(proto)
        if ct.utils._is_macos():
            prediction = model.predict({'x': np.random.rand(1, 100, 100, 3).astype(np.float32)})
            assert len(prediction) == 1