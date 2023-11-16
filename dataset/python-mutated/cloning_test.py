import numpy as np
import pytest
import tree
from absl.testing import parameterized
from keras import layers
from keras import models
from keras import testing
from keras.models.cloning import clone_model

def get_mlp_functional_model(shared_layers=False):
    if False:
        print('Hello World!')
    inputs = layers.Input(shape=(3,))
    x = layers.Dense(2)(inputs)
    if shared_layers:
        layer = layers.Dense(2, name='shared')
        x = layer(x)
        x = layer(x)
    outputs = layers.Dense(2)(x)
    model = models.Model(inputs, outputs)
    return model

def get_cnn_functional_model(shared_layers=False):
    if False:
        for i in range(10):
            print('nop')
    inputs = layers.Input(shape=(7, 3))
    x = layers.Conv1D(2, 2, padding='same')(inputs)
    if shared_layers:
        layer = layers.Conv1D(2, 2, padding='same', name='shared')
        x = layer(x)
        x = layer(x)
    outputs = layers.Conv1D(2, 2, padding='same')(x)
    model = models.Model(inputs, outputs)
    return model

def get_sequential_model(explicit_input=True):
    if False:
        return 10
    model = models.Sequential()
    if explicit_input:
        model.add(layers.Input(shape=(3,)))
    model.add(layers.Dense(2))
    model.add(layers.Dense(2))
    return model

def get_subclassed_model():
    if False:
        print('Hello World!')

    class ExampleModel(models.Model):

        def __init__(self, **kwargs):
            if False:
                i = 10
                return i + 15
            super().__init__(**kwargs)
            self.d1 = layers.Dense(2)
            self.d2 = layers.Dense(2)

        def call(self, x):
            if False:
                while True:
                    i = 10
            return self.d2(self.d1(x))
    return ExampleModel()

@pytest.mark.requires_trainable_backend
class CloneModelTest(testing.TestCase, parameterized.TestCase):

    @parameterized.named_parameters(('mlp_functional', get_mlp_functional_model), ('cnn_functional', get_cnn_functional_model, True), ('sequential', get_sequential_model), ('deferred_sequential', lambda : get_sequential_model(explicit_input=False)), ('subclassed', get_subclassed_model))
    def test_cloning_correctness(self, model_fn, is_conv=False):
        if False:
            print('Hello World!')
        ref_input = np.random.random((2, 7, 3) if is_conv else (2, 3))
        model = model_fn()
        new_model = clone_model(model)
        ref_output = model(ref_input)
        new_model(ref_input)
        new_model.set_weights(model.get_weights())
        output = new_model(ref_input)
        self.assertAllClose(ref_output, output)

    @parameterized.named_parameters(('mlp_functional', get_mlp_functional_model), ('cnn_functional', get_cnn_functional_model), ('sequential', get_sequential_model))
    def test_custom_clone_function(self, model_fn):
        if False:
            for i in range(10):
                print('nop')

        def clone_function(layer):
            if False:
                print('Hello World!')
            config = layer.get_config()
            config['name'] = config['name'] + '_custom'
            return layer.__class__.from_config(config)
        model = model_fn()
        new_model = clone_model(model, clone_function=clone_function)
        for (l1, l2) in zip(model.layers, new_model.layers):
            if not isinstance(l1, layers.InputLayer):
                self.assertEqual(l2.name, l1.name + '_custom')

    def test_shared_layers_cloning(self):
        if False:
            return 10
        model = get_mlp_functional_model(shared_layers=True)
        new_model = clone_model(model)
        self.assertLen(new_model.layers, 4)

    def test_structured_io_cloning(self):
        if False:
            for i in range(10):
                print('nop')
        x = layers.Input((3,))
        y = layers.Input((3,))
        z1 = x + y
        z2 = layers.Dense(5)(z1)
        inputs = dict(x=x, y=y)
        outputs = dict(z1=z1, z2=z2)
        model0 = models.Model(inputs, outputs)
        model = clone_model(model0)
        tree.assert_same_structure(model.input, inputs)
        tree.assert_same_structure(model.output, outputs)
        model = clone_model(model0, input_tensors=inputs)
        tree.assert_same_structure(model.input, inputs)
        tree.assert_same_structure(model.output, outputs)
        with self.assertRaisesRegex(ValueError, '`input_tensors` must have the same structure as model.input'):
            model = clone_model(model0, input_tensors=(x, y))