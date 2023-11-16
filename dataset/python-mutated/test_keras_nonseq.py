import unittest
import pytest
from coremltools._deps import _HAS_KERAS_TF
if _HAS_KERAS_TF:
    from keras.models import Model
    from keras.layers import Dense, Input, merge
    from coremltools.converters import keras

@unittest.skipIf(not _HAS_KERAS_TF, 'Missing keras. Skipping tests.')
@pytest.mark.keras1
class KerasNonSequentialModelTest(unittest.TestCase):
    """
    Unit test class for testing non-sequential Keras models.
    """

    @classmethod
    def setUpClass(self):
        if False:
            print('Hello World!')
        '\n        Set up the unit test by loading common utilities.\n        '

    def test_simple_merge(self):
        if False:
            return 10
        '\n        Test the following Keras model\n               |- dense-|\n        dense -|        |- merge - dense\n               |- dense-|\n        '
        input_tensor = Input(shape=(3,))
        x1 = Dense(4)(input_tensor)
        x2 = Dense(5)(x1)
        x3 = Dense(6)(x1)
        x4 = merge([x2, x3], mode='concat')
        x5 = Dense(7)(x4)
        model = Model(input=[input_tensor], output=[x5])
        input_names = ['data']
        output_names = ['output']
        spec = keras.convert(model, input_names, output_names).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetwork'))
        self.assertEqual(len(spec.description.input), len(input_names))
        self.assertEqual(sorted(input_names), sorted(map(lambda x: x.name, spec.description.input)))
        self.assertEqual(len(spec.description.output), len(output_names))
        self.assertEqual(sorted(output_names), sorted(map(lambda x: x.name, spec.description.output)))

    def test_merge_add(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the following Keras model\n               |- dense-|\n        dense -|        |- merge - dense\n               |- dense-|\n        '
        input_tensor = Input(shape=(3,))
        x1 = Dense(4)(input_tensor)
        x2 = Dense(5)(x1)
        x3 = Dense(5)(x1)
        x4 = merge([x2, x3], mode='sum')
        x5 = Dense(7)(x4)
        model = Model(input=[input_tensor], output=[x5])
        input_names = ['data']
        output_names = ['output']
        spec = keras.convert(model, input_names, output_names).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetwork'))
        self.assertEqual(len(spec.description.input), len(input_names))
        self.assertEqual(sorted(input_names), sorted(map(lambda x: x.name, spec.description.input)))
        self.assertEqual(len(spec.description.output), len(output_names))
        self.assertEqual(sorted(output_names), sorted(map(lambda x: x.name, spec.description.output)))

    def test_merge_multiply(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the following Keras model\n               |- dense-|\n        dense -|        |- merge - dense\n               |- dense-|\n        '
        input_tensor = Input(shape=(3,))
        x1 = Dense(4)(input_tensor)
        x2 = Dense(5)(x1)
        x3 = Dense(5)(x1)
        x4 = merge([x2, x3], mode='mul')
        x5 = Dense(7)(x4)
        model = Model(input=[input_tensor], output=[x5])
        input_names = ['data']
        output_names = ['output']
        spec = keras.convert(model, input_names, output_names).get_spec()
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.description)
        self.assertTrue(spec.HasField('neuralNetwork'))
        self.assertEqual(len(spec.description.input), len(input_names))
        self.assertEqual(sorted(input_names), sorted(map(lambda x: x.name, spec.description.input)))
        self.assertEqual(len(spec.description.output), len(output_names))
        self.assertEqual(sorted(output_names), sorted(map(lambda x: x.name, spec.description.output)))