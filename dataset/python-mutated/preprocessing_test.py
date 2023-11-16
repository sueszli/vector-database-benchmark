import keras_tuner
import tensorflow as tf
from keras_tuner.engine import hyperparameters
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils

def test_augment_build_return_tensor():
    if False:
        print('Hello World!')
    block = blocks.ImageAugmentation(rotation_factor=0.2)
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(32, 32, 3), dtype=tf.float32))
    assert len(nest.flatten(outputs)) == 1

def test_augment_build_with_translation_factor_range_return_tensor():
    if False:
        while True:
            i = 10
    block = blocks.ImageAugmentation(translation_factor=(0, 0.1))
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(32, 32, 3), dtype=tf.float32))
    assert len(nest.flatten(outputs)) == 1

def test_augment_build_with_no_flip_return_tensor():
    if False:
        while True:
            i = 10
    block = blocks.ImageAugmentation(vertical_flip=False, horizontal_flip=False)
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(32, 32, 3), dtype=tf.float32))
    assert len(nest.flatten(outputs)) == 1

def test_augment_build_with_vflip_only_return_tensor():
    if False:
        print('Hello World!')
    block = blocks.ImageAugmentation(vertical_flip=True, horizontal_flip=False)
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(32, 32, 3), dtype=tf.float32))
    assert len(nest.flatten(outputs)) == 1

def test_augment_build_with_zoom_factor_return_tensor():
    if False:
        for i in range(10):
            print('nop')
    block = blocks.ImageAugmentation(zoom_factor=0.1)
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(32, 32, 3), dtype=tf.float32))
    assert len(nest.flatten(outputs)) == 1

def test_augment_build_with_contrast_factor_return_tensor():
    if False:
        print('Hello World!')
    block = blocks.ImageAugmentation(contrast_factor=0.1)
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(32, 32, 3), dtype=tf.float32))
    assert len(nest.flatten(outputs)) == 1

def test_augment_deserialize_to_augment():
    if False:
        return 10
    serialized_block = blocks.serialize(blocks.ImageAugmentation(zoom_factor=0.1, contrast_factor=hyperparameters.Float('contrast_factor', 0.1, 0.5)))
    block = blocks.deserialize(serialized_block)
    assert isinstance(block, blocks.ImageAugmentation)
    assert block.zoom_factor == 0.1
    assert isinstance(block.contrast_factor, hyperparameters.Float)

def test_augment_get_config_has_all_attributes():
    if False:
        i = 10
        return i + 15
    block = blocks.ImageAugmentation()
    config = block.get_config()
    assert test_utils.get_func_args(blocks.ImageAugmentation.__init__).issubset(config.keys())

def test_ngram_build_return_tensor():
    if False:
        while True:
            i = 10
    block = blocks.TextToNgramVector()
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(1,), dtype=tf.string))
    assert len(nest.flatten(outputs)) == 1

def test_ngram_build_with_ngrams_return_tensor():
    if False:
        i = 10
        return i + 15
    block = blocks.TextToNgramVector(ngrams=2)
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(1,), dtype=tf.string))
    assert len(nest.flatten(outputs)) == 1

def test_ngram_deserialize_to_ngram():
    if False:
        print('Hello World!')
    serialized_block = blocks.serialize(blocks.TextToNgramVector())
    block = blocks.deserialize(serialized_block)
    assert isinstance(block, blocks.TextToNgramVector)

def test_ngram_get_config_has_all_attributes():
    if False:
        while True:
            i = 10
    block = blocks.TextToNgramVector()
    config = block.get_config()
    assert test_utils.get_func_args(blocks.TextToNgramVector.__init__).issubset(config.keys())

def test_int_seq_build_return_tensor():
    if False:
        while True:
            i = 10
    block = blocks.TextToIntSequence()
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(1,), dtype=tf.string))
    assert len(nest.flatten(outputs)) == 1

def test_int_seq_build_with_seq_len_return_tensor():
    if False:
        print('Hello World!')
    block = blocks.TextToIntSequence(output_sequence_length=50)
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(1,), dtype=tf.string))
    assert len(nest.flatten(outputs)) == 1

def test_int_seq_deserialize_to_int_seq():
    if False:
        i = 10
        return i + 15
    serialized_block = blocks.serialize(blocks.TextToIntSequence())
    block = blocks.deserialize(serialized_block)
    assert isinstance(block, blocks.TextToIntSequence)

def test_int_seq_get_config_has_all_attributes():
    if False:
        for i in range(10):
            print('nop')
    block = blocks.TextToIntSequence()
    config = block.get_config()
    assert test_utils.get_func_args(blocks.TextToIntSequence.__init__).issubset(config.keys())

def test_cat_to_num_build_return_tensor():
    if False:
        while True:
            i = 10
    block = blocks.CategoricalToNumerical()
    block.column_names = ['a']
    block.column_types = {'a': 'num'}
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(1,), dtype=tf.string))
    assert len(nest.flatten(outputs)) == 1

def test_cat_to_num_deserialize_to_cat_to_num():
    if False:
        i = 10
        return i + 15
    serialized_block = blocks.serialize(blocks.CategoricalToNumerical())
    block = blocks.deserialize(serialized_block)
    assert isinstance(block, blocks.CategoricalToNumerical)

def test_cat_to_num_get_config_has_all_attributes():
    if False:
        while True:
            i = 10
    block = blocks.CategoricalToNumerical()
    config = block.get_config()
    assert test_utils.get_func_args(blocks.CategoricalToNumerical.__init__).issubset(config.keys())