import keras_tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import analysers
from autokeras import blocks
from autokeras import test_utils

def test_image_build_return_tensor():
    if False:
        while True:
            i = 10
    block = blocks.ImageBlock()
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(32, 32, 3), dtype=tf.float32))
    assert len(nest.flatten(outputs)) == 1

def test_image_block_xception_return_tensor():
    if False:
        return 10
    block = blocks.ImageBlock(block_type='xception')
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(32, 32, 3), dtype=tf.float32))
    assert len(nest.flatten(outputs)) == 1

def test_image_block_normalize_return_tensor():
    if False:
        print('Hello World!')
    block = blocks.ImageBlock(normalize=True)
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(32, 32, 3), dtype=tf.float32))
    assert len(nest.flatten(outputs)) == 1

def test_image_block_augment_return_tensor():
    if False:
        i = 10
        return i + 15
    block = blocks.ImageBlock(augment=True)
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(32, 32, 3), dtype=tf.float32))
    assert len(nest.flatten(outputs)) == 1

def test_image_deserialize_to_image():
    if False:
        for i in range(10):
            print('nop')
    serialized_block = blocks.serialize(blocks.ImageBlock())
    block = blocks.deserialize(serialized_block)
    assert isinstance(block, blocks.ImageBlock)

def test_image_get_config_has_all_attributes():
    if False:
        for i in range(10):
            print('nop')
    block = blocks.ImageBlock()
    config = block.get_config()
    assert test_utils.get_func_args(blocks.ImageBlock.__init__).issubset(config.keys())

def test_text_build_return_tensor():
    if False:
        while True:
            i = 10
    block = blocks.TextBlock()
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(1,), dtype=tf.string))
    assert len(nest.flatten(outputs)) == 1

def test_text_block_ngram_return_tensor():
    if False:
        i = 10
        return i + 15
    block = blocks.TextBlock(block_type='ngram')
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(1,), dtype=tf.string))
    assert len(nest.flatten(outputs)) == 1

def test_text_block_transformer_return_tensor():
    if False:
        i = 10
        return i + 15
    block = blocks.TextBlock(block_type='transformer')
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(1,), dtype=tf.string))
    assert len(nest.flatten(outputs)) == 1

def test_text_deserialize_to_text():
    if False:
        while True:
            i = 10
    serialized_block = blocks.serialize(blocks.TextBlock())
    block = blocks.deserialize(serialized_block)
    assert isinstance(block, blocks.TextBlock)

def test_text_get_config_has_all_attributes():
    if False:
        return 10
    block = blocks.TextBlock()
    config = block.get_config()
    assert test_utils.get_func_args(blocks.TextBlock.__init__).issubset(config.keys())

def test_structured_build_return_tensor():
    if False:
        print('Hello World!')
    block = blocks.StructuredDataBlock()
    block.column_names = ['0', '1']
    block.column_types = {'0': analysers.NUMERICAL, '1': analysers.NUMERICAL}
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(2,), dtype=tf.string))
    assert len(nest.flatten(outputs)) == 1

def test_structured_block_normalize_return_tensor():
    if False:
        return 10
    block = blocks.StructuredDataBlock(normalize=True)
    block.column_names = ['0', '1']
    block.column_types = {'0': analysers.NUMERICAL, '1': analysers.NUMERICAL}
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(2,), dtype=tf.string))
    assert len(nest.flatten(outputs)) == 1

def test_structured_block_search_normalize_return_tensor():
    if False:
        return 10
    block = blocks.StructuredDataBlock(name='a')
    block.column_names = ['0', '1']
    block.column_types = {'0': analysers.NUMERICAL, '1': analysers.NUMERICAL}
    hp = keras_tuner.HyperParameters()
    hp.values['a/' + blocks.wrapper.NORMALIZE] = True
    outputs = block.build(hp, keras.Input(shape=(2,), dtype=tf.string))
    assert len(nest.flatten(outputs)) == 1

def test_structured_deserialize_to_structured():
    if False:
        print('Hello World!')
    serialized_block = blocks.serialize(blocks.StructuredDataBlock())
    block = blocks.deserialize(serialized_block)
    assert isinstance(block, blocks.StructuredDataBlock)

def test_structured_get_config_has_all_attributes():
    if False:
        for i in range(10):
            print('nop')
    block = blocks.StructuredDataBlock()
    config = block.get_config()
    assert test_utils.get_func_args(blocks.StructuredDataBlock.__init__).issubset(config.keys())

def test_timeseries_build_return_tensor():
    if False:
        for i in range(10):
            print('nop')
    block = blocks.TimeseriesBlock()
    block.column_names = ['0', '1']
    block.column_types = {'0': analysers.NUMERICAL, '1': analysers.NUMERICAL}
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(32, 2), dtype=tf.float32))
    assert len(nest.flatten(outputs)) == 1

def test_timeseries_deserialize_to_timeseries():
    if False:
        return 10
    serialized_block = blocks.serialize(blocks.TimeseriesBlock())
    block = blocks.deserialize(serialized_block)
    assert isinstance(block, blocks.TimeseriesBlock)

def test_timeseries_get_config_has_all_attributes():
    if False:
        for i in range(10):
            print('nop')
    block = blocks.TimeseriesBlock()
    config = block.get_config()
    assert test_utils.get_func_args(blocks.TimeseriesBlock.__init__).issubset(config.keys())