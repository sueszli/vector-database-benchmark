import keras_tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from autokeras import blocks
from autokeras import test_utils

def test_merge_build_return_tensor():
    if False:
        return 10
    block = blocks.Merge()
    outputs = block.build(keras_tuner.HyperParameters(), [keras.Input(shape=(32,), dtype=tf.float32), keras.Input(shape=(4, 8), dtype=tf.float32)])
    assert len(nest.flatten(outputs)) == 1

def test_merge_single_input_return_tensor():
    if False:
        i = 10
        return i + 15
    block = blocks.Merge()
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(32,), dtype=tf.float32))
    assert len(nest.flatten(outputs)) == 1

def test_merge_inputs_with_same_shape_return_tensor():
    if False:
        return 10
    block = blocks.Merge()
    outputs = block.build(keras_tuner.HyperParameters(), [keras.Input(shape=(32,), dtype=tf.float32), keras.Input(shape=(32,), dtype=tf.float32)])
    assert len(nest.flatten(outputs)) == 1

def test_merge_deserialize_to_merge():
    if False:
        print('Hello World!')
    serialized_block = blocks.serialize(blocks.Merge())
    block = blocks.deserialize(serialized_block)
    assert isinstance(block, blocks.Merge)

def test_merge_get_config_has_all_attributes():
    if False:
        while True:
            i = 10
    block = blocks.Merge()
    config = block.get_config()
    assert test_utils.get_func_args(blocks.Merge.__init__).issubset(config.keys())

def test_temporal_build_return_tensor():
    if False:
        while True:
            i = 10
    block = blocks.TemporalReduction()
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(32, 10), dtype=tf.float32))
    assert len(nest.flatten(outputs)) == 1

def test_temporal_global_max_return_tensor():
    if False:
        print('Hello World!')
    block = blocks.TemporalReduction(reduction_type='global_max')
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(32, 10), dtype=tf.float32))
    assert len(nest.flatten(outputs)) == 1

def test_temporal_global_avg_return_tensor():
    if False:
        for i in range(10):
            print('nop')
    block = blocks.TemporalReduction(reduction_type='global_avg')
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(32, 10), dtype=tf.float32))
    assert len(nest.flatten(outputs)) == 1

def test_reduction_2d_tensor_return_input_node():
    if False:
        i = 10
        return i + 15
    block = blocks.TemporalReduction()
    input_node = keras.Input(shape=(32,), dtype=tf.float32)
    outputs = block.build(keras_tuner.HyperParameters(), input_node)
    assert len(nest.flatten(outputs)) == 1
    assert nest.flatten(outputs)[0] is input_node

def test_temporal_deserialize_to_temporal():
    if False:
        for i in range(10):
            print('nop')
    serialized_block = blocks.serialize(blocks.TemporalReduction())
    block = blocks.deserialize(serialized_block)
    assert isinstance(block, blocks.TemporalReduction)

def test_temporal_get_config_has_all_attributes():
    if False:
        print('Hello World!')
    block = blocks.TemporalReduction()
    config = block.get_config()
    assert test_utils.get_func_args(blocks.TemporalReduction.__init__).issubset(config.keys())

def test_spatial_build_return_tensor():
    if False:
        print('Hello World!')
    block = blocks.SpatialReduction()
    outputs = block.build(keras_tuner.HyperParameters(), keras.Input(shape=(32, 32, 3), dtype=tf.float32))
    assert len(nest.flatten(outputs)) == 1

def test_spatial_deserialize_to_spatial():
    if False:
        while True:
            i = 10
    serialized_block = blocks.serialize(blocks.SpatialReduction())
    block = blocks.deserialize(serialized_block)
    assert isinstance(block, blocks.SpatialReduction)

def test_spatial_get_config_has_all_attributes():
    if False:
        while True:
            i = 10
    block = blocks.SpatialReduction()
    config = block.get_config()
    assert test_utils.get_func_args(blocks.SpatialReduction.__init__).issubset(config.keys())