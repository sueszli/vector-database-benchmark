import keras_tuner
from autokeras import blocks
from autokeras import nodes

def test_input_get_block_return_general_block():
    if False:
        while True:
            i = 10
    input_node = nodes.Input()
    assert isinstance(input_node.get_block(), blocks.GeneralBlock)

def test_time_series_input_node_build_no_error():
    if False:
        while True:
            i = 10
    node = nodes.TimeseriesInput(lookback=2, shape=(32,))
    hp = keras_tuner.HyperParameters()
    input_node = node.build_node(hp)
    node.build(hp, input_node)

def test_time_series_input_node_deserialize_build_no_error():
    if False:
        for i in range(10):
            print('nop')
    node = nodes.TimeseriesInput(lookback=2, shape=(32,))
    node = nodes.deserialize(nodes.serialize(node))
    hp = keras_tuner.HyperParameters()
    input_node = node.build_node(hp)
    node.build(hp, input_node)