import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import random
from nvidia.dali.pipeline import Pipeline
from test_utils import RandomlyShapedDataIterator
from test_utils import compare_pipelines

class LookupTablePipeline(Pipeline):

    def __init__(self, device, batch_size, iterator, data_shape, data_layout, dtype, num_threads=1, device_id=0, dictionary={}, default_value=0.0):
        if False:
            print('Hello World!')
        super().__init__(batch_size, num_threads, device_id)
        self.device = device
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.data_shape = data_shape
        self.data_layout = data_layout
        if dictionary:
            keys = [k for k in dictionary.keys()]
            values = [dictionary[k] for k in keys]
            self.lookup = ops.LookupTable(device=self.device, dtype=dtype, default_value=default_value, keys=keys, values=values)
        else:
            self.lookup = ops.LookupTable(device=self.device, dtype=dtype, default_value=default_value)

    def define_graph(self):
        if False:
            for i in range(10):
                print('nop')
        self.data = self.inputs()
        input_data = self.data.gpu() if self.device == 'gpu' else self.data
        out = self.lookup(input_data)
        return out

    def iter_setup(self):
        if False:
            while True:
                i = 10
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.data_layout)

class LookupTablePythonOpPipeline(Pipeline):

    def __init__(self, function, batch_size, iterator, data_shape, data_layout, dtype, num_threads=1, device_id=0, dictionary={}, default_value=0.0):
        if False:
            print('Hello World!')
        super().__init__(batch_size, num_threads, device_id, exec_async=False, exec_pipelined=False)
        self.iterator = iterator
        self.inputs = ops.ExternalSource()
        self.data_shape = data_shape
        self.data_layout = data_layout

        def lookup_table_func(input_data):
            if False:
                i = 10
                return i + 15
            return function(input_data, dictionary=dictionary, default_value=default_value)
        self.lookup = ops.PythonFunction(function=lookup_table_func, output_layouts=data_layout, batch_processing=False)
        self.cast = ops.Cast(dtype=dtype)

    def define_graph(self):
        if False:
            i = 10
            return i + 15
        self.data = self.inputs()
        out = self.lookup(self.data)
        out = self.cast(out)
        return out

    def iter_setup(self):
        if False:
            while True:
                i = 10
        data = self.iterator.next()
        self.feed_input(self.data, data, layout=self.data_layout)

def lookup_func(image, dictionary, default_value):
    if False:
        return 10
    arr = [default_value for k in range(4096)]
    for k in dictionary.keys():
        arr[k] = dictionary[k]
    lut = np.array(arr)
    return lut[image]

def check_lookup_table_vs_python_op(device, batch_size, layout, shape, dtype, dictionary_type, default_value):
    if False:
        i = 10
        return i + 15
    eii1 = RandomlyShapedDataIterator(batch_size, max_shape=shape)
    eii2 = RandomlyShapedDataIterator(batch_size, max_shape=shape)
    if dictionary_type == 'empty':
        dictionary = {}
    elif dictionary_type == 'random':
        dictionary = {k: random.random() for k in range(256)}
    elif dictionary_type == 'small':
        dictionary = {0: 0.1, 200: 0.99}
    else:
        assert False
    compare_pipelines(LookupTablePipeline(device, batch_size, iter(eii1), data_shape=shape, data_layout=layout, dtype=dtype, dictionary=dictionary, default_value=default_value), LookupTablePythonOpPipeline(lookup_func, batch_size, iter(eii2), data_shape=shape, data_layout=layout, dtype=dtype, dictionary=dictionary, default_value=default_value), batch_size=batch_size, N_iterations=3)

def test_lookup_table_vs_python_op():
    if False:
        i = 10
        return i + 15
    layout = types.NHWC
    for device in {'cpu', 'gpu'}:
        for dtype in {types.FLOAT, types.FLOAT16, types.INT64}:
            for (batch_size, shape, dictionary_type, default_value) in [(1, (300, 300, 3), 'random', 0.0), (1, (300, 300, 3), 'empty', 0.33), (10, (300, 300, 3), 'random', 0.9), (3, (300, 300, 3), 'small', 0.4)]:
                yield (check_lookup_table_vs_python_op, device, batch_size, layout, shape, dtype, dictionary_type, default_value)