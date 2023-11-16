"""A microbenchmark for measuring performance of coders.

This runs a sequence of encode-decode operations on random inputs
to collect performance of various coders.

To evaluate coders performance we approximate the behavior
how the coders are used in PCollections: we encode and decode
a list of elements. An element can be a string, a list of integers,
a windowed value, or any other object we want a coder to process.

Run as:
  python -m apache_beam.tools.coders_microbenchmark

"""
import argparse
import logging
import random
import re
import string
import sys
import apache_beam as beam
from apache_beam.coders import proto2_coder_test_messages_pb2 as test_message
from apache_beam.coders import coder_impl
from apache_beam.coders import coders
from apache_beam.coders import row_coder
from apache_beam.coders import typecoders
from apache_beam.tools import utils
from apache_beam.transforms import window
from apache_beam.typehints import trivial_inference
from apache_beam.typehints.pandas_type_compatibility import DataFrameBatchConverterDropIndex
from apache_beam.utils import windowed_value

def coder_benchmark_factory(coder, generate_fn):
    if False:
        while True:
            i = 10
    'Creates a benchmark that encodes and decodes a list of elements.\n\n  Args:\n    coder: coder to use to encode an element.\n    generate_fn: a callable that generates an element.\n  '

    class CoderBenchmark(object):

        def __init__(self, num_elements_per_benchmark):
            if False:
                return 10
            self._coder = coders.IterableCoder(coder)
            self._list = [generate_fn() for _ in range(num_elements_per_benchmark)]

        def __call__(self):
            if False:
                while True:
                    i = 10
            _ = self._coder.decode(self._coder.encode(self._list))
    CoderBenchmark.__name__ = '%s, %s' % (generate_fn.__name__, str(coder))
    return CoderBenchmark

def batch_row_coder_benchmark_factory(generate_fn, use_batch):
    if False:
        i = 10
        return i + 15
    'Creates a benchmark that encodes and decodes a list of elements.\n\n  Args:\n    coder: coder to use to encode an element.\n    generate_fn: a callable that generates an element.\n  '

    class CoderBenchmark(object):

        def __init__(self, num_elements_per_benchmark):
            if False:
                print('Hello World!')
            self._use_batch = use_batch
            row_instance = generate_fn()
            row_type = trivial_inference.instance_to_type(row_instance)
            self._row_coder = get_row_coder(row_instance)
            self._batch_converter = DataFrameBatchConverterDropIndex(row_type)
            self._seq_coder = coders.IterableCoder(self._row_coder)
            self._data = self._batch_converter.produce_batch([generate_fn() for _ in range(num_elements_per_benchmark)])

        def __call__(self):
            if False:
                for i in range(10):
                    print('nop')
            if self._use_batch:
                impl = self._row_coder.get_impl()
                columnar = {col: self._data[col].to_numpy() for col in self._data.columns}
                output_stream = coder_impl.create_OutputStream()
                impl.encode_batch_to_stream(columnar, output_stream)
                impl.decode_batch_from_stream(columnar, coder_impl.create_InputStream(output_stream.get()))
            else:
                self._batch_converter.produce_batch(self._seq_coder.decode(self._seq_coder.encode(self._batch_converter.explode_batch(self._data))))
    CoderBenchmark.__name__ = '%s, BatchRowCoder%s' % (generate_fn.__name__, use_batch)
    return CoderBenchmark

def small_int():
    if False:
        for i in range(10):
            print('nop')
    return random.randint(0, 127)

def large_int():
    if False:
        return 10
    return random.randint(sys.maxsize >> 2, sys.maxsize)

def random_string(length):
    if False:
        for i in range(10):
            print('nop')
    return ''.join((random.choice(string.ascii_letters + string.digits) for _ in range(length)))

def small_string():
    if False:
        return 10
    return random_string(4)

def large_string():
    if False:
        for i in range(10):
            print('nop')
    return random_string(100)

def list_int(size):
    if False:
        i = 10
        return i + 15
    return [small_int() for _ in range(size)]

def dict_int_int(size):
    if False:
        i = 10
        return i + 15
    return {i: i for i in list_int(size)}

def small_list():
    if False:
        for i in range(10):
            print('nop')
    return list_int(10)

def large_list():
    if False:
        return 10
    return [bool(k) for k in list_int(1000)]

def small_tuple():
    if False:
        for i in range(10):
            print('nop')
    return tuple(list_int(2))

def large_tuple():
    if False:
        for i in range(10):
            print('nop')
    return tuple(large_list())

def small_dict():
    if False:
        print('Hello World!')
    return {i: i for i in small_list()}

def large_dict():
    if False:
        print('Hello World!')
    return {i: i for i in large_list()}

def large_iterable():
    if False:
        for i in range(10):
            print('nop')
    yield ('a' * coders.coder_impl.SequenceCoderImpl._DEFAULT_BUFFER_SIZE)
    for k in range(1000):
        yield k

def random_message_with_map(size):
    if False:
        i = 10
        return i + 15
    message = test_message.MessageWithMap()
    keys = list_int(size)
    random.shuffle(keys)
    for key in keys:
        message.field1[str(key)].field1 = small_string()
    return message

def small_message_with_map():
    if False:
        return 10
    return random_message_with_map(5)

def large_message_with_map():
    if False:
        print('Hello World!')
    return random_message_with_map(20)

def globally_windowed_value():
    if False:
        i = 10
        return i + 15
    return windowed_value.WindowedValue(value=small_int(), timestamp=12345678, windows=(window.GlobalWindow(),))

def random_windowed_value(num_windows):
    if False:
        print('Hello World!')
    return windowed_value.WindowedValue(value=small_int(), timestamp=12345678, windows=tuple((window.IntervalWindow(i * 10, i * 10 + small_int()) for i in range(num_windows))))

def wv_with_one_window():
    if False:
        i = 10
        return i + 15
    return random_windowed_value(num_windows=1)

def wv_with_multiple_windows():
    if False:
        return 10
    return random_windowed_value(num_windows=32)

def tiny_row():
    if False:
        return 10
    return beam.Row(int_value=1)

def large_row():
    if False:
        for i in range(10):
            print('nop')
    return beam.Row(**{f'int_{ix}': ix for ix in range(20)})

def nullable_row():
    if False:
        i = 10
        return i + 15
    return beam.Row(**{f'int_{ix}': ix if ix % 2 else None for ix in range(20)})

def diverse_row():
    if False:
        i = 10
        return i + 15
    return beam.Row(int_value=1, float_value=3.14159, str_value='beam', row_value=beam.Row(int_value=2, float_value=2.718281828))

def get_row_coder(row_instance):
    if False:
        while True:
            i = 10
    coder = typecoders.registry.get_coder(trivial_inference.instance_to_type(row_instance))
    assert isinstance(coder, row_coder.RowCoder)
    return coder

def row_coder_benchmark_factory(generate_fn):
    if False:
        for i in range(10):
            print('nop')
    return coder_benchmark_factory(get_row_coder(generate_fn()), generate_fn)

def run_coder_benchmarks(num_runs, input_size, seed, verbose, filter_regex='.*'):
    if False:
        i = 10
        return i + 15
    random.seed(seed)
    benchmarks = [coder_benchmark_factory(coders.FastPrimitivesCoder(), small_int), coder_benchmark_factory(coders.FastPrimitivesCoder(), large_int), coder_benchmark_factory(coders.FastPrimitivesCoder(), small_string), coder_benchmark_factory(coders.FastPrimitivesCoder(), large_string), coder_benchmark_factory(coders.FastPrimitivesCoder(), small_list), coder_benchmark_factory(coders.IterableCoder(coders.FastPrimitivesCoder()), small_list), coder_benchmark_factory(coders.FastPrimitivesCoder(), large_list), coder_benchmark_factory(coders.IterableCoder(coders.FastPrimitivesCoder()), large_list), coder_benchmark_factory(coders.IterableCoder(coders.FastPrimitivesCoder()), large_iterable), coder_benchmark_factory(coders.FastPrimitivesCoder(), small_tuple), coder_benchmark_factory(coders.FastPrimitivesCoder(), large_tuple), coder_benchmark_factory(coders.FastPrimitivesCoder(), small_dict), coder_benchmark_factory(coders.FastPrimitivesCoder(), large_dict), coder_benchmark_factory(coders.ProtoCoder(test_message.MessageWithMap), small_message_with_map), coder_benchmark_factory(coders.ProtoCoder(test_message.MessageWithMap), large_message_with_map), coder_benchmark_factory(coders.DeterministicProtoCoder(test_message.MessageWithMap), small_message_with_map), coder_benchmark_factory(coders.DeterministicProtoCoder(test_message.MessageWithMap), large_message_with_map), coder_benchmark_factory(coders.WindowedValueCoder(coders.FastPrimitivesCoder()), wv_with_one_window), coder_benchmark_factory(coders.WindowedValueCoder(coders.FastPrimitivesCoder(), coders.IntervalWindowCoder()), wv_with_multiple_windows), coder_benchmark_factory(coders.WindowedValueCoder(coders.FastPrimitivesCoder(), coders.GlobalWindowCoder()), globally_windowed_value), coder_benchmark_factory(coders.LengthPrefixCoder(coders.FastPrimitivesCoder()), small_int), row_coder_benchmark_factory(tiny_row), row_coder_benchmark_factory(large_row), row_coder_benchmark_factory(nullable_row), row_coder_benchmark_factory(diverse_row), batch_row_coder_benchmark_factory(tiny_row, False), batch_row_coder_benchmark_factory(tiny_row, True), batch_row_coder_benchmark_factory(large_row, False), batch_row_coder_benchmark_factory(large_row, True), batch_row_coder_benchmark_factory(nullable_row, False), batch_row_coder_benchmark_factory(nullable_row, True), batch_row_coder_benchmark_factory(diverse_row, False), batch_row_coder_benchmark_factory(diverse_row, True)]
    suite = [utils.BenchmarkConfig(b, input_size, num_runs) for b in benchmarks if re.search(filter_regex, b.__name__, flags=re.I)]
    utils.run_benchmarks(suite, verbose=verbose)
if __name__ == '__main__':
    logging.basicConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', default='.*')
    parser.add_argument('--num_runs', default=20, type=int)
    parser.add_argument('--num_elements_per_benchmark', default=1000, type=int)
    parser.add_argument('--seed', default=42, type=int)
    options = parser.parse_args()
    utils.check_compiled('apache_beam.coders.coder_impl')
    num_runs = 20
    num_elements_per_benchmark = 1000
    seed = 42
    run_coder_benchmarks(options.num_runs, options.num_elements_per_benchmark, options.seed, verbose=True, filter_regex=options.filter)