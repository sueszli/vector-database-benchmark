import itertools
import numpy as np
import random as random
import tensorflow as tf
from nose.tools import with_setup
from nose_utils import raises
from test_dali_tf_dataset_pipelines import FixedSampleIterator, external_source_tester, external_source_converter_with_fixed_value, external_source_converter_with_callback, RandomSampleIterator, external_source_converter_multiple, get_min_shape_helper, external_source_tester_multiple
from test_dali_tf_es_pipelines import external_source_to_tf_dataset, gen_tf_with_dali_external_source, get_external_source_pipe
from test_utils_tensorflow import run_tf_dataset_graph, skip_inputs_for_incompatible_tf, run_dataset_in_graph, run_tf_dataset_multigpu_graph_manual_placement, get_dali_dataset_from_pipeline, get_image_pipeline
tf.compat.v1.disable_eager_execution()

def test_tf_dataset_gpu():
    if False:
        print('Hello World!')
    run_tf_dataset_graph('gpu')

def test_tf_dataset_cpu():
    if False:
        print('Hello World!')
    run_tf_dataset_graph('cpu')

def run_tf_dataset_with_constant_input(dev, shape, value, dtype, batch):
    if False:
        while True:
            i = 10
    tensor = np.full(shape, value, dtype)
    run_tf_dataset_graph(dev, get_pipeline_desc=external_source_tester(shape, dtype, FixedSampleIterator(tensor), batch=batch), to_dataset=external_source_converter_with_fixed_value(shape, dtype, tensor, batch))

@with_setup(skip_inputs_for_incompatible_tf)
def test_tf_dataset_with_constant_input():
    if False:
        return 10
    for dev in ['cpu', 'gpu']:
        for shape in [(7, 42), (64, 64, 3), (3, 40, 40, 4)]:
            for dtype in [np.uint8, np.int32, np.float32]:
                for batch in ['dataset', True, False, None]:
                    value = random.choice([42, 255])
                    yield (run_tf_dataset_with_constant_input, dev, shape, value, dtype, batch)

def run_tf_dataset_with_random_input(dev, max_shape, dtype, batch):
    if False:
        return 10
    min_shape = get_min_shape_helper(batch, max_shape)
    iterator = RandomSampleIterator(max_shape, dtype(0), min_shape=min_shape)
    run_tf_dataset_graph(dev, get_pipeline_desc=external_source_tester(max_shape, dtype, iterator, batch=batch), to_dataset=external_source_converter_with_callback(RandomSampleIterator, max_shape, dtype, 0, 10000000000.0, min_shape, batch=batch))

@with_setup(skip_inputs_for_incompatible_tf)
def test_tf_dataset_with_random_input():
    if False:
        print('Hello World!')
    for dev in ['cpu', 'gpu']:
        for max_shape in [(10, 20), (120, 120, 3), (3, 40, 40, 4)]:
            for dtype in [np.uint8, np.int32, np.float32]:
                for batch in ['dataset', True, False, None]:
                    yield (run_tf_dataset_with_random_input, dev, max_shape, dtype, batch)

def run_tf_dataset_with_random_input_gpu(max_shape, dtype, batch):
    if False:
        return 10
    min_shape = get_min_shape_helper(batch, max_shape)
    iterator = RandomSampleIterator(max_shape, dtype(0), min_shape=min_shape)
    run_tf_dataset_graph('gpu', get_pipeline_desc=external_source_tester(max_shape, dtype, iterator, 'gpu', batch=batch), to_dataset=external_source_converter_with_callback(RandomSampleIterator, max_shape, dtype, 0, 10000000000.0, min_shape, batch=batch))

@with_setup(skip_inputs_for_incompatible_tf)
def test_tf_dataset_with_random_input_gpu():
    if False:
        for i in range(10):
            print('nop')
    for max_shape in [(10, 20), (120, 120, 3), (3, 40, 40, 4)]:
        for dtype in [np.uint8, np.int32, np.float32]:
            for batch in ['dataset', True, False, None]:
                yield (run_tf_dataset_with_random_input_gpu, max_shape, dtype, batch)

def run_tf_dataset_no_copy(max_shape, dtype, dataset_dev, es_dev, no_copy):
    if False:
        for i in range(10):
            print('nop')
    run_tf_dataset_graph(dataset_dev, get_pipeline_desc=external_source_tester(max_shape, dtype, RandomSampleIterator(max_shape, dtype(0)), es_dev, no_copy), to_dataset=external_source_converter_with_callback(RandomSampleIterator, max_shape, dtype))

@with_setup(skip_inputs_for_incompatible_tf)
def test_tf_dataset_with_no_copy():
    if False:
        return 10
    for max_shape in [(10, 20), (120, 120, 3)]:
        for dataset_dev in ['cpu', 'gpu']:
            for es_dev in ['cpu', 'gpu']:
                if dataset_dev == 'cpu' and es_dev == 'gpu':
                    continue
                for no_copy in [True, False, None]:
                    yield (run_tf_dataset_no_copy, max_shape, np.uint8, dataset_dev, es_dev, no_copy)

def run_tf_dataset_with_stop_iter(dev, max_shape, dtype, stop_samples):
    if False:
        print('Hello World!')
    run_tf_dataset_graph(dev, to_stop_iter=True, get_pipeline_desc=external_source_tester(max_shape, dtype, RandomSampleIterator(max_shape, dtype(0), start=0, stop=stop_samples)), to_dataset=external_source_converter_with_callback(RandomSampleIterator, max_shape, dtype, 0, stop_samples))

@with_setup(skip_inputs_for_incompatible_tf)
def test_tf_dataset_with_stop_iter():
    if False:
        for i in range(10):
            print('nop')
    batch_size = 12
    for dev in ['cpu', 'gpu']:
        for max_shape in [(10, 20), (120, 120, 3), (3, 40, 40, 4)]:
            for dtype in [np.uint8, np.int32, np.float32]:
                for iters in [1, 2, 3, 4, 5]:
                    yield (run_tf_dataset_with_stop_iter, dev, max_shape, dtype, iters * batch_size - 3)

def run_tf_dataset_multi_input(dev, start_values, input_names, batches):
    if False:
        while True:
            i = 10
    run_tf_dataset_graph(dev, get_pipeline_desc=external_source_tester_multiple(start_values, input_names, batches), to_dataset=external_source_converter_multiple(start_values, input_names, batches))
start_values = [[np.full((2, 4), -42, dtype=np.int64), np.full((3, 5), -123.0, dtype=np.float32)], [np.full((3, 5), -3.14, dtype=np.float32)], [np.full((2, 4), -42, dtype=np.int64), np.full((3, 5), -666.0, dtype=np.float32), np.full((1, 7), 5, dtype=np.int8)]]
input_names = [['input_{}'.format(i) for (i, _) in enumerate(vals)] for vals in start_values]

@with_setup(skip_inputs_for_incompatible_tf)
def test_tf_dataset_multi_input():
    if False:
        for i in range(10):
            print('nop')
    for dev in ['cpu', 'gpu']:
        for (starts, names) in zip(start_values, input_names):
            yield (run_tf_dataset_multi_input, dev, starts, names, ['dataset' for _ in input_names])
            for batches in list(itertools.product([True, False], repeat=len(input_names))):
                yield (run_tf_dataset_multi_input, dev, starts, names, batches)

def run_tf_with_dali_external_source(dev, es_args, ed_dev, dtype, *_):
    if False:
        while True:
            i = 10
    run_tf_dataset_graph(dev, get_pipeline_desc=get_external_source_pipe(es_args, dtype, ed_dev), to_dataset=external_source_to_tf_dataset, to_stop_iter=True)

@with_setup(skip_inputs_for_incompatible_tf)
def test_tf_with_dali_external_source():
    if False:
        i = 10
        return i + 15
    yield from gen_tf_with_dali_external_source(run_tf_with_dali_external_source)
tf_dataset_wrong_placement_error_msg = 'TF device and DALI device mismatch. TF device: [\\w]*, DALI device: [\\w]* for output'

@raises(Exception, regex=tf_dataset_wrong_placement_error_msg)
def test_tf_dataset_wrong_placement_cpu():
    if False:
        i = 10
        return i + 15
    batch_size = 12
    num_threads = 4
    iterations = 10
    pipeline = get_image_pipeline(batch_size, num_threads, 'cpu', 0)
    with tf.device('/gpu:0'):
        dataset = get_dali_dataset_from_pipeline(pipeline, 'gpu', 0)
    run_dataset_in_graph(dataset, iterations)

@raises(Exception, regex=tf_dataset_wrong_placement_error_msg)
def test_tf_dataset_wrong_placement_gpu():
    if False:
        while True:
            i = 10
    batch_size = 12
    num_threads = 4
    iterations = 10
    pipeline = get_image_pipeline(batch_size, num_threads, 'gpu', 0)
    with tf.device('/cpu:0'):
        dataset = get_dali_dataset_from_pipeline(pipeline, 'cpu', 0)
    run_dataset_in_graph(dataset, iterations)

def _test_tf_dataset_other_gpu():
    if False:
        return 10
    run_tf_dataset_graph('gpu', 1)

def _test_tf_dataset_multigpu_manual_placement():
    if False:
        i = 10
        return i + 15
    run_tf_dataset_multigpu_graph_manual_placement()