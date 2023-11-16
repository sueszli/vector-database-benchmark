from utils import iterator_function_def
from nvidia.dali.plugin.jax import DALIGenericIterator, data_iterator
from test_iterator import run_and_assert_sequential_iterator
import inspect
batch_size = 3

def test_dali_iterator_decorator_functional():
    if False:
        i = 10
        return i + 15
    iter = data_iterator(iterator_function_def, output_map=['data'], reader_name='reader')(batch_size=batch_size, device_id=0, num_threads=4)
    run_and_assert_sequential_iterator(iter)

def test_dali_iterator_decorator_declarative():
    if False:
        print('Hello World!')

    @data_iterator(output_map=['data'], reader_name='reader')
    def iterator_function():
        if False:
            while True:
                i = 10
        return iterator_function_def()
    iter = iterator_function(num_threads=4, device_id=0, batch_size=batch_size)
    run_and_assert_sequential_iterator(iter)

def test_dali_iterator_decorator_declarative_with_default_args():
    if False:
        while True:
            i = 10

    @data_iterator(output_map=['data'], reader_name='reader')
    def iterator_function():
        if False:
            print('Hello World!')
        return iterator_function_def()
    iter = iterator_function(batch_size=batch_size)
    run_and_assert_sequential_iterator(iter)

def test_dali_iterator_decorator_declarative_pipeline_fn_with_argument():
    if False:
        while True:
            i = 10

    @data_iterator(output_map=['data'], reader_name='reader')
    def iterator_function(num_shards):
        if False:
            print('Hello World!')
        return iterator_function_def(num_shards=num_shards)
    iter = iterator_function(num_shards=2, num_threads=4, device_id=0, batch_size=batch_size)
    run_and_assert_sequential_iterator(iter)
    assert iter.size == 24

def test_iterator_decorator_api_match_iterator_init():
    if False:
        for i in range(10):
            print('nop')
    iterator_init_args = inspect.getfullargspec(DALIGenericIterator.__init__).args
    iterator_init_args.remove('self')
    iterator_init_args.remove('pipelines')
    iterator_decorator_args = inspect.getfullargspec(data_iterator).args
    iterator_decorator_args.remove('pipeline_fn')
    assert iterator_decorator_args == iterator_init_args, 'Arguments for the iterator decorator and the iterator __init__ method do not match'
    iterator_decorator_docs = inspect.getdoc(data_iterator)
    iterator_decorator_docs = iterator_decorator_docs.split('output_map')[1]
    iterator_init_docs = inspect.getdoc(DALIGenericIterator)
    iterator_init_docs = iterator_init_docs.split('output_map')[1]
    assert iterator_decorator_docs == iterator_init_docs, 'Documentation for the iterator decorator and the iterator __init__ method does not match'