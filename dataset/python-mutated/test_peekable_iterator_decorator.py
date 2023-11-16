from nvidia.dali.plugin.jax.clu import DALIGenericPeekableIterator
from nvidia.dali.plugin.jax.clu import peekable_data_iterator
from utils import iterator_function_def
from test_iterator import run_and_assert_sequential_iterator
import inspect
batch_size = 3
batch_shape = (batch_size, 1)

def test_dali_iterator_decorator_all_pipeline_args_in_call():
    if False:
        for i in range(10):
            print('nop')
    iter = peekable_data_iterator(iterator_function_def, output_map=['data'], reader_name='reader')(batch_size=batch_size, device_id=0, num_threads=4)
    run_and_assert_sequential_iterator(iter)

def test_dali_iterator_decorator_declarative():
    if False:
        while True:
            i = 10

    @peekable_data_iterator(output_map=['data'], reader_name='reader')
    def iterator_function():
        if False:
            return 10
        return iterator_function_def()
    iter = iterator_function(num_threads=4, device_id=0, batch_size=batch_size)
    run_and_assert_sequential_iterator(iter)

def test_dali_iterator_decorator_declarative_pipeline_fn_with_argument():
    if False:
        i = 10
        return i + 15

    @peekable_data_iterator(output_map=['data'], reader_name='reader')
    def iterator_function(num_shards):
        if False:
            i = 10
            return i + 15
        return iterator_function_def(num_shards=num_shards)
    iter = iterator_function(num_shards=2, num_threads=4, device_id=0, batch_size=batch_size)
    run_and_assert_sequential_iterator(iter)
    assert iter.size == 24

def test_iterator_decorator_api_match_iterator_init():
    if False:
        return 10
    iterator_init_args = inspect.getfullargspec(DALIGenericPeekableIterator.__init__).args
    iterator_init_args.remove('self')
    iterator_init_args.remove('pipelines')
    iterator_decorator_args = inspect.getfullargspec(peekable_data_iterator).args
    iterator_decorator_args.remove('pipeline_fn')
    assert iterator_decorator_args == iterator_init_args, 'Arguments for the iterator decorator and the iterator __init__ method do not match'
    iterator_decorator_docs = inspect.getdoc(peekable_data_iterator)
    iterator_decorator_docs = iterator_decorator_docs.split('output_map')[1]
    iterator_init_docs = inspect.getdoc(DALIGenericPeekableIterator)
    iterator_init_docs = iterator_init_docs.split('output_map')[1]
    assert iterator_decorator_docs == iterator_init_docs, 'Documentation for the iterator decorator and the iterator __init__ method does not match'