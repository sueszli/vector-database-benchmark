import jax.numpy as jnp
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.jax.clu import DALIGenericPeekableIterator as DALIPeekableIterator
from nvidia.dali.plugin.jax.iterator import DALIGenericIterator
from utils import iterator_function_def
from clu.data.dataset_iterator import ArraySpec
from nose_utils import raises
import time
import inspect
from utils import pipeline_with_variable_shape_output
batch_size = 3
batch_shape = (batch_size, 1)

def test_jax_peekable_iterator_peek():
    if False:
        while True:
            i = 10
    pipe = pipeline_def(iterator_function_def)(batch_size=batch_size, num_threads=4, device_id=0)
    iterator = DALIPeekableIterator([pipe], ['data'], reader_name='reader')
    assert iterator.element_spec == {'data': ArraySpec(dtype=jnp.int32, shape=batch_shape)}
    for i in range(5):
        peeked_output = iterator.peek()
        output = iterator.next()
        assert jnp.array_equal(output['data'], peeked_output['data'])

def test_jax_peekable_iterator_peek_async_result_before_next():
    if False:
        print('Hello World!')
    pipe = pipeline_def(iterator_function_def)(batch_size=batch_size, num_threads=4, device_id=0)
    iterator = DALIPeekableIterator([pipe], ['data'], reader_name='reader')
    assert iterator.element_spec == {'data': ArraySpec(dtype=jnp.int32, shape=batch_shape)}
    for i in range(5):
        peeked_output = iterator.peek_async()
        peeked_output = peeked_output.result()
        output = iterator.next()
        assert jnp.array_equal(output['data'], peeked_output['data']), f"output: {output['data']}, peeked_output: {peeked_output['data']}"

def test_jax_peekable_iterator_peek_async_result_after_next():
    if False:
        while True:
            i = 10
    'This test is not deterministic, but it should pass most of the time.'
    pipe = pipeline_def(iterator_function_def)(batch_size=batch_size, num_threads=4, device_id=0)
    iterator = DALIPeekableIterator([pipe], ['data'], reader_name='reader')
    assert iterator.element_spec == {'data': ArraySpec(dtype=jnp.int32, shape=batch_shape)}
    for i in range(5):
        peeked_output = iterator.peek_async()
        time.sleep(0.1)
        output = iterator.next()
        peeked_output = peeked_output.result()
        assert jnp.array_equal(output['data'], peeked_output['data']), f"output: {output['data']}, peeked_output: {peeked_output['data']}"

@raises(ValueError, glob='The shape or type of the output changed between iterations.')
def test_jax_peekable_iterator_with_variable_shapes_pipeline():
    if False:
        for i in range(10):
            print('nop')
    batch_size = 1
    pipe = pipeline_with_variable_shape_output(batch_size)
    iterator = DALIPeekableIterator([pipe], ['data'], size=batch_size * 100)
    iterator.next()
    iterator.next()

def test_iterators_init_method_api_compatibility():
    if False:
        i = 10
        return i + 15
    iterator_init_args = inspect.getfullargspec(DALIGenericIterator.__init__).args
    peekalbe_iterator_init_args = inspect.getfullargspec(DALIPeekableIterator.__init__).args
    assert iterator_init_args == peekalbe_iterator_init_args
    iterator_decorator_docs = inspect.getdoc(DALIGenericIterator)
    iterator_decorator_docs = iterator_decorator_docs.split('output_map')[1]
    iterator_init_docs = inspect.getdoc(DALIPeekableIterator)
    iterator_init_docs = iterator_init_docs.split('output_map')[1]
    assert iterator_decorator_docs == iterator_init_docs