from numba import types
from cudf.utils import cudautils

def setup_function():
    if False:
        for i in range(10):
            print('nop')
    cudautils._udf_code_cache.clear()

def assert_cache_size(size):
    if False:
        print('Hello World!')
    assert cudautils._udf_code_cache.currsize == size

def test_first_compile_sets_cache_entry():
    if False:
        for i in range(10):
            print('nop')
    cudautils.compile_udf(lambda x: x + 1, (types.float32,))
    assert_cache_size(1)

def test_code_cache_same_code_different_function_hit():
    if False:
        return 10
    cudautils.compile_udf(lambda x: x + 1, (types.float32,))
    assert_cache_size(1)
    cudautils.compile_udf(lambda x: x + 1, (types.float32,))
    assert_cache_size(1)

def test_code_cache_different_types_miss():
    if False:
        return 10
    cudautils.compile_udf(lambda x: x + 1, (types.float32,))
    assert_cache_size(1)
    cudautils.compile_udf(lambda x: x + 1, (types.float64,))
    assert_cache_size(2)

def test_code_cache_different_cvars_miss():
    if False:
        print('Hello World!')

    def gen_closure(y):
        if False:
            i = 10
            return i + 15
        return lambda x: x + y
    cudautils.compile_udf(gen_closure(1), (types.float32,))
    assert_cache_size(1)
    cudautils.compile_udf(gen_closure(2), (types.float32,))
    assert_cache_size(2)

def test_lambda_in_loop_code_cached():
    if False:
        while True:
            i = 10
    for i in range(3):
        cudautils.compile_udf(lambda x: x + 1, (types.float32,))
    assert_cache_size(1)