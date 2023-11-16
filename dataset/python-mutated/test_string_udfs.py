import numba
import numpy as np
import pandas as pd
import pytest
from numba import cuda
from numba.core.typing import signature as nb_signature
from numba.types import CPointer, void
import rmm
import cudf
from cudf._lib.strings_udf import column_from_udf_string_array, column_to_string_view_array
from cudf.core.udf.strings_typing import str_view_arg_handler, string_view, udf_string
from cudf.core.udf.utils import _PTX_FILE, _get_extensionty_size
from cudf.testing._utils import assert_eq, sv_to_udf_str
from cudf.utils._numba import _CUDFNumbaConfig

def get_kernels(func, dtype, size):
    if False:
        return 10
    "\n    Create two kernels for testing a single scalar string function.\n    The first tests the function's action on a string_view object and\n    the second tests the same except using a udf_string object.\n    Allocates an output vector with a dtype specified by the caller\n    The returned kernels execute the input function on each data\n    element of the input and returns the output into the output vector\n    "
    func = cuda.jit(device=True)(func)
    if dtype == 'str':
        outty = CPointer(udf_string)
    else:
        outty = numba.np.numpy_support.from_dtype(dtype)[::1]
    sig = nb_signature(void, CPointer(string_view), outty)

    @cuda.jit(sig, link=[_PTX_FILE], extensions=[str_view_arg_handler])
    def string_view_kernel(input_strings, output_col):
        if False:
            while True:
                i = 10
        id = cuda.grid(1)
        if id < size:
            st = input_strings[id]
            result = func(st)
            output_col[id] = result

    @cuda.jit(sig, link=[_PTX_FILE], extensions=[str_view_arg_handler])
    def udf_string_kernel(input_strings, output_col):
        if False:
            return 10
        id = cuda.grid(1)
        if id < size:
            st = input_strings[id]
            st = sv_to_udf_str(st)
            result = func(st)
            output_col[id] = result
    return (string_view_kernel, udf_string_kernel)

def run_udf_test(data, func, dtype):
    if False:
        print('Hello World!')
    '\n    Run a test kernel on a set of input data\n    Converts the input data to a cuDF column and subsequently\n    to an array of cudf::string_view objects. It then creates\n    a CUDA kernel using get_kernel which calls the input function,\n    and then assembles the result back into a cuDF series before\n    comparing it with the equivalent pandas result\n    '
    if dtype == 'str':
        output = rmm.DeviceBuffer(size=len(data) * _get_extensionty_size(udf_string))
    else:
        dtype = np.dtype(dtype)
        output = cudf.core.column.column_empty(len(data), dtype=dtype)
    cudf_column = cudf.core.column.as_column(data)
    str_views = column_to_string_view_array(cudf_column)
    (sv_kernel, udf_str_kernel) = get_kernels(func, dtype, len(data))
    expect = pd.Series(data).apply(func)
    with _CUDFNumbaConfig():
        sv_kernel.forall(len(data))(str_views, output)
    if dtype == 'str':
        result = column_from_udf_string_array(output)
    else:
        result = output
    got = cudf.Series(result, dtype=dtype)
    assert_eq(expect, got, check_dtype=False)
    with _CUDFNumbaConfig():
        udf_str_kernel.forall(len(data))(str_views, output)
    if dtype == 'str':
        result = column_from_udf_string_array(output)
    else:
        result = output
    got = cudf.Series(result, dtype=dtype)
    assert_eq(expect, got, check_dtype=False)

@pytest.fixture(scope='module')
def data():
    if False:
        print('Hello World!')
    return ['abc', 'ABC', 'AbC', '123', '123aBc', '123@.!', '', 'rapids ai', 'gpu', 'True', 'False', '1.234', '.123a', '0.013', '1.0', '01', '20010101', 'cudf', 'cuda', 'gpu', 'This Is A Title', 'This is Not a Title', 'Neither is This a Title', 'NoT a TiTlE', '123 Title Works']

@pytest.fixture(params=['cudf', 'cuda', 'gpucudf', 'abc'])
def rhs(request):
    if False:
        i = 10
        return i + 15
    return request.param

@pytest.fixture(params=['c', 'cu', '2', 'abc', '', 'gpu'])
def substr(request):
    if False:
        print('Hello World!')
    return request.param

def test_string_udf_eq(data, rhs):
    if False:
        return 10

    def func(st):
        if False:
            print('Hello World!')
        return st == rhs
    run_udf_test(data, func, 'bool')

def test_string_udf_ne(data, rhs):
    if False:
        for i in range(10):
            print('nop')

    def func(st):
        if False:
            print('Hello World!')
        return st != rhs
    run_udf_test(data, func, 'bool')

def test_string_udf_ge(data, rhs):
    if False:
        while True:
            i = 10

    def func(st):
        if False:
            for i in range(10):
                print('nop')
        return st >= rhs
    run_udf_test(data, func, 'bool')

def test_string_udf_le(data, rhs):
    if False:
        i = 10
        return i + 15

    def func(st):
        if False:
            return 10
        return st <= rhs
    run_udf_test(data, func, 'bool')

def test_string_udf_gt(data, rhs):
    if False:
        for i in range(10):
            print('nop')

    def func(st):
        if False:
            print('Hello World!')
        return st > rhs
    run_udf_test(data, func, 'bool')

def test_string_udf_lt(data, rhs):
    if False:
        for i in range(10):
            print('nop')

    def func(st):
        if False:
            for i in range(10):
                print('nop')
        return st < rhs
    run_udf_test(data, func, 'bool')

def test_string_udf_contains(data, substr):
    if False:
        while True:
            i = 10

    def func(st):
        if False:
            for i in range(10):
                print('nop')
        return substr in st
    run_udf_test(data, func, 'bool')

def test_string_udf_count(data, substr):
    if False:
        print('Hello World!')

    def func(st):
        if False:
            while True:
                i = 10
        return st.count(substr)
    run_udf_test(data, func, 'int32')

def test_string_udf_find(data, substr):
    if False:
        while True:
            i = 10

    def func(st):
        if False:
            i = 10
            return i + 15
        return st.find(substr)
    run_udf_test(data, func, 'int32')

def test_string_udf_endswith(data, substr):
    if False:
        while True:
            i = 10

    def func(st):
        if False:
            for i in range(10):
                print('nop')
        return st.endswith(substr)
    run_udf_test(data, func, 'bool')

def test_string_udf_isalnum(data):
    if False:
        i = 10
        return i + 15

    def func(st):
        if False:
            print('Hello World!')
        return st.isalnum()
    run_udf_test(data, func, 'bool')

def test_string_udf_isalpha(data):
    if False:
        return 10

    def func(st):
        if False:
            return 10
        return st.isalpha()
    run_udf_test(data, func, 'bool')

def test_string_udf_isdecimal(data):
    if False:
        print('Hello World!')

    def func(st):
        if False:
            while True:
                i = 10
        return st.isdecimal()
    run_udf_test(data, func, 'bool')

def test_string_udf_isdigit(data):
    if False:
        return 10

    def func(st):
        if False:
            for i in range(10):
                print('nop')
        return st.isdigit()
    run_udf_test(data, func, 'bool')

def test_string_udf_islower(data):
    if False:
        return 10

    def func(st):
        if False:
            for i in range(10):
                print('nop')
        return st.islower()
    run_udf_test(data, func, 'bool')

def test_string_udf_isnumeric(data):
    if False:
        while True:
            i = 10

    def func(st):
        if False:
            for i in range(10):
                print('nop')
        return st.isnumeric()
    run_udf_test(data, func, 'bool')

def test_string_udf_isspace(data):
    if False:
        while True:
            i = 10

    def func(st):
        if False:
            while True:
                i = 10
        return st.isspace()
    run_udf_test(data, func, 'bool')

def test_string_udf_isupper(data):
    if False:
        for i in range(10):
            print('nop')

    def func(st):
        if False:
            return 10
        return st.isupper()
    run_udf_test(data, func, 'bool')

def test_string_udf_istitle(data):
    if False:
        for i in range(10):
            print('nop')

    def func(st):
        if False:
            for i in range(10):
                print('nop')
        return st.istitle()
    run_udf_test(data, func, 'bool')

def test_string_udf_len(data):
    if False:
        while True:
            i = 10

    def func(st):
        if False:
            i = 10
            return i + 15
        return len(st)
    run_udf_test(data, func, 'int64')

def test_string_udf_rfind(data, substr):
    if False:
        for i in range(10):
            print('nop')

    def func(st):
        if False:
            for i in range(10):
                print('nop')
        return st.rfind(substr)
    run_udf_test(data, func, 'int32')

def test_string_udf_startswith(data, substr):
    if False:
        for i in range(10):
            print('nop')

    def func(st):
        if False:
            i = 10
            return i + 15
        return st.startswith(substr)
    run_udf_test(data, func, 'bool')

def test_string_udf_return_string(data):
    if False:
        i = 10
        return i + 15

    def func(st):
        if False:
            while True:
                i = 10
        return st
    run_udf_test(data, func, 'str')

@pytest.mark.parametrize('strip_char', ['1', 'a', '12', ' ', '', '.', '@'])
def test_string_udf_strip(data, strip_char):
    if False:
        i = 10
        return i + 15

    def func(st):
        if False:
            return 10
        return st.strip(strip_char)
    run_udf_test(data, func, 'str')

@pytest.mark.parametrize('strip_char', ['1', 'a', '12', ' ', '', '.', '@'])
def test_string_udf_lstrip(data, strip_char):
    if False:
        i = 10
        return i + 15

    def func(st):
        if False:
            while True:
                i = 10
        return st.lstrip(strip_char)
    run_udf_test(data, func, 'str')

@pytest.mark.parametrize('strip_char', ['1', 'a', '12', ' ', '', '.', '@'])
def test_string_udf_rstrip(data, strip_char):
    if False:
        while True:
            i = 10

    def func(st):
        if False:
            for i in range(10):
                print('nop')
        return st.rstrip(strip_char)
    run_udf_test(data, func, 'str')

def test_string_udf_upper(data):
    if False:
        while True:
            i = 10

    def func(st):
        if False:
            i = 10
            return i + 15
        return st.upper()
    run_udf_test(data, func, 'str')

def test_string_udf_lower(data):
    if False:
        while True:
            i = 10

    def func(st):
        if False:
            print('Hello World!')
        return st.lower()
    run_udf_test(data, func, 'str')

@pytest.mark.parametrize('concat_char', ['1', 'a', '12', ' ', '', '.', '@'])
def test_string_udf_concat(data, concat_char):
    if False:
        i = 10
        return i + 15

    def func(st):
        if False:
            for i in range(10):
                print('nop')
        return st + concat_char
    run_udf_test(data, func, 'str')

@pytest.mark.parametrize('concat_char', ['1', 'a', '12', ' ', '', '.', '@'])
def test_string_udf_concat_reflected(data, concat_char):
    if False:
        while True:
            i = 10

    def func(st):
        if False:
            while True:
                i = 10
        return concat_char + st
    run_udf_test(data, func, 'str')

@pytest.mark.parametrize('to_replace', ['a', '1', '', '@'])
@pytest.mark.parametrize('replacement', ['a', '1', '', '@'])
def test_string_udf_replace(data, to_replace, replacement):
    if False:
        return 10

    def func(st):
        if False:
            for i in range(10):
                print('nop')
        return st.replace(to_replace, replacement)
    run_udf_test(data, func, 'str')