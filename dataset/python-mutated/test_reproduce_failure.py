import base64
import re
import zlib
import pytest
from hypothesis import Verbosity, __version__, example, given, reject, reproduce_failure, settings, strategies as st
from hypothesis.core import decode_failure, encode_failure
from hypothesis.errors import DidNotReproduce, InvalidArgument, UnsatisfiedAssumption
from tests.common.utils import capture_out, no_shrink

@example(bytes(20))
@example(bytes(3))
@given(st.binary() | st.binary(min_size=100))
def test_encoding_loop(b):
    if False:
        for i in range(10):
            print('nop')
    assert decode_failure(encode_failure(b)) == b

@example(base64.b64encode(b'\x02\x03\x04'))
@example(b'\t')
@example(base64.b64encode(b'\x01\x00'))
@given(st.binary())
def test_decoding_may_fail(t):
    if False:
        print('Hello World!')
    try:
        decode_failure(t)
        reject()
    except UnsatisfiedAssumption:
        raise
    except InvalidArgument:
        pass
    except Exception as e:
        raise AssertionError('Expected an InvalidArgument exception') from e

def test_invalid_base_64_gives_invalid_argument():
    if False:
        return 10
    with pytest.raises(InvalidArgument) as exc_info:
        decode_failure(b'/')
    assert 'Invalid base64 encoded' in exc_info.value.args[0]

def test_reproduces_the_failure():
    if False:
        while True:
            i = 10
    b = b'hello world'
    n = len(b)

    @reproduce_failure(__version__, encode_failure(b))
    @given(st.binary(min_size=n, max_size=n))
    def test_outer(x):
        if False:
            return 10
        assert x != b

    @given(st.binary(min_size=n, max_size=n))
    @reproduce_failure(__version__, encode_failure(b))
    def test_inner(x):
        if False:
            i = 10
            return i + 15
        assert x != b
    with pytest.raises(AssertionError):
        test_outer()
    with pytest.raises(AssertionError):
        test_inner()

def test_errors_if_provided_example_does_not_reproduce_failure():
    if False:
        while True:
            i = 10
    b = b'hello world'
    n = len(b)

    @reproduce_failure(__version__, encode_failure(b))
    @given(st.binary(min_size=n, max_size=n))
    def test(x):
        if False:
            while True:
                i = 10
        assert x == b
    with pytest.raises(DidNotReproduce):
        test()

def test_errors_with_did_not_reproduce_if_the_shape_changes():
    if False:
        return 10
    b = b'hello world'
    n = len(b)

    @reproduce_failure(__version__, encode_failure(b))
    @given(st.binary(min_size=n + 1, max_size=n + 1))
    def test(x):
        if False:
            return 10
        assert x == b
    with pytest.raises(DidNotReproduce):
        test()

def test_errors_with_did_not_reproduce_if_rejected():
    if False:
        while True:
            i = 10
    b = b'hello world'
    n = len(b)

    @reproduce_failure(__version__, encode_failure(b))
    @given(st.binary(min_size=n, max_size=n))
    def test(x):
        if False:
            i = 10
            return i + 15
        reject()
    with pytest.raises(DidNotReproduce):
        test()

def test_prints_reproduction_if_requested():
    if False:
        for i in range(10):
            print('nop')
    failing_example = [None]

    @settings(print_blob=True, database=None, max_examples=100)
    @given(st.integers())
    def test(i):
        if False:
            for i in range(10):
                print('nop')
        if failing_example[0] is None and i != 0:
            failing_example[0] = i
        assert i not in failing_example
    with pytest.raises(AssertionError) as err:
        test()
    notes = '\n'.join(err.value.__notes__)
    assert '@reproduce_failure' in notes
    exp = re.compile('reproduce_failure\\(([^)]+)\\)', re.MULTILINE)
    extract = exp.search(notes)
    reproduction = eval(extract.group(0))
    test = reproduction(test)
    with pytest.raises(AssertionError):
        test()

def test_does_not_print_reproduction_for_simple_examples_by_default():
    if False:
        for i in range(10):
            print('nop')

    @settings(print_blob=False)
    @given(st.integers())
    def test(i):
        if False:
            for i in range(10):
                print('nop')
        raise AssertionError
    with capture_out() as o:
        with pytest.raises(AssertionError):
            test()
    assert '@reproduce_failure' not in o.getvalue()

def test_does_not_print_reproduction_for_simple_data_examples_by_default():
    if False:
        for i in range(10):
            print('nop')

    @settings(print_blob=False)
    @given(st.data())
    def test(data):
        if False:
            return 10
        data.draw(st.integers())
        raise AssertionError
    with capture_out() as o:
        with pytest.raises(AssertionError):
            test()
    assert '@reproduce_failure' not in o.getvalue()

def test_does_not_print_reproduction_for_large_data_examples_by_default():
    if False:
        for i in range(10):
            print('nop')

    @settings(phases=no_shrink, print_blob=False)
    @given(st.data())
    def test(data):
        if False:
            i = 10
            return i + 15
        b = data.draw(st.binary(min_size=1000, max_size=1000))
        if len(zlib.compress(b)) > 1000:
            raise ValueError
    with capture_out() as o:
        with pytest.raises(ValueError):
            test()
    assert '@reproduce_failure' not in o.getvalue()

class Foo:

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'not a valid python expression'

def test_does_not_print_reproduction_if_told_not_to():
    if False:
        for i in range(10):
            print('nop')

    @settings(print_blob=False)
    @given(st.integers().map(lambda x: Foo()))
    def test(i):
        if False:
            for i in range(10):
                print('nop')
        raise ValueError
    with capture_out() as o:
        with pytest.raises(ValueError):
            test()
    assert '@reproduce_failure' not in o.getvalue()

def test_raises_invalid_if_wrong_version():
    if False:
        for i in range(10):
            print('nop')
    b = b'hello world'
    n = len(b)

    @reproduce_failure('1.0.0', encode_failure(b))
    @given(st.binary(min_size=n, max_size=n))
    def test(x):
        if False:
            print('Hello World!')
        pass
    with pytest.raises(InvalidArgument):
        test()

def test_does_not_print_reproduction_if_verbosity_set_to_quiet():
    if False:
        while True:
            i = 10

    @given(st.data())
    @settings(verbosity=Verbosity.quiet, print_blob=False)
    def test_always_fails(data):
        if False:
            for i in range(10):
                print('nop')
        assert data.draw(st.just(False))
    with capture_out() as out:
        with pytest.raises(AssertionError):
            test_always_fails()
    assert '@reproduce_failure' not in out.getvalue()