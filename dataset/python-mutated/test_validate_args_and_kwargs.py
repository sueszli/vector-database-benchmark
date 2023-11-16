import pytest
from pandas.util._validators import validate_args_and_kwargs

@pytest.fixture
def _fname():
    if False:
        i = 10
        return i + 15
    return 'func'

def test_invalid_total_length_max_length_one(_fname):
    if False:
        i = 10
        return i + 15
    compat_args = ('foo',)
    kwargs = {'foo': 'FOO'}
    args = ('FoO', 'BaZ')
    min_fname_arg_count = 0
    max_length = len(compat_args) + min_fname_arg_count
    actual_length = len(kwargs) + len(args) + min_fname_arg_count
    msg = f'{_fname}\\(\\) takes at most {max_length} argument \\({actual_length} given\\)'
    with pytest.raises(TypeError, match=msg):
        validate_args_and_kwargs(_fname, args, kwargs, min_fname_arg_count, compat_args)

def test_invalid_total_length_max_length_multiple(_fname):
    if False:
        i = 10
        return i + 15
    compat_args = ('foo', 'bar', 'baz')
    kwargs = {'foo': 'FOO', 'bar': 'BAR'}
    args = ('FoO', 'BaZ')
    min_fname_arg_count = 2
    max_length = len(compat_args) + min_fname_arg_count
    actual_length = len(kwargs) + len(args) + min_fname_arg_count
    msg = f'{_fname}\\(\\) takes at most {max_length} arguments \\({actual_length} given\\)'
    with pytest.raises(TypeError, match=msg):
        validate_args_and_kwargs(_fname, args, kwargs, min_fname_arg_count, compat_args)

@pytest.mark.parametrize('args,kwargs', [((), {'foo': -5, 'bar': 2}), ((-5, 2), {})])
def test_missing_args_or_kwargs(args, kwargs, _fname):
    if False:
        for i in range(10):
            print('nop')
    bad_arg = 'bar'
    min_fname_arg_count = 2
    compat_args = {'foo': -5, bad_arg: 1}
    msg = f"the '{bad_arg}' parameter is not supported in the pandas implementation of {_fname}\\(\\)"
    with pytest.raises(ValueError, match=msg):
        validate_args_and_kwargs(_fname, args, kwargs, min_fname_arg_count, compat_args)

def test_duplicate_argument(_fname):
    if False:
        while True:
            i = 10
    min_fname_arg_count = 2
    compat_args = {'foo': None, 'bar': None, 'baz': None}
    kwargs = {'foo': None, 'bar': None}
    args = (None,)
    msg = f"{_fname}\\(\\) got multiple values for keyword argument 'foo'"
    with pytest.raises(TypeError, match=msg):
        validate_args_and_kwargs(_fname, args, kwargs, min_fname_arg_count, compat_args)

def test_validation(_fname):
    if False:
        while True:
            i = 10
    compat_args = {'foo': 1, 'bar': None, 'baz': -2}
    kwargs = {'baz': -2}
    args = (1, None)
    min_fname_arg_count = 2
    validate_args_and_kwargs(_fname, args, kwargs, min_fname_arg_count, compat_args)