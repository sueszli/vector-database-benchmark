import re
import sys
import pytest
from _pytest.outcomes import Failed
from _pytest.pytester import Pytester

class TestRaises:

    def test_check_callable(self) -> None:
        if False:
            print('Hello World!')
        with pytest.raises(TypeError, match='.* must be callable'):
            pytest.raises(RuntimeError, "int('qwe')")

    def test_raises(self):
        if False:
            print('Hello World!')
        excinfo = pytest.raises(ValueError, int, 'qwe')
        assert 'invalid literal' in str(excinfo.value)

    def test_raises_function(self):
        if False:
            return 10
        excinfo = pytest.raises(ValueError, int, 'hello')
        assert 'invalid literal' in str(excinfo.value)

    def test_raises_does_not_allow_none(self):
        if False:
            while True:
                i = 10
        with pytest.raises(ValueError, match='Expected an exception type or'):
            pytest.raises(expected_exception=None)

    def test_raises_does_not_allow_empty_tuple(self):
        if False:
            print('Hello World!')
        with pytest.raises(ValueError, match='Expected an exception type or'):
            pytest.raises(expected_exception=())

    def test_raises_callable_no_exception(self) -> None:
        if False:
            i = 10
            return i + 15

        class A:

            def __call__(self):
                if False:
                    i = 10
                    return i + 15
                pass
        try:
            pytest.raises(ValueError, A())
        except pytest.fail.Exception:
            pass

    def test_raises_falsey_type_error(self) -> None:
        if False:
            print('Hello World!')
        with pytest.raises(TypeError):
            with pytest.raises(AssertionError, match=0):
                raise AssertionError('ohai')

    def test_raises_repr_inflight(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure repr() on an exception info inside a pytest.raises with block works (#4386)'

        class E(Exception):
            pass
        with pytest.raises(E) as excinfo:
            print(str(excinfo))
            print(repr(excinfo))
            import pprint
            pprint.pprint(excinfo)
            raise E()

    def test_raises_as_contextmanager(self, pytester: Pytester) -> None:
        if False:
            i = 10
            return i + 15
        pytester.makepyfile('\n            import pytest\n            import _pytest._code\n\n            def test_simple():\n                with pytest.raises(ZeroDivisionError) as excinfo:\n                    assert isinstance(excinfo, _pytest._code.ExceptionInfo)\n                    1/0\n                print(excinfo)\n                assert excinfo.type == ZeroDivisionError\n                assert isinstance(excinfo.value, ZeroDivisionError)\n\n            def test_noraise():\n                with pytest.raises(pytest.raises.Exception):\n                    with pytest.raises(ValueError):\n                           int()\n\n            def test_raise_wrong_exception_passes_by():\n                with pytest.raises(ZeroDivisionError):\n                    with pytest.raises(ValueError):\n                           1/0\n        ')
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*3 passed*'])

    def test_does_not_raise(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile("\n            from contextlib import nullcontext as does_not_raise\n            import pytest\n\n            @pytest.mark.parametrize('example_input,expectation', [\n                (3, does_not_raise()),\n                (2, does_not_raise()),\n                (1, does_not_raise()),\n                (0, pytest.raises(ZeroDivisionError)),\n            ])\n            def test_division(example_input, expectation):\n                '''Test how much I know division.'''\n                with expectation:\n                    assert (6 / example_input) is not None\n        ")
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*4 passed*'])

    def test_does_not_raise_does_raise(self, pytester: Pytester) -> None:
        if False:
            while True:
                i = 10
        pytester.makepyfile("\n            from contextlib import nullcontext as does_not_raise\n            import pytest\n\n            @pytest.mark.parametrize('example_input,expectation', [\n                (0, does_not_raise()),\n                (1, pytest.raises(ZeroDivisionError)),\n            ])\n            def test_division(example_input, expectation):\n                '''Test how much I know division.'''\n                with expectation:\n                    assert (6 / example_input) is not None\n        ")
        result = pytester.runpytest()
        result.stdout.fnmatch_lines(['*2 failed*'])

    def test_noclass(self) -> None:
        if False:
            return 10
        with pytest.raises(TypeError):
            pytest.raises('wrong', lambda : None)

    def test_invalid_arguments_to_raises(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(TypeError, match='unknown'):
            with pytest.raises(TypeError, unknown='bogus'):
                raise ValueError()

    def test_tuple(self):
        if False:
            return 10
        with pytest.raises((KeyError, ValueError)):
            raise KeyError('oops')

    def test_no_raise_message(self) -> None:
        if False:
            return 10
        try:
            pytest.raises(ValueError, int, '0')
        except pytest.fail.Exception as e:
            assert e.msg == f'DID NOT RAISE {repr(ValueError)}'
        else:
            assert False, 'Expected pytest.raises.Exception'
        try:
            with pytest.raises(ValueError):
                pass
        except pytest.fail.Exception as e:
            assert e.msg == f'DID NOT RAISE {repr(ValueError)}'
        else:
            assert False, 'Expected pytest.raises.Exception'

    @pytest.mark.parametrize('method', ['function', 'function_match', 'with'])
    def test_raises_cyclic_reference(self, method):
        if False:
            for i in range(10):
                print('nop')
        'Ensure pytest.raises does not leave a reference cycle (#1965).'
        import gc

        class T:

            def __call__(self):
                if False:
                    for i in range(10):
                        print('nop')
                raise ValueError
        t = T()
        refcount = len(gc.get_referrers(t))
        if method == 'function':
            pytest.raises(ValueError, t)
        elif method == 'function_match':
            pytest.raises(ValueError, t).match('^$')
        else:
            with pytest.raises(ValueError):
                t()
        assert sys.exc_info() == (None, None, None)
        assert refcount == len(gc.get_referrers(t))

    def test_raises_match(self) -> None:
        if False:
            while True:
                i = 10
        msg = 'with base \\d+'
        with pytest.raises(ValueError, match=msg):
            int('asdf')
        msg = 'with base 10'
        with pytest.raises(ValueError, match=msg):
            int('asdf')
        msg = 'with base 16'
        expr = f'''Regex pattern did not match.\n Regex: {msg!r}\n Input: "invalid literal for int() with base 10: 'asdf'"'''
        with pytest.raises(AssertionError, match='(?m)' + re.escape(expr)):
            with pytest.raises(ValueError, match=msg):
                int('asdf', base=10)
        pytest.raises(ValueError, int, 'asdf').match('invalid literal')
        with pytest.raises(AssertionError) as excinfo:
            pytest.raises(ValueError, int, 'asdf').match(msg)
        assert str(excinfo.value) == expr
        pytest.raises(TypeError, int, match='invalid')

        def tfunc(match):
            if False:
                for i in range(10):
                    print('nop')
            raise ValueError(f'match={match}')
        pytest.raises(ValueError, tfunc, match='asdf').match('match=asdf')
        pytest.raises(ValueError, tfunc, match='').match('match=')

    def test_match_failure_string_quoting(self):
        if False:
            print('Hello World!')
        with pytest.raises(AssertionError) as excinfo:
            with pytest.raises(AssertionError, match="'foo"):
                raise AssertionError("'bar")
        (msg,) = excinfo.value.args
        assert msg == 'Regex pattern did not match.\n Regex: "\'foo"\n Input: "\'bar"'

    def test_match_failure_exact_string_message(self):
        if False:
            for i in range(10):
                print('nop')
        message = 'Oh here is a message with (42) numbers in parameters'
        with pytest.raises(AssertionError) as excinfo:
            with pytest.raises(AssertionError, match=message):
                raise AssertionError(message)
        (msg,) = excinfo.value.args
        assert msg == "Regex pattern did not match.\n Regex: 'Oh here is a message with (42) numbers in parameters'\n Input: 'Oh here is a message with (42) numbers in parameters'\n Did you mean to `re.escape()` the regex?"

    def test_raises_match_wrong_type(self):
        if False:
            print('Hello World!')
        'Raising an exception with the wrong type and match= given.\n\n        pytest should throw the unexpected exception - the pattern match is not\n        really relevant if we got a different exception.\n        '
        with pytest.raises(ValueError):
            with pytest.raises(IndexError, match='nomatch'):
                int('asdf')

    def test_raises_exception_looks_iterable(self):
        if False:
            i = 10
            return i + 15

        class Meta(type):

            def __getitem__(self, item):
                if False:
                    print('Hello World!')
                return 1 / 0

            def __len__(self):
                if False:
                    i = 10
                    return i + 15
                return 1

        class ClassLooksIterableException(Exception, metaclass=Meta):
            pass
        with pytest.raises(Failed, match="DID NOT RAISE <class 'raises(\\..*)*ClassLooksIterableException'>"):
            pytest.raises(ClassLooksIterableException, lambda : None)

    def test_raises_with_raising_dunder_class(self) -> None:
        if False:
            print('Hello World!')
        'Test current behavior with regard to exceptions via __class__ (#4284).'

        class CrappyClass(Exception):

            @property
            def __class__(self):
                if False:
                    return 10
                assert False, 'via __class__'
        with pytest.raises(AssertionError) as excinfo:
            with pytest.raises(CrappyClass()):
                pass
        assert 'via __class__' in excinfo.value.args[0]

    def test_raises_context_manager_with_kwargs(self):
        if False:
            i = 10
            return i + 15
        with pytest.raises(TypeError) as excinfo:
            with pytest.raises(Exception, foo='bar'):
                pass
        assert 'Unexpected keyword arguments' in str(excinfo.value)

    def test_expected_exception_is_not_a_baseexception(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(TypeError) as excinfo:
            with pytest.raises('hello'):
                pass
        assert 'must be a BaseException type, not str' in str(excinfo.value)

        class NotAnException:
            pass
        with pytest.raises(TypeError) as excinfo:
            with pytest.raises(NotAnException):
                pass
        assert 'must be a BaseException type, not NotAnException' in str(excinfo.value)
        with pytest.raises(TypeError) as excinfo:
            with pytest.raises(('hello', NotAnException)):
                pass
        assert 'must be a BaseException type, not str' in str(excinfo.value)