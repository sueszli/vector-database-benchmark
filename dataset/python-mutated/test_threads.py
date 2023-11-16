from unittest.mock import patch
import pytest
from celery.utils.threads import Local, LocalManager, _FastLocalStack, _LocalStack, bgThread
from t.unit import conftest

class test_bgThread:

    def test_crash(self):
        if False:
            for i in range(10):
                print('nop')

        class T(bgThread):

            def body(self):
                if False:
                    return 10
                raise KeyError()
        with patch('os._exit') as _exit:
            with conftest.stdouts():
                _exit.side_effect = ValueError()
                t = T()
                with pytest.raises(ValueError):
                    t.run()
                _exit.assert_called_with(1)

    def test_interface(self):
        if False:
            for i in range(10):
                print('nop')
        x = bgThread()
        with pytest.raises(NotImplementedError):
            x.body()

class test_Local:

    def test_iter(self):
        if False:
            print('Hello World!')
        x = Local()
        x.foo = 'bar'
        ident = x.__ident_func__()
        assert (ident, {'foo': 'bar'}) in list(iter(x))
        delattr(x, 'foo')
        assert (ident, {'foo': 'bar'}) not in list(iter(x))
        with pytest.raises(AttributeError):
            delattr(x, 'foo')
        assert x(lambda : 'foo') is not None

class test_LocalStack:

    def test_stack(self):
        if False:
            while True:
                i = 10
        x = _LocalStack()
        assert x.pop() is None
        x.__release_local__()
        ident = x.__ident_func__
        x.__ident_func__ = ident
        with pytest.raises(RuntimeError):
            x()[0]
        x.push(['foo'])
        assert x()[0] == 'foo'
        x.pop()
        with pytest.raises(RuntimeError):
            x()[0]

class test_FastLocalStack:

    def test_stack(self):
        if False:
            while True:
                i = 10
        x = _FastLocalStack()
        x.push(['foo'])
        x.push(['bar'])
        assert x.top == ['bar']
        assert len(x) == 2
        x.pop()
        assert x.top == ['foo']
        x.pop()
        assert x.top is None

class test_LocalManager:

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        x = LocalManager()
        assert x.locals == []
        assert x.ident_func

        def ident():
            if False:
                i = 10
                return i + 15
            return 1
        loc = Local()
        x = LocalManager([loc], ident_func=ident)
        assert x.locals == [loc]
        x = LocalManager(loc, ident_func=ident)
        assert x.locals == [loc]
        assert x.ident_func is ident
        assert x.locals[0].__ident_func__ is ident
        assert x.get_ident() == 1
        with patch('celery.utils.threads.release_local') as release:
            x.cleanup()
            release.assert_called_with(loc)
        assert repr(x)