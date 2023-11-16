import pytest
import sysconfig
import numpy as np
from numpy.testing import assert_, assert_raises, IS_WASM
hosttype = sysconfig.get_config_var('HOST_GNU_TYPE')
arm_softfloat = False if hosttype is None else hosttype.endswith('gnueabi')

class TestErrstate:

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.skipif(arm_softfloat, reason='platform/cpu issue with FPU (gh-413,-15562)')
    def test_invalid(self):
        if False:
            while True:
                i = 10
        with np.errstate(all='raise', under='ignore'):
            a = -np.arange(3)
            with np.errstate(invalid='ignore'):
                np.sqrt(a)
            with assert_raises(FloatingPointError):
                np.sqrt(a)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.skipif(arm_softfloat, reason='platform/cpu issue with FPU (gh-15562)')
    def test_divide(self):
        if False:
            while True:
                i = 10
        with np.errstate(all='raise', under='ignore'):
            a = -np.arange(3)
            with np.errstate(divide='ignore'):
                a // 0
            with assert_raises(FloatingPointError):
                a // 0
            with assert_raises(FloatingPointError):
                a // a

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.skipif(arm_softfloat, reason='platform/cpu issue with FPU (gh-15562)')
    def test_errcall(self):
        if False:
            print('Hello World!')
        count = 0

        def foo(*args):
            if False:
                while True:
                    i = 10
            nonlocal count
            count += 1
        olderrcall = np.geterrcall()
        with np.errstate(call=foo):
            assert np.geterrcall() is foo
            with np.errstate(call=None):
                assert np.geterrcall() is None
        assert np.geterrcall() is olderrcall
        assert count == 0
        with np.errstate(call=foo, invalid='call'):
            np.array(np.inf) - np.array(np.inf)
        assert count == 1

    def test_errstate_decorator(self):
        if False:
            print('Hello World!')

        @np.errstate(all='ignore')
        def foo():
            if False:
                i = 10
                return i + 15
            a = -np.arange(3)
            a // 0
        foo()

    def test_errstate_enter_once(self):
        if False:
            i = 10
            return i + 15
        errstate = np.errstate(invalid='warn')
        with errstate:
            pass
        with pytest.raises(TypeError, match='Cannot enter `np.errstate` twice'):
            with errstate:
                pass

    @pytest.mark.skipif(IS_WASM, reason="wasm doesn't support asyncio")
    def test_asyncio_safe(self):
        if False:
            i = 10
            return i + 15
        asyncio = pytest.importorskip('asyncio')

        @np.errstate(invalid='ignore')
        def decorated():
            if False:
                return 10
            assert np.geterr()['invalid'] == 'ignore'

        async def func1():
            decorated()
            await asyncio.sleep(0.1)
            decorated()

        async def func2():
            with np.errstate(invalid='raise'):
                assert np.geterr()['invalid'] == 'raise'
                await asyncio.sleep(0.125)
                assert np.geterr()['invalid'] == 'raise'

        async def func3():
            with np.errstate(invalid='print'):
                assert np.geterr()['invalid'] == 'print'
                await asyncio.sleep(0.11)
                assert np.geterr()['invalid'] == 'print'

        async def main():
            await asyncio.gather(func1(), func2(), func3(), func1(), func2(), func3(), func1(), func2(), func3(), func1(), func2(), func3())
        loop = asyncio.new_event_loop()
        with np.errstate(invalid='warn'):
            asyncio.run(main())
            assert np.geterr()['invalid'] == 'warn'
        assert np.geterr()['invalid'] == 'warn'
        loop.close()