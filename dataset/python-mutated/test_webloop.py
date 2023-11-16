import pytest
from pytest_pyodide.decorator import run_in_pyodide

def run_with_resolve(selenium, code):
    if False:
        i = 10
        return i + 15
    selenium.run_js(f'\n        try {{\n            let promise = new Promise((resolve) => self.resolve = resolve);\n            pyodide.runPython({code!r});\n            await promise;\n        }} finally {{\n            delete self.resolve;\n        }}\n        ')

def test_asyncio_sleep(selenium):
    if False:
        return 10
    run_with_resolve(selenium, "\n        import asyncio\n        from js import resolve\n        async def sleep_task():\n            print('start sleeping for 1s')\n            await asyncio.sleep(1)\n            print('sleeping done')\n            resolve()\n        asyncio.ensure_future(sleep_task())\n        None\n        ")

def test_cancel_handle(selenium_standalone):
    if False:
        print('Hello World!')
    selenium_standalone.run_js('\n        await pyodide.runPythonAsync(`\n            import asyncio\n            loop = asyncio.get_event_loop()\n            exc = []\n            def exception_handler(loop, context):\n                exc.append(context)\n            loop.set_exception_handler(exception_handler)\n            try:\n                await asyncio.wait_for(asyncio.sleep(1), 2)\n            finally:\n                loop.set_exception_handler(None)\n            assert not exc\n        `);\n        ')

def test_return_result(selenium):
    if False:
        for i in range(10):
            print('nop')
    run_with_resolve(selenium, '\n        from js import resolve\n        async def foo(arg):\n            return arg\n\n        def check_result(fut):\n            result = fut.result()\n            if result == 998:\n                resolve()\n            else:\n                raise Exception(f"Unexpected result {result!r}")\n        import asyncio\n        fut = asyncio.ensure_future(foo(998))\n        fut.add_done_callback(check_result)\n\n        ')

def test_capture_exception(selenium):
    if False:
        print('Hello World!')
    run_with_resolve(selenium, "\n        from unittest import TestCase\n        raises = TestCase().assertRaises\n        from js import resolve\n        class MyException(Exception):\n            pass\n        async def foo(arg):\n            raise MyException('oops')\n\n        def capture_exception(fut):\n            with raises(MyException):\n                fut.result()\n            resolve()\n        import asyncio\n        fut = asyncio.ensure_future(foo(998))\n        fut.add_done_callback(capture_exception)\n        ")

def test_await_js_promise(selenium):
    if False:
        print('Hello World!')
    run_with_resolve(selenium, "\n        from js import fetch, resolve\n        async def fetch_task():\n            print('fetching data...')\n            result = await fetch('console.html')\n            resolve()\n        import asyncio\n        asyncio.ensure_future(fetch_task())\n        None\n        ")

def test_call_soon(selenium):
    if False:
        i = 10
        return i + 15
    run_with_resolve(selenium, '\n        from js import resolve\n        def foo(arg):\n            if arg == \'bar\':\n                resolve()\n            else:\n                raise Exception("Expected arg == \'bar\'...")\n        import asyncio\n        asyncio.get_event_loop().call_soon(foo, \'bar\')\n        None\n        ')

def test_contextvars(selenium):
    if False:
        while True:
            i = 10
    run_with_resolve(selenium, '\n        from js import resolve\n        import contextvars\n        request_id = contextvars.ContextVar(\'Id of request.\')\n        request_id.set(123)\n        ctx = contextvars.copy_context()\n        request_id.set(456)\n        def func_ctx():\n            if request_id.get() == 123:\n                resolve()\n            else:\n                raise Exception(f"Expected request_id.get() == \'123\', got {request_id.get()!r}")\n        import asyncio\n        asyncio.get_event_loop().call_soon(func_ctx, context=ctx)\n        None\n        ')

def test_asyncio_exception(selenium):
    if False:
        print('Hello World!')
    run_with_resolve(selenium, '\n        from unittest import TestCase\n        raises = TestCase().assertRaises\n        from js import resolve\n        async def dummy_task():\n            raise ValueError("oops!")\n        async def capture_exception():\n            with raises(ValueError):\n                await dummy_task()\n            resolve()\n        import asyncio\n        asyncio.ensure_future(capture_exception())\n        None\n        ')

@pytest.mark.skip_pyproxy_check
def test_run_in_executor(selenium):
    if False:
        for i in range(10):
            print('nop')
    selenium.run_js('\n        pyodide.runPythonAsync(`\n            from concurrent.futures import ThreadPoolExecutor\n            import asyncio\n            def f():\n                return 5\n            result = await asyncio.get_event_loop().run_in_executor(ThreadPoolExecutor(), f)\n            assert result == 5\n        `);\n        ')

@pytest.mark.xfail(reason='Works locally but failing in test suite as of #2022.')
def test_webloop_exception_handler(selenium_standalone):
    if False:
        for i in range(10):
            print('nop')
    selenium = selenium_standalone
    selenium.run_async('\n        import asyncio\n        async def test():\n            raise Exception("test")\n        asyncio.ensure_future(test())\n        await asyncio.sleep(0.2)\n        ')
    assert 'Task exception was never retrieved' in selenium.logs
    try:
        selenium.run_js('\n            pyodide.runPython(`\n                import asyncio\n                loop = asyncio.get_event_loop()\n                exc = []\n                def exception_handler(loop, context):\n                    exc.append(context)\n                loop.set_exception_handler(exception_handler)\n\n                async def test():\n                    raise Exception("blah")\n                asyncio.ensure_future(test());\n                1\n            `);\n            await sleep(100)\n            pyodide.runPython(`\n                assert exc[0]["exception"].args[0] == "blah"\n            `)\n            ')
    finally:
        selenium.run('loop.set_exception_handler(None)')

@pytest.mark.asyncio
async def test_pyodide_future():
    import asyncio
    from pyodide.webloop import PyodideFuture
    fut: PyodideFuture[int]
    fut = PyodideFuture()
    increment = lambda x: x + 1
    tostring = lambda x: repr(x)

    def raises(x):
        if False:
            return 10
        raise Exception(x)
    rf = fut.then(increment).then(increment)
    fut.set_result(5)
    assert await rf == 7
    e = Exception('oops')
    fut = PyodideFuture()
    rf = fut.then(increment, tostring)
    fut.set_exception(e)
    assert await rf == repr(e)
    e = Exception('oops')
    fut = PyodideFuture()
    rf = fut.catch(tostring)
    fut.set_exception(e)
    assert await rf == repr(e)

    async def f(x):
        await asyncio.sleep(0.1)
        return x + 1
    fut = PyodideFuture()
    rf = fut.then(f)
    fut.set_result(6)
    assert await rf == 7
    fut = PyodideFuture()
    rf = fut.then(raises)
    fut.set_result(6)
    try:
        await rf
    except Exception:
        pass
    assert repr(rf.exception()) == repr(Exception(6))
    x = 0

    def incx():
        if False:
            for i in range(10):
                print('nop')
        nonlocal x
        x += 1
    fut = PyodideFuture()
    rf = fut.then(increment).then(increment).finally_(incx).finally_(incx)
    assert x == 0
    fut.set_result(5)
    await rf
    assert x == 2
    fut = PyodideFuture()
    rf = fut.then(increment).then(increment).finally_(incx).finally_(incx)
    fut.set_exception(e)
    try:
        await rf
    except Exception:
        pass
    assert x == 4

    async def f1(x):
        if x == 0:
            return 7
        await asyncio.sleep(0.1)
        return f1(x - 1)
    fut = PyodideFuture()
    rf = fut.then(f1)
    fut.set_result(3)
    assert await rf == 7

    async def f2():
        await asyncio.sleep(0.1)
        raise e
    fut = PyodideFuture()
    rf = fut.finally_(f2)
    fut.set_result(3)
    try:
        await rf
    except Exception:
        pass
    assert rf.exception() == e
    fut = PyodideFuture()
    rf = fut.finally_(f2)
    fut.set_exception(Exception('oops!'))
    try:
        await rf
    except Exception:
        pass
    assert rf.exception() == e

@run_in_pyodide
async def test_pyodide_future2(selenium):
    from js import fetch
    from pyodide.ffi import JsFetchResponse, JsProxy

    async def get_json(x: JsFetchResponse) -> JsProxy:
        return await x.json()

    def get_name(x: JsProxy) -> str:
        if False:
            while True:
                i = 10
        return x.info.name
    url = 'https://pypi.org/pypi/pytest/json'
    b = fetch(url).then(get_json)
    name = await b.then(get_name)
    assert name == 'pytest'

@run_in_pyodide
async def test_pyodide_task(selenium):
    from asyncio import Future, ensure_future, sleep

    async def taskify(fut):
        return await fut

    def do_the_thing():
        if False:
            while True:
                i = 10
        d = dict(did_onresolve=None, did_onreject=None, did_onfinally=False)
        f: Future[int] = Future()
        t = ensure_future(taskify(f))
        t.then(lambda v: d.update(did_onresolve=v), lambda e: d.update(did_onreject=e)).finally_(lambda : d.update(did_onfinally=True))
        return (f, d)
    (f, d) = do_the_thing()
    f.set_result(7)
    await sleep(0.1)
    assert d == dict(did_onresolve=7, did_onreject=None, did_onfinally=True)
    (f, d) = do_the_thing()
    e = Exception('Oops!')
    f.set_exception(e)
    assert d == dict(did_onresolve=None, did_onreject=None, did_onfinally=False)
    await sleep(0.1)
    assert d == dict(did_onresolve=None, did_onreject=e, did_onfinally=True)

@run_in_pyodide
async def test_inprogress(selenium):
    import asyncio
    from pyodide.webloop import WebLoop
    loop: WebLoop = asyncio.get_event_loop()
    loop._in_progress = 0
    ran_no_in_progress_handler = False

    def _no_in_progress_handler():
        if False:
            while True:
                i = 10
        nonlocal ran_no_in_progress_handler
        ran_no_in_progress_handler = True
    ran_keyboard_interrupt_handler = False

    def _keyboard_interrupt_handler():
        if False:
            while True:
                i = 10
        print('_keyboard_interrupt_handler')
        nonlocal ran_keyboard_interrupt_handler
        ran_keyboard_interrupt_handler = True
    system_exit_code = None

    def _system_exit_handler(exit_code):
        if False:
            while True:
                i = 10
        nonlocal system_exit_code
        system_exit_code = exit_code
    try:
        loop._no_in_progress_handler = _no_in_progress_handler
        loop._keyboard_interrupt_handler = _keyboard_interrupt_handler
        loop._system_exit_handler = _system_exit_handler
        fut = loop.create_future()

        async def temp():
            await fut
        fut2 = asyncio.ensure_future(temp())
        await asyncio.sleep(0)
        assert loop._in_progress == 2
        fut.set_result(0)
        await fut2
        assert loop._in_progress == 0
        assert ran_no_in_progress_handler
        assert not ran_keyboard_interrupt_handler
        assert not system_exit_code
        ran_no_in_progress_handler = False
        fut = loop.create_future()

        async def temp():
            await fut
        fut2 = asyncio.ensure_future(temp())
        assert loop._in_progress == 2
        fut.set_exception(KeyboardInterrupt())
        try:
            await fut2
        except KeyboardInterrupt:
            pass
        assert loop._in_progress == 0
        assert ran_no_in_progress_handler
        assert ran_keyboard_interrupt_handler
        assert not system_exit_code
        ran_no_in_progress_handler = False
        ran_keyboard_interrupt_handler = False
        fut = loop.create_future()

        async def temp():
            await fut
        fut2 = asyncio.ensure_future(temp())
        assert loop._in_progress == 2
        fut.set_exception(SystemExit(2))
        try:
            await fut2
        except SystemExit:
            pass
        assert loop._in_progress == 0
        assert ran_no_in_progress_handler
        assert not ran_keyboard_interrupt_handler
        assert system_exit_code == 2
        ran_no_in_progress_handler = False
        system_exit_code = None
    finally:
        loop._in_progress = 1
        loop._no_in_progress_handler = None
        loop._keyboard_interrupt_handler = None
        loop._system_exit_handler = None