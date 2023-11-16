import pytest
from .support import PyScriptTest, filter_inner_text, only_main

class TestAsync(PyScriptTest):
    coroutine_script = '\n        <script type="py">\n        import js\n        import asyncio\n        js.console.log("first")\n        async def main():\n            await asyncio.sleep(1)\n            js.console.log("third")\n        asyncio.{func}(main())\n        js.console.log("second")\n        </script>\n        '

    def test_asyncio_ensure_future(self):
        if False:
            for i in range(10):
                print('nop')
        self.pyscript_run(self.coroutine_script.format(func='ensure_future'))
        self.wait_for_console('third')
        assert self.console.log.lines[-3:] == ['first', 'second', 'third']

    def test_asyncio_create_task(self):
        if False:
            print('Hello World!')
        self.pyscript_run(self.coroutine_script.format(func='create_task'))
        self.wait_for_console('third')
        assert self.console.log.lines[-3:] == ['first', 'second', 'third']

    def test_asyncio_gather(self):
        if False:
            i = 10
            return i + 15
        self.pyscript_run('\n            <script type="py" id="pys">\n            import asyncio\n            import js\n            from pyodide.ffi import to_js\n\n            async def coro(delay):\n                await asyncio.sleep(delay)\n                return(delay)\n\n            async def get_results():\n                results = await asyncio.gather(*[coro(d) for d in range(3,0,-1)])\n                js.console.log(str(results)) #Compare to string representation, not Proxy\n                js.console.log("DONE")\n\n            asyncio.ensure_future(get_results())\n            </script>\n            ')
        self.wait_for_console('DONE')
        assert self.console.log.lines[-2:] == ['[3, 2, 1]', 'DONE']

    @only_main
    def test_multiple_async(self):
        if False:
            i = 10
            return i + 15
        self.pyscript_run('\n        <script type="py">\n            import js\n            import asyncio\n            async def a_func():\n                for i in range(3):\n                    js.console.log(\'A\', i)\n                    await asyncio.sleep(0.1)\n            asyncio.ensure_future(a_func())\n        </script>\n\n        <script type="py">\n            import js\n            import asyncio\n            async def b_func():\n                for i in range(3):\n                    js.console.log(\'B\', i)\n                    await asyncio.sleep(0.1)\n                js.console.log(\'b func done\')\n            asyncio.ensure_future(b_func())\n        </script>\n        ')
        self.wait_for_console('b func done')
        assert self.console.log.lines == ['A 0', 'B 0', 'A 1', 'B 1', 'A 2', 'B 2', 'b func done']

    @only_main
    def test_multiple_async_multiple_display_targeted(self):
        if False:
            while True:
                i = 10
        self.pyscript_run('\n                <script type="py" id="pyA">\n                    from pyscript import display\n                    import js\n                    import asyncio\n\n                    async def a_func():\n                        for i in range(2):\n                            display(f\'A{i}\', target=\'pyA\', append=True)\n                            js.console.log("A", i)\n                            await asyncio.sleep(0.1)\n                    asyncio.ensure_future(a_func())\n\n                </script>\n\n                <script type="py" id="pyB">\n                    from pyscript import display\n                    import js\n                    import asyncio\n\n                    async def a_func():\n                        for i in range(2):\n                            display(f\'B{i}\', target=\'pyB\', append=True)\n                            js.console.log("B", i)\n                            await asyncio.sleep(0.1)\n                        js.console.log("B DONE")\n\n                    asyncio.ensure_future(a_func())\n                </script>\n            ')
        self.wait_for_console('B DONE')
        inner_text = self.page.inner_text('html')
        assert 'A0\nA1\nB0\nB1' in filter_inner_text(inner_text)

    def test_async_display_untargeted(self):
        if False:
            while True:
                i = 10
        self.pyscript_run('\n                <script type="py">\n                    from pyscript import display\n                    import asyncio\n                    import js\n\n                    async def a_func():\n                        display(\'A\')\n                        await asyncio.sleep(1)\n                        js.console.log("DONE")\n\n                    asyncio.ensure_future(a_func())\n                </script>\n            ')
        self.wait_for_console('DONE')
        assert self.page.locator('script-py').inner_text() == 'A'

    @only_main
    def test_sync_and_async_order(self):
        if False:
            print('Hello World!')
        '\n        The order of execution is defined as follows:\n          1. first, we execute all the script tags in order\n          2. then, we start all the tasks which were scheduled with create_task\n\n        Note that tasks are started *AFTER* all py-script tags have been\n        executed. That\'s why the console.log() inside mytask1 and mytask2 are\n        executed after e.g. js.console.log("6").\n        '
        src = '\n                <script type="py">\n                    import js\n                    js.console.log("1")\n                </script>\n\n                <script type="py">\n                    import asyncio\n                    import js\n\n                    async def mytask1():\n                        js.console.log("7")\n                        await asyncio.sleep(0)\n                        js.console.log("9")\n\n                    js.console.log("2")\n                    asyncio.create_task(mytask1())\n                    js.console.log("3")\n                </script>\n\n                <script type="py">\n                    import js\n                    js.console.log("4")\n                </script>\n\n                <script type="py">\n                    import asyncio\n                    import js\n\n                    async def mytask2():\n                        js.console.log("8")\n                        await asyncio.sleep(0)\n                        js.console.log("10")\n                        js.console.log("DONE")\n\n                    js.console.log("5")\n                    asyncio.create_task(mytask2())\n                    js.console.log("6")\n                </script>\n            '
        self.pyscript_run(src, wait_for_pyscript=False)
        self.wait_for_console('DONE')
        lines = self.console.log.lines[-11:]
        assert lines == ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'DONE']