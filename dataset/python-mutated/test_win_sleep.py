import asyncio
import time
import sys
import pytest
from textual.app import App
pytestmark = pytest.mark.skipif(sys.platform != 'win32', reason='We only need to test this on Windows.')

def test_win_sleep_timer_is_cancellable():
    if False:
        while True:
            i = 10
    'Regression test for https://github.com/Textualize/textual/issues/2711.\n\n    When we exit an app with a "long" timer, everything asyncio-related\n    should shutdown quickly. So, we create an app with a timer that triggers\n    every SLEEP_FOR seconds and we shut the app down immediately after creating\n    it. `asyncio` should be done quickly (i.e., the timer was cancelled) and\n    thus the total time this takes should be considerably lesser than the time\n    we originally set the timer for.\n    '
    SLEEP_FOR = 10

    class WindowsIntervalBugApp(App[None]):

        def on_mount(self) -> None:
            if False:
                return 10
            self.set_interval(SLEEP_FOR, lambda : None)

        def key_e(self):
            if False:
                i = 10
                return i + 15
            self.exit()

    async def actual_test():
        async with WindowsIntervalBugApp().run_test() as pilot:
            await pilot.press('e')
    start = time.perf_counter()
    asyncio.run(actual_test())
    end = time.perf_counter()
    assert end - start < 1