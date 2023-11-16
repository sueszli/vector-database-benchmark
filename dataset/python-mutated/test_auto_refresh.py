import asyncio
from time import time
from textual.app import App
from textual.pilot import Pilot

class RefreshApp(App[float]):

    def __init__(self):
        if False:
            return 10
        self.count = 0
        super().__init__()

    def on_mount(self):
        if False:
            while True:
                i = 10
        self.start = time()
        self.auto_refresh = 0.1

    def _automatic_refresh(self):
        if False:
            print('Hello World!')
        self.count += 1
        if self.count == 3:
            self.exit(time() - self.start)
        super()._automatic_refresh()

def test_auto_refresh():
    if False:
        i = 10
        return i + 15
    app = RefreshApp()

    async def quit_after(pilot: Pilot) -> None:
        await asyncio.sleep(1)
    elapsed = app.run(auto_pilot=quit_after, headless=True)
    assert elapsed is not None
    assert 0.2 <= elapsed < 0.8