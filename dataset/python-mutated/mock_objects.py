from asciimatics.effects import Effect
from asciimatics.exceptions import StopApplication, NextScene

class MockEffect(Effect):
    """
    Dummy Effect use for some UTs.
    """

    def __init__(self, count=10, stop=True, swallow=False, next_scene=None, frame_rate=1, stop_frame=5, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param count: When to stop effect\n        :param stop: Whether to stop the application or skip to next scene.\n        :param swallow: Whether to swallow any events or not.\n        :param next_scene: The next scene to move to (if stop=False)\n        :param frame_rate: The frame rate for updates.\n        '
        super().__init__(None, **kwargs)
        self.stop_called = False
        self.reset_called = False
        self.event_called = False
        self.save_called = False
        self.update_called = False
        self._count = count
        self._stop = stop
        self._swallow = swallow
        self._next_scene = next_scene
        self._frame_rate = frame_rate
        self._my_stop_frame = stop_frame

    @property
    def stop_frame(self):
        if False:
            return 10
        self.stop_called = True
        return self._my_stop_frame

    @property
    def frame_update_count(self):
        if False:
            for i in range(10):
                print('nop')
        return self._frame_rate

    def _update(self, frame_no):
        if False:
            i = 10
            return i + 15
        self.update_called = True
        self._count -= 1
        if self._count <= 0:
            if self._stop:
                raise StopApplication('End of test')
            else:
                raise NextScene(self._next_scene)

    def reset(self):
        if False:
            while True:
                i = 10
        self.reset_called = True

    def process_event(self, event):
        if False:
            print('Hello World!')
        self.event_called = True
        return None if self._swallow else event

    def save(self):
        if False:
            i = 10
            return i + 15
        self.save_called = True