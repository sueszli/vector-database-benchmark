import numpy as np
from vispy import gloo
from vispy.app import Application, Canvas
from vispy.app.backends import _jupyter_rfb
from vispy.testing import run_tests_if_main, requires_application
import pytest
try:
    import jupyter_rfb
except ImportError:
    jupyter_rfb = None

def test_rfb_app():
    if False:
        while True:
            i = 10
    app_backend = _jupyter_rfb.ApplicationBackend()
    app_backend._vispy_run()
    app_backend._vispy_quit()

class MyCanvas(Canvas):

    def on_draw(self, event):
        if False:
            i = 10
            return i + 15
        gloo.set_clear_color((0, 1, 0))
        gloo.clear()

@pytest.mark.skipif(jupyter_rfb is None, reason='jupyter_rfb is not installed')
@requires_application()
def test_rfb_canvas():
    if False:
        for i in range(10):
            print('nop')
    app = Application('jupyter_rfb')
    canvas = MyCanvas(app=app)
    canvas_backend = canvas.native
    assert isinstance(canvas_backend, _jupyter_rfb.CanvasBackend)
    assert '42' not in canvas_backend.css_width
    canvas.size = (42, 42)
    assert canvas_backend.css_width == '42px'
    canvas_backend.handle_event({'event_type': 'resize', 'width': 50, 'height': 50, 'pixel_ratio': 2.0})
    assert canvas.size == (50, 50)
    assert canvas.physical_size == (100, 100)
    frame = canvas_backend.get_frame()
    assert frame.shape[:2] == (100, 100)
    assert np.all(frame[:, :, 0] == 0)
    assert np.all(frame[:, :, 1] == 255)
    canvas_backend.handle_event({'event_type': 'resize', 'width': 60, 'height': 60, 'pixel_ratio': 1.0})
    assert canvas.size == (60, 60)
    assert canvas.physical_size == (60, 60)
    frame = canvas_backend.get_frame()
    assert frame.shape[:2] == (60, 60)
    assert np.all(frame[:, :, 0] == 0)
    assert np.all(frame[:, :, 1] == 255)
    events = []
    canvas.events.mouse_press.connect(lambda e: events.append(e))
    canvas_backend.handle_event({'event_type': 'pointer_down', 'x': 11, 'y': 12, 'button': 1, 'modifiers': []})
    assert len(events) == 1
    assert tuple(events[0].pos) == (11, 12)
run_tests_if_main()