import time
import uiautomator2 as u2

def test_swipe_duration(d: u2.Device):
    if False:
        while True:
            i = 10
    (w, h) = d.window_size()
    start = time.time()
    d.debug = True
    d.swipe(w // 2, h // 2, w - 1, h // 2, 2.0)
    duration = time.time() - start
    assert duration >= 1.5