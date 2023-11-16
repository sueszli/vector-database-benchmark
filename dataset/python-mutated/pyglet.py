"""Enable pyglet to be used interactively with prompt_toolkit
"""
import sys
import time
from timeit import default_timer as clock
import pyglet
if sys.platform.startswith('linux'):

    def flip(window):
        if False:
            return 10
        try:
            window.flip()
        except AttributeError:
            pass
else:

    def flip(window):
        if False:
            while True:
                i = 10
        window.flip()

def inputhook(context):
    if False:
        i = 10
        return i + 15
    'Run the pyglet event loop by processing pending events only.\n\n    This keeps processing pending events until stdin is ready.  After\n    processing all pending events, a call to time.sleep is inserted.  This is\n    needed, otherwise, CPU usage is at 100%.  This sleep time should be tuned\n    though for best performance.\n    '
    try:
        t = clock()
        while not context.input_is_ready():
            pyglet.clock.tick()
            for window in pyglet.app.windows:
                window.switch_to()
                window.dispatch_events()
                window.dispatch_event('on_draw')
                flip(window)
            used_time = clock() - t
            if used_time > 10.0:
                time.sleep(1.0)
            elif used_time > 0.1:
                time.sleep(0.05)
            else:
                time.sleep(0.001)
    except KeyboardInterrupt:
        pass