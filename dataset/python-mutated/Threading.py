import functools
import threading
from cura.CuraApplication import CuraApplication

class InterCallObject:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.finish_event = threading.Event()
        self.result = None

def call_on_qt_thread(func):
    if False:
        return 10

    @functools.wraps(func)
    def _call_on_qt_thread_wrapper(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        current_thread = threading.current_thread()
        if isinstance(current_thread, threading._MainThread):
            return func(*args, **kwargs)

        def _handle_call(ico, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            ico.result = func(*args, **kwargs)
            ico.finish_event.set()
        inter_call_object = InterCallObject()
        new_args = tuple([inter_call_object] + list(args)[:])
        CuraApplication.getInstance().callLater(_handle_call, *new_args, **kwargs)
        inter_call_object.finish_event.wait()
        return inter_call_object.result
    return _call_on_qt_thread_wrapper