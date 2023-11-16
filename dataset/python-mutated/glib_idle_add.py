from functools import wraps
from gi.repository import GLib

def glib_idle_add(fn):
    if False:
        while True:
            i = 10
    '\n    Schedules fn to run in the main GTK loop\n    '

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if False:
            while True:
                i = 10
        GLib.idle_add(fn, *args, **kwargs)
    wrapper.original = fn
    return wrapper