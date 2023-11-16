"""
prompt_toolkit input hook for GTK 4.
"""
from gi.repository import GLib

class _InputHook:

    def __init__(self, context):
        if False:
            i = 10
            return i + 15
        self._quit = False
        GLib.io_add_watch(context.fileno(), GLib.PRIORITY_DEFAULT, GLib.IO_IN, self.quit)

    def quit(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self._quit = True
        return False

    def run(self):
        if False:
            i = 10
            return i + 15
        context = GLib.MainContext.default()
        while not self._quit:
            context.iteration(True)

def inputhook(context):
    if False:
        i = 10
        return i + 15
    hook = _InputHook(context)
    hook.run()