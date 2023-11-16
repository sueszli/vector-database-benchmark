"""
if setproctitle is installed.
set Unix thread name with the Python thread name
"""
try:
    import setproctitle
except ImportError:
    pass
else:
    import threading
    old_thread_init = threading.Thread.__init__

    def new_thread_init(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        old_thread_init(self, *args, **kwargs)
        setproctitle.setthreadtitle(self._name)
    threading.Thread.__init__ = new_thread_init