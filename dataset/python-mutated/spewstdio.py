from __future__ import annotations
import atexit
import os
import sys
from ansible.plugins.callback import CallbackBase
from ansible.utils.display import Display
from threading import Thread

class CallbackModule(CallbackBase):
    CALLBACK_VERSION = 2.0
    CALLBACK_NAME = 'spewstdio'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.display = Display()
        if os.environ.get('SPEWSTDIO_ENABLED', '0') != '1':
            self.display.warning('spewstdio test plugin loaded but disabled; set SPEWSTDIO_ENABLED=1 to enable')
            return
        self.display = Display()
        self._keep_spewing = True
        os.register_at_fork(after_in_child=lambda : print(f'hi from forked child pid {os.getpid()}'))
        atexit.register(self.stop_spew)
        self._spew_thread = Thread(target=self.spew, daemon=True)
        self._spew_thread.start()

    def stop_spew(self):
        if False:
            return 10
        self._keep_spewing = False

    def spew(self):
        if False:
            for i in range(10):
                print('nop')
        self.display.warning('spewstdio STARTING NONPRINTING SPEW ON BACKGROUND THREAD')
        while self._keep_spewing:
            sys.stdout.write('\x1b[K')
            sys.stdout.flush()
        self.display.warning('spewstdio STOPPING SPEW')