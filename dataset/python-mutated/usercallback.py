from __future__ import annotations
from ansible.plugins.callback import CallbackBase
DOCUMENTATION = '\n    callback: usercallback\n    callback_type: notification\n    short_description: does stuff\n    description:\n      - does some stuff\n'

class CallbackModule(CallbackBase):
    CALLBACK_VERSION = 2.0
    CALLBACK_TYPE = 'aggregate'
    CALLBACK_NAME = 'usercallback'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(CallbackModule, self).__init__()
        self._display.display('loaded usercallback from collection, yay')

    def v2_runner_on_ok(self, result):
        if False:
            for i in range(10):
                print('nop')
        self._display.display('usercallback says ok')