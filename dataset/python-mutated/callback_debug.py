from __future__ import annotations
from ansible.plugins.callback import CallbackBase

class CallbackModule(CallbackBase):
    CALLBACK_VERSION = 2.0
    CALLBACK_TYPE = 'stdout'
    CALLBACK_NAME = 'callback_debug'

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(CallbackModule, self).__init__(*args, **kwargs)
        self._display.display('__init__')
        for cb in [x for x in dir(CallbackBase) if x.startswith('v2_')]:
            delattr(CallbackBase, cb)

    def __getattr__(self, name):
        if False:
            while True:
                i = 10
        if name.startswith('v2_'):
            return lambda *args, **kwargs: self._display.display(name)