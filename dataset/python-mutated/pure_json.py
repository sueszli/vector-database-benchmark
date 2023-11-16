from __future__ import annotations
DOCUMENTATION = '\n    name: pure_json\n    type: stdout\n    short_description: only outputs the module results as json\n'
import json
from ansible.plugins.callback import CallbackBase

class CallbackModule(CallbackBase):
    CALLBACK_VERSION = 2.0
    CALLBACK_TYPE = 'stdout'
    CALLBACK_NAME = 'pure_json'

    def v2_runner_on_failed(self, result, ignore_errors=False):
        if False:
            while True:
                i = 10
        self._display.display(json.dumps(result._result))

    def v2_runner_on_ok(self, result):
        if False:
            i = 10
            return i + 15
        self._display.display(json.dumps(result._result))

    def v2_runner_on_skipped(self, result):
        if False:
            while True:
                i = 10
        self._display.display(json.dumps(result._result))