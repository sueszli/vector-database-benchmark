from __future__ import annotations
from ansible.plugins.callback import CallbackBase
import os

class CallbackModule(CallbackBase):
    CALLBACK_VERSION = 2.0
    CALLBACK_TYPE = 'stdout'
    CALLBACK_NAME = 'callback_meta'

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(CallbackModule, self).__init__(*args, **kwargs)
        self.wants_implicit_tasks = os.environ.get('CB_WANTS_IMPLICIT', False)

    def v2_playbook_on_task_start(self, task, is_conditional):
        if False:
            for i in range(10):
                print('nop')
        if task.implicit:
            self._display.display('saw implicit task')
        self._display.display(task.get_name())