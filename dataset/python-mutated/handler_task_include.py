from __future__ import annotations
from ansible.playbook.handler import Handler
from ansible.playbook.task_include import TaskInclude

class HandlerTaskInclude(Handler, TaskInclude):
    VALID_INCLUDE_KEYWORDS = TaskInclude.VALID_INCLUDE_KEYWORDS.union(('listen',))

    @staticmethod
    def load(data, block=None, role=None, task_include=None, variable_manager=None, loader=None):
        if False:
            while True:
                i = 10
        t = HandlerTaskInclude(block=block, role=role, task_include=task_include)
        handler = t.check_options(t.load_data(data, variable_manager=variable_manager, loader=loader), data)
        return handler