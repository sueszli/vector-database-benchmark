from __future__ import annotations
DOCUMENTATION = '\n    name: display_resolved_action\n    type: aggregate\n    short_description: Displays the requested and resolved actions at the end of a playbook.\n    description:\n        - Displays the requested and resolved actions in the format "requested == resolved".\n    requirements:\n      - Enable in configuration.\n'
from ansible.plugins.callback import CallbackBase

class CallbackModule(CallbackBase):
    CALLBACK_VERSION = 2.0
    CALLBACK_TYPE = 'aggregate'
    CALLBACK_NAME = 'display_resolved_action'
    CALLBACK_NEEDS_ENABLED = True

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(CallbackModule, self).__init__(*args, **kwargs)
        self.requested_to_resolved = {}

    def v2_playbook_on_task_start(self, task, is_conditional):
        if False:
            print('Hello World!')
        self.requested_to_resolved[task.action] = task.resolved_action

    def v2_playbook_on_stats(self, stats):
        if False:
            return 10
        for (requested, resolved) in self.requested_to_resolved.items():
            self._display.display('%s == %s' % (requested, resolved), screen_only=True)