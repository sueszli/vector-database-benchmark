from __future__ import annotations
from ansible.plugins.action.normal import ActionModule as NormalAction

class ActionModule(NormalAction):

    def run(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        result = super(ActionModule, self).run(*args, **kwargs)
        result['hacked'] = 'I got run under a subclassed normal, yay'
        return result