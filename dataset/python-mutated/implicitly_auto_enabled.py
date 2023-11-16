from __future__ import annotations
from ansible.plugins.vars import BaseVarsPlugin

class VarsModule(BaseVarsPlugin):

    def get_vars(self, loader, path, entities):
        if False:
            for i in range(10):
                print('nop')
        return {'implicitly_auto_enabled': True}