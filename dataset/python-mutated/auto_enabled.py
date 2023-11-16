from __future__ import annotations
from ansible.plugins.vars import BaseVarsPlugin

class VarsModule(BaseVarsPlugin):
    REQUIRES_ENABLED = False

    def get_vars(self, loader, path, entities):
        if False:
            while True:
                i = 10
        return {'explicitly_auto_enabled': True}