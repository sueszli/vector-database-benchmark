from __future__ import annotations
from ansible.plugins.vars import BaseVarsPlugin

class VarsModule(BaseVarsPlugin):
    REQUIRES_ENABLED = True

    def get_vars(self, loader, path, entities):
        if False:
            while True:
                i = 10
        return {'require_enabled': True}