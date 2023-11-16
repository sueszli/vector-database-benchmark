from __future__ import annotations
DOCUMENTATION = '\n    vars: v1_vars_plugin\n    version_added: "2.10"\n    short_description: load host and group vars\n    description:\n      - 3rd party vars plugin to test loading host and group vars without requiring whitelisting and without a plugin-specific stage option\n    options:\n'
from ansible.plugins.vars import BaseVarsPlugin

class VarsModule(BaseVarsPlugin):

    def get_vars(self, loader, path, entities, cache=True):
        if False:
            return 10
        super(VarsModule, self).get_vars(loader, path, entities)
        return {'collection': False, 'name': 'v1_vars_plugin', 'v1_vars_plugin': True}