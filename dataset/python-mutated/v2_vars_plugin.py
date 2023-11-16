from __future__ import annotations
DOCUMENTATION = '\n    vars: v2_vars_plugin\n    version_added: "2.10"\n    short_description: load host and group vars\n    description:\n      - 3rd party vars plugin to test loading host and group vars without requiring whitelisting and with a plugin-specific stage option\n    options:\n      stage:\n        choices: [\'all\', \'inventory\', \'task\']\n        type: str\n        ini:\n          - key: stage\n            section: other_vars_plugin\n        env:\n          - name: ANSIBLE_VARS_PLUGIN_STAGE\n'
from ansible.plugins.vars import BaseVarsPlugin

class VarsModule(BaseVarsPlugin):

    def get_vars(self, loader, path, entities, cache=True):
        if False:
            for i in range(10):
                print('nop')
        super(VarsModule, self).get_vars(loader, path, entities)
        return {'collection': False, 'name': 'v2_vars_plugin', 'v2_vars_plugin': True}