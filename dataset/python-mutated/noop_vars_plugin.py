from __future__ import annotations
DOCUMENTATION = "\n    vars: noop_vars_plugin\n    short_description: Do NOT load host and group vars\n    description: don't test loading host and group vars from a collection\n    options:\n      stage:\n        default: all\n        choices: ['all', 'inventory', 'task']\n        type: str\n        ini:\n          - key: stage\n            section: testns.testcol.noop_vars_plugin\n        env:\n          - name: ANSIBLE_VARS_PLUGIN_STAGE\n    extends_documentation_fragment:\n        - testns.testcol2.deprecation\n"
from ansible.plugins.vars import BaseVarsPlugin

class VarsModule(BaseVarsPlugin):

    def get_vars(self, loader, path, entities, cache=True):
        if False:
            print('Hello World!')
        super(VarsModule, self).get_vars(loader, path, entities)
        return {'collection': 'yes', 'notreal': 'value'}