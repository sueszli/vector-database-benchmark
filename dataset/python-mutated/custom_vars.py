from __future__ import annotations
DOCUMENTATION = '\n    vars: custom_vars\n    version_added: "2.10"\n    short_description: load host and group vars\n    description: test loading host and group vars from a collection\n    options:\n      stage:\n        choices: [\'all\', \'inventory\', \'task\']\n        type: str\n        ini:\n          - key: stage\n            section: custom_vars\n        env:\n          - name: ANSIBLE_VARS_PLUGIN_STAGE\n'
from ansible.plugins.vars import BaseVarsPlugin

class VarsModule(BaseVarsPlugin):
    REQUIRES_ENABLED = False

    def get_vars(self, loader, path, entities, cache=True):
        if False:
            i = 10
            return i + 15
        super(VarsModule, self).get_vars(loader, path, entities)
        return {'collection': 'collection_root_user'}