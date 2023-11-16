from __future__ import annotations
DOCUMENTATION = '\n    name: bogus\n    author: Ansible Core Team\n    version_added: histerical\n    short_description: returns what you gave it\n    description:\n      - this is mostly a noop\n    options:\n        _terms:\n            description: stuff to pass through\n        test_list:\n            description: does nothihng, just for testing values\n            type: list\n            choices:\n                - Dan\n                - Yevgeni\n                - Carla\n                - Manuela\n'
EXAMPLES = "\n- name: like some other plugins, this is mostly useless\n  debug: msg={{ q('bogus', [1,2,3])}}\n"
RETURN = '\n  _list:\n    description: basically the same as you fed in\n    type: list\n    elements: raw\n'
from ansible.plugins.lookup import LookupBase

class LookupModule(LookupBase):

    def run(self, terms, variables=None, **kwargs):
        if False:
            while True:
                i = 10
        self.set_options(var_options=variables, direct=kwargs)
        dump = self.get_option('test_list')
        return terms