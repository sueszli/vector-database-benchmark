from __future__ import annotations
DOCUMENTATION = '\n    name: lookup2\n    author: Ansible Core Team\n    short_description: hello test lookup\n    description:\n        - Hello test lookup.\n    options: {}\n'
EXAMPLES = '\n- minimal:\n'
RETURN = '\n'
from ansible.plugins.lookup import LookupBase

class LookupModule(LookupBase):

    def run(self, terms, variables=None, **kwargs):
        if False:
            while True:
                i = 10
        return []