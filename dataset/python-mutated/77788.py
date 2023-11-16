from __future__ import annotations
from ansible.plugins.lookup import LookupBase

class LookupModule(LookupBase):

    def run(self, terms, variables, **kwargs):
        if False:
            return 10
        return {'one': 1, 'two': 2}