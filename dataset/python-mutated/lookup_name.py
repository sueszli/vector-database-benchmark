from __future__ import annotations
from ansible.plugins.lookup import LookupBase

class LookupModule(LookupBase):

    def run(self, terms, variables, **kwargs):
        if False:
            i = 10
            return i + 15
        return [__name__]