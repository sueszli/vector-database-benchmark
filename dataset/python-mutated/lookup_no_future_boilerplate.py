from __future__ import annotations
from ansible.plugins.lookup import LookupBase

class LookupModule(LookupBase):

    def run(self, terms, variables, **kwargs):
        if False:
            print('Hello World!')
        return [__name__]