from __future__ import annotations
from ansible.plugins.lookup import LookupBase

class LookupModule(LookupBase):

    def run(self, terms, variables, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return ['mylookup_from_user_dir']