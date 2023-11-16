from __future__ import annotations
from ansible.plugins.lookup import LookupBase
DOCUMENTATION = '\nname: import_order_lookup\nshort_description: Import order lookup\ndescription: Import order lookup.\n'

class LookupModule(LookupBase):

    def run(self, terms, variables=None, **kwargs):
        if False:
            print('Hello World!')
        return []