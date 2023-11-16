from __future__ import annotations
import pkg_resources
from ansible.plugins.lookup import LookupBase

class LookupModule(LookupBase):

    def run(self, terms, variables, **kwargs):
        if False:
            i = 10
            return i + 15
        return ['ok']