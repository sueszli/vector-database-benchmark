from __future__ import annotations
DOCUMENTATION = '\nname: vendor1\nshort_description: lookup\ndescription: Lookup.\nauthor:\n  - Ansible Core Team\n'
EXAMPLES = '#'
RETURN = '#'
from ansible.plugins.lookup import LookupBase
from ansible.plugins import loader
try:
    import demo
except ImportError:
    pass
else:
    raise Exception('demo import found when it should not be')

class LookupModule(LookupBase):

    def run(self, terms, variables, **kwargs):
        if False:
            i = 10
            return i + 15
        self.set_options(var_options=variables, direct=kwargs)
        return terms