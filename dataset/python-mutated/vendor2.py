from __future__ import annotations
DOCUMENTATION = '\nname: vendor2\nshort_description: lookup\ndescription: Lookup.\nauthor:\n  - Ansible Core Team\n'
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
            for i in range(10):
                print('nop')
        self.set_options(var_options=variables, direct=kwargs)
        return terms