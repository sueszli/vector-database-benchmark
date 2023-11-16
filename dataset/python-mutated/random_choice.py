from __future__ import annotations
DOCUMENTATION = '\n    name: random_choice\n    author: Michael DeHaan\n    version_added: "1.1"\n    short_description: return random element from list\n    description:\n      - The \'random_choice\' feature can be used to pick something at random. While it\'s not a load balancer (there are modules for those),\n        it can somewhat be used as a poor man\'s load balancer in a MacGyver like situation.\n      - At a more basic level, they can be used to add chaos and excitement to otherwise predictable automation environments.\n'
EXAMPLES = '\n- name: Magic 8 ball for MUDs\n  ansible.builtin.debug:\n    msg: "{{ item }}"\n  with_random_choice:\n     - "go through the door"\n     - "drink from the goblet"\n     - "press the red button"\n     - "do nothing"\n'
RETURN = '\n  _raw:\n    description:\n      - random item\n    type: raw\n'
import random
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native
from ansible.plugins.lookup import LookupBase

class LookupModule(LookupBase):

    def run(self, terms, variables=None, **kwargs):
        if False:
            print('Hello World!')
        ret = terms
        if terms:
            try:
                ret = [random.choice(terms)]
            except Exception as e:
                raise AnsibleError('Unable to choose random term: %s' % to_native(e))
        return ret