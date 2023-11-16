from __future__ import annotations
DOCUMENTATION = '\n    name: indexed_items\n    author: Michael DeHaan\n    version_added: "1.3"\n    short_description: rewrites lists to return \'indexed items\'\n    description:\n      - use this lookup if you want to loop over an array and also get the numeric index of where you are in the array as you go\n      - any list given will be transformed with each resulting element having the it\'s previous position in item.0 and its value in item.1\n    options:\n      _terms:\n        description: list of items\n        required: True\n'
EXAMPLES = '\n- name: indexed loop demo\n  ansible.builtin.debug:\n    msg: "at array position {{ item.0 }} there is a value {{ item.1 }}"\n  with_indexed_items:\n    - "{{ some_list }}"\n'
RETURN = '\n  _raw:\n    description:\n      - list with each item.0 giving you the position and item.1 the value\n    type: list\n    elements: list\n'
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase

class LookupModule(LookupBase):

    def __init__(self, basedir=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.basedir = basedir

    def run(self, terms, variables, **kwargs):
        if False:
            while True:
                i = 10
        if not isinstance(terms, list):
            raise AnsibleError('with_indexed_items expects a list')
        items = self._flatten(terms)
        return list(zip(range(len(items)), items))