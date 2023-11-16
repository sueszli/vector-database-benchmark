from __future__ import annotations
DOCUMENTATION = '\n    name: varnames\n    author: Ansible Core Team\n    version_added: "2.8"\n    short_description: Lookup matching variable names\n    description:\n      - Retrieves a list of matching Ansible variable names.\n    options:\n      _terms:\n        description: List of Python regex patterns to search for in variable names.\n        required: True\n'
EXAMPLES = '\n- name: List variables that start with qz_\n  ansible.builtin.debug: msg="{{ lookup(\'ansible.builtin.varnames\', \'^qz_.+\')}}"\n  vars:\n    qz_1: hello\n    qz_2: world\n    qa_1: "I won\'t show"\n    qz_: "I won\'t show either"\n\n- name: Show all variables\n  ansible.builtin.debug: msg="{{ lookup(\'ansible.builtin.varnames\', \'.+\')}}"\n\n- name: Show variables with \'hosts\' in their names\n  ansible.builtin.debug: msg="{{ lookup(\'ansible.builtin.varnames\', \'hosts\')}}"\n\n- name: Find several related variables that end specific way\n  ansible.builtin.debug: msg="{{ lookup(\'ansible.builtin.varnames\', \'.+_zone$\', \'.+_location$\') }}"\n\n'
RETURN = '\n_value:\n  description:\n    - List of the variable names requested.\n  type: list\n'
import re
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible.plugins.lookup import LookupBase

class LookupModule(LookupBase):

    def run(self, terms, variables=None, **kwargs):
        if False:
            i = 10
            return i + 15
        if variables is None:
            raise AnsibleError('No variables available to search')
        self.set_options(var_options=variables, direct=kwargs)
        ret = []
        variable_names = list(variables.keys())
        for term in terms:
            if not isinstance(term, string_types):
                raise AnsibleError('Invalid setting identifier, "%s" is not a string, it is a %s' % (term, type(term)))
            try:
                name = re.compile(term)
            except Exception as e:
                raise AnsibleError('Unable to use "%s" as a search parameter: %s' % (term, to_native(e)))
            for varname in variable_names:
                if name.search(varname):
                    ret.append(varname)
        return ret