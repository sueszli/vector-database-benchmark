from __future__ import annotations
DOCUMENTATION = '\n    name: unvault\n    author: Ansible Core Team\n    version_added: "2.10"\n    short_description: read vaulted file(s) contents\n    description:\n        - This lookup returns the contents from vaulted (or not) file(s) on the Ansible controller\'s file system.\n    options:\n      _terms:\n        description: path(s) of files to read\n        required: True\n    notes:\n      - This lookup does not understand \'globbing\' nor shell environment variables.\n    seealso:\n      - ref: playbook_task_paths\n        description: Search paths used for relative files.\n'
EXAMPLES = '\n- ansible.builtin.debug: msg="the value of foo.txt is {{ lookup(\'ansible.builtin.unvault\', \'/etc/foo.txt\') | string | trim }}"\n'
RETURN = '\n  _raw:\n    description:\n      - content of file(s) as bytes\n    type: list\n    elements: raw\n'
from ansible.errors import AnsibleParserError
from ansible.plugins.lookup import LookupBase
from ansible.module_utils.common.text.converters import to_text
from ansible.utils.display import Display
display = Display()

class LookupModule(LookupBase):

    def run(self, terms, variables=None, **kwargs):
        if False:
            while True:
                i = 10
        ret = []
        self.set_options(var_options=variables, direct=kwargs)
        for term in terms:
            display.debug('Unvault lookup term: %s' % term)
            lookupfile = self.find_file_in_search_path(variables, 'files', term)
            display.vvvv(u'Unvault lookup found %s' % lookupfile)
            if lookupfile:
                actual_file = self._loader.get_real_file(lookupfile, decrypt=True)
                with open(actual_file, 'rb') as f:
                    b_contents = f.read()
                ret.append(to_text(b_contents))
            else:
                raise AnsibleParserError('Unable to find file matching "%s" ' % term)
        return ret