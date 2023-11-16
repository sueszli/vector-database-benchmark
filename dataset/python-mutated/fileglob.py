from __future__ import annotations
DOCUMENTATION = '\n    name: fileglob\n    author: Michael DeHaan\n    version_added: "1.4"\n    short_description: list files matching a pattern\n    description:\n        - Matches all files in a single directory, non-recursively, that match a pattern.\n          It calls Python\'s "glob" library.\n    options:\n      _terms:\n        description: path(s) of files to read\n        required: True\n    notes:\n      - Patterns are only supported on files, not directory/paths.\n      - See R(Ansible task paths,playbook_task_paths) to understand how file lookup occurs with paths.\n      - Matching is against local system files on the Ansible controller.\n        To iterate a list of files on a remote node, use the M(ansible.builtin.find) module.\n      - Returns a string list of paths joined by commas, or an empty list if no files match. For a \'true list\' pass O(ignore:wantlist=True) to the lookup.\n    seealso:\n      - ref: playbook_task_paths\n        description: Search paths used for relative files.\n'
EXAMPLES = '\n- name: Display paths of all .txt files in dir\n  ansible.builtin.debug: msg={{ lookup(\'ansible.builtin.fileglob\', \'/my/path/*.txt\') }}\n\n- name: Copy each file over that matches the given pattern\n  ansible.builtin.copy:\n    src: "{{ item }}"\n    dest: "/etc/fooapp/"\n    owner: "root"\n    mode: 0600\n  with_fileglob:\n    - "/playbooks/files/fooapp/*"\n'
RETURN = '\n  _list:\n    description:\n      - list of files\n    type: list\n    elements: path\n'
import os
import glob
from ansible.plugins.lookup import LookupBase
from ansible.module_utils.common.text.converters import to_bytes, to_text

class LookupModule(LookupBase):

    def run(self, terms, variables=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ret = []
        for term in terms:
            term_file = os.path.basename(term)
            found_paths = []
            if term_file != term:
                found_paths.append(self.find_file_in_search_path(variables, 'files', os.path.dirname(term)))
            else:
                if 'ansible_search_path' in variables:
                    paths = variables['ansible_search_path']
                else:
                    paths = [self.get_basedir(variables)]
                for p in paths:
                    found_paths.append(os.path.join(p, 'files'))
                    found_paths.append(p)
            for dwimmed_path in found_paths:
                if dwimmed_path:
                    globbed = glob.glob(to_bytes(os.path.join(dwimmed_path, term_file), errors='surrogate_or_strict'))
                    term_results = [to_text(g, errors='surrogate_or_strict') for g in globbed if os.path.isfile(g)]
                    if term_results:
                        ret.extend(term_results)
                        break
        return ret