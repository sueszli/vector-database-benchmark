from __future__ import annotations
DOCUMENTATION = "\n    name: first_found\n    author: Seth Vidal (!UNKNOWN) <skvidal@fedoraproject.org>\n    version_added: historical\n    short_description: return first file found from list\n    description:\n      - This lookup checks a list of files and paths and returns the full path to the first combination found.\n      - As all lookups, when fed relative paths it will try use the current task's location first and go up the chain\n        to the containing locations of role / play / include and so on.\n      - The list of files has precedence over the paths searched.\n        For example, A task in a role has a 'file1' in the play's relative path, this will be used, 'file2' in role's relative path will not.\n      - Either a list of files O(_terms) or a key O(files) with a list of files is required for this plugin to operate.\n    notes:\n      - This lookup can be used in 'dual mode', either passing a list of file names or a dictionary that has O(files) and O(paths).\n    options:\n      _terms:\n        description: A list of file names.\n      files:\n        description: A list of file names.\n        type: list\n        elements: string\n        default: []\n      paths:\n        description: A list of paths in which to look for the files.\n        type: list\n        elements: string\n        default: []\n      skip:\n        type: boolean\n        default: False\n        description:\n          - When V(True), return an empty list when no files are matched.\n          - This is useful when used with C(with_first_found), as an empty list return to C(with_) calls\n            causes the calling task to be skipped.\n          - When used as a template via C(lookup) or C(query), setting O(skip=True) will *not* cause the task to skip.\n            Tasks must handle the empty list return from the template.\n          - When V(False) and C(lookup) or C(query) specifies O(ignore:errors='ignore') all errors (including no file found,\n            but potentially others) return an empty string or an empty list respectively.\n          - When V(True) and C(lookup) or C(query) specifies O(ignore:errors='ignore'), no file found will return an empty\n            list and other potential errors return an empty string or empty list depending on the template call\n            (in other words return values of C(lookup) vs C(query)).\n    seealso:\n      - ref: playbook_task_paths\n        description: Search paths used for relative paths/files.\n"
EXAMPLES = '\n- name: Set _found_file to the first existing file, raising an error if a file is not found\n  ansible.builtin.set_fact:\n    _found_file: "{{ lookup(\'ansible.builtin.first_found\', findme) }}"\n  vars:\n    findme:\n      - /path/to/foo.txt\n      - bar.txt  # will be looked in files/ dir relative to role and/or play\n      - /path/to/biz.txt\n\n- name: Set _found_file to the first existing file, or an empty list if no files found\n  ansible.builtin.set_fact:\n    _found_file: "{{ lookup(\'ansible.builtin.first_found\', files, paths=[\'/extra/path\'], skip=True) }}"\n  vars:\n    files:\n      - /path/to/foo.txt\n      - /path/to/bar.txt\n\n- name: Include tasks only if one of the files exist, otherwise skip the task\n  ansible.builtin.include_tasks:\n    file: "{{ item }}"\n  with_first_found:\n    - files:\n      - path/tasks.yaml\n      - path/other_tasks.yaml\n      skip: True\n\n- name: Include tasks only if one of the files exists, otherwise skip\n  ansible.builtin.include_tasks: \'{{ tasks_file }}\'\n  when: tasks_file != ""\n  vars:\n    tasks_file: "{{ lookup(\'ansible.builtin.first_found\', files=[\'tasks.yaml\', \'other_tasks.yaml\'], errors=\'ignore\') }}"\n\n- name: |\n        copy first existing file found to /some/file,\n        looking in relative directories from where the task is defined and\n        including any play objects that contain it\n  ansible.builtin.copy:\n    src: "{{ lookup(\'ansible.builtin.first_found\', findme) }}"\n    dest: /some/file\n  vars:\n    findme:\n      - foo\n      - "{{ inventory_hostname }}"\n      - bar\n\n- name: same copy but specific paths\n  ansible.builtin.copy:\n    src: "{{ lookup(\'ansible.builtin.first_found\', params) }}"\n    dest: /some/file\n  vars:\n    params:\n      files:\n        - foo\n        - "{{ inventory_hostname }}"\n        - bar\n      paths:\n        - /tmp/production\n        - /tmp/staging\n\n- name: INTERFACES | Create Ansible header for /etc/network/interfaces\n  ansible.builtin.template:\n    src: "{{ lookup(\'ansible.builtin.first_found\', findme)}}"\n    dest: "/etc/foo.conf"\n  vars:\n    findme:\n      - "{{ ansible_virtualization_type }}_foo.conf"\n      - "default_foo.conf"\n\n- name: read vars from first file found, use \'vars/\' relative subdir\n  ansible.builtin.include_vars: "{{lookup(\'ansible.builtin.first_found\', params)}}"\n  vars:\n    params:\n      files:\n        - \'{{ ansible_distribution }}.yml\'\n        - \'{{ ansible_os_family }}.yml\'\n        - default.yml\n      paths:\n        - \'vars\'\n'
RETURN = '\n  _raw:\n    description:\n      - path to file found\n    type: list\n    elements: path\n'
import os
import re
from collections.abc import Mapping, Sequence
from jinja2.exceptions import UndefinedError
from ansible.errors import AnsibleLookupError, AnsibleUndefinedVariable
from ansible.module_utils.six import string_types
from ansible.plugins.lookup import LookupBase

def _split_on(terms, spliters=','):
    if False:
        while True:
            i = 10
    termlist = []
    if isinstance(terms, string_types):
        termlist = re.split('[%s]' % ''.join(map(re.escape, spliters)), terms)
    else:
        for t in terms:
            termlist.extend(_split_on(t, spliters))
    return termlist

class LookupModule(LookupBase):

    def _process_terms(self, terms, variables, kwargs):
        if False:
            while True:
                i = 10
        total_search = []
        skip = False
        for term in terms:
            if isinstance(term, Mapping):
                self.set_options(var_options=variables, direct=term)
                files = self.get_option('files')
            elif isinstance(term, string_types):
                files = [term]
            elif isinstance(term, Sequence):
                (partial, skip) = self._process_terms(term, variables, kwargs)
                total_search.extend(partial)
                continue
            else:
                raise AnsibleLookupError('Invalid term supplied, can handle string, mapping or list of strings but got: %s for %s' % (type(term), term))
            paths = self.get_option('paths')
            skip = self.get_option('skip')
            filelist = _split_on(files, ',;')
            pathlist = _split_on(paths, ',:;')
            if pathlist:
                for path in pathlist:
                    for fn in filelist:
                        f = os.path.join(path, fn)
                        total_search.append(f)
            elif filelist:
                total_search.extend(filelist)
            else:
                total_search.append(term)
        return (total_search, skip)

    def run(self, terms, variables, **kwargs):
        if False:
            i = 10
            return i + 15
        if not terms:
            self.set_options(var_options=variables, direct=kwargs)
            terms = self.get_option('files')
        (total_search, skip) = self._process_terms(terms, variables, kwargs)
        subdir = getattr(self, '_subdir', 'files')
        path = None
        for fn in total_search:
            try:
                fn = self._templar.template(fn)
            except (AnsibleUndefinedVariable, UndefinedError):
                continue
            path = self.find_file_in_search_path(variables, subdir, fn, ignore_missing=True)
            if path is not None:
                return [path]
        if skip:
            return []
        raise AnsibleLookupError('No file was found when using first_found.')