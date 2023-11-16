from __future__ import annotations
DOCUMENTATION = '\n    name: subelements\n    author: Serge van Ginderachter (!UNKNOWN) <serge@vanginderachter.be>\n    version_added: "1.4"\n    short_description: traverse nested key from a list of dictionaries\n    description:\n      - Subelements walks a list of hashes (aka dictionaries) and then traverses a list with a given (nested sub-)key inside of those records.\n    options:\n      _terms:\n         description: tuple of list of dictionaries and dictionary key to extract\n         required: True\n      skip_missing:\n        default: False\n        description:\n          - Lookup accepts this flag from a dictionary as optional. See Example section for more information.\n          - If set to V(True), the lookup plugin will skip the lists items that do not contain the given subkey.\n          - If set to V(False), the plugin will yield an error and complain about the missing subkey.\n'
EXAMPLES = '\n- name: show var structure as it is needed for example to make sense\n  hosts: all\n  vars:\n    users:\n      - name: alice\n        authorized:\n          - /tmp/alice/onekey.pub\n          - /tmp/alice/twokey.pub\n        mysql:\n            password: mysql-password\n            hosts:\n              - "%"\n              - "127.0.0.1"\n              - "::1"\n              - "localhost"\n            privs:\n              - "*.*:SELECT"\n              - "DB1.*:ALL"\n        groups:\n          - wheel\n      - name: bob\n        authorized:\n          - /tmp/bob/id_rsa.pub\n        mysql:\n            password: other-mysql-password\n            hosts:\n              - "db1"\n            privs:\n              - "*.*:SELECT"\n              - "DB2.*:ALL"\n  tasks:\n    - name: Set authorized ssh key, extracting just that data from \'users\'\n      ansible.posix.authorized_key:\n        user: "{{ item.0.name }}"\n        key: "{{ lookup(\'file\', item.1) }}"\n      with_subelements:\n         - "{{ users }}"\n         - authorized\n\n    - name: Setup MySQL users, given the mysql hosts and privs subkey lists\n      community.mysql.mysql_user:\n        name: "{{ item.0.name }}"\n        password: "{{ item.0.mysql.password }}"\n        host: "{{ item.1 }}"\n        priv: "{{ item.0.mysql.privs | join(\'/\') }}"\n      with_subelements:\n        - "{{ users }}"\n        - mysql.hosts\n\n    - name: list groups for users that have them, don\'t error if groups key is missing\n      ansible.builtin.debug: var=item\n      loop: "{{ q(\'ansible.builtin.subelements\', users, \'groups\', {\'skip_missing\': True}) }}"\n'
RETURN = '\n_list:\n  description: list of subelements extracted\n'
from ansible.errors import AnsibleError
from ansible.module_utils.six import string_types
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.plugins.lookup import LookupBase
from ansible.utils.listify import listify_lookup_plugin_terms
FLAGS = ('skip_missing',)

class LookupModule(LookupBase):

    def run(self, terms, variables, **kwargs):
        if False:
            print('Hello World!')

        def _raise_terms_error(msg=''):
            if False:
                print('Hello World!')
            raise AnsibleError('subelements lookup expects a list of two or three items, ' + msg)
        terms[0] = listify_lookup_plugin_terms(terms[0], templar=self._templar)
        if not isinstance(terms, list) or not 2 <= len(terms) <= 3:
            _raise_terms_error()
        if not isinstance(terms[0], (list, dict)) or not isinstance(terms[1], string_types):
            _raise_terms_error('first a dict or a list, second a string pointing to the subkey')
        subelements = terms[1].split('.')
        if isinstance(terms[0], dict):
            if terms[0].get('skipped', False) is not False:
                return []
            elementlist = []
            for key in terms[0]:
                elementlist.append(terms[0][key])
        else:
            elementlist = terms[0]
        flags = {}
        if len(terms) == 3:
            flags = terms[2]
        if not isinstance(flags, dict) and (not all((isinstance(key, string_types) and key in FLAGS for key in flags))):
            _raise_terms_error('the optional third item must be a dict with flags %s' % FLAGS)
        ret = []
        for item0 in elementlist:
            if not isinstance(item0, dict):
                raise AnsibleError("subelements lookup expects a dictionary, got '%s'" % item0)
            if item0.get('skipped', False) is not False:
                continue
            skip_missing = boolean(flags.get('skip_missing', False), strict=False)
            subvalue = item0
            lastsubkey = False
            sublist = []
            for subkey in subelements:
                if subkey == subelements[-1]:
                    lastsubkey = True
                if subkey not in subvalue:
                    if skip_missing:
                        continue
                    else:
                        raise AnsibleError("could not find '%s' key in iterated item '%s'" % (subkey, subvalue))
                if not lastsubkey:
                    if not isinstance(subvalue[subkey], dict):
                        if skip_missing:
                            continue
                        else:
                            raise AnsibleError("the key %s should point to a dictionary, got '%s'" % (subkey, subvalue[subkey]))
                    else:
                        subvalue = subvalue[subkey]
                elif not isinstance(subvalue[subkey], list):
                    raise AnsibleError("the key %s should point to a list, got '%s'" % (subkey, subvalue[subkey]))
                else:
                    sublist = subvalue.pop(subkey, [])
            for item1 in sublist:
                ret.append((item0, item1))
        return ret