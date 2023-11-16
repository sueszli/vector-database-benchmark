from __future__ import annotations
DOCUMENTATION = '\n    name: together\n    author:  Bradley Young (!UNKNOWN) <young.bradley@gmail.com>\n    version_added: \'1.3\'\n    short_description: merges lists into synchronized list\n    description:\n      - Creates a list with the iterated elements of the supplied lists\n      - "To clarify with an example, [ \'a\', \'b\' ] and [ 1, 2 ] turn into [ (\'a\',1), (\'b\', 2) ]"\n      - This is basically the same as the \'zip_longest\' filter and Python function\n      - Any \'unbalanced\' elements will be substituted with \'None\'\n    options:\n      _terms:\n        description: list of lists to merge\n        required: True\n'
EXAMPLES = '\n- name: item.0 returns from the \'a\' list, item.1 returns from the \'1\' list\n  ansible.builtin.debug:\n    msg: "{{ item.0 }} and {{ item.1 }}"\n  with_together:\n    - [\'a\', \'b\', \'c\', \'d\']\n    - [1, 2, 3, 4]\n'
RETURN = '\n  _list:\n    description: synchronized list\n    type: list\n    elements: list\n'
import itertools
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.utils.listify import listify_lookup_plugin_terms

class LookupModule(LookupBase):
    """
    Transpose a list of arrays:
    [1, 2, 3], [4, 5, 6] -> [1, 4], [2, 5], [3, 6]
    Replace any empty spots in 2nd array with None:
    [1, 2], [3] -> [1, 3], [2, None]
    """

    def _lookup_variables(self, terms):
        if False:
            i = 10
            return i + 15
        results = []
        for x in terms:
            intermediate = listify_lookup_plugin_terms(x, templar=self._templar)
            results.append(intermediate)
        return results

    def run(self, terms, variables=None, **kwargs):
        if False:
            return 10
        terms = self._lookup_variables(terms)
        my_list = terms[:]
        if len(my_list) == 0:
            raise AnsibleError('with_together requires at least one element in each list')
        return [self._flatten(x) for x in itertools.zip_longest(*my_list, fillvalue=None)]