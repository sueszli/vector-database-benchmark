from __future__ import annotations
DOCUMENTATION = '\n    name: csvfile\n    author: Jan-Piet Mens (@jpmens) <jpmens(at)gmail.com>\n    version_added: "1.5"\n    short_description: read data from a TSV or CSV file\n    description:\n      - The csvfile lookup reads the contents of a file in CSV (comma-separated value) format.\n        The lookup looks for the row where the first column matches keyname (which can be multiple words)\n        and returns the value in the O(col) column (default 1, which indexed from 0 means the second column in the file).\n    options:\n      col:\n        description:  column to return (0 indexed).\n        default: "1"\n      default:\n        description: what to return if the value is not found in the file.\n      delimiter:\n        description: field separator in the file, for a tab you can specify V(TAB) or V(\\\\t).\n        default: TAB\n      file:\n        description: name of the CSV/TSV file to open.\n        default: ansible.csv\n      encoding:\n        description: Encoding (character set) of the used CSV file.\n        default: utf-8\n        version_added: "2.1"\n    notes:\n      - The default is for TSV files (tab delimited) not CSV (comma delimited) ... yes the name is misleading.\n      - As of version 2.11, the search parameter (text that must match the first column of the file) and filename parameter can be multi-word.\n      - For historical reasons, in the search keyname, quotes are treated\n        literally and cannot be used around the string unless they appear\n        (escaped as required) in the first column of the file you are parsing.\n    seealso:\n      - ref: playbook_task_paths\n        description: Search paths used for relative files.\n'
EXAMPLES = '\n- name:  Match \'Li\' on the first column, return the second column (0 based index)\n  ansible.builtin.debug: msg="The atomic number of Lithium is {{ lookup(\'ansible.builtin.csvfile\', \'Li file=elements.csv delimiter=,\') }}"\n\n- name: msg="Match \'Li\' on the first column, but return the 3rd column (columns start counting after the match)"\n  ansible.builtin.debug: msg="The atomic mass of Lithium is {{ lookup(\'ansible.builtin.csvfile\', \'Li file=elements.csv delimiter=, col=2\') }}"\n\n- name: Define Values From CSV File, this reads file in one go, but you could also use col= to read each in it\'s own lookup.\n  ansible.builtin.set_fact:\n    loop_ip: "{{ csvline[0] }}"\n    int_ip: "{{ csvline[1] }}"\n    int_mask: "{{ csvline[2] }}"\n    int_name: "{{ csvline[3] }}"\n    local_as: "{{ csvline[4] }}"\n    neighbor_as: "{{ csvline[5] }}"\n    neigh_int_ip: "{{ csvline[6] }}"\n  vars:\n    csvline: "{{ lookup(\'ansible.builtin.csvfile\', bgp_neighbor_ip, file=\'bgp_neighbors.csv\', delimiter=\',\') }}"\n  delegate_to: localhost\n'
RETURN = '\n  _raw:\n    description:\n      - value(s) stored in file column\n    type: list\n    elements: str\n'
import codecs
import csv
from collections.abc import MutableSequence
from ansible.errors import AnsibleError, AnsibleAssertionError
from ansible.parsing.splitter import parse_kv
from ansible.plugins.lookup import LookupBase
from ansible.module_utils.six import PY2
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text

class CSVRecoder:
    """
    Iterator that reads an encoded stream and encodes the input to UTF-8
    """

    def __init__(self, f, encoding='utf-8'):
        if False:
            while True:
                i = 10
        self.reader = codecs.getreader(encoding)(f)

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self

    def __next__(self):
        if False:
            i = 10
            return i + 15
        return next(self.reader).encode('utf-8')
    next = __next__

class CSVReader:
    """
    A CSV reader which will iterate over lines in the CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding='utf-8', **kwds):
        if False:
            while True:
                i = 10
        if PY2:
            f = CSVRecoder(f, encoding)
        else:
            f = codecs.getreader(encoding)(f)
        self.reader = csv.reader(f, dialect=dialect, **kwds)

    def __next__(self):
        if False:
            while True:
                i = 10
        row = next(self.reader)
        return [to_text(s) for s in row]
    next = __next__

    def __iter__(self):
        if False:
            return 10
        return self

class LookupModule(LookupBase):

    def read_csv(self, filename, key, delimiter, encoding='utf-8', dflt=None, col=1):
        if False:
            i = 10
            return i + 15
        try:
            f = open(to_bytes(filename), 'rb')
            creader = CSVReader(f, delimiter=to_native(delimiter), encoding=encoding)
            for row in creader:
                if len(row) and row[0] == key:
                    return row[int(col)]
        except Exception as e:
            raise AnsibleError('csvfile: %s' % to_native(e))
        return dflt

    def run(self, terms, variables=None, **kwargs):
        if False:
            while True:
                i = 10
        ret = []
        self.set_options(var_options=variables, direct=kwargs)
        paramvals = self.get_options()
        for term in terms:
            kv = parse_kv(term)
            if '_raw_params' not in kv:
                raise AnsibleError('Search key is required but was not found')
            key = kv['_raw_params']
            try:
                for (name, value) in kv.items():
                    if name == '_raw_params':
                        continue
                    if name not in paramvals:
                        raise AnsibleAssertionError('%s is not a valid option' % name)
                    self._deprecate_inline_kv()
                    paramvals[name] = value
            except (ValueError, AssertionError) as e:
                raise AnsibleError(e)
            if paramvals['delimiter'] == 'TAB':
                paramvals['delimiter'] = '\t'
            lookupfile = self.find_file_in_search_path(variables, 'files', paramvals['file'])
            var = self.read_csv(lookupfile, key, paramvals['delimiter'], paramvals['encoding'], paramvals['default'], paramvals['col'])
            if var is not None:
                if isinstance(var, MutableSequence):
                    for v in var:
                        ret.append(v)
                else:
                    ret.append(var)
        return ret