from __future__ import annotations
DOCUMENTATION = '\n    name: ini\n    version_added: "2.4"\n    short_description: Uses an Ansible INI file as inventory source.\n    description:\n        - INI file based inventory, sections are groups or group related with special C(:modifiers).\n        - Entries in sections C([group_1]) are hosts, members of the group.\n        - Hosts can have variables defined inline as key/value pairs separated by C(=).\n        - The C(children) modifier indicates that the section contains groups.\n        - The C(vars) modifier indicates that the section contains variables assigned to members of the group.\n        - Anything found outside a section is considered an \'ungrouped\' host.\n        - Values passed in the INI format using the C(key=value) syntax are interpreted differently depending on where they are declared within your inventory.\n        - When declared inline with the host, INI values are processed by Python\'s ast.literal_eval function\n          (U(https://docs.python.org/3/library/ast.html#ast.literal_eval)) and interpreted as Python literal structures\n         (strings, numbers, tuples, lists, dicts, booleans, None). If you want a number to be treated as a string, you must quote it.\n          Host lines accept multiple C(key=value) parameters per line.\n          Therefore they need a way to indicate that a space is part of a value rather than a separator.\n        - When declared in a C(:vars) section, INI values are interpreted as strings. For example C(var=FALSE) would create a string equal to C(FALSE).\n          Unlike host lines, C(:vars) sections accept only a single entry per line, so everything after the C(=) must be the value for the entry.\n        - Do not rely on types set during definition, always make sure you specify type with a filter when needed when consuming the variable.\n        - See the Examples for proper quoting to prevent changes to variable type.\n    notes:\n        - Enabled in configuration by default.\n        - Consider switching to YAML format for inventory sources to avoid confusion on the actual type of a variable.\n          The YAML inventory plugin processes variable values consistently and correctly.\n'
EXAMPLES = '# fmt: ini\n# Example 1\n[web]\nhost1\nhost2 ansible_port=222 # defined inline, interpreted as an integer\n\n[web:vars]\nhttp_port=8080 # all members of \'web\' will inherit these\nmyvar=23 # defined in a :vars section, interpreted as a string\n\n[web:children] # child groups will automatically add their hosts to parent group\napache\nnginx\n\n[apache]\ntomcat1\ntomcat2 myvar=34 # host specific vars override group vars\ntomcat3 mysecret="\'03#pa33w0rd\'" # proper quoting to prevent value changes\n\n[nginx]\njenkins1\n\n[nginx:vars]\nhas_java = True # vars in child groups override same in parent\n\n[all:vars]\nhas_java = False # \'all\' is \'top\' parent\n\n# Example 2\nhost1 # this is \'ungrouped\'\n\n# both hosts have same IP but diff ports, also \'ungrouped\'\nhost2 ansible_host=127.0.0.1 ansible_port=44\nhost3 ansible_host=127.0.0.1 ansible_port=45\n\n[g1]\nhost4\n\n[g2]\nhost4 # same host as above, but member of 2 groups, will inherit vars from both\n      # inventory hostnames are unique\n'
import ast
import re
import warnings
from ansible.inventory.group import to_safe_group_name
from ansible.plugins.inventory import BaseFileInventoryPlugin
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.utils.shlex import shlex_split

class InventoryModule(BaseFileInventoryPlugin):
    """
    Takes an INI-format inventory file and builds a list of groups and subgroups
    with their associated hosts and variable settings.
    """
    NAME = 'ini'
    _COMMENT_MARKERS = frozenset((u';', u'#'))
    b_COMMENT_MARKERS = frozenset((b';', b'#'))

    def __init__(self):
        if False:
            print('Hello World!')
        super(InventoryModule, self).__init__()
        self.patterns = {}
        self._filename = None

    def parse(self, inventory, loader, path, cache=True):
        if False:
            for i in range(10):
                print('nop')
        super(InventoryModule, self).parse(inventory, loader, path)
        self._filename = path
        try:
            if self.loader:
                (b_data, private) = self.loader._get_file_contents(path)
            else:
                b_path = to_bytes(path, errors='surrogate_or_strict')
                with open(b_path, 'rb') as fh:
                    b_data = fh.read()
            try:
                data = to_text(b_data, errors='surrogate_or_strict').splitlines()
            except UnicodeError:
                data = []
                for line in b_data.splitlines():
                    if line and line[0] in self.b_COMMENT_MARKERS:
                        data.append(u'')
                    else:
                        data.append(to_text(line, errors='surrogate_or_strict'))
            self._parse(path, data)
        except Exception as e:
            raise AnsibleParserError(e)

    def _raise_error(self, message):
        if False:
            for i in range(10):
                print('nop')
        raise AnsibleError('%s:%d: ' % (self._filename, self.lineno) + message)

    def _parse(self, path, lines):
        if False:
            print('Hello World!')
        '\n        Populates self.groups from the given array of lines. Raises an error on\n        any parse failure.\n        '
        self._compile_patterns()
        pending_declarations = {}
        groupname = 'ungrouped'
        state = 'hosts'
        self.lineno = 0
        for line in lines:
            self.lineno += 1
            line = line.strip()
            if not line or line[0] in self._COMMENT_MARKERS:
                continue
            m = self.patterns['section'].match(line)
            if m:
                (groupname, state) = m.groups()
                groupname = to_safe_group_name(groupname)
                state = state or 'hosts'
                if state not in ['hosts', 'children', 'vars']:
                    title = ':'.join(m.groups())
                    self._raise_error('Section [%s] has unknown type: %s' % (title, state))
                if groupname not in self.inventory.groups:
                    if state == 'vars' and groupname not in pending_declarations:
                        pending_declarations[groupname] = dict(line=self.lineno, state=state, name=groupname)
                    self.inventory.add_group(groupname)
                if groupname in pending_declarations and state != 'vars':
                    if pending_declarations[groupname]['state'] == 'children':
                        self._add_pending_children(groupname, pending_declarations)
                    elif pending_declarations[groupname]['state'] == 'vars':
                        del pending_declarations[groupname]
                continue
            elif line.startswith('[') and line.endswith(']'):
                self._raise_error("Invalid section entry: '%s'. Please make sure that there are no spaces" % line + ' ' + 'in the section entry, and that there are no other invalid characters')
            if state == 'hosts':
                (hosts, port, variables) = self._parse_host_definition(line)
                self._populate_host_vars(hosts, variables, groupname, port)
            elif state == 'vars':
                (k, v) = self._parse_variable_definition(line)
                self.inventory.set_variable(groupname, k, v)
            elif state == 'children':
                child = self._parse_group_name(line)
                if child not in self.inventory.groups:
                    if child not in pending_declarations:
                        pending_declarations[child] = dict(line=self.lineno, state=state, name=child, parents=[groupname])
                    else:
                        pending_declarations[child]['parents'].append(groupname)
                else:
                    self.inventory.add_child(groupname, child)
            else:
                self._raise_error('Entered unhandled state: %s' % state)
        for g in pending_declarations:
            decl = pending_declarations[g]
            if decl['state'] == 'vars':
                raise AnsibleError('%s:%d: Section [%s:vars] not valid for undefined group: %s' % (path, decl['line'], decl['name'], decl['name']))
            elif decl['state'] == 'children':
                raise AnsibleError('%s:%d: Section [%s:children] includes undefined group: %s' % (path, decl['line'], decl['parents'].pop(), decl['name']))

    def _add_pending_children(self, group, pending):
        if False:
            print('Hello World!')
        for parent in pending[group]['parents']:
            self.inventory.add_child(parent, group)
            if parent in pending and pending[parent]['state'] == 'children':
                self._add_pending_children(parent, pending)
        del pending[group]

    def _parse_group_name(self, line):
        if False:
            i = 10
            return i + 15
        '\n        Takes a single line and tries to parse it as a group name. Returns the\n        group name if successful, or raises an error.\n        '
        m = self.patterns['groupname'].match(line)
        if m:
            return m.group(1)
        self._raise_error('Expected group name, got: %s' % line)

    def _parse_variable_definition(self, line):
        if False:
            print('Hello World!')
        '\n        Takes a string and tries to parse it as a variable definition. Returns\n        the key and value if successful, or raises an error.\n        '
        if '=' in line:
            (k, v) = [e.strip() for e in line.split('=', 1)]
            return (k, self._parse_value(v))
        self._raise_error('Expected key=value, got: %s' % line)

    def _parse_host_definition(self, line):
        if False:
            return 10
        '\n        Takes a single line and tries to parse it as a host definition. Returns\n        a list of Hosts if successful, or raises an error.\n        '
        try:
            tokens = shlex_split(line, comments=True)
        except ValueError as e:
            self._raise_error("Error parsing host definition '%s': %s" % (line, e))
        (hostnames, port) = self._expand_hostpattern(tokens[0])
        variables = {}
        for t in tokens[1:]:
            if '=' not in t:
                self._raise_error('Expected key=value host variable assignment, got: %s' % t)
            (k, v) = t.split('=', 1)
            variables[k] = self._parse_value(v)
        return (hostnames, port, variables)

    def _expand_hostpattern(self, hostpattern):
        if False:
            return 10
        '\n        do some extra checks over normal processing\n        '
        (hostnames, port) = super(InventoryModule, self)._expand_hostpattern(hostpattern)
        if hostpattern.strip().endswith(':') and port is None:
            raise AnsibleParserError("Invalid host pattern '%s' supplied, ending in ':' is not allowed, this character is reserved to provide a port." % hostpattern)
        for pattern in hostnames:
            if pattern.strip() == '---':
                raise AnsibleParserError("Invalid host pattern '%s' supplied, '---' is normally a sign this is a YAML file." % hostpattern)
        return (hostnames, port)

    @staticmethod
    def _parse_value(v):
        if False:
            i = 10
            return i + 15
        '\n        Attempt to transform the string value from an ini file into a basic python object\n        (int, dict, list, unicode string, etc).\n        '
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', SyntaxWarning)
                v = ast.literal_eval(v)
        except ValueError:
            pass
        except SyntaxError:
            pass
        return to_text(v, nonstring='passthru', errors='surrogate_or_strict')

    def _compile_patterns(self):
        if False:
            print('Hello World!')
        '\n        Compiles the regular expressions required to parse the inventory and\n        stores them in self.patterns.\n        '
        self.patterns['section'] = re.compile(to_text('^\\[\n                    ([^:\\]\\s]+)             # group name (see groupname below)\n                    (?::(\\w+))?             # optional : and tag name\n                \\]\n                \\s*                         # ignore trailing whitespace\n                (?:\\#.*)?                   # and/or a comment till the\n                $                           # end of the line\n            ', errors='surrogate_or_strict'), re.X)
        self.patterns['groupname'] = re.compile(to_text('^\n                ([^:\\]\\s]+)\n                \\s*                         # ignore trailing whitespace\n                (?:\\#.*)?                   # and/or a comment till the\n                $                           # end of the line\n            ', errors='surrogate_or_strict'), re.X)