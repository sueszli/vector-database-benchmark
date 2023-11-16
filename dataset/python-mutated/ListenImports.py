import os
try:
    basestring
except NameError:
    basestring = str

class ListenImports:
    ROBOT_LISTENER_API_VERSION = 2

    def __init__(self, imports):
        if False:
            for i in range(10):
                print('nop')
        self.imports = open(imports, 'w')

    def library_import(self, name, attrs):
        if False:
            i = 10
            return i + 15
        self._imported('Library', name, attrs)

    def resource_import(self, name, attrs):
        if False:
            print('Hello World!')
        self._imported('Resource', name, attrs)

    def variables_import(self, name, attrs):
        if False:
            return 10
        self._imported('Variables', name, attrs)

    def _imported(self, import_type, name, attrs):
        if False:
            print('Hello World!')
        self.imports.write('Imported %s\n\tname: %s\n' % (import_type, name))
        for name in sorted(attrs):
            self.imports.write('\t%s: %s\n' % (name, self._pretty(attrs[name])))

    def _pretty(self, entry):
        if False:
            print('Hello World!')
        if isinstance(entry, list):
            return '[%s]' % ', '.join(entry)
        if isinstance(entry, basestring) and os.path.isabs(entry):
            entry = entry.replace('$py.class', '.py').replace('.pyc', '.py')
            tokens = entry.split(os.sep)
            index = -1 if tokens[-1] != '__init__.py' else -2
            return '//' + '/'.join(tokens[index:])
        return entry

    def close(self):
        if False:
            return 10
        self.imports.close()