import unittest
from Cython import Shadow
from Cython.Compiler import Options, CythonScope

class TestShadow(unittest.TestCase):

    def test_all_directives_in_shadow(self):
        if False:
            for i in range(10):
                print('nop')
        missing_directives = []
        extra_directives = []
        for full_directive in Options.directive_types.keys():
            split_directive = full_directive.split('.')
            (directive, rest) = (split_directive[0], split_directive[1:])
            scope = Options.directive_scopes.get(full_directive)
            if scope and len(scope) == 1 and (scope[0] == 'module'):
                if hasattr(Shadow, directive):
                    extra_directives.append(full_directive)
                continue
            if full_directive == 'collection_type':
                continue
            if full_directive == 'staticmethod':
                continue
            if not hasattr(Shadow, directive):
                missing_directives.append(full_directive)
            elif rest:
                directive_value = getattr(Shadow, directive)
                for subdirective in rest:
                    if hasattr(type(directive_value), '__getattr__') or hasattr(type(directive_value), '__getattribute__'):
                        break
        self.assertEqual(missing_directives, [])
        self.assertEqual(extra_directives, [])

    def test_all_types_in_shadow(self):
        if False:
            while True:
                i = 10
        cython_scope = CythonScope.create_cython_scope(None)
        missing_types = []
        for key in cython_scope.entries.keys():
            if key.startswith('__') and key.endswith('__'):
                continue
            if key in ('PyTypeObject', 'PyObject_TypeCheck'):
                continue
            if not hasattr(Shadow, key):
                missing_types.append(key)
        self.assertEqual(missing_types, [])