"""
Generates code for C++ testing, mostly the table to look up symbols from test
names.
"""
import collections

class Namespace:
    """
    Represents a C++ namespace, which contains other namespaces and functions.

    gen_prototypes() generates the code for the namespace.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.namespaces = collections.defaultdict(self.__class__)
        self.functions = []

    def add_functionname(self, path):
        if False:
            print('Hello World!')
        '\n        Adds a function to the namespace.\n\n        Path is the qualified function "path" (e.g., openage::test::foo)\n        has the path ["openage", "test", "foo"].\n\n        Descends recursively, creating subnamespaces as required.\n        '
        if len(path) == 1:
            self.functions.append(path[0])
        else:
            subnamespace = self.namespaces[path[0]]
            subnamespace.add_functionname(path[1:])

    def gen_prototypes(self):
        if False:
            while True:
                i = 10
        '\n        Generates the actual C++ code for this namespace,\n        including all sub-namespaces and function prototypes.\n        '
        for name in self.functions:
            yield f'void {name}();\n'
        for (namespacename, namespace) in sorted(self.namespaces.items()):
            yield f'namespace {namespacename} {{\n'
            for line in namespace.gen_prototypes():
                yield line
            yield f'}} // {namespacename}\n\n'

    def get_functionnames(self):
        if False:
            i = 10
            return i + 15
        '\n        Yields all function names in this namespace,\n        as well as all subnamespaces.\n        '
        for name in self.functions:
            yield name
        for (namespacename, namespace) in sorted(self.namespaces.items()):
            for name in namespace.get_functionnames():
                yield (namespacename + '::' + name)

def generate_testlist(projectdir):
    if False:
        i = 10
        return i + 15
    '\n    Generates the test/demo method symbol lookup file from tests_cpp.\n\n    projectdir is a util.fslike.path.Path.\n    '
    root_namespace = Namespace()
    from ..testing.list_processor import list_targets_cpp
    for (testname, _, _, _) in list_targets_cpp():
        root_namespace.add_functionname(testname.split('::'))
    func_prototypes = list(root_namespace.gen_prototypes())
    method_mappings = [f'{{"{functionname}", ::{functionname}}}' for functionname in root_namespace.get_functionnames()]
    tmpl_path = projectdir.joinpath('libopenage/testing/testlist.cpp.template')
    with tmpl_path.open() as tmpl:
        content = tmpl.read()
    content = content.replace('FUNCTION_PROTOTYPES', ''.join(func_prototypes))
    content = content.replace('METHOD_MAPPINGS', ',\n\t'.join(method_mappings))
    gen_path = projectdir.joinpath('libopenage/testing/testlist.gen.cpp')
    with gen_path.open('w') as gen:
        gen.write(content)