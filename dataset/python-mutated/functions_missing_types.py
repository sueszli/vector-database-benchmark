"""Find the functions in a module missing type annotations.

To use it run

./functions_missing_types.py <module>

and it will print out a list of functions in the module that don't
have types.

"""
import argparse
import ast
import importlib
import os
NUMPY_ROOT = os.path.dirname(os.path.join(os.path.abspath(__file__), '..'))
EXCLUDE_LIST = {'numpy': {'absolute_import', 'division', 'print_function', 'warnings', 'sys', 'os', 'math', 'Tester', '_core', 'get_array_wrap', 'int_asbuffer', 'numarray', 'oldnumeric', 'safe_eval', 'test', 'typeDict', 'bool', 'complex', 'float', 'int', 'long', 'object', 'str', 'unicode', 'alltrue', 'sometrue'}}

class FindAttributes(ast.NodeVisitor):
    """Find top-level attributes/functions/classes in stubs files.

    Do this by walking the stubs ast. See e.g.

    https://greentreesnakes.readthedocs.io/en/latest/index.html

    for more information on working with Python's ast.

    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.attributes = set()

    def visit_FunctionDef(self, node):
        if False:
            return 10
        if node.name == '__getattr__':
            return
        self.attributes.add(node.name)
        return

    def visit_ClassDef(self, node):
        if False:
            i = 10
            return i + 15
        if not node.name.startswith('_'):
            self.attributes.add(node.name)
        return

    def visit_AnnAssign(self, node):
        if False:
            i = 10
            return i + 15
        self.attributes.add(node.target.id)

def find_missing(module_name):
    if False:
        return 10
    module_path = os.path.join(NUMPY_ROOT, module_name.replace('.', os.sep), '__init__.pyi')
    module = importlib.import_module(module_name)
    module_attributes = {attribute for attribute in dir(module) if not attribute.startswith('_')}
    if os.path.isfile(module_path):
        with open(module_path) as f:
            tree = ast.parse(f.read())
        ast_visitor = FindAttributes()
        ast_visitor.visit(tree)
        stubs_attributes = ast_visitor.attributes
    else:
        stubs_attributes = set()
    exclude_list = EXCLUDE_LIST.get(module_name, set())
    missing = module_attributes - stubs_attributes - exclude_list
    print('\n'.join(sorted(missing)))

def main():
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser()
    parser.add_argument('module')
    args = parser.parse_args()
    find_missing(args.module)
if __name__ == '__main__':
    main()