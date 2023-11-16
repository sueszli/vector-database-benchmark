""" Handle bytecode and compile source code to bytecode. """
import ast
from nuitka.Options import hasPythonFlagNoAsserts, hasPythonFlagNoDocStrings
from nuitka.tree.TreeHelpers import getKind
doc_having = tuple((getattr(ast, candidate) for candidate in ('FunctionDef', 'ClassDef', 'AsyncFunctionDef') if hasattr(ast, candidate)))

def _removeDocFromBody(node):
    if False:
        while True:
            i = 10
    if node.body and getKind(node.body[0]) == 'Expr':
        if getKind(node.body[0].value) == 'Str':
            node.body[0].value.s = ''
        elif getKind(node.body[0].value) == 'Constant':
            node.body[0].value.value = ''

def compileSourceToBytecode(source_code, filename):
    if False:
        i = 10
        return i + 15
    'Compile given source code into bytecode.'
    tree = ast.parse(source_code, filename)
    remove_doc_strings_from_tree = hasPythonFlagNoDocStrings()
    remove_asserts_from_tree = hasPythonFlagNoAsserts() and str is bytes
    if remove_doc_strings_from_tree or remove_asserts_from_tree:
        if remove_doc_strings_from_tree:
            _removeDocFromBody(tree)
        for node in ast.walk(tree):
            if remove_asserts_from_tree:
                node_type = type(node)
                if node_type is ast.Name:
                    if node.id == '__debug__':
                        node.id = 'False'
                elif node_type is ast.Assert:
                    node.test = ast.Num()
                    node.test.n = 1
                    node.test.lineno = node.lineno
                    node.test.col_offset = node.col_offset
                    node.msg = None
            if remove_doc_strings_from_tree and isinstance(node, doc_having):
                _removeDocFromBody(node)
    if str is bytes:
        bytecode = compile(tree, filename=filename, mode='exec', dont_inherit=True)
    else:
        optimize = 0
        if hasPythonFlagNoAsserts():
            optimize = 1
        bytecode = compile(tree, filename=filename, mode='exec', dont_inherit=True, optimize=optimize)
    return bytecode

def loadCodeObjectData(bytecode_filename):
    if False:
        print('Hello World!')
    'Load bytecode from a file.'
    with open(bytecode_filename, 'rb') as f:
        return f.read()[8 if str is bytes else 16:]