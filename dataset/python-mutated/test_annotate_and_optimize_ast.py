import ast as python_ast
from vyper.ast.annotation import annotate_python_ast
from vyper.ast.pre_parser import pre_parse

class AssertionVisitor(python_ast.NodeVisitor):

    def assert_about_node(self, node):
        if False:
            while True:
                i = 10
        raise AssertionError()

    def generic_visit(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.assert_about_node(node)
        super().generic_visit(node)
TEST_CONTRACT_SOURCE_CODE = '\nstruct S:\n    a: bool\n    b: int128\n\ninterface ERC20Contract:\n    def name() -> String[64]: view\n\n@external\ndef foo() -> int128:\n    return -(-(-1))\n'

def get_contract_info(source_code):
    if False:
        i = 10
        return i + 15
    (_, class_types, reformatted_code) = pre_parse(source_code)
    py_ast = python_ast.parse(reformatted_code)
    annotate_python_ast(py_ast, reformatted_code, class_types)
    return (py_ast, reformatted_code)

def test_it_annotates_ast_with_source_code():
    if False:
        for i in range(10):
            print('nop')
    (contract_ast, reformatted_code) = get_contract_info(TEST_CONTRACT_SOURCE_CODE)

    class AssertSourceCodePresent(AssertionVisitor):

        def assert_about_node(self, node):
            if False:
                for i in range(10):
                    print('nop')
            assert node.full_source_code is reformatted_code
    AssertSourceCodePresent().visit(contract_ast)

def test_it_annotates_ast_with_class_types():
    if False:
        i = 10
        return i + 15
    (contract_ast, _) = get_contract_info(TEST_CONTRACT_SOURCE_CODE)
    struct_def = contract_ast.body[0]
    contract_def = contract_ast.body[1]
    assert struct_def.ast_type == 'StructDef'
    assert contract_def.ast_type == 'InterfaceDef'

def test_it_rewrites_unary_subtractions():
    if False:
        i = 10
        return i + 15
    (contract_ast, _) = get_contract_info(TEST_CONTRACT_SOURCE_CODE)
    function_def = contract_ast.body[2]
    return_stmt = function_def.body[0]
    assert isinstance(return_stmt.value, python_ast.Num)
    assert return_stmt.value.n == -1