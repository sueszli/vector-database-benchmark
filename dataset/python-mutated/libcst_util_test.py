from pathlib import Path
import testslide
from ...language_server import protocol as lsp
from .. import libcst_util
test_root = '/test_root'
test_path = '/test_root/test_project/test_module.py'

def create_lsp_range(line: int, start: int, end: int) -> lsp.LspRange:
    if False:
        while True:
            i = 10
    line = line - 1
    start = start - 1
    end = end - 1
    return lsp.LspRange(start=lsp.LspPosition(line, start), end=lsp.LspPosition(line, end))

class LibcstUtilTest(testslide.TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.maxDiff = None

    def test_success_case(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Tests success cases for:\n        1. TODO:import statement\n        2. function name\n        3. imported Types\n        4. global scoped variables\n        5. out of order variables\n        '
        test_code: str = '\nimport os\nfrom pathlib import Path as TestPath\n\ntest_path: str = "TEST_PATH"\n\ndef get_path() -> TestPath:\n    return TestPath(os.environ[test_path])\n\ndef count_level() -> int:\n    return x.split("")\n\nx = get_path()\nprint(count_level())\n'
        visitor: libcst_util.QualifiedNameWithPositionVisitor = libcst_util.generate_qualified_name_with_position_visitor(Path(test_path), Path(test_root), test_code, lsp.PyrePosition(line=7, character=4))
        results = visitor.find_references()
        self.assertEqual(results, [create_lsp_range(7, 5, 13), create_lsp_range(13, 5, 13)])
        visitor = libcst_util.generate_qualified_name_with_position_visitor(Path(test_path), Path(test_root), test_code, lsp.PyrePosition(line=7, character=19))
        results = visitor.find_references()
        self.assertEqual(results, [create_lsp_range(7, 19, 27), create_lsp_range(8, 12, 20)])
        visitor = libcst_util.generate_qualified_name_with_position_visitor(Path(test_path), Path(test_root), test_code, lsp.PyrePosition(line=5, character=1))
        results = visitor.find_references()
        self.assertEqual(results, [create_lsp_range(5, 1, 10), create_lsp_range(8, 32, 41)])
        visitor = libcst_util.generate_qualified_name_with_position_visitor(Path(test_path), Path(test_root), test_code, lsp.PyrePosition(line=11, character=12))
        references_to_x = [create_lsp_range(11, 12, 13), create_lsp_range(13, 1, 2)]
        results = visitor.find_references()
        self.assertEqual(results, references_to_x)
        visitor = libcst_util.generate_qualified_name_with_position_visitor(Path(test_path), Path(test_root), test_code, lsp.PyrePosition(line=13, character=1))
        results = visitor.find_references()
        self.assertEqual(results, references_to_x)
    "\n    Things we dont' expect to return references for:\n    1. Keywords\n    2. Literals\n        a. string\n        b. int\n        c. bool\n    "

    def test_keyword(self) -> None:
        if False:
            return 10
        test_code: str = '\nfor x in y:\n    print(x)\n\nfor foo in bar:\n    pass\n'
        visitor: libcst_util.QualifiedNameWithPositionVisitor = libcst_util.generate_qualified_name_with_position_visitor(Path(test_path), Path(test_root), test_code, lsp.PyrePosition(line=2, character=1))
        results = visitor.find_references()
        self.assertEqual(results, [])
        visitor = libcst_util.generate_qualified_name_with_position_visitor(Path(test_path), Path(test_root), test_code, lsp.PyrePosition(line=6, character=4))
        results = visitor.find_references()
        self.assertEqual(results, [])

    def test_int(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        test_code: str = '\na : int = 1\nb : int = 1 + 1\n'
        visitor: libcst_util.QualifiedNameWithPositionVisitor = libcst_util.generate_qualified_name_with_position_visitor(Path(test_path), Path(test_root), test_code, lsp.PyrePosition(line=2, character=11))
        results = visitor.find_references()
        self.assertEqual(results, [])

    def test_booleans(self) -> None:
        if False:
            print('Hello World!')
        test_code: str = '\ndef foo() -> None:\n    if True:\n        return False\n    elif False:\n        pass\n    return True\n'
        visitor: libcst_util.QualifiedNameWithPositionVisitor = libcst_util.generate_qualified_name_with_position_visitor(Path(test_path), Path(test_root), test_code, lsp.PyrePosition(line=3, character=8))
        results = visitor.find_references()
        self.assertEqual(results, [])

    def test_string(self) -> None:
        if False:
            while True:
                i = 10
        test_code: str = '\nc: string = "hello"\nd: string = "hello" + "world"\n'
        visitor: libcst_util.QualifiedNameWithPositionVisitor = libcst_util.generate_qualified_name_with_position_visitor(Path(test_path), Path(test_root), test_code, lsp.PyrePosition(line=2, character=14))
        results = visitor.find_references()
        self.assertEqual(results, [])