import unittest
from robot.parsing import get_model, Token
from robot.parsing.model.statements import Break, Continue, Error, ReturnStatement
from parsing_test_utils import assert_model, RemoveNonDataTokensVisitor

def remove_non_data_nodes_and_assert(node, expected, data_only):
    if False:
        for i in range(10):
            print('nop')
    if not data_only:
        RemoveNonDataTokensVisitor().visit(node)
    assert_model(node, expected)

class TestReturn(unittest.TestCase):

    def test_in_test_case_body(self):
        if False:
            return 10
        for data_only in [True, False]:
            with self.subTest(data_only=data_only):
                model = get_model('*** Test Cases ***\nExample\n    RETURN', data_only=data_only)
                node = model.sections[0].body[0].body[0]
                expected = Error([Token(Token.ERROR, 'RETURN', 3, 4, 'RETURN is not allowed in this context.')])
                remove_non_data_nodes_and_assert(node, expected, data_only)

    def test_in_test_case_body_inside_for(self):
        if False:
            while True:
                i = 10
        for data_only in [True, False]:
            with self.subTest(data_only=data_only):
                model = get_model('*** Test Cases ***\nExample\n    FOR    ${i}    IN    1    2\n        RETURN\n    END\n        ', data_only=data_only)
                node = model.sections[0].body[0].body[0].body[0]
                expected = ReturnStatement([Token(Token.RETURN_STATEMENT, 'RETURN', 4, 8)], errors=('RETURN can only be used inside a user keyword.',))
                remove_non_data_nodes_and_assert(node, expected, data_only)

    def test_in_test_case_body_inside_while(self):
        if False:
            i = 10
            return i + 15
        for data_only in [True, False]:
            with self.subTest(data_only=data_only):
                model = get_model('*** Test Cases ***\nExample\n    WHILE    True\n        RETURN\n    END\n        ', data_only=data_only)
                node = model.sections[0].body[0].body[0].body[0]
                expected = ReturnStatement([Token(Token.RETURN_STATEMENT, 'RETURN', 4, 8)], errors=('RETURN can only be used inside a user keyword.',))
                remove_non_data_nodes_and_assert(node, expected, data_only)

    def test_in_test_case_body_inside_if_else(self):
        if False:
            i = 10
            return i + 15
        for data_only in [True, False]:
            with self.subTest(data_only=data_only):
                model = get_model('*** Test Cases ***\nExample\n    IF    True\n        RETURN\n    ELSE IF    False\n        RETURN\n    ELSE\n        RETURN\n    END\n        ', data_only=data_only)
                ifroot = model.sections[0].body[0].body[0]
                node = ifroot.body[0]
                expected = ReturnStatement([Token(Token.RETURN_STATEMENT, 'RETURN', 4, 8)], errors=('RETURN can only be used inside a user keyword.',))
                remove_non_data_nodes_and_assert(node, expected, data_only)
                expected.tokens[0].lineno = 6
                remove_non_data_nodes_and_assert(ifroot.orelse.body[0], expected, data_only)
                expected.tokens[0].lineno = 8
                remove_non_data_nodes_and_assert(ifroot.orelse.orelse.body[0], expected, data_only)

    def test_in_test_case_body_inside_try_except(self):
        if False:
            for i in range(10):
                print('nop')
        for data_only in [True, False]:
            with self.subTest(data_only=data_only):
                model = get_model('*** Test Cases ***\nExample\n    TRY\n        RETURN\n    EXCEPT\n        RETURN\n    ELSE\n        RETURN\n    FINALLY\n        RETURN\n    END\n        ', data_only=data_only)
                tryroot = model.sections[0].body[0].body[0]
                node = tryroot.body[0]
                expected = ReturnStatement([Token(Token.RETURN_STATEMENT, 'RETURN', 4, 8)], errors=('RETURN can only be used inside a user keyword.',))
                remove_non_data_nodes_and_assert(node, expected, data_only)
                expected.tokens[0].lineno = 6
                remove_non_data_nodes_and_assert(tryroot.next.body[0], expected, data_only)
                expected.tokens[0].lineno = 8
                remove_non_data_nodes_and_assert(tryroot.next.next.body[0], expected, data_only)
                expected.tokens[0].lineno = 10
                expected.errors += ('RETURN cannot be used in FINALLY branch.',)
                remove_non_data_nodes_and_assert(tryroot.next.next.next.body[0], expected, data_only)

    def test_in_finally_in_uk(self):
        if False:
            i = 10
            return i + 15
        for data_only in [True, False]:
            with self.subTest(data_only=data_only):
                model = get_model('*** Keywords ***\nExample\n    TRY\n        No operation\n    EXCEPT\n        No operation\n    FINALLY\n        RETURN\n    END\n        ', data_only=data_only)
                node = model.sections[0].body[0].body[0].next.next.body[0]
                expected = ReturnStatement([Token(Token.RETURN_STATEMENT, 'RETURN', 8, 8)], errors=('RETURN cannot be used in FINALLY branch.',))
                remove_non_data_nodes_and_assert(node, expected, data_only)

    def test_in_nested_finally_in_uk(self):
        if False:
            i = 10
            return i + 15
        for data_only in [True, False]:
            with self.subTest(data_only=data_only):
                model = get_model('*** Keywords ***\nExample\n    IF    True\n        TRY\n            No operation\n        EXCEPT\n            No operation\n        FINALLY\n            RETURN\n        END\n    END', data_only=data_only)
                node = model.sections[0].body[0].body[0].body[0].next.next.body[0]
                expected = ReturnStatement([Token(Token.RETURN_STATEMENT, 'RETURN', 9, 12)], errors=('RETURN cannot be used in FINALLY branch.',))
                remove_non_data_nodes_and_assert(node, expected, data_only)

class TestBreak(unittest.TestCase):

    def test_in_test_case_body(self):
        if False:
            while True:
                i = 10
        for data_only in [True, False]:
            with self.subTest(data_only=data_only):
                model = get_model('*** Test Cases ***\nExample\n    BREAK', data_only=data_only)
                node = model.sections[0].body[0].body[0]
                expected = Error([Token(Token.ERROR, 'BREAK', 3, 4, 'BREAK is not allowed in this context.')])
                remove_non_data_nodes_and_assert(node, expected, data_only)

    def test_in_if_test_case_body(self):
        if False:
            i = 10
            return i + 15
        for data_only in [True, False]:
            with self.subTest(data_only=data_only):
                model = get_model('*** Test Cases ***\nExample\n    IF    True\n        BREAK\n    END', data_only=data_only)
                node = model.sections[0].body[0].body[0].body[0]
                expected = Break([Token(Token.BREAK, 'BREAK', 4, 8)], errors=('BREAK can only be used inside a loop.',))
                remove_non_data_nodes_and_assert(node, expected, data_only)

    def test_in_try_test_case_body(self):
        if False:
            for i in range(10):
                print('nop')
        for data_only in [True, False]:
            with self.subTest(data_only=data_only):
                model = get_model('*** Test Cases ***\nExample\n    TRY    \n        BREAK\n    EXCEPT\n        no operation\n    END', data_only=data_only)
                node = model.sections[0].body[0].body[0].body[0]
                expected = Break([Token(Token.BREAK, 'BREAK', 4, 8)], errors=('BREAK can only be used inside a loop.',))
                remove_non_data_nodes_and_assert(node, expected, data_only)

    def test_in_finally_inside_loop(self):
        if False:
            return 10
        for data_only in [True, False]:
            with self.subTest(data_only=data_only):
                model = get_model('*** Test Cases ***\nExample\n    WHILE    True\n        TRY    \n            Fail\n        EXCEPT\n            no operation\n        FINALLY\n           BREAK\n        END     \n    END', data_only=data_only)
                node = model.sections[0].body[0].body[0].body[0].next.next.body[0]
                expected = Break([Token(Token.BREAK, 'BREAK', 9, 11)], errors=('BREAK cannot be used in FINALLY branch.',))
                remove_non_data_nodes_and_assert(node, expected, data_only)

    def test_in_uk_body(self):
        if False:
            i = 10
            return i + 15
        for data_only in [True, False]:
            with self.subTest(data_only=data_only):
                model = get_model('*** Keywords ***\nExample\n    BREAK', data_only=data_only)
                node = model.sections[0].body[0].body[0]
                expected = Error([Token(Token.ERROR, 'BREAK', 3, 4, 'BREAK is not allowed in this context.')])
                remove_non_data_nodes_and_assert(node, expected, data_only)

    def test_in_if_uk_body(self):
        if False:
            print('Hello World!')
        for data_only in [True, False]:
            with self.subTest(data_only=data_only):
                model = get_model('*** Keywords ***\nExample\n    IF    True\n        BREAK\n    END', data_only=data_only)
                node = model.sections[0].body[0].body[0].body[0]
                expected = Break([Token(Token.BREAK, 'BREAK', 4, 8)], errors=('BREAK can only be used inside a loop.',))
                remove_non_data_nodes_and_assert(node, expected, data_only)

    def test_in_try_uk_body(self):
        if False:
            i = 10
            return i + 15
        for data_only in [True, False]:
            with self.subTest(data_only=data_only):
                model = get_model('*** Keywords ***\nExample\n    TRY    \n        BREAK\n    EXCEPT\n        no operation\n    END', data_only=data_only)
                node = model.sections[0].body[0].body[0].body[0]
                expected = Break([Token(Token.BREAK, 'BREAK', 4, 8)], errors=('BREAK can only be used inside a loop.',))
                remove_non_data_nodes_and_assert(node, expected, data_only)

class TestContinue(unittest.TestCase):

    def test_in_test_case_body(self):
        if False:
            i = 10
            return i + 15
        for data_only in [True, False]:
            with self.subTest(data_only=data_only):
                model = get_model('*** Test Cases ***\nExample\n    CONTINUE', data_only=data_only)
                node = model.sections[0].body[0].body[0]
                expected = Error([Token(Token.ERROR, 'CONTINUE', 3, 4, 'CONTINUE is not allowed in this context.')])
                remove_non_data_nodes_and_assert(node, expected, data_only)

    def test_in_if_test_case_body(self):
        if False:
            for i in range(10):
                print('nop')
        for data_only in [True, False]:
            with self.subTest(data_only=data_only):
                model = get_model('*** Test Cases ***\nExample\n    IF    True\n        CONTINUE\n    END', data_only=data_only)
                node = model.sections[0].body[0].body[0].body[0]
                expected = Continue([Token(Token.CONTINUE, 'CONTINUE', 4, 8)], errors=('CONTINUE can only be used inside a loop.',))
                remove_non_data_nodes_and_assert(node, expected, data_only)

    def test_in_try_test_case_body(self):
        if False:
            for i in range(10):
                print('nop')
        for data_only in [True, False]:
            with self.subTest(data_only=data_only):
                model = get_model('*** Test Cases ***\nExample\n    TRY    \n        CONTINUE\n    EXCEPT\n        no operation\n    END', data_only=data_only)
                node = model.sections[0].body[0].body[0].body[0]
                expected = Continue([Token(Token.CONTINUE, 'CONTINUE', 4, 8)], errors=('CONTINUE can only be used inside a loop.',))
                remove_non_data_nodes_and_assert(node, expected, data_only)

    def test_in_finally_inside_loop(self):
        if False:
            return 10
        for data_only in [True, False]:
            with self.subTest(data_only=data_only):
                model = get_model('*** Test Cases ***\nExample\n    WHILE    True\n        TRY    \n            Fail\n        EXCEPT\n            no operation\n        FINALLY\n           CONTINUE\n        END     \n    END', data_only=data_only)
                node = model.sections[0].body[0].body[0].body[0].next.next.body[0]
                expected = Continue([Token(Token.CONTINUE, 'CONTINUE', 9, 11)], errors=('CONTINUE cannot be used in FINALLY branch.',))
                remove_non_data_nodes_and_assert(node, expected, data_only)

    def test_in_uk_body(self):
        if False:
            return 10
        for data_only in [True, False]:
            with self.subTest(data_only=data_only):
                model = get_model('*** Keywords ***\nExample\n    CONTINUE', data_only=data_only)
                node = model.sections[0].body[0].body[0]
                expected = Error([Token(Token.ERROR, 'CONTINUE', 3, 4, 'CONTINUE is not allowed in this context.')])
                remove_non_data_nodes_and_assert(node, expected, data_only)

    def test_in_if_uk_body(self):
        if False:
            i = 10
            return i + 15
        for data_only in [True, False]:
            with self.subTest(data_only=data_only):
                model = get_model('*** Keywords ***\nExample\n    IF    True\n        CONTINUE\n    END', data_only=data_only)
                node = model.sections[0].body[0].body[0].body[0]
                expected = Continue([Token(Token.CONTINUE, 'CONTINUE', 4, 8)], errors=('CONTINUE can only be used inside a loop.',))
                remove_non_data_nodes_and_assert(node, expected, data_only)

    def test_in_try_uk_body(self):
        if False:
            i = 10
            return i + 15
        for data_only in [True, False]:
            with self.subTest(data_only=data_only):
                model = get_model('*** Keywords ***\nExample\n    TRY    \n        CONTINUE\n    EXCEPT\n        no operation\n    END', data_only=data_only)
                node = model.sections[0].body[0].body[0].body[0]
                expected = Continue([Token(Token.CONTINUE, 'CONTINUE', 4, 8)], errors=('CONTINUE can only be used inside a loop.',))
                remove_non_data_nodes_and_assert(node, expected, data_only)
if __name__ == '__main__':
    unittest.main()