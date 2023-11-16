"""Tests for yapf.blank_line_calculator."""
import textwrap
import unittest
from yapf.yapflib import reformatter
from yapf.yapflib import style
from yapf.yapflib import yapf_api
from yapftests import yapf_test_helper

class BasicBlankLineCalculatorTest(yapf_test_helper.YAPFTest):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        style.SetGlobalStyle(style.CreateYapfStyle())

    def testDecorators(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        @bork()\n\n        def foo():\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        @bork()\n        def foo():\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testComplexDecorators(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        import sys\n        @bork()\n\n        def foo():\n          pass\n        @fork()\n\n        class moo(object):\n          @bar()\n          @baz()\n\n          def method(self):\n            pass\n    ')
        expected_formatted_code = textwrap.dedent('        import sys\n\n\n        @bork()\n        def foo():\n          pass\n\n\n        @fork()\n        class moo(object):\n\n          @bar()\n          @baz()\n          def method(self):\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testCodeAfterFunctionsAndClasses(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent('        def foo():\n          pass\n        top_level_code = True\n        class moo(object):\n          def method_1(self):\n            pass\n          ivar_a = 42\n          ivar_b = 13\n          def method_2(self):\n            pass\n        try:\n          raise Error\n        except Error as error:\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        def foo():\n          pass\n\n\n        top_level_code = True\n\n\n        class moo(object):\n\n          def method_1(self):\n            pass\n\n          ivar_a = 42\n          ivar_b = 13\n\n          def method_2(self):\n            pass\n\n\n        try:\n          raise Error\n        except Error as error:\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testCommentSpacing(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent("        # This is the first comment\n        # And it's multiline\n\n        # This is the second comment\n\n        def foo():\n          pass\n\n        # multiline before a\n        # class definition\n\n        # This is the second comment\n\n        class qux(object):\n          pass\n\n\n        # An attached comment.\n        class bar(object):\n          '''class docstring'''\n          # Comment attached to\n          # function\n          def foo(self):\n            '''Another docstring.'''\n            # Another multiline\n            # comment\n            pass\n    ")
        expected_formatted_code = textwrap.dedent("        # This is the first comment\n        # And it's multiline\n\n        # This is the second comment\n\n\n        def foo():\n          pass\n\n\n        # multiline before a\n        # class definition\n\n        # This is the second comment\n\n\n        class qux(object):\n          pass\n\n\n        # An attached comment.\n        class bar(object):\n          '''class docstring'''\n\n          # Comment attached to\n          # function\n          def foo(self):\n            '''Another docstring.'''\n            # Another multiline\n            # comment\n            pass\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testCommentBeforeMethod(self):
        if False:
            return 10
        code = textwrap.dedent('        class foo(object):\n\n          # pylint: disable=invalid-name\n          def f(self):\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testCommentsBeforeClassDefs(self):
        if False:
            return 10
        code = textwrap.dedent('        """Test."""\n\n        # Comment\n\n\n        class Foo(object):\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testCommentsBeforeDecorator(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent('        # The @foo operator adds bork to a().\n        @foo()\n        def a():\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))
        code = textwrap.dedent('        # Hello world\n\n\n        @foo()\n        def a():\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testCommentsAfterDecorator(self):
        if False:
            for i in range(10):
                print('nop')
        code = textwrap.dedent('        class _():\n\n          def _():\n            pass\n\n          @pytest.mark.xfail(reason="#709 and #710")\n          # also\n          #@pytest.mark.xfail(setuptools.tests.is_ascii,\n          #    reason="https://github.com/pypa/setuptools/issues/706")\n          def test_unicode_filename_in_sdist(self, sdist_unicode, tmpdir, monkeypatch):\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testInnerClasses(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('      class DeployAPIClient(object):\n          class Error(Exception): pass\n\n          class TaskValidationError(Error): pass\n\n          class DeployAPIHTTPError(Error): pass\n    ')
        expected_formatted_code = textwrap.dedent('      class DeployAPIClient(object):\n\n        class Error(Exception):\n          pass\n\n        class TaskValidationError(Error):\n          pass\n\n        class DeployAPIHTTPError(Error):\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testLinesOnRangeBoundary(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        def A():\n          pass\n\n        def B():  # 4\n          pass  # 5\n\n        def C():\n          pass\n        def D():  # 9\n          pass  # 10\n        def E():\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        def A():\n          pass\n\n\n        def B():  # 4\n          pass  # 5\n\n        def C():\n          pass\n\n\n        def D():  # 9\n          pass  # 10\n        def E():\n          pass\n    ')
        (code, changed) = yapf_api.FormatCode(unformatted_code, lines=[(4, 5), (9, 10)])
        self.assertCodeEqual(expected_formatted_code, code)
        self.assertTrue(changed)

    def testLinesRangeBoundaryNotOutside(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        def A():\n          pass\n\n\n\n        def B():  # 6\n          pass  # 7\n\n\n\n        def C():\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        def A():\n          pass\n\n\n\n        def B():  # 6\n          pass  # 7\n\n\n\n        def C():\n          pass\n    ')
        (code, changed) = yapf_api.FormatCode(unformatted_code, lines=[(6, 7)])
        self.assertCodeEqual(expected_formatted_code, code)
        self.assertFalse(changed)

    def testLinesRangeRemove(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        def A():\n          pass\n\n\n\n        def B():  # 6\n          pass  # 7\n\n\n\n\n        def C():\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        def A():\n          pass\n\n\n        def B():  # 6\n          pass  # 7\n\n\n\n\n        def C():\n          pass\n    ')
        (code, changed) = yapf_api.FormatCode(unformatted_code, lines=[(5, 9)])
        self.assertCodeEqual(expected_formatted_code, code)
        self.assertTrue(changed)

    def testLinesRangeRemoveSome(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        def A():\n          pass\n\n\n\n\n        def B():  # 7\n          pass  # 8\n\n\n\n\n        def C():\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        def A():\n          pass\n\n\n\n        def B():  # 7\n          pass  # 8\n\n\n\n\n        def C():\n          pass\n    ')
        (code, changed) = yapf_api.FormatCode(unformatted_code, lines=[(6, 9)])
        self.assertCodeEqual(expected_formatted_code, code)
        self.assertTrue(changed)
if __name__ == '__main__':
    unittest.main()