"""Basic tests for yapf.reformatter."""
import sys
import textwrap
import unittest
from yapf.yapflib import reformatter
from yapf.yapflib import style
from yapftests import yapf_test_helper

class BasicReformatterTest(yapf_test_helper.YAPFTest):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        style.SetGlobalStyle(style.CreateYapfStyle())

    def testSplittingAllArgs(self):
        if False:
            i = 10
            return i + 15
        style.SetGlobalStyle(style.CreateStyleFromConfig('{split_all_comma_separated_values: true, column_limit: 40}'))
        unformatted_code = textwrap.dedent('        responseDict = {"timestamp": timestamp, "someValue":   value, "whatever": 120}\n    ')
        expected_formatted_code = textwrap.dedent('        responseDict = {\n            "timestamp": timestamp,\n            "someValue": value,\n            "whatever": 120\n        }\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent("        yes = { 'yes': 'no', 'no': 'yes', }\n    ")
        expected_formatted_code = textwrap.dedent("        yes = {\n            'yes': 'no',\n            'no': 'yes',\n        }\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent('        def foo(long_arg, really_long_arg, really_really_long_arg, cant_keep_all_these_args):\n            pass\n    ')
        expected_formatted_code = textwrap.dedent('        def foo(long_arg,\n                really_long_arg,\n                really_really_long_arg,\n                cant_keep_all_these_args):\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent('        foo_tuple = [long_arg, really_long_arg, really_really_long_arg, cant_keep_all_these_args]\n    ')
        expected_formatted_code = textwrap.dedent('        foo_tuple = [\n            long_arg,\n            really_long_arg,\n            really_really_long_arg,\n            cant_keep_all_these_args\n        ]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent('        foo_tuple = [short, arg]\n    ')
        expected_formatted_code = textwrap.dedent('        foo_tuple = [short, arg]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent('        values = [ lambda arg1, arg2: arg1 + arg2 ]\n    ')
        expected_formatted_code = textwrap.dedent('        values = [\n            lambda arg1, arg2: arg1 + arg2\n        ]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent('        values = [\n            (some_arg1, some_arg2) for some_arg1, some_arg2 in values\n        ]\n    ')
        expected_formatted_code = textwrap.dedent('        values = [\n            (some_arg1,\n             some_arg2)\n            for some_arg1, some_arg2 in values\n        ]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent('        someLongFunction(this_is_a_very_long_parameter,\n            abc=(a, this_will_just_fit_xxxxxxx))\n    ')
        expected_formatted_code = textwrap.dedent('        someLongFunction(\n            this_is_a_very_long_parameter,\n            abc=(a,\n                 this_will_just_fit_xxxxxxx))\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testSplittingTopLevelAllArgs(self):
        if False:
            for i in range(10):
                print('nop')
        style_dict = style.CreateStyleFromConfig('{split_all_top_level_comma_separated_values: true, column_limit: 40}')
        style.SetGlobalStyle(style_dict)
        unformatted_code = textwrap.dedent('        responseDict = {"timestamp": timestamp, "someValue":   value, "whatever": 120}\n    ')
        expected_formatted_code = textwrap.dedent('        responseDict = {\n            "timestamp": timestamp,\n            "someValue": value,\n            "whatever": 120\n        }\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent('        def foo(long_arg, really_long_arg, really_really_long_arg, cant_keep_all_these_args):\n              pass\n    ')
        expected_formatted_code = textwrap.dedent('        def foo(long_arg,\n                really_long_arg,\n                really_really_long_arg,\n                cant_keep_all_these_args):\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent('        foo_tuple = [long_arg, really_long_arg, really_really_long_arg, cant_keep_all_these_args]\n    ')
        expected_formatted_code = textwrap.dedent('        foo_tuple = [\n            long_arg,\n            really_long_arg,\n            really_really_long_arg,\n            cant_keep_all_these_args\n        ]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent('        foo_tuple = [short, arg]\n    ')
        expected_formatted_code = textwrap.dedent('        foo_tuple = [short, arg]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent('        values = [ lambda arg1, arg2: arg1 + arg2 ]\n    ')
        expected_formatted_code = textwrap.dedent('        values = [\n            lambda arg1, arg2: arg1 + arg2\n        ]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent('        values = [\n            (some_arg1, some_arg2) for some_arg1, some_arg2 in values\n        ]\n    ')
        expected_formatted_code = textwrap.dedent('        values = [\n            (some_arg1, some_arg2)\n            for some_arg1, some_arg2 in values\n        ]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent('        someLongFunction(this_is_a_very_long_parameter,\n            abc=(a, this_will_just_fit_xxxxxxx))\n    ')
        expected_formatted_code = textwrap.dedent('        someLongFunction(\n            this_is_a_very_long_parameter,\n            abc=(a, this_will_just_fit_xxxxxxx))\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        actual_formatted_code = reformatter.Reformat(llines)
        self.assertEqual(40, len(actual_formatted_code.splitlines()[-1]))
        self.assertCodeEqual(expected_formatted_code, actual_formatted_code)
        unformatted_code = textwrap.dedent('        someLongFunction(this_is_a_very_long_parameter,\n            abc=(a, this_will_not_fit_xxxxxxxxx))\n    ')
        expected_formatted_code = textwrap.dedent('        someLongFunction(\n            this_is_a_very_long_parameter,\n            abc=(a,\n                 this_will_not_fit_xxxxxxxxx))\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        original_multiline = style_dict['FORCE_MULTILINE_DICT']
        style_dict['FORCE_MULTILINE_DICT'] = False
        style.SetGlobalStyle(style_dict)
        unformatted_code = textwrap.dedent('          someLongFunction(this_is_a_very_long_parameter,\n              abc={a: b, b: c})\n          ')
        expected_formatted_code = textwrap.dedent('          someLongFunction(\n              this_is_a_very_long_parameter,\n              abc={\n                  a: b, b: c\n              })\n          ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        actual_formatted_code = reformatter.Reformat(llines)
        self.assertCodeEqual(expected_formatted_code, actual_formatted_code)
        style_dict['FORCE_MULTILINE_DICT'] = True
        style.SetGlobalStyle(style_dict)
        unformatted_code = textwrap.dedent('          someLongFunction(this_is_a_very_long_parameter,\n              abc={a: b, b: c})\n          ')
        expected_formatted_code = textwrap.dedent('          someLongFunction(\n              this_is_a_very_long_parameter,\n              abc={\n                  a: b,\n                  b: c\n              })\n          ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        actual_formatted_code = reformatter.Reformat(llines)
        self.assertCodeEqual(expected_formatted_code, actual_formatted_code)
        style_dict['FORCE_MULTILINE_DICT'] = original_multiline
        style.SetGlobalStyle(style_dict)
        unformatted_code = textwrap.dedent('        a, b = f(\n            a_very_long_parameter, yet_another_one, and_another)\n    ')
        expected_formatted_code = textwrap.dedent('        a, b = f(\n            a_very_long_parameter, yet_another_one, and_another)\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent("        KO = {\n            'ABC': Abc, # abc\n            'DEF': Def, # def\n            'LOL': Lol, # wtf\n            'GHI': Ghi,\n            'JKL': Jkl,\n        }\n    ")
        expected_formatted_code = textwrap.dedent("        KO = {\n            'ABC': Abc,  # abc\n            'DEF': Def,  # def\n            'LOL': Lol,  # wtf\n            'GHI': Ghi,\n            'JKL': Jkl,\n        }\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testSimpleFunctionsWithTrailingComments(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent("        def g():  # Trailing comment\n          if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and\n              xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n            pass\n\n        def f(  # Intermediate comment\n        ):\n          if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and\n              xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n            pass\n    ")
        expected_formatted_code = textwrap.dedent("        def g():  # Trailing comment\n          if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and\n              xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n            pass\n\n\n        def f(  # Intermediate comment\n        ):\n          if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and\n              xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n            pass\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testParamListWithTrailingComments(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        def f(a,\n              b, #\n              c):\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        def f(a, b,  #\n              c):\n          pass\n    ')
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: yapf, disable_split_list_with_comment: True}'))
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())

    def testBlankLinesBetweenTopLevelImportsAndVariables(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent("        import foo as bar\n        VAR = 'baz'\n    ")
        expected_formatted_code = textwrap.dedent("        import foo as bar\n\n        VAR = 'baz'\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent("        import foo as bar\n\n        VAR = 'baz'\n    ")
        expected_formatted_code = textwrap.dedent("        import foo as bar\n\n\n        VAR = 'baz'\n    ")
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: yapf, blank_lines_between_top_level_imports_and_variables: 2}'))
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())
        unformatted_code = textwrap.dedent('        import foo as bar\n        # Some comment\n    ')
        expected_formatted_code = textwrap.dedent('        import foo as bar\n        # Some comment\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent('        import foo as bar\n        class Baz():\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        import foo as bar\n\n\n        class Baz():\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent('        import foo as bar\n        def foobar():\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        import foo as bar\n\n\n        def foobar():\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent('        def foobar():\n          from foo import Bar\n          Bar.baz()\n    ')
        expected_formatted_code = textwrap.dedent('        def foobar():\n          from foo import Bar\n          Bar.baz()\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testBlankLinesAtEndOfFile(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        def foobar(): # foo\n         pass\n\n\n\n    ')
        expected_formatted_code = textwrap.dedent('        def foobar():  # foo\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent("        x = {  'a':37,'b':42,\n\n        'c':927}\n\n    ")
        expected_formatted_code = textwrap.dedent("        x = {'a': 37, 'b': 42, 'c': 927}\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testIndentBlankLines(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        class foo(object):\n\n          def foobar(self):\n\n            pass\n\n          def barfoo(self, x, y):  # bar\n\n            if x:\n\n              return y\n\n\n        def bar():\n\n          return 0\n    ')
        expected_formatted_code = 'class foo(object):\n  \n  def foobar(self):\n    \n    pass\n  \n  def barfoo(self, x, y):  # bar\n    \n    if x:\n      \n      return y\n\n\ndef bar():\n  \n  return 0\n'
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: yapf, indent_blank_lines: true}'))
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())
        (unformatted_code, expected_formatted_code) = (expected_formatted_code, unformatted_code)
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testMultipleUgliness(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent("        x = {  'a':37,'b':42,\n\n        'c':927}\n\n        y = 'hello ''world'\n        z = 'hello '+'world'\n        a = 'hello {}'.format('world')\n        class foo  (     object  ):\n          def f    (self   ):\n            return       37*-+2\n          def g(self, x,y=42):\n              return y\n        def f  (   a ) :\n          return      37+-+a[42-x :  y**3]\n    ")
        expected_formatted_code = textwrap.dedent("        x = {'a': 37, 'b': 42, 'c': 927}\n\n        y = 'hello ' 'world'\n        z = 'hello ' + 'world'\n        a = 'hello {}'.format('world')\n\n\n        class foo(object):\n\n          def f(self):\n            return 37 * -+2\n\n          def g(self, x, y=42):\n            return y\n\n\n        def f(a):\n          return 37 + -+a[42 - x:y**3]\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testComments(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent('        class Foo(object):\n          pass\n\n        # Attached comment\n        class Bar(object):\n          pass\n\n        global_assignment = 42\n\n        # Comment attached to class with decorator.\n        # Comment attached to class with decorator.\n        @noop\n        @noop\n        class Baz(object):\n          pass\n\n        # Intermediate comment\n\n        class Qux(object):\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        class Foo(object):\n          pass\n\n\n        # Attached comment\n        class Bar(object):\n          pass\n\n\n        global_assignment = 42\n\n\n        # Comment attached to class with decorator.\n        # Comment attached to class with decorator.\n        @noop\n        @noop\n        class Baz(object):\n          pass\n\n\n        # Intermediate comment\n\n\n        class Qux(object):\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testSingleComment(self):
        if False:
            return 10
        code = textwrap.dedent('        # Thing 1\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testCommentsWithTrailingSpaces(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        # Thing 1    \n# Thing 2    \n')
        expected_formatted_code = textwrap.dedent('        # Thing 1\n        # Thing 2\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testCommentsInDataLiteral(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent("        def f():\n          return collections.OrderedDict({\n              # First comment.\n              'fnord': 37,\n\n              # Second comment.\n              # Continuation of second comment.\n              'bork': 42,\n\n              # Ending comment.\n          })\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testEndingWhitespaceAfterSimpleStatement(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent('        import foo as bar\n        # Thing 1\n        # Thing 2\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testDocstrings(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        u"""Module-level docstring."""\n        import os\n        class Foo(object):\n\n          """Class-level docstring."""\n          # A comment for qux.\n          def qux(self):\n\n\n            """Function-level docstring.\n\n            A multiline function docstring.\n            """\n            print(\'hello {}\'.format(\'world\'))\n            return 42\n    ')
        expected_formatted_code = textwrap.dedent('        u"""Module-level docstring."""\n        import os\n\n\n        class Foo(object):\n          """Class-level docstring."""\n\n          # A comment for qux.\n          def qux(self):\n            """Function-level docstring.\n\n            A multiline function docstring.\n            """\n            print(\'hello {}\'.format(\'world\'))\n            return 42\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testDocstringAndMultilineComment(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        """Hello world"""\n        # A multiline\n        # comment\n        class bar(object):\n          """class docstring"""\n          # class multiline\n          # comment\n          def foo(self):\n            """Another docstring."""\n            # Another multiline\n            # comment\n            pass\n    ')
        expected_formatted_code = textwrap.dedent('        """Hello world"""\n\n\n        # A multiline\n        # comment\n        class bar(object):\n          """class docstring"""\n\n          # class multiline\n          # comment\n          def foo(self):\n            """Another docstring."""\n            # Another multiline\n            # comment\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testMultilineDocstringAndMultilineComment(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        """Hello world\n\n        RIP Dennis Richie.\n        """\n        # A multiline\n        # comment\n        class bar(object):\n          """class docstring\n\n          A classy class.\n          """\n          # class multiline\n          # comment\n          def foo(self):\n            """Another docstring.\n\n            A functional function.\n            """\n            # Another multiline\n            # comment\n            pass\n    ')
        expected_formatted_code = textwrap.dedent('        """Hello world\n\n        RIP Dennis Richie.\n        """\n\n\n        # A multiline\n        # comment\n        class bar(object):\n          """class docstring\n\n          A classy class.\n          """\n\n          # class multiline\n          # comment\n          def foo(self):\n            """Another docstring.\n\n            A functional function.\n            """\n            # Another multiline\n            # comment\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testTupleCommaBeforeLastParen(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        a = ( 1, )\n    ')
        expected_formatted_code = textwrap.dedent('        a = (1,)\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testNoBreakOutsideOfBracket(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent("        def f():\n          assert port >= minimum, 'Unexpected port %d when minimum was %d.' % (port, minimum)\n        ")
        expected_formatted_code = textwrap.dedent("        def f():\n          assert port >= minimum, 'Unexpected port %d when minimum was %d.' % (port,\n                                                                               minimum)\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testBlankLinesBeforeDecorators(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        @foo()\n        class A(object):\n          @bar()\n          @baz()\n          def x(self):\n            pass\n    ')
        expected_formatted_code = textwrap.dedent('        @foo()\n        class A(object):\n\n          @bar()\n          @baz()\n          def x(self):\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testCommentBetweenDecorators(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        @foo()\n        # frob\n        @bar\n        def x  (self):\n            pass\n    ')
        expected_formatted_code = textwrap.dedent('        @foo()\n        # frob\n        @bar\n        def x(self):\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testListComprehension(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        def given(y):\n            [k for k in ()\n              if k in y]\n    ')
        expected_formatted_code = textwrap.dedent('        def given(y):\n          [k for k in () if k in y]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testListComprehensionPreferOneLine(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        def given(y):\n            long_variable_name = [\n                long_var_name + 1\n                for long_var_name in ()\n                if long_var_name == 2]\n    ')
        expected_formatted_code = textwrap.dedent('        def given(y):\n          long_variable_name = [\n              long_var_name + 1 for long_var_name in () if long_var_name == 2\n          ]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testListComprehensionPreferOneLineOverArithmeticSplit(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        def given(used_identifiers):\n          return (sum(len(identifier)\n                      for identifier in used_identifiers) / len(used_identifiers))\n    ')
        expected_formatted_code = textwrap.dedent('        def given(used_identifiers):\n          return (sum(len(identifier) for identifier in used_identifiers) /\n                  len(used_identifiers))\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testListComprehensionPreferThreeLinesForLineWrap(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        def given(y):\n            long_variable_name = [\n                long_var_name + 1\n                for long_var_name, number_two in ()\n                if long_var_name == 2 and number_two == 3]\n    ')
        expected_formatted_code = textwrap.dedent('        def given(y):\n          long_variable_name = [\n              long_var_name + 1\n              for long_var_name, number_two in ()\n              if long_var_name == 2 and number_two == 3\n          ]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testListComprehensionPreferNoBreakForTrivialExpression(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        def given(y):\n            long_variable_name = [\n                long_var_name\n                for long_var_name, number_two in ()\n                if long_var_name == 2 and number_two == 3]\n    ')
        expected_formatted_code = textwrap.dedent('        def given(y):\n          long_variable_name = [\n              long_var_name for long_var_name, number_two in ()\n              if long_var_name == 2 and number_two == 3\n          ]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testOpeningAndClosingBrackets(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        foo( (1, ) )\n        foo( ( 1, 2, 3  ) )\n        foo( ( 1, 2, 3, ) )\n    ')
        expected_formatted_code = textwrap.dedent('        foo((1,))\n        foo((1, 2, 3))\n        foo((\n            1,\n            2,\n            3,\n        ))\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testSingleLineFunctions(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        def foo():  return 42\n    ')
        expected_formatted_code = textwrap.dedent('        def foo():\n          return 42\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testNoQueueSeletionInMiddleOfLine(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        find_symbol(node.type) + "< " + " ".join(find_pattern(n) for n in node.child) + " >"\n    ')
        expected_formatted_code = textwrap.dedent('        find_symbol(node.type) + "< " + " ".join(\n            find_pattern(n) for n in node.child) + " >"\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testNoSpacesBetweenSubscriptsAndCalls(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        aaaaaaaaaa = bbbbbbbb.ccccccccc() [42] (a, 2)\n    ')
        expected_formatted_code = textwrap.dedent('        aaaaaaaaaa = bbbbbbbb.ccccccccc()[42](a, 2)\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testNoSpacesBetweenOpeningBracketAndStartingOperator(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent('        aaaaaaaaaa = bbbbbbbb.ccccccccc[ -1 ]( -42 )\n    ')
        expected_formatted_code = textwrap.dedent('        aaaaaaaaaa = bbbbbbbb.ccccccccc[-1](-42)\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent('        aaaaaaaaaa = bbbbbbbb.ccccccccc( *varargs )\n        aaaaaaaaaa = bbbbbbbb.ccccccccc( **kwargs )\n    ')
        expected_formatted_code = textwrap.dedent('        aaaaaaaaaa = bbbbbbbb.ccccccccc(*varargs)\n        aaaaaaaaaa = bbbbbbbb.ccccccccc(**kwargs)\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testMultilineCommentReformatted(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        if True:\n            # This is a multiline\n            # comment.\n            pass\n    ')
        expected_formatted_code = textwrap.dedent('        if True:\n          # This is a multiline\n          # comment.\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testDictionaryMakerFormatting(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent("        _PYTHON_STATEMENTS = frozenset({\n            lambda x, y: 'simple_stmt': 'small_stmt', 'expr_stmt': 'print_stmt', 'del_stmt':\n            'pass_stmt', lambda: 'break_stmt': 'continue_stmt', 'return_stmt': 'raise_stmt',\n            'yield_stmt': 'import_stmt', lambda: 'global_stmt': 'exec_stmt', 'assert_stmt':\n            'if_stmt', 'while_stmt': 'for_stmt',\n        })\n    ")
        expected_formatted_code = textwrap.dedent("        _PYTHON_STATEMENTS = frozenset({\n            lambda x, y: 'simple_stmt': 'small_stmt',\n            'expr_stmt': 'print_stmt',\n            'del_stmt': 'pass_stmt',\n            lambda: 'break_stmt': 'continue_stmt',\n            'return_stmt': 'raise_stmt',\n            'yield_stmt': 'import_stmt',\n            lambda: 'global_stmt': 'exec_stmt',\n            'assert_stmt': 'if_stmt',\n            'while_stmt': 'for_stmt',\n        })\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testSimpleMultilineCode(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        if True:\n          aaaaaaaaaaaaaa.bbbbbbbbbbbbbb.ccccccc(zzzzzzzzzzzz, xxxxxxxxxxx, yyyyyyyyyyyy, vvvvvvvvv)\n          aaaaaaaaaaaaaa.bbbbbbbbbbbbbb.ccccccc(zzzzzzzzzzzz, xxxxxxxxxxx, yyyyyyyyyyyy, vvvvvvvvv)\n        ')
        expected_formatted_code = textwrap.dedent('        if True:\n          aaaaaaaaaaaaaa.bbbbbbbbbbbbbb.ccccccc(zzzzzzzzzzzz, xxxxxxxxxxx, yyyyyyyyyyyy,\n                                                vvvvvvvvv)\n          aaaaaaaaaaaaaa.bbbbbbbbbbbbbb.ccccccc(zzzzzzzzzzzz, xxxxxxxxxxx, yyyyyyyyyyyy,\n                                                vvvvvvvvv)\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testMultilineComment(self):
        if False:
            for i in range(10):
                print('nop')
        code = textwrap.dedent('        if Foo:\n          # Hello world\n          # Yo man.\n          # Yo man.\n          # Yo man.\n          # Yo man.\n          a = 42\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testSpaceBetweenStringAndParentheses(self):
        if False:
            return 10
        code = textwrap.dedent("        b = '0' ('hello')\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testMultilineString(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent("        code = textwrap.dedent('''            if Foo:\n              # Hello world\n              # Yo man.\n              # Yo man.\n              # Yo man.\n              # Yo man.\n              a = 42\n            ''')\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent('        def f():\n            email_text += """<html>This is a really long docstring that goes over the column limit and is multi-line.<br><br>\n        <b>Czar: </b>"""+despot["Nicholas"]+"""<br>\n        <b>Minion: </b>"""+serf["Dmitri"]+"""<br>\n        <b>Residence: </b>"""+palace["Winter"]+"""<br>\n        </body>\n        </html>"""\n    ')
        expected_formatted_code = textwrap.dedent('        def f():\n          email_text += """<html>This is a really long docstring that goes over the column limit and is multi-line.<br><br>\n        <b>Czar: </b>""" + despot["Nicholas"] + """<br>\n        <b>Minion: </b>""" + serf["Dmitri"] + """<br>\n        <b>Residence: </b>""" + palace["Winter"] + """<br>\n        </body>\n        </html>"""\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testSimpleMultilineWithComments(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        if (  # This is the first comment\n            a and  # This is the second comment\n            # This is the third comment\n            b):  # A trailing comment\n          # Whoa! A normal comment!!\n          pass  # Another trailing comment\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testMatchingParenSplittingMatching(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent("        def f():\n          raise RuntimeError('unable to find insertion point for target node',\n                             (target,))\n    ")
        expected_formatted_code = textwrap.dedent("        def f():\n          raise RuntimeError('unable to find insertion point for target node',\n                             (target,))\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testContinuationIndent(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent('        class F:\n          def _ProcessArgLists(self, node):\n            """Common method for processing argument lists."""\n            for child in node.children:\n              if isinstance(child, pytree.Leaf):\n                self._SetTokenSubtype(\n                    child, subtype=_ARGLIST_TOKEN_TO_SUBTYPE.get(\n                        child.value, format_token.Subtype.NONE))\n    ')
        expected_formatted_code = textwrap.dedent('        class F:\n\n          def _ProcessArgLists(self, node):\n            """Common method for processing argument lists."""\n            for child in node.children:\n              if isinstance(child, pytree.Leaf):\n                self._SetTokenSubtype(\n                    child,\n                    subtype=_ARGLIST_TOKEN_TO_SUBTYPE.get(child.value,\n                                                          format_token.Subtype.NONE))\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testTrailingCommaAndBracket(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        a = { 42, }\n        b = ( 42, )\n        c = [ 42, ]\n    ')
        expected_formatted_code = textwrap.dedent('        a = {\n            42,\n        }\n        b = (42,)\n        c = [\n            42,\n        ]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testI18n(self):
        if False:
            for i in range(10):
                print('nop')
        code = textwrap.dedent("        N_('Some years ago - never mind how long precisely - having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world.')  # A comment is here.\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))
        code = textwrap.dedent("        foo('Fake function call')  #. Some years ago - never mind how long precisely - having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world.\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testI18nCommentsInDataLiteral(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent("        def f():\n          return collections.OrderedDict({\n              #. First i18n comment.\n              'bork': 'foo',\n\n              #. Second i18n comment.\n              'snork': 'bar#.*=\\\\0',\n          })\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testClosingBracketIndent(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent("        def f():\n\n          def g():\n            while (xxxxxxxxxxxxxxxxxxxxx(yyyyyyyyyyyyy[zzzzz]) == 'aaaaaaaaaaa' and\n                   xxxxxxxxxxxxxxxxxxxxx(\n                       yyyyyyyyyyyyy[zzzzz].aaaaaaaa[0]) == 'bbbbbbb'):\n              pass\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testClosingBracketsInlinedInCall(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        class Foo(object):\n\n          def bar(self):\n            self.aaaaaaaa = xxxxxxxxxxxxxxxxxxx.yyyyyyyyyyyyy(\n                self.cccccc.ddddddddd.eeeeeeee,\n                options={\n                    "forkforkfork": 1,\n                    "borkborkbork": 2,\n                    "corkcorkcork": 3,\n                    "horkhorkhork": 4,\n                    "porkporkpork": 5,\n                    })\n    ')
        expected_formatted_code = textwrap.dedent('        class Foo(object):\n\n          def bar(self):\n            self.aaaaaaaa = xxxxxxxxxxxxxxxxxxx.yyyyyyyyyyyyy(\n                self.cccccc.ddddddddd.eeeeeeee,\n                options={\n                    "forkforkfork": 1,\n                    "borkborkbork": 2,\n                    "corkcorkcork": 3,\n                    "horkhorkhork": 4,\n                    "porkporkpork": 5,\n                })\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testLineWrapInForExpression(self):
        if False:
            return 10
        code = textwrap.dedent('        class A:\n\n          def x(self, node, name, n=1):\n            for i, child in enumerate(\n                itertools.ifilter(lambda c: pytree_utils.NodeName(c) == name,\n                                  node.pre_order())):\n              pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testFunctionCallContinuationLine(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent('        class foo:\n\n          def bar(self, node, name, n=1):\n            if True:\n              if True:\n                return [(aaaaaaaaaa,\n                         bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb(\n                             cccc, ddddddddddddddddddddddddddddddddddddd))]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testI18nNonFormatting(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent("        class F(object):\n\n          def __init__(self, fieldname,\n                       #. Error message indicating an invalid e-mail address.\n                       message=N_('Please check your email address.'), **kwargs):\n            pass\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testNoSpaceBetweenUnaryOpAndOpeningParen(self):
        if False:
            return 10
        code = textwrap.dedent('        if ~(a or b):\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testCommentBeforeFuncDef(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        class Foo(object):\n\n          a = 42\n\n          # This is a comment.\n          def __init__(self,\n                       xxxxxxx,\n                       yyyyy=0,\n                       zzzzzzz=None,\n                       aaaaaaaaaaaaaaaaaa=False,\n                       bbbbbbbbbbbbbbb=False):\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testExcessLineCountWithDefaultKeywords(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        class Fnord(object):\n          def Moo(self):\n            aaaaaaaaaaaaaaaa = self._bbbbbbbbbbbbbbbbbbbbbbb(\n                ccccccccccccc=ccccccccccccc, ddddddd=ddddddd, eeee=eeee,\n                fffff=fffff, ggggggg=ggggggg, hhhhhhhhhhhhh=hhhhhhhhhhhhh,\n                iiiiiii=iiiiiiiiiiiiii)\n    ')
        expected_formatted_code = textwrap.dedent('        class Fnord(object):\n\n          def Moo(self):\n            aaaaaaaaaaaaaaaa = self._bbbbbbbbbbbbbbbbbbbbbbb(\n                ccccccccccccc=ccccccccccccc,\n                ddddddd=ddddddd,\n                eeee=eeee,\n                fffff=fffff,\n                ggggggg=ggggggg,\n                hhhhhhhhhhhhh=hhhhhhhhhhhhh,\n                iiiiiii=iiiiiiiiiiiiii)\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testSpaceAfterNotOperator(self):
        if False:
            return 10
        code = textwrap.dedent('        if not (this and that):\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testNoPenaltySplitting(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent('        def f():\n          if True:\n            if True:\n              python_files.extend(\n                  os.path.join(filename, f)\n                  for f in os.listdir(filename)\n                  if IsPythonFile(os.path.join(filename, f)))\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testExpressionPenalties(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent("        def f():\n          if ((left.value == '(' and right.value == ')') or\n              (left.value == '[' and right.value == ']') or\n              (left.value == '{' and right.value == '}')):\n            return False\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testLineDepthOfSingleLineStatement(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        while True: continue\n        for x in range(3): continue\n        try: a = 42\n        except: b = 42\n        with open(a) as fd: a = fd.read()\n    ')
        expected_formatted_code = textwrap.dedent('        while True:\n          continue\n        for x in range(3):\n          continue\n        try:\n          a = 42\n        except:\n          b = 42\n        with open(a) as fd:\n          a = fd.read()\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testSplitListWithTerminatingComma(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent("        FOO = ['bar', 'baz', 'mux', 'qux', 'quux', 'quuux', 'quuuux',\n          'quuuuux', 'quuuuuux', 'quuuuuuux', lambda a, b: 37,]\n    ")
        expected_formatted_code = textwrap.dedent("        FOO = [\n            'bar',\n            'baz',\n            'mux',\n            'qux',\n            'quux',\n            'quuux',\n            'quuuux',\n            'quuuuux',\n            'quuuuuux',\n            'quuuuuuux',\n            lambda a, b: 37,\n        ]\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testSplitListWithInterspersedComments(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent("        FOO = [\n            'bar',  # bar\n            'baz',  # baz\n            'mux',  # mux\n            'qux',  # qux\n            'quux',  # quux\n            'quuux',  # quuux\n            'quuuux',  # quuuux\n            'quuuuux',  # quuuuux\n            'quuuuuux',  # quuuuuux\n            'quuuuuuux',  # quuuuuuux\n            lambda a, b: 37  # lambda\n        ]\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testRelativeImportStatements(self):
        if False:
            return 10
        code = textwrap.dedent('        from ... import bork\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testSingleLineList(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb = aaaaaaaaaaa(\n            ("...", "."), "..",\n            ".............................................."\n        )\n    ')
        expected_formatted_code = textwrap.dedent('        bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb = aaaaaaaaaaa(\n            ("...", "."), "..", "..............................................")\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testBlankLinesBeforeFunctionsNotInColumnZero(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        import signal\n\n\n        try:\n          signal.SIGALRM\n          # ..................................................................\n          # ...............................................................\n\n\n          def timeout(seconds=1):\n            pass\n        except:\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        import signal\n\n        try:\n          signal.SIGALRM\n\n          # ..................................................................\n          # ...............................................................\n\n\n          def timeout(seconds=1):\n            pass\n        except:\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testNoKeywordArgumentBreakage(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        class A(object):\n\n          def b(self):\n            if self.aaaaaaaaaaaaaaaaaaaa not in self.bbbbbbbbbb(\n                cccccccccccccccccccc=True):\n              pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testTrailerOnSingleLine(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent("        urlpatterns = patterns('', url(r'^$', 'homepage_view'),\n                               url(r'^/login/$', 'login_view'),\n                               url(r'^/login/$', 'logout_view'),\n                               url(r'^/user/(?P<username>\\w+)/$', 'profile_view'))\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testIfConditionalParens(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        class Foo:\n\n          def bar():\n            if True:\n              if (child.type == grammar_token.NAME and\n                  child.value in substatement_names):\n                pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testContinuationMarkers(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent('        text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec a diam lectus. "\\\n               "Sed sit amet ipsum mauris. Maecenas congue ligula ac quam viverra nec consectetur "\\\n               "ante hendrerit. Donec et mollis dolor. Praesent et diam eget libero egestas mattis "\\\n               "sit amet vitae augue. Nam tincidunt congue enim, ut porta lorem lacinia consectetur. "\\\n               "Donec ut libero sed arcu vehicula ultricies a non tortor. Lorem ipsum dolor sit amet"\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))
        code = textwrap.dedent('        from __future__ import nested_scopes, generators, division, absolute_import, with_statement, \\\n            print_function, unicode_literals\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))
        code = textwrap.dedent('        if aaaaaaaaa == 42 and bbbbbbbbbbbbbb == 42 and \\\n           cccccccc == 42:\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testCommentsWithContinuationMarkers(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        def fn(arg):\n          v = fn2(key1=True,\n                  #c1\n                  key2=arg)\\\n                        .fn3()\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testMultipleContinuationMarkers(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent('        xyz = \\\n            \\\n            some_thing()\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testContinuationMarkerAfterStringWithContinuation(self):
        if False:
            for i in range(10):
                print('nop')
        code = textwrap.dedent("        s = 'foo \\\n            bar' \\\n            .format()\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testEmptyContainers(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent("        flags.DEFINE_list(\n            'output_dirs', [],\n            'Lorem ipsum dolor sit amet, consetetur adipiscing elit. Donec a diam lectus. '\n            'Sed sit amet ipsum mauris. Maecenas congue.')\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testSplitStringsIfSurroundedByParens(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent("        a = foo.bar({'xxxxxxxxxxxxxxxxxxxxxxx' 'yyyyyyyyyyyyyyyyyyyyyyyyyy': baz[42]} + 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' 'bbbbbbbbbbbbbbbbbbbbbbbbbb' 'cccccccccccccccccccccccccccccccc' 'ddddddddddddddddddddddddddddd')\n    ")
        expected_formatted_code = textwrap.dedent("        a = foo.bar({'xxxxxxxxxxxxxxxxxxxxxxx'\n                     'yyyyyyyyyyyyyyyyyyyyyyyyyy': baz[42]} +\n                    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'\n                    'bbbbbbbbbbbbbbbbbbbbbbbbbb'\n                    'cccccccccccccccccccccccccccccccc'\n                    'ddddddddddddddddddddddddddddd')\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        code = textwrap.dedent("        a = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' 'bbbbbbbbbbbbbbbbbbbbbbbbbb' 'cccccccccccccccccccccccccccccccc' 'ddddddddddddddddddddddddddddd'\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testMultilineShebang(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent('        #!/bin/sh\n        if "true" : \'\'\'\'\n        then\n\n        export FOO=123\n        exec /usr/bin/env python "$0" "$@"\n\n        exit 127\n        fi\n        \'\'\'\n\n        import os\n\n        assert os.environ[\'FOO\'] == \'123\'\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testNoSplittingAroundTermOperators(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent('        a_very_long_function_call_yada_yada_etc_etc_etc(long_arg1,\n                                                        long_arg2 / long_arg3)\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testNoSplittingAroundCompOperators(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent('        c = (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa is not bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb)\n        c = (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa in bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb)\n        c = (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa not in bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb)\n\n        c = (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa is bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb)\n        c = (aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa <= bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb)\n    ')
        expected_code = textwrap.dedent('        c = (\n            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n            is not bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb)\n        c = (\n            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n            in bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb)\n        c = (\n            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n            not in bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb)\n\n        c = (\n            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n            is bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb)\n        c = (\n            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n            <= bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb)\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(expected_code, reformatter.Reformat(llines))

    def testNoSplittingWithinSubscriptList(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent("        somequitelongvariablename.somemember[(a, b)] = {\n            'somelongkey': 1,\n            'someotherlongkey': 2\n        }\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testExcessCharacters(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent("        class foo:\n\n          def bar(self):\n            self.write(s=[\n                '%s%s %s' % ('many of really', 'long strings', '+ just makes up 81')\n            ])\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent('        def _():\n          if True:\n            if True:\n              if contract == allow_contract and attr_dict.get(if_attribute) == has_value:\n                return True\n    ')
        expected_code = textwrap.dedent('        def _():\n          if True:\n            if True:\n              if contract == allow_contract and attr_dict.get(\n                  if_attribute) == has_value:\n                return True\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_code, reformatter.Reformat(llines))

    def testDictSetGenerator(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent("        foo = {\n            variable: 'hello world. How are you today?'\n            for variable in fnord\n            if variable != 37\n        }\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent('        foo = {\n            x: x\n            for x in fnord\n        }\n    ')
        expected_code = textwrap.dedent('        foo = {x: x for x in fnord}\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_code, reformatter.Reformat(llines))

    def testUnaryOpInDictionaryValue(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        beta = "123"\n\n        test = {\'alpha\': beta[-1]}\n\n        print(beta[-1])\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testUnaryNotOperator(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent('        if True:\n          if True:\n            if True:\n              if True:\n                remote_checksum = self.get_checksum(conn, tmp, dest, inject,\n                                                    not directory_prepended, source)\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testRelaxArraySubscriptAffinity(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent("        class A(object):\n\n          def f(self, aaaaaaaaa, bbbbbbbbbbbbb, row):\n            if True:\n              if True:\n                if True:\n                  if True:\n                    if row[4] is None or row[5] is None:\n                      bbbbbbbbbbbbb[\n                          '..............'] = row[5] if row[5] is not None else 5\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testFunctionCallInDict(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent("        a = {'a': b(c=d, **e)}\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testFunctionCallInNestedDict(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent("        a = {'a': {'a': {'a': b(c=d, **e)}}}\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testUnbreakableNot(self):
        if False:
            return 10
        code = textwrap.dedent('        def test():\n          if not "Foooooooooooooooooooooooooooooo" or "Foooooooooooooooooooooooooooooo" == "Foooooooooooooooooooooooooooooo":\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testSplitListWithComment(self):
        if False:
            for i in range(10):
                print('nop')
        code = textwrap.dedent("        a = [\n            'a',\n            'b',\n            'c'  # hello world\n        ]\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testOverColumnLimit(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent("        class Test:\n\n          def testSomething(self):\n            expected = {\n                ('aaaaaaaaaaaaa', 'bbbb'): 'ccccccccccccccccccccccccccccccccccccccccccc',\n                ('aaaaaaaaaaaaa', 'bbbb'): 'ccccccccccccccccccccccccccccccccccccccccccc',\n                ('aaaaaaaaaaaaa', 'bbbb'): 'ccccccccccccccccccccccccccccccccccccccccccc',\n            }\n    ")
        expected_formatted_code = textwrap.dedent("        class Test:\n\n          def testSomething(self):\n            expected = {\n                ('aaaaaaaaaaaaa', 'bbbb'):\n                    'ccccccccccccccccccccccccccccccccccccccccccc',\n                ('aaaaaaaaaaaaa', 'bbbb'):\n                    'ccccccccccccccccccccccccccccccccccccccccccc',\n                ('aaaaaaaaaaaaa', 'bbbb'):\n                    'ccccccccccccccccccccccccccccccccccccccccccc',\n            }\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testEndingComment(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent('        a = f(\n            a="something",\n            b="something requiring comment which is quite long",  # comment about b (pushes line over 79)\n            c="something else, about which comment doesn\'t make sense")\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testContinuationSpaceRetention(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        def fn():\n          return module \\\n                 .method(Object(data,\n                     fn2(arg)\n                 ))\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testIfExpressionWithFunctionCall(self):
        if False:
            return 10
        code = textwrap.dedent('        if x or z.y(\n            a,\n            c,\n            aaaaaaaaaaaaaaaaaaaaa=aaaaaaaaaaaaaaaaaa,\n            bbbbbbbbbbbbbbbbbbbbb=bbbbbbbbbbbbbbbbbb):\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testUnformattedAfterMultilineString(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent("        def foo():\n          com_text = \\\n        '''\n        TEST\n        ''' % (input_fname, output_fname)\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testNoSpacesAroundKeywordDefaultValues(self):
        if False:
            for i in range(10):
                print('nop')
        code = textwrap.dedent("        sources = {\n            'json': request.get_json(silent=True) or {},\n            'json2': request.get_json(silent=True),\n        }\n        json = request.get_json(silent=True) or {}\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testNoSplittingBeforeEndingSubscriptBracket(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent("        if True:\n          if True:\n            status = cf.describe_stacks(StackName=stackname)[u'Stacks'][0][u'StackStatus']\n    ")
        expected_formatted_code = textwrap.dedent("        if True:\n          if True:\n            status = cf.describe_stacks(\n                StackName=stackname)[u'Stacks'][0][u'StackStatus']\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testNoSplittingOnSingleArgument(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent("        xxxxxxxxxxxxxx = (re.search(r'(\\d+\\.\\d+\\.\\d+\\.)\\d+',\n                                    aaaaaaa.bbbbbbbbbbbb).group(1) +\n                          re.search(r'\\d+\\.\\d+\\.\\d+\\.(\\d+)',\n                                    ccccccc).group(1))\n        xxxxxxxxxxxxxx = (re.search(r'(\\d+\\.\\d+\\.\\d+\\.)\\d+',\n                                    aaaaaaa.bbbbbbbbbbbb).group(a.b) +\n                          re.search(r'\\d+\\.\\d+\\.\\d+\\.(\\d+)',\n                                    ccccccc).group(c.d))\n    ")
        expected_formatted_code = textwrap.dedent("        xxxxxxxxxxxxxx = (\n            re.search(r'(\\d+\\.\\d+\\.\\d+\\.)\\d+', aaaaaaa.bbbbbbbbbbbb).group(1) +\n            re.search(r'\\d+\\.\\d+\\.\\d+\\.(\\d+)', ccccccc).group(1))\n        xxxxxxxxxxxxxx = (\n            re.search(r'(\\d+\\.\\d+\\.\\d+\\.)\\d+', aaaaaaa.bbbbbbbbbbbb).group(a.b) +\n            re.search(r'\\d+\\.\\d+\\.\\d+\\.(\\d+)', ccccccc).group(c.d))\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testSplittingArraysSensibly(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent("        while True:\n          while True:\n            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = list['bbbbbbbbbbbbbbbbbbbbbbbbb'].split(',')\n            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = list('bbbbbbbbbbbbbbbbbbbbbbbbb').split(',')\n    ")
        expected_formatted_code = textwrap.dedent("        while True:\n          while True:\n            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = list[\n                'bbbbbbbbbbbbbbbbbbbbbbbbb'].split(',')\n            aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa = list(\n                'bbbbbbbbbbbbbbbbbbbbbbbbb').split(',')\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testComprehensionForAndIf(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent("        class f:\n\n          def __repr__(self):\n            tokens_repr = ','.join(['{0}({1!r})'.format(tok.name, tok.value) for tok in self._tokens])\n    ")
        expected_formatted_code = textwrap.dedent("        class f:\n\n          def __repr__(self):\n            tokens_repr = ','.join(\n                ['{0}({1!r})'.format(tok.name, tok.value) for tok in self._tokens])\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testFunctionCallArguments(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        def f():\n          if True:\n            pytree_utils.InsertNodesBefore(_CreateCommentsFromPrefix(\n                comment_prefix, comment_lineno, comment_column,\n                standalone=True), ancestor_at_indent)\n            pytree_utils.InsertNodesBefore(_CreateCommentsFromPrefix(\n                comment_prefix, comment_lineno, comment_column,\n                standalone=True))\n    ')
        expected_formatted_code = textwrap.dedent('        def f():\n          if True:\n            pytree_utils.InsertNodesBefore(\n                _CreateCommentsFromPrefix(\n                    comment_prefix, comment_lineno, comment_column, standalone=True),\n                ancestor_at_indent)\n            pytree_utils.InsertNodesBefore(\n                _CreateCommentsFromPrefix(\n                    comment_prefix, comment_lineno, comment_column, standalone=True))\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testBinaryOperators(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        a = b ** 37\n        c = (20 ** -3) / (_GRID_ROWS ** (code_length - 10))\n    ')
        expected_formatted_code = textwrap.dedent('        a = b**37\n        c = (20**-3) / (_GRID_ROWS**(code_length - 10))\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        code = textwrap.dedent("        def f():\n          if True:\n            if (self.stack[-1].split_before_closing_bracket and\n                # FIXME(morbo): Use the 'matching_bracket' instead of this.\n                # FIXME(morbo): Don't forget about tuples!\n                current.value in ']}'):\n              pass\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testContiguousList(self):
        if False:
            for i in range(10):
                print('nop')
        code = textwrap.dedent('        [retval1, retval2] = a_very_long_function(argument_1, argument2, argument_3,\n                                                  argument_4)\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testArgsAndKwargsFormatting(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent('        a(a=aaaaaaaaaaaaaaaaaaaaa,\n          b=aaaaaaaaaaaaaaaaaaaaaaaa,\n          c=aaaaaaaaaaaaaaaaaa,\n          *d,\n          **e)\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))
        code = textwrap.dedent("        def foo():\n          return [\n              Bar(xxx='some string',\n                  yyy='another long string',\n                  zzz='a third long string')\n          ]\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testCommentColumnLimitOverflow(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent("        def f():\n          if True:\n            TaskManager.get_tags = MagicMock(\n                name='get_tags_mock',\n                return_value=[157031694470475],\n                # side_effect=[(157031694470475), (157031694470475),],\n            )\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testMultilineLambdas(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        class SomeClass(object):\n          do_something = True\n\n          def succeeded(self, dddddddddddddd):\n            d = defer.succeed(None)\n\n            if self.do_something:\n              d.addCallback(lambda _: self.aaaaaa.bbbbbbbbbbbbbbbb.cccccccccccccccccccccccccccccccc(dddddddddddddd))\n            return d\n    ')
        expected_formatted_code = textwrap.dedent('        class SomeClass(object):\n          do_something = True\n\n          def succeeded(self, dddddddddddddd):\n            d = defer.succeed(None)\n\n            if self.do_something:\n              d.addCallback(lambda _: self.aaaaaa.bbbbbbbbbbbbbbbb.\n                            cccccccccccccccccccccccccccccccc(dddddddddddddd))\n            return d\n    ')
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: yapf, allow_multiline_lambdas: true}'))
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())

    def testMultilineDictionaryKeys(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent("        MAP_WITH_LONG_KEYS = {\n            ('lorem ipsum', 'dolor sit amet'):\n                1,\n            ('consectetur adipiscing elit.', 'Vestibulum mauris justo, ornare eget dolor eget'):\n                2,\n            ('vehicula convallis nulla. Vestibulum dictum nisl in malesuada finibus.',):\n                3\n        }\n    ")
        expected_formatted_code = textwrap.dedent("        MAP_WITH_LONG_KEYS = {\n            ('lorem ipsum', 'dolor sit amet'):\n                1,\n            ('consectetur adipiscing elit.',\n             'Vestibulum mauris justo, ornare eget dolor eget'):\n                2,\n            ('vehicula convallis nulla. Vestibulum dictum nisl in malesuada finibus.',):\n                3\n        }\n    ")
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: yapf, allow_multiline_dictionary_keys: true}'))
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())

    def testStableDictionaryFormatting(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent("        class A(object):\n\n          def method(self):\n            filters = {\n                'expressions': [{\n                    'field': {\n                        'search_field': {\n                            'user_field': 'latest_party__number_of_guests'\n                        },\n                    }\n                }]\n            }\n    ")
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: pep8, indent_width: 2, continuation_indent_width: 4, indent_dictionary_value: True}'))
            llines = yapf_test_helper.ParseAndUnwrap(code)
            reformatted_code = reformatter.Reformat(llines)
            self.assertCodeEqual(code, reformatted_code)
            llines = yapf_test_helper.ParseAndUnwrap(reformatted_code)
            reformatted_code = reformatter.Reformat(llines)
            self.assertCodeEqual(code, reformatted_code)
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())

    def testStableInlinedDictionaryFormatting(self):
        if False:
            return 10
        try:
            style.SetGlobalStyle(style.CreatePEP8Style())
            unformatted_code = textwrap.dedent('          def _():\n              url = "http://{0}/axis-cgi/admin/param.cgi?{1}".format(\n                  value, urllib.urlencode({\'action\': \'update\', \'parameter\': value}))\n      ')
            expected_formatted_code = textwrap.dedent('          def _():\n              url = "http://{0}/axis-cgi/admin/param.cgi?{1}".format(\n                  value, urllib.urlencode({\n                      \'action\': \'update\',\n                      \'parameter\': value\n                  }))\n      ')
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            reformatted_code = reformatter.Reformat(llines)
            self.assertCodeEqual(expected_formatted_code, reformatted_code)
            llines = yapf_test_helper.ParseAndUnwrap(reformatted_code)
            reformatted_code = reformatter.Reformat(llines)
            self.assertCodeEqual(expected_formatted_code, reformatted_code)
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())

    def testDontSplitKeywordValueArguments(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        def mark_game_scored(gid):\n          _connect.execute(_games.update().where(_games.c.gid == gid).values(\n              scored=True))\n    ')
        expected_formatted_code = textwrap.dedent('        def mark_game_scored(gid):\n          _connect.execute(\n              _games.update().where(_games.c.gid == gid).values(scored=True))\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testDontAddBlankLineAfterMultilineString(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent('        query = \'\'\'SELECT id\n        FROM table\n        WHERE day in {}\'\'\'\n        days = ",".join(days)\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testFormattingListComprehensions(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent('        def a():\n          if True:\n            if True:\n              if True:\n                columns = [\n                    x for x, y in self._heap_this_is_very_long if x.route[0] == choice\n                ]\n                self._heap = [x for x in self._heap if x.route and x.route[0] == choice]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testNoSplittingWhenBinPacking(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent('        a_very_long_function_name(\n            long_argument_name_1=1,\n            long_argument_name_2=2,\n            long_argument_name_3=3,\n            long_argument_name_4=4,\n        )\n\n        a_very_long_function_name(\n            long_argument_name_1=1, long_argument_name_2=2, long_argument_name_3=3,\n            long_argument_name_4=4\n        )\n    ')
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: pep8, indent_width: 2, continuation_indent_width: 4, indent_dictionary_value: True, dedent_closing_brackets: True, split_before_named_assigns: False}'))
            llines = yapf_test_helper.ParseAndUnwrap(code)
            reformatted_code = reformatter.Reformat(llines)
            self.assertCodeEqual(code, reformatted_code)
            llines = yapf_test_helper.ParseAndUnwrap(reformatted_code)
            reformatted_code = reformatter.Reformat(llines)
            self.assertCodeEqual(code, reformatted_code)
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())

    def testNotSplittingAfterSubscript(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent("        if not aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.b(c == d[\n                'eeeeee']).ffffff():\n          pass\n    ")
        expected_formatted_code = textwrap.dedent("        if not aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.b(\n            c == d['eeeeee']).ffffff():\n          pass\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testSplittingOneArgumentList(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        def _():\n          if True:\n            if True:\n              if True:\n                if True:\n                  if True:\n                    boxes[id_] = np.concatenate((points.min(axis=0), qoints.max(axis=0)))\n    ')
        expected_formatted_code = textwrap.dedent('        def _():\n          if True:\n            if True:\n              if True:\n                if True:\n                  if True:\n                    boxes[id_] = np.concatenate(\n                        (points.min(axis=0), qoints.max(axis=0)))\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testSplittingBeforeFirstElementListArgument(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent('        class _():\n          @classmethod\n          def _pack_results_for_constraint_or(cls, combination, constraints):\n            if True:\n              if True:\n                if True:\n                  return cls._create_investigation_result(\n                          (\n                                  clue for clue in combination if not clue == Verifier.UNMATCHED\n                          ), constraints, InvestigationResult.OR\n                  )\n    ')
        expected_formatted_code = textwrap.dedent('        class _():\n\n          @classmethod\n          def _pack_results_for_constraint_or(cls, combination, constraints):\n            if True:\n              if True:\n                if True:\n                  return cls._create_investigation_result(\n                      (clue for clue in combination if not clue == Verifier.UNMATCHED),\n                      constraints, InvestigationResult.OR)\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testSplittingArgumentsTerminatedByComma(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        function_name(argument_name_1=1, argument_name_2=2, argument_name_3=3)\n\n        function_name(argument_name_1=1, argument_name_2=2, argument_name_3=3,)\n\n        a_very_long_function_name(long_argument_name_1=1, long_argument_name_2=2, long_argument_name_3=3, long_argument_name_4=4)\n\n        a_very_long_function_name(long_argument_name_1, long_argument_name_2, long_argument_name_3, long_argument_name_4,)\n\n        r =f0 (1,  2,3,)\n\n        r =f0 (1,)\n\n        r =f0 (a=1,)\n    ')
        expected_formatted_code = textwrap.dedent('        function_name(argument_name_1=1, argument_name_2=2, argument_name_3=3)\n\n        function_name(\n            argument_name_1=1,\n            argument_name_2=2,\n            argument_name_3=3,\n        )\n\n        a_very_long_function_name(\n            long_argument_name_1=1,\n            long_argument_name_2=2,\n            long_argument_name_3=3,\n            long_argument_name_4=4)\n\n        a_very_long_function_name(\n            long_argument_name_1,\n            long_argument_name_2,\n            long_argument_name_3,\n            long_argument_name_4,\n        )\n\n        r = f0(\n            1,\n            2,\n            3,\n        )\n\n        r = f0(\n            1,\n        )\n\n        r = f0(\n            a=1,\n        )\n    ')
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: yapf, split_arguments_when_comma_terminated: True}'))
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            reformatted_code = reformatter.Reformat(llines)
            self.assertCodeEqual(expected_formatted_code, reformatted_code)
            llines = yapf_test_helper.ParseAndUnwrap(reformatted_code)
            reformatted_code = reformatter.Reformat(llines)
            self.assertCodeEqual(expected_formatted_code, reformatted_code)
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())

    def testImportAsList(self):
        if False:
            for i in range(10):
                print('nop')
        code = textwrap.dedent('        from toto import titi, tata, tutu  # noqa\n        from toto import titi, tata, tutu\n        from toto import (titi, tata, tutu)\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testDictionaryValuesOnOwnLines(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent("        a = {\n        'aaaaaaaaaaaaaaaaaaaaaaaa':\n            Check('ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ', '=', True),\n        'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb':\n            Check('YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY', '=', True),\n        'ccccccccccccccc':\n            Check('XXXXXXXXXXXXXXXXXXX', '!=', 'SUSPENDED'),\n        'dddddddddddddddddddddddddddddd':\n            Check('WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW', '=', False),\n        'eeeeeeeeeeeeeeeeeeeeeeeeeeeee':\n            Check('VVVVVVVVVVVVVVVVVVVVVVVVVVVVVV', '=', False),\n        'ffffffffffffffffffffffffff':\n            Check('UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU', '=', True),\n        'ggggggggggggggggg':\n            Check('TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT', '=', True),\n        'hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh':\n            Check('SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS', '=', True),\n        'iiiiiiiiiiiiiiiiiiiiiiii':\n            Check('RRRRRRRRRRRRRRRRRRRRRRRRRRR', '=', True),\n        'jjjjjjjjjjjjjjjjjjjjjjjjjj':\n            Check('QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ', '=', False),\n        }\n    ")
        expected_formatted_code = textwrap.dedent("        a = {\n            'aaaaaaaaaaaaaaaaaaaaaaaa':\n                Check('ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ', '=', True),\n            'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb':\n                Check('YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY', '=', True),\n            'ccccccccccccccc':\n                Check('XXXXXXXXXXXXXXXXXXX', '!=', 'SUSPENDED'),\n            'dddddddddddddddddddddddddddddd':\n                Check('WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW', '=', False),\n            'eeeeeeeeeeeeeeeeeeeeeeeeeeeee':\n                Check('VVVVVVVVVVVVVVVVVVVVVVVVVVVVVV', '=', False),\n            'ffffffffffffffffffffffffff':\n                Check('UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU', '=', True),\n            'ggggggggggggggggg':\n                Check('TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT', '=', True),\n            'hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh':\n                Check('SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS', '=', True),\n            'iiiiiiiiiiiiiiiiiiiiiiii':\n                Check('RRRRRRRRRRRRRRRRRRRRRRRRRRR', '=', True),\n            'jjjjjjjjjjjjjjjjjjjjjjjjjj':\n                Check('QQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQQ', '=', False),\n        }\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testDictionaryOnOwnLine(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent("        doc = test_utils.CreateTestDocumentViaController(\n            content={ 'a': 'b' },\n            branch_key=branch.key,\n            collection_key=collection.key)\n    ")
        expected_formatted_code = textwrap.dedent("        doc = test_utils.CreateTestDocumentViaController(\n            content={'a': 'b'}, branch_key=branch.key, collection_key=collection.key)\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent("        doc = test_utils.CreateTestDocumentViaController(\n            content={ 'a': 'b' },\n            branch_key=branch.key,\n            collection_key=collection.key,\n            collection_key2=collection.key2)\n    ")
        expected_formatted_code = textwrap.dedent("        doc = test_utils.CreateTestDocumentViaController(\n            content={'a': 'b'},\n            branch_key=branch.key,\n            collection_key=collection.key,\n            collection_key2=collection.key2)\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testNestedListsInDictionary(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent("        _A = {\n            'cccccccccc': ('^^1',),\n            'rrrrrrrrrrrrrrrrrrrrrrrrr': ('^7913',  # AAAAAAAAAAAAAA.\n                                         ),\n            'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee': ('^6242',  # BBBBBBBBBBBBBBB.\n                                                  ),\n            'vvvvvvvvvvvvvvvvvvv': ('^27959',  # CCCCCCCCCCCCCCCCCC.\n                                    '^19746',  # DDDDDDDDDDDDDDDDDDDDDDD.\n                                    '^22907',  # EEEEEEEEEEEEEEEEEEEEEEEE.\n                                    '^21098',  # FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF.\n                                    '^22826',  # GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG.\n                                    '^22769',  # HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH.\n                                    '^22935',  # IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII.\n                                    '^3982',  # JJJJJJJJJJJJJ.\n                                   ),\n            'uuuuuuuuuuuu': ('^19745',  # LLLLLLLLLLLLLLLLLLLLLLLLLL.\n                             '^21324',  # MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM.\n                             '^22831',  # NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN.\n                             '^17081',  # OOOOOOOOOOOOOOOOOOOOO.\n                            ),\n            'eeeeeeeeeeeeee': (\n                '^9416',  # Reporter email. Not necessarily the reporter.\n                '^^3',  # This appears to be the raw email field.\n            ),\n            'cccccccccc': ('^21109',  # PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP.\n                          ),\n        }\n    ")
        expected_formatted_code = textwrap.dedent("        _A = {\n            'cccccccccc': ('^^1',),\n            'rrrrrrrrrrrrrrrrrrrrrrrrr': (\n                '^7913',  # AAAAAAAAAAAAAA.\n            ),\n            'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee': (\n                '^6242',  # BBBBBBBBBBBBBBB.\n            ),\n            'vvvvvvvvvvvvvvvvvvv': (\n                '^27959',  # CCCCCCCCCCCCCCCCCC.\n                '^19746',  # DDDDDDDDDDDDDDDDDDDDDDD.\n                '^22907',  # EEEEEEEEEEEEEEEEEEEEEEEE.\n                '^21098',  # FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF.\n                '^22826',  # GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG.\n                '^22769',  # HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH.\n                '^22935',  # IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII.\n                '^3982',  # JJJJJJJJJJJJJ.\n            ),\n            'uuuuuuuuuuuu': (\n                '^19745',  # LLLLLLLLLLLLLLLLLLLLLLLLLL.\n                '^21324',  # MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM.\n                '^22831',  # NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN.\n                '^17081',  # OOOOOOOOOOOOOOOOOOOOO.\n            ),\n            'eeeeeeeeeeeeee': (\n                '^9416',  # Reporter email. Not necessarily the reporter.\n                '^^3',  # This appears to be the raw email field.\n            ),\n            'cccccccccc': (\n                '^21109',  # PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP.\n            ),\n        }\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testNestedDictionary(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        class _():\n          def _():\n            breadcrumbs = [{\'name\': \'Admin\',\n                            \'url\': url_for(".home")},\n                           {\'title\': title},]\n            breadcrumbs = [{\'name\': \'Admin\',\n                            \'url\': url_for(".home")},\n                           {\'title\': title}]\n    ')
        expected_formatted_code = textwrap.dedent('        class _():\n          def _():\n            breadcrumbs = [\n                {\n                    \'name\': \'Admin\',\n                    \'url\': url_for(".home")\n                },\n                {\n                    \'title\': title\n                },\n            ]\n            breadcrumbs = [{\'name\': \'Admin\', \'url\': url_for(".home")}, {\'title\': title}]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testDictionaryElementsOnOneLine(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent("        class _():\n\n          @mock.patch.dict(\n              os.environ,\n              {'HTTP_' + xsrf._XSRF_TOKEN_HEADER.replace('-', '_'): 'atoken'})\n          def _():\n            pass\n\n\n        AAAAAAAAAAAAAAAAAAAAAAAA = {\n            Environment.XXXXXXXXXX: 'some text more text even more tex',\n            Environment.YYYYYYY: 'some text more text even more text yet ag',\n            Environment.ZZZZZZZZZZZ: 'some text more text even more text yet again tex',\n        }\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testNotInParams(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        list("a long line to break the line. a long line to break the brk a long lin", not True)\n    ')
        expected_code = textwrap.dedent('        list("a long line to break the line. a long line to break the brk a long lin",\n             not True)\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_code, reformatter.Reformat(llines))

    def testNamedAssignNotAtEndOfLine(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent("        def _():\n          if True:\n            with py3compat.open_with_encoding(filename, mode='w',\n                                              encoding=encoding) as fd:\n              pass\n    ")
        expected_code = textwrap.dedent("        def _():\n          if True:\n            with py3compat.open_with_encoding(\n                filename, mode='w', encoding=encoding) as fd:\n              pass\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_code, reformatter.Reformat(llines))

    def testBlankLineBeforeClassDocstring(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        class A:\n\n          """Does something.\n\n          Also, here are some details.\n          """\n\n          def __init__(self):\n            pass\n    ')
        expected_code = textwrap.dedent('        class A:\n          """Does something.\n\n          Also, here are some details.\n          """\n\n          def __init__(self):\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent('        class A:\n\n          """Does something.\n\n          Also, here are some details.\n          """\n\n          def __init__(self):\n            pass\n    ')
        expected_formatted_code = textwrap.dedent('        class A:\n\n          """Does something.\n\n          Also, here are some details.\n          """\n\n          def __init__(self):\n            pass\n    ')
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: yapf, blank_line_before_class_docstring: True}'))
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())

    def testBlankLineBeforeModuleDocstring(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent('        #!/usr/bin/env python\n        # -*- coding: utf-8 name> -*-\n\n        """Some module docstring."""\n\n\n        def foobar():\n          pass\n    ')
        expected_code = textwrap.dedent('        #!/usr/bin/env python\n        # -*- coding: utf-8 name> -*-\n        """Some module docstring."""\n\n\n        def foobar():\n          pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent('        #!/usr/bin/env python\n        # -*- coding: utf-8 name> -*-\n        """Some module docstring."""\n\n\n        def foobar():\n            pass\n    ')
        expected_formatted_code = textwrap.dedent('        #!/usr/bin/env python\n        # -*- coding: utf-8 name> -*-\n\n        """Some module docstring."""\n\n\n        def foobar():\n            pass\n    ')
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: pep8, blank_line_before_module_docstring: True}'))
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())

    def testTupleCohesion(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent("        def f():\n          this_is_a_very_long_function_name(an_extremely_long_variable_name, (\n              'a string that may be too long %s' % 'M15'))\n    ")
        expected_code = textwrap.dedent("        def f():\n          this_is_a_very_long_function_name(\n              an_extremely_long_variable_name,\n              ('a string that may be too long %s' % 'M15'))\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_code, reformatter.Reformat(llines))

    def testSubscriptExpression(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent('        foo = d[not a]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testSubscriptExpressionTerminatedByComma(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent('        A[B, C,]\n    ')
        expected_code = textwrap.dedent('        A[\n            B,\n            C,\n        ]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_code, reformatter.Reformat(llines))

    def testListWithFunctionCalls(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent("        def foo():\n          return [\n              Bar(\n                  xxx='some string',\n                  yyy='another long string',\n                  zzz='a third long string'), Bar(\n                      xxx='some string',\n                      yyy='another long string',\n                      zzz='a third long string')\n          ]\n    ")
        expected_code = textwrap.dedent("        def foo():\n          return [\n              Bar(xxx='some string',\n                  yyy='another long string',\n                  zzz='a third long string'),\n              Bar(xxx='some string',\n                  yyy='another long string',\n                  zzz='a third long string')\n          ]\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_code, reformatter.Reformat(llines))

    def testEllipses(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        X=...\n        Y = X if ... else X\n    ')
        expected_code = textwrap.dedent('        X = ...\n        Y = X if ... else X\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_code, reformatter.Reformat(llines))

    def testPseudoParens(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent("        my_dict = {\n            'key':  # Some comment about the key\n                {'nested_key': 1, },\n        }\n    ")
        expected_code = textwrap.dedent("        my_dict = {\n            'key':  # Some comment about the key\n                {\n                    'nested_key': 1,\n                },\n        }\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_code, reformatter.Reformat(llines))

    def testSplittingBeforeFirstArgumentOnFunctionCall(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests split_before_first_argument on a function call.'
        unformatted_code = textwrap.dedent('        a_very_long_function_name("long string with formatting {0:s}".format(\n            "mystring"))\n    ')
        expected_formatted_code = textwrap.dedent('        a_very_long_function_name(\n            "long string with formatting {0:s}".format("mystring"))\n    ')
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: yapf, split_before_first_argument: True}'))
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())

    def testSplittingBeforeFirstArgumentOnFunctionDefinition(self):
        if False:
            return 10
        'Tests split_before_first_argument on a function definition.'
        unformatted_code = textwrap.dedent('        def _GetNumberOfSecondsFromElements(year, month, day, hours,\n                                            minutes, seconds, microseconds):\n          return\n    ')
        expected_formatted_code = textwrap.dedent('        def _GetNumberOfSecondsFromElements(\n            year, month, day, hours, minutes, seconds, microseconds):\n          return\n    ')
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: yapf, split_before_first_argument: True}'))
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())

    def testSplittingBeforeFirstArgumentOnCompoundStatement(self):
        if False:
            while True:
                i = 10
        'Tests split_before_first_argument on a compound statement.'
        unformatted_code = textwrap.dedent('        if (long_argument_name_1 == 1 or\n            long_argument_name_2 == 2 or\n            long_argument_name_3 == 3 or\n            long_argument_name_4 == 4):\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        if (long_argument_name_1 == 1 or long_argument_name_2 == 2 or\n            long_argument_name_3 == 3 or long_argument_name_4 == 4):\n          pass\n    ')
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: yapf, split_before_first_argument: True}'))
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())

    def testCoalesceBracketsOnDict(self):
        if False:
            i = 10
            return i + 15
        'Tests coalesce_brackets on a dictionary.'
        unformatted_code = textwrap.dedent("        date_time_values = (\n            {\n                u'year': year,\n                u'month': month,\n                u'day_of_month': day_of_month,\n                u'hours': hours,\n                u'minutes': minutes,\n                u'seconds': seconds\n            }\n        )\n    ")
        expected_formatted_code = textwrap.dedent("        date_time_values = ({\n            u'year': year,\n            u'month': month,\n            u'day_of_month': day_of_month,\n            u'hours': hours,\n            u'minutes': minutes,\n            u'seconds': seconds\n        })\n    ")
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: yapf, coalesce_brackets: True}'))
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())

    def testSplitAfterComment(self):
        if False:
            i = 10
            return i + 15
        code = textwrap.dedent('        if __name__ == "__main__":\n          with another_resource:\n            account = {\n                "validUntil":\n                    int(time() + (6 * 7 * 24 * 60 * 60))  # in 6 weeks time\n            }\n    ')
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: yapf, coalesce_brackets: True, dedent_closing_brackets: true}'))
            llines = yapf_test_helper.ParseAndUnwrap(code)
            self.assertCodeEqual(code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())

    def testDisableEndingCommaHeuristic(self):
        if False:
            for i in range(10):
                print('nop')
        code = textwrap.dedent('        x = [1, 2, 3, 4, 5, 6, 7,]\n    ')
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: yapf, disable_ending_comma_heuristic: True}'))
            llines = yapf_test_helper.ParseAndUnwrap(code)
            self.assertCodeEqual(code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())

    def testDedentClosingBracketsWithTypeAnnotationExceedingLineLength(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        def function(first_argument_xxxxxxxxxxxxxxxx=(0,), second_argument=None) -> None:\n          pass\n\n\n        def function(first_argument_xxxxxxxxxxxxxxxxxxxxxxx=(0,), second_argument=None) -> None:\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        def function(\n            first_argument_xxxxxxxxxxxxxxxx=(0,), second_argument=None\n        ) -> None:\n          pass\n\n\n        def function(\n            first_argument_xxxxxxxxxxxxxxxxxxxxxxx=(0,), second_argument=None\n        ) -> None:\n          pass\n    ')
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: yapf, dedent_closing_brackets: True}'))
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())

    def testIndentClosingBracketsWithTypeAnnotationExceedingLineLength(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        def function(first_argument_xxxxxxxxxxxxxxxx=(0,), second_argument=None) -> None:\n          pass\n\n\n        def function(first_argument_xxxxxxxxxxxxxxxxxxxxxxx=(0,), second_argument=None) -> None:\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        def function(\n            first_argument_xxxxxxxxxxxxxxxx=(0,), second_argument=None\n            ) -> None:\n          pass\n\n\n        def function(\n            first_argument_xxxxxxxxxxxxxxxxxxxxxxx=(0,), second_argument=None\n            ) -> None:\n          pass\n    ')
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: yapf, indent_closing_brackets: True}'))
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())

    def testIndentClosingBracketsInFunctionCall(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent('        def function(first_argument_xxxxxxxxxxxxxxxx=(0,), second_argument=None, third_and_final_argument=True):\n          pass\n\n\n        def function(first_argument_xxxxxxxxxxxxxxxxxxxxxxx=(0,), second_and_last_argument=None):\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        def function(\n            first_argument_xxxxxxxxxxxxxxxx=(0,),\n            second_argument=None,\n            third_and_final_argument=True\n            ):\n          pass\n\n\n        def function(\n            first_argument_xxxxxxxxxxxxxxxxxxxxxxx=(0,), second_and_last_argument=None\n            ):\n          pass\n    ')
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: yapf, indent_closing_brackets: True}'))
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())

    def testIndentClosingBracketsInTuple(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent("        def function():\n          some_var = ('a long element', 'another long element', 'short element', 'really really long element')\n          return True\n\n        def function():\n          some_var = ('a couple', 'small', 'elemens')\n          return False\n    ")
        expected_formatted_code = textwrap.dedent("        def function():\n          some_var = (\n              'a long element', 'another long element', 'short element',\n              'really really long element'\n              )\n          return True\n\n\n        def function():\n          some_var = ('a couple', 'small', 'elemens')\n          return False\n    ")
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: yapf, indent_closing_brackets: True}'))
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())

    def testIndentClosingBracketsInList(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent("        def function():\n          some_var = ['a long element', 'another long element', 'short element', 'really really long element']\n          return True\n\n        def function():\n          some_var = ['a couple', 'small', 'elemens']\n          return False\n    ")
        expected_formatted_code = textwrap.dedent("        def function():\n          some_var = [\n              'a long element', 'another long element', 'short element',\n              'really really long element'\n              ]\n          return True\n\n\n        def function():\n          some_var = ['a couple', 'small', 'elemens']\n          return False\n    ")
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: yapf, indent_closing_brackets: True}'))
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())

    def testIndentClosingBracketsInDict(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent("        def function():\n          some_var = {1: ('a long element', 'and another really really long element that is really really amazingly long'), 2: 'another long element', 3: 'short element', 4: 'really really long element'}\n          return True\n\n        def function():\n          some_var = {1: 'a couple', 2: 'small', 3: 'elemens'}\n          return False\n    ")
        expected_formatted_code = textwrap.dedent("        def function():\n          some_var = {\n              1:\n                  (\n                      'a long element',\n                      'and another really really long element that is really really amazingly long'\n                      ),\n              2: 'another long element',\n              3: 'short element',\n              4: 'really really long element'\n              }\n          return True\n\n\n        def function():\n          some_var = {1: 'a couple', 2: 'small', 3: 'elemens'}\n          return False\n    ")
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: yapf, indent_closing_brackets: True}'))
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())

    def testMultipleDictionariesInList(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        class A:\n            def b():\n                d = {\n                    "123456": [\n                        {\n                            "12": "aa"\n                        },\n                        {\n                            "12": "bb"\n                        },\n                        {\n                            "12": "cc",\n                            "1234567890": {\n                                "1234567": [{\n                                    "12": "dd",\n                                    "12345": "text 1"\n                                }, {\n                                    "12": "ee",\n                                    "12345": "text 2"\n                                }]\n                            }\n                        }\n                    ]\n                }\n    ')
        expected_formatted_code = textwrap.dedent('        class A:\n\n          def b():\n            d = {\n                "123456": [{\n                    "12": "aa"\n                }, {\n                    "12": "bb"\n                }, {\n                    "12": "cc",\n                    "1234567890": {\n                        "1234567": [{\n                            "12": "dd",\n                            "12345": "text 1"\n                        }, {\n                            "12": "ee",\n                            "12345": "text 2"\n                        }]\n                    }\n                }]\n            }\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testForceMultilineDict_True(self):
        if False:
            return 10
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{force_multiline_dict: true}'))
            unformatted_code = textwrap.dedent("          responseDict = {'childDict': {'spam': 'eggs'}}\n          generatedDict = {x: x for x in 'value'}\n      ")
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            actual = reformatter.Reformat(llines)
            expected = textwrap.dedent("          responseDict = {\n              'childDict': {\n                  'spam': 'eggs'\n              }\n          }\n          generatedDict = {\n              x: x for x in 'value'\n          }\n      ")
            self.assertCodeEqual(expected, actual)
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())

    def testForceMultilineDict_False(self):
        if False:
            i = 10
            return i + 15
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{force_multiline_dict: false}'))
            unformatted_code = textwrap.dedent("          responseDict = {'childDict': {'spam': 'eggs'}}\n          generatedDict = {x: x for x in 'value'}\n      ")
            expected_formatted_code = unformatted_code
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreateYapfStyle())

    def testWalrus(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent("        if (x  :=  len([1]*1000)>100):\n          print(f'{x} is pretty big' )\n    ")
        expected = textwrap.dedent("        if (x := len([1] * 1000) > 100):\n          print(f'{x} is pretty big')\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected, reformatter.Reformat(llines))

    def testStructuredPatternMatching(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        match command.split():\n          case[action   ]:\n            ...  # interpret single-verb action\n          case[action,    obj]:\n            ...  # interpret action, obj\n    ')
        expected = textwrap.dedent('        match command.split():\n          case [action]:\n            ...  # interpret single-verb action\n          case [action, obj]:\n            ...  # interpret action, obj\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected, reformatter.Reformat(llines))

    def testParenthesizedContextManagers(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent("        with (cert_authority.cert_pem.tempfile() as ca_temp_path, patch.object(os, 'environ', os.environ | {'REQUESTS_CA_BUNDLE': ca_temp_path}),):\n            httpserver_url = httpserver.url_for('/resource.jar')\n    ")
        expected = textwrap.dedent("        with (\n            cert_authority.cert_pem.tempfile() as ca_temp_path,\n            patch.object(os, 'environ',\n                         os.environ | {'REQUESTS_CA_BUNDLE': ca_temp_path}),\n        ):\n          httpserver_url = httpserver.url_for('/resource.jar')\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected, reformatter.Reformat(llines))
if __name__ == '__main__':
    unittest.main()