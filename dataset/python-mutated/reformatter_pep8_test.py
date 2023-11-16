"""PEP8 tests for yapf.reformatter."""
import textwrap
import unittest
from yapf.yapflib import reformatter
from yapf.yapflib import style
from yapftests import yapf_test_helper

class TestsForPEP8Style(yapf_test_helper.YAPFTest):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        style.SetGlobalStyle(style.CreatePEP8Style())

    def testIndent4(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        if a+b:\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        if a + b:\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testSingleLineIfStatements(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent('        if True: a = 42\n        elif False: b = 42\n        else: c = 42\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testBlankBetweenClassAndDef(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        class Foo:\n          def joe():\n            pass\n    ')
        expected_formatted_code = textwrap.dedent('        class Foo:\n\n            def joe():\n                pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testBlankBetweenDefsInClass(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        class TestClass:\n            def __init__(self):\n                self.running = False\n            def run(self):\n                """Override in subclass"""\n            def is_running(self):\n                return self.running\n    ')
        expected_formatted_code = textwrap.dedent('        class TestClass:\n\n            def __init__(self):\n                self.running = False\n\n            def run(self):\n                """Override in subclass"""\n\n            def is_running(self):\n                return self.running\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testSingleWhiteBeforeTrailingComment(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        if a+b: # comment\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        if a + b:  # comment\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testSpaceBetweenEndingCommandAndClosingBracket(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent('        a = (\n            1,\n        )\n    ')
        expected_formatted_code = textwrap.dedent('        a = (1, )\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testContinuedNonOutdentedLine(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent("        class eld(d):\n            if str(geom.geom_type).upper(\n            ) != self.geom_type and not self.geom_type == 'GEOMETRY':\n                ror(code='om_type')\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testWrappingPercentExpressions(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent("        def f():\n            if True:\n                zzzzz = '%s-%s' % (xxxxxxxxxxxxxxxxxxxxxxxxxx + 1, xxxxxxxxxxxxxxxxx.yyy + 1)\n                zzzzz = '%s-%s'.ww(xxxxxxxxxxxxxxxxxxxxxxxxxx + 1, xxxxxxxxxxxxxxxxx.yyy + 1)\n                zzzzz = '%s-%s' % (xxxxxxxxxxxxxxxxxxxxxxx + 1, xxxxxxxxxxxxxxxxxxxxx + 1)\n                zzzzz = '%s-%s'.ww(xxxxxxxxxxxxxxxxxxxxxxx + 1, xxxxxxxxxxxxxxxxxxxxx + 1)\n    ")
        expected_formatted_code = textwrap.dedent("        def f():\n            if True:\n                zzzzz = '%s-%s' % (xxxxxxxxxxxxxxxxxxxxxxxxxx + 1,\n                                   xxxxxxxxxxxxxxxxx.yyy + 1)\n                zzzzz = '%s-%s'.ww(xxxxxxxxxxxxxxxxxxxxxxxxxx + 1,\n                                   xxxxxxxxxxxxxxxxx.yyy + 1)\n                zzzzz = '%s-%s' % (xxxxxxxxxxxxxxxxxxxxxxx + 1,\n                                   xxxxxxxxxxxxxxxxxxxxx + 1)\n                zzzzz = '%s-%s'.ww(xxxxxxxxxxxxxxxxxxxxxxx + 1,\n                                   xxxxxxxxxxxxxxxxxxxxx + 1)\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testAlignClosingBracketWithVisualIndentation(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent("        TEST_LIST = ('foo', 'bar',  # first comment\n                     'baz'  # second comment\n                    )\n    ")
        expected_formatted_code = textwrap.dedent("        TEST_LIST = (\n            'foo',\n            'bar',  # first comment\n            'baz'  # second comment\n        )\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        unformatted_code = textwrap.dedent("        def f():\n\n          def g():\n            while (xxxxxxxxxxxxxxxxxxxx(yyyyyyyyyyyyy[zzzzz]) == 'aaaaaaaaaaa' and\n                   xxxxxxxxxxxxxxxxxxxx(yyyyyyyyyyyyy[zzzzz].aaaaaaaa[0]) == 'bbbbbbb'\n                  ):\n              pass\n    ")
        expected_formatted_code = textwrap.dedent("        def f():\n\n            def g():\n                while (xxxxxxxxxxxxxxxxxxxx(yyyyyyyyyyyyy[zzzzz]) == 'aaaaaaaaaaa'\n                       and xxxxxxxxxxxxxxxxxxxx(\n                           yyyyyyyyyyyyy[zzzzz].aaaaaaaa[0]) == 'bbbbbbb'):\n                    pass\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testIndentSizeChanging(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        if True:\n          runtime_mins = (program_end_time - program_start_time).total_seconds() / 60.0\n    ')
        expected_formatted_code = textwrap.dedent('        if True:\n            runtime_mins = (program_end_time -\n                            program_start_time).total_seconds() / 60.0\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testHangingIndentCollision(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent("        if (aaaaaaaaaaaaaa + bbbbbbbbbbbbbbbb == ccccccccccccccccc and xxxxxxxxxxxxx or yyyyyyyyyyyyyyyyy):\n            pass\n        elif (xxxxxxxxxxxxxxx(aaaaaaaaaaa, bbbbbbbbbbbbbb, cccccccccccc, dddddddddd=None)):\n            pass\n\n\n        def h():\n            if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n                pass\n\n            for connection in itertools.chain(branch.contact, branch.address, morestuff.andmore.andmore.andmore.andmore.andmore.andmore.andmore):\n                dosomething(connection)\n    ")
        expected_formatted_code = textwrap.dedent("        if (aaaaaaaaaaaaaa + bbbbbbbbbbbbbbbb == ccccccccccccccccc and xxxxxxxxxxxxx\n                or yyyyyyyyyyyyyyyyy):\n            pass\n        elif (xxxxxxxxxxxxxxx(aaaaaaaaaaa,\n                              bbbbbbbbbbbbbb,\n                              cccccccccccc,\n                              dddddddddd=None)):\n            pass\n\n\n        def h():\n            if (xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0]) == 'aaaaaaaaaaa' and\n                    xxxxxxxxxxxx.yyyyyyyy(zzzzzzzzzzzzz[0].mmmmmmmm[0]) == 'bbbbbbb'):\n                pass\n\n            for connection in itertools.chain(\n                    branch.contact, branch.address,\n                    morestuff.andmore.andmore.andmore.andmore.andmore.andmore.andmore):\n                dosomething(connection)\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testSplittingBeforeLogicalOperator(self):
        if False:
            i = 10
            return i + 15
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: pep8, split_before_logical_operator: True}'))
            unformatted_code = textwrap.dedent('          def foo():\n              return bool(update.message.new_chat_member or update.message.left_chat_member or\n                          update.message.new_chat_title or update.message.new_chat_photo or\n                          update.message.delete_chat_photo or update.message.group_chat_created or\n                          update.message.supergroup_chat_created or update.message.channel_chat_created\n                          or update.message.migrate_to_chat_id or update.message.migrate_from_chat_id or\n                          update.message.pinned_message)\n      ')
            expected_formatted_code = textwrap.dedent('          def foo():\n              return bool(\n                  update.message.new_chat_member or update.message.left_chat_member\n                  or update.message.new_chat_title or update.message.new_chat_photo\n                  or update.message.delete_chat_photo\n                  or update.message.group_chat_created\n                  or update.message.supergroup_chat_created\n                  or update.message.channel_chat_created\n                  or update.message.migrate_to_chat_id\n                  or update.message.migrate_from_chat_id\n                  or update.message.pinned_message)\n      ')
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreatePEP8Style())

    def testContiguousListEndingWithComment(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        if True:\n            if True:\n                keys.append(aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)  # may be unassigned.\n    ')
        expected_formatted_code = textwrap.dedent('        if True:\n            if True:\n                keys.append(\n                    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)  # may be unassigned.\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testSplittingBeforeFirstArgument(self):
        if False:
            while True:
                i = 10
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: pep8, split_before_first_argument: True}'))
            unformatted_code = textwrap.dedent('          a_very_long_function_name(long_argument_name_1=1, long_argument_name_2=2,\n                                    long_argument_name_3=3, long_argument_name_4=4)\n      ')
            expected_formatted_code = textwrap.dedent('          a_very_long_function_name(\n              long_argument_name_1=1,\n              long_argument_name_2=2,\n              long_argument_name_3=3,\n              long_argument_name_4=4)\n      ')
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreatePEP8Style())

    def testSplittingExpressionsInsideSubscripts(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent("        def foo():\n            df = df[(df['campaign_status'] == 'LIVE') & (df['action_status'] == 'LIVE')]\n    ")
        expected_formatted_code = textwrap.dedent("        def foo():\n            df = df[(df['campaign_status'] == 'LIVE')\n                    & (df['action_status'] == 'LIVE')]\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testSplitListsAndDictSetMakersIfCommaTerminated(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        DJANGO_TEMPLATES_OPTIONS = {"context_processors": []}\n        DJANGO_TEMPLATES_OPTIONS = {"context_processors": [],}\n        x = ["context_processors"]\n        x = ["context_processors",]\n    ')
        expected_formatted_code = textwrap.dedent('        DJANGO_TEMPLATES_OPTIONS = {"context_processors": []}\n        DJANGO_TEMPLATES_OPTIONS = {\n            "context_processors": [],\n        }\n        x = ["context_processors"]\n        x = [\n            "context_processors",\n        ]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testSplitAroundNamedAssigns(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        class a():\n\n            def a(): return a(\n             aaaaaaaaaa=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)\n    ')
        expected_formatted_code = textwrap.dedent('        class a():\n\n            def a():\n                return a(\n                    aaaaaaaaaa=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n                )\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testUnaryOperator(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        if not -3 < x < 3:\n          pass\n        if -3 < x < 3:\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        if not -3 < x < 3:\n            pass\n        if -3 < x < 3:\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testNoSplitBeforeDictValue(self):
        if False:
            print('Hello World!')
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: pep8, allow_split_before_dict_value: false, coalesce_brackets: true, dedent_closing_brackets: true, each_dict_entry_on_separate_line: true, split_before_logical_operator: true}'))
            unformatted_code = textwrap.dedent('          some_dict = {\n              \'title\': _("I am example data"),\n              \'description\': _("Lorem ipsum dolor met sit amet elit, si vis pacem para bellum "\n                               "elites nihi very long string."),\n          }\n      ')
            expected_formatted_code = textwrap.dedent('          some_dict = {\n              \'title\': _("I am example data"),\n              \'description\': _(\n                  "Lorem ipsum dolor met sit amet elit, si vis pacem para bellum "\n                  "elites nihi very long string."\n              ),\n          }\n      ')
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
            unformatted_code = textwrap.dedent("          X = {'a': 1, 'b': 2, 'key': this_is_a_function_call_that_goes_over_the_column_limit_im_pretty_sure()}\n      ")
            expected_formatted_code = textwrap.dedent("          X = {\n              'a': 1,\n              'b': 2,\n              'key': this_is_a_function_call_that_goes_over_the_column_limit_im_pretty_sure()\n          }\n      ")
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
            unformatted_code = textwrap.dedent('          attrs = {\n              \'category\': category,\n              \'role\': forms.ModelChoiceField(label=_("Role"), required=False, queryset=category_roles, initial=selected_role, empty_label=_("No access"),),\n          }\n      ')
            expected_formatted_code = textwrap.dedent('          attrs = {\n              \'category\': category,\n              \'role\': forms.ModelChoiceField(\n                  label=_("Role"),\n                  required=False,\n                  queryset=category_roles,\n                  initial=selected_role,\n                  empty_label=_("No access"),\n              ),\n          }\n      ')
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
            unformatted_code = textwrap.dedent('          css_class = forms.CharField(\n              label=_("CSS class"),\n              required=False,\n              help_text=_("Optional CSS class used to customize this category appearance from templates."),\n          )\n      ')
            expected_formatted_code = textwrap.dedent('          css_class = forms.CharField(\n              label=_("CSS class"),\n              required=False,\n              help_text=_(\n                  "Optional CSS class used to customize this category appearance from templates."\n              ),\n          )\n      ')
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreatePEP8Style())

    def testBitwiseOperandSplitting(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent("        def _():\n            include_values = np.where(\n                        (cdffile['Quality_Flag'][:] >= 5) & (\n                        cdffile['Day_Night_Flag'][:] == 1) & (\n                        cdffile['Longitude'][:] >= select_lon - radius) & (\n                        cdffile['Longitude'][:] <= select_lon + radius) & (\n                        cdffile['Latitude'][:] >= select_lat - radius) & (\n                        cdffile['Latitude'][:] <= select_lat + radius))\n    ")
        expected_code = textwrap.dedent("        def _():\n            include_values = np.where(\n                (cdffile['Quality_Flag'][:] >= 5) & (cdffile['Day_Night_Flag'][:] == 1)\n                & (cdffile['Longitude'][:] >= select_lon - radius)\n                & (cdffile['Longitude'][:] <= select_lon + radius)\n                & (cdffile['Latitude'][:] >= select_lat - radius)\n                & (cdffile['Latitude'][:] <= select_lat + radius))\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertEqual(expected_code, reformatter.Reformat(llines))

    def testNoBlankLinesOnlyForFirstNestedObject(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent('        class Demo:\n            """\n            Demo docs\n            """\n            def foo(self):\n                """\n                foo docs\n                """\n            def bar(self):\n                """\n                bar docs\n                """\n    ')
        expected_code = textwrap.dedent('        class Demo:\n            """\n            Demo docs\n            """\n\n            def foo(self):\n                """\n                foo docs\n                """\n\n            def bar(self):\n                """\n                bar docs\n                """\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertEqual(expected_code, reformatter.Reformat(llines))

    def testSplitBeforeArithmeticOperators(self):
        if False:
            while True:
                i = 10
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: pep8, split_before_arithmetic_operator: true}'))
            unformatted_code = textwrap.dedent("        def _():\n            raise ValueError('This is a long message that ends with an argument: ' + str(42))\n      ")
            expected_formatted_code = textwrap.dedent("        def _():\n            raise ValueError('This is a long message that ends with an argument: '\n                             + str(42))\n      ")
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreatePEP8Style())

    def testListSplitting(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        foo([(1,1), (1,1), (1,1), (1,1), (1,1), (1,1), (1,1),\n             (1,1), (1,1), (1,1), (1,1), (1,1), (1,1), (1,1),\n             (1,10), (1,11), (1, 10), (1,11), (10,11)])\n    ')
        expected_code = textwrap.dedent('        foo([(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1),\n             (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 10), (1, 11), (1, 10),\n             (1, 11), (10, 11)])\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_code, reformatter.Reformat(llines))

    def testNoBlankLineBeforeNestedFuncOrClass(self):
        if False:
            print('Hello World!')
        try:
            style.SetGlobalStyle(style.CreateStyleFromConfig('{based_on_style: pep8, blank_line_before_nested_class_or_def: false}'))
            unformatted_code = textwrap.dedent('        def normal_function():\n            """Return the nested function."""\n\n            def nested_function():\n                """Do nothing just nest within."""\n\n                @nested(klass)\n                class nested_class():\n                    pass\n\n                pass\n\n            return nested_function\n      ')
            expected_formatted_code = textwrap.dedent('        def normal_function():\n            """Return the nested function."""\n            def nested_function():\n                """Do nothing just nest within."""\n                @nested(klass)\n                class nested_class():\n                    pass\n\n                pass\n\n            return nested_function\n      ')
            llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
            self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
        finally:
            style.SetGlobalStyle(style.CreatePEP8Style())

    def testParamListIndentationCollision1(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent("        class _():\n\n            def __init__(self, title: Optional[str], diffs: Collection[BinaryDiff] = (), charset: Union[Type[AsciiCharset], Type[LineCharset]] = AsciiCharset, preprocess: Callable[[str], str] = identity,\n                    # TODO(somebody): Make this a Literal type.\n                    justify: str = 'rjust'):\n                self._cs = charset\n                self._preprocess = preprocess\n    ")
        expected_formatted_code = textwrap.dedent("        class _():\n\n            def __init__(\n                    self,\n                    title: Optional[str],\n                    diffs: Collection[BinaryDiff] = (),\n                    charset: Union[Type[AsciiCharset],\n                                   Type[LineCharset]] = AsciiCharset,\n                    preprocess: Callable[[str], str] = identity,\n                    # TODO(somebody): Make this a Literal type.\n                    justify: str = 'rjust'):\n                self._cs = charset\n                self._preprocess = preprocess\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testParamListIndentationCollision2(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent('        def simple_pass_function_with_an_extremely_long_name_and_some_arguments(\n                argument0, argument1):\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testParamListIndentationCollision3(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent('        def func1(\n            arg1,\n            arg2,\n        ) -> None:\n            pass\n\n\n        def func2(\n            arg1,\n            arg2,\n        ):\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testTwoWordComparisonOperators(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent('        _ = (klsdfjdklsfjksdlfjdklsfjdslkfjsdkl is not ksldfjsdklfjdklsfjdklsfjdklsfjdsklfjdklsfj)\n        _ = (klsdfjdklsfjksdlfjdklsfjdslkfjsdkl not in {ksldfjsdklfjdklsfjdklsfjdklsfjdsklfjdklsfj})\n    ')
        expected_formatted_code = textwrap.dedent('        _ = (klsdfjdklsfjksdlfjdklsfjdslkfjsdkl\n             is not ksldfjsdklfjdklsfjdklsfjdklsfjdsklfjdklsfj)\n        _ = (klsdfjdklsfjksdlfjdklsfjdslkfjsdkl\n             not in {ksldfjsdklfjdklsfjdklsfjdklsfjdsklfjdklsfj})\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testStableInlinedDictionaryFormatting(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        def _():\n            url = "http://{0}/axis-cgi/admin/param.cgi?{1}".format(\n                value, urllib.urlencode({\'action\': \'update\', \'parameter\': value}))\n    ')
        expected_formatted_code = textwrap.dedent('        def _():\n            url = "http://{0}/axis-cgi/admin/param.cgi?{1}".format(\n                value, urllib.urlencode({\n                    \'action\': \'update\',\n                    \'parameter\': value\n                }))\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        reformatted_code = reformatter.Reformat(llines)
        self.assertCodeEqual(expected_formatted_code, reformatted_code)
        llines = yapf_test_helper.ParseAndUnwrap(reformatted_code)
        reformatted_code = reformatter.Reformat(llines)
        self.assertCodeEqual(expected_formatted_code, reformatted_code)

class TestsForSpacesInsideBrackets(yapf_test_helper.YAPFTest):
    """Test the SPACE_INSIDE_BRACKETS style option."""
    unformatted_code = textwrap.dedent('      foo()\n      foo(1)\n      foo(1,2)\n      foo((1,))\n      foo((1, 2))\n      foo((1, 2,))\n      foo(bar[\'baz\'][0])\n      set1 = {1, 2, 3}\n      dict1 = {1: 1, foo: 2, 3: bar}\n      dict2 = {\n          1: 1,\n          foo: 2,\n          3: bar,\n      }\n      dict3[3][1][get_index(*args,**kwargs)]\n      dict4[3][1][get_index(**kwargs)]\n      x = dict5[4](foo(*args))\n      a = list1[:]\n      b = list2[slice_start:]\n      c = list3[slice_start:slice_end]\n      d = list4[slice_start:slice_end:]\n      e = list5[slice_start:slice_end:slice_step]\n      # Print gets special handling\n      print(set2)\n      compound = ((10+3)/(5-2**(6+x)))\n      string_idx = "mystring"[3]\n  ')

    def testEnabled(self):
        if False:
            i = 10
            return i + 15
        style.SetGlobalStyle(style.CreateStyleFromConfig('{space_inside_brackets: True}'))
        expected_formatted_code = textwrap.dedent('        foo()\n        foo( 1 )\n        foo( 1, 2 )\n        foo( ( 1, ) )\n        foo( ( 1, 2 ) )\n        foo( (\n            1,\n            2,\n        ) )\n        foo( bar[ \'baz\' ][ 0 ] )\n        set1 = { 1, 2, 3 }\n        dict1 = { 1: 1, foo: 2, 3: bar }\n        dict2 = {\n            1: 1,\n            foo: 2,\n            3: bar,\n        }\n        dict3[ 3 ][ 1 ][ get_index( *args, **kwargs ) ]\n        dict4[ 3 ][ 1 ][ get_index( **kwargs ) ]\n        x = dict5[ 4 ]( foo( *args ) )\n        a = list1[ : ]\n        b = list2[ slice_start: ]\n        c = list3[ slice_start:slice_end ]\n        d = list4[ slice_start:slice_end: ]\n        e = list5[ slice_start:slice_end:slice_step ]\n        # Print gets special handling\n        print( set2 )\n        compound = ( ( 10 + 3 ) / ( 5 - 2**( 6 + x ) ) )\n        string_idx = "mystring"[ 3 ]\n   ')
        llines = yapf_test_helper.ParseAndUnwrap(self.unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testDefault(self):
        if False:
            return 10
        style.SetGlobalStyle(style.CreatePEP8Style())
        expected_formatted_code = textwrap.dedent('        foo()\n        foo(1)\n        foo(1, 2)\n        foo((1, ))\n        foo((1, 2))\n        foo((\n            1,\n            2,\n        ))\n        foo(bar[\'baz\'][0])\n        set1 = {1, 2, 3}\n        dict1 = {1: 1, foo: 2, 3: bar}\n        dict2 = {\n            1: 1,\n            foo: 2,\n            3: bar,\n        }\n        dict3[3][1][get_index(*args, **kwargs)]\n        dict4[3][1][get_index(**kwargs)]\n        x = dict5[4](foo(*args))\n        a = list1[:]\n        b = list2[slice_start:]\n        c = list3[slice_start:slice_end]\n        d = list4[slice_start:slice_end:]\n        e = list5[slice_start:slice_end:slice_step]\n        # Print gets special handling\n        print(set2)\n        compound = ((10 + 3) / (5 - 2**(6 + x)))\n        string_idx = "mystring"[3]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(self.unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

class TestsForSpacesAroundSubscriptColon(yapf_test_helper.YAPFTest):
    """Test the SPACES_AROUND_SUBSCRIPT_COLON style option."""
    unformatted_code = textwrap.dedent('      a = list1[ : ]\n      b = list2[ slice_start: ]\n      c = list3[ slice_start:slice_end ]\n      d = list4[ slice_start:slice_end: ]\n      e = list5[ slice_start:slice_end:slice_step ]\n      a1 = list1[ : ]\n      b1 = list2[ 1: ]\n      c1 = list3[ 1:20 ]\n      d1 = list4[ 1:20: ]\n      e1 = list5[ 1:20:3 ]\n  ')

    def testEnabled(self):
        if False:
            while True:
                i = 10
        style.SetGlobalStyle(style.CreateStyleFromConfig('{spaces_around_subscript_colon: True}'))
        expected_formatted_code = textwrap.dedent('        a = list1[:]\n        b = list2[slice_start :]\n        c = list3[slice_start : slice_end]\n        d = list4[slice_start : slice_end :]\n        e = list5[slice_start : slice_end : slice_step]\n        a1 = list1[:]\n        b1 = list2[1 :]\n        c1 = list3[1 : 20]\n        d1 = list4[1 : 20 :]\n        e1 = list5[1 : 20 : 3]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(self.unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testWithSpaceInsideBrackets(self):
        if False:
            print('Hello World!')
        style.SetGlobalStyle(style.CreateStyleFromConfig('{spaces_around_subscript_colon: true, space_inside_brackets: true,}'))
        expected_formatted_code = textwrap.dedent('        a = list1[ : ]\n        b = list2[ slice_start : ]\n        c = list3[ slice_start : slice_end ]\n        d = list4[ slice_start : slice_end : ]\n        e = list5[ slice_start : slice_end : slice_step ]\n        a1 = list1[ : ]\n        b1 = list2[ 1 : ]\n        c1 = list3[ 1 : 20 ]\n        d1 = list4[ 1 : 20 : ]\n        e1 = list5[ 1 : 20 : 3 ]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(self.unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testDefault(self):
        if False:
            i = 10
            return i + 15
        style.SetGlobalStyle(style.CreatePEP8Style())
        expected_formatted_code = textwrap.dedent('        a = list1[:]\n        b = list2[slice_start:]\n        c = list3[slice_start:slice_end]\n        d = list4[slice_start:slice_end:]\n        e = list5[slice_start:slice_end:slice_step]\n        a1 = list1[:]\n        b1 = list2[1:]\n        c1 = list3[1:20]\n        d1 = list4[1:20:]\n        e1 = list5[1:20:3]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(self.unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
if __name__ == '__main__':
    unittest.main()