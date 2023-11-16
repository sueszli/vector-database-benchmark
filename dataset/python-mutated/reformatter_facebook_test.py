"""Facebook tests for yapf.reformatter."""
import textwrap
import unittest
from yapf.yapflib import reformatter
from yapf.yapflib import style
from yapftests import yapf_test_helper

class TestsForFacebookStyle(yapf_test_helper.YAPFTest):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        style.SetGlobalStyle(style.CreateFacebookStyle())

    def testNoNeedForLineBreaks(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        def overly_long_function_name(\n          just_one_arg, **kwargs):\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        def overly_long_function_name(just_one_arg, **kwargs):\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testDedentClosingBracket(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        def overly_long_function_name(\n          first_argument_on_the_same_line,\n          second_argument_makes_the_line_too_long):\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        def overly_long_function_name(\n            first_argument_on_the_same_line, second_argument_makes_the_line_too_long\n        ):\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testBreakAfterOpeningBracketIfContentsTooBig(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        def overly_long_function_name(a, b, c, d, e, f, g, h, i, j, k, l, m,\n          n, o, p, q, r, s, t, u, v, w, x, y, z):\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        def overly_long_function_name(\n            a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z\n        ):\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testDedentClosingBracketWithComments(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent('        def overly_long_function_name(\n          # comment about the first argument\n          first_argument_with_a_very_long_name_or_so,\n          # comment about the second argument\n          second_argument_makes_the_line_too_long):\n          pass\n    ')
        expected_formatted_code = textwrap.dedent('        def overly_long_function_name(\n            # comment about the first argument\n            first_argument_with_a_very_long_name_or_so,\n            # comment about the second argument\n            second_argument_makes_the_line_too_long\n        ):\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testDedentImportAsNames(self):
        if False:
            return 10
        code = textwrap.dedent('        from module import (\n            internal_function as function,\n            SOME_CONSTANT_NUMBER1,\n            SOME_CONSTANT_NUMBER2,\n            SOME_CONSTANT_NUMBER3,\n        )\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testDedentTestListGexp(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        try:\n            pass\n        except (\n            IOError, OSError, LookupError, RuntimeError, OverflowError\n        ) as exception:\n            pass\n\n        try:\n            pass\n        except (\n            IOError, OSError, LookupError, RuntimeError, OverflowError,\n        ) as exception:\n            pass\n    ')
        expected_formatted_code = textwrap.dedent('        try:\n            pass\n        except (\n            IOError, OSError, LookupError, RuntimeError, OverflowError\n        ) as exception:\n            pass\n\n        try:\n            pass\n        except (\n            IOError,\n            OSError,\n            LookupError,\n            RuntimeError,\n            OverflowError,\n        ) as exception:\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testBrokenIdempotency(self):
        if False:
            for i in range(10):
                print('nop')
        pass0_code = textwrap.dedent('        try:\n            pass\n        except (IOError, OSError, LookupError, RuntimeError, OverflowError) as exception:\n            pass\n    ')
        pass1_code = textwrap.dedent('        try:\n            pass\n        except (\n            IOError, OSError, LookupError, RuntimeError, OverflowError\n        ) as exception:\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(pass0_code)
        self.assertCodeEqual(pass1_code, reformatter.Reformat(llines))
        pass2_code = textwrap.dedent('        try:\n            pass\n        except (\n            IOError, OSError, LookupError, RuntimeError, OverflowError\n        ) as exception:\n            pass\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(pass1_code)
        self.assertCodeEqual(pass2_code, reformatter.Reformat(llines))

    def testIfExprHangingIndent(self):
        if False:
            print('Hello World!')
        unformatted_code = textwrap.dedent("        if True:\n            if True:\n                if True:\n                    if not self.frobbies and (\n                       self.foobars.counters['db.cheeses'] != 1 or\n                       self.foobars.counters['db.marshmellow_skins'] != 1):\n                        pass\n    ")
        expected_formatted_code = textwrap.dedent("        if True:\n            if True:\n                if True:\n                    if not self.frobbies and (\n                        self.foobars.counters['db.cheeses'] != 1 or\n                        self.foobars.counters['db.marshmellow_skins'] != 1\n                    ):\n                        pass\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testSimpleDedenting(self):
        if False:
            return 10
        unformatted_code = textwrap.dedent('        if True:\n            self.assertEqual(result.reason_not_added, "current preflight is still running")\n    ')
        expected_formatted_code = textwrap.dedent('        if True:\n            self.assertEqual(\n                result.reason_not_added, "current preflight is still running"\n            )\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testDedentingWithSubscripts(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        class Foo:\n            class Bar:\n                @classmethod\n                def baz(cls, clues_list, effect, constraints, constraint_manager):\n                    if clues_lists:\n                       return cls.single_constraint_not(clues_lists, effect, constraints[0], constraint_manager)\n\n    ')
        expected_formatted_code = textwrap.dedent('        class Foo:\n            class Bar:\n                @classmethod\n                def baz(cls, clues_list, effect, constraints, constraint_manager):\n                    if clues_lists:\n                        return cls.single_constraint_not(\n                            clues_lists, effect, constraints[0], constraint_manager\n                        )\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testDedentingCallsWithInnerLists(self):
        if False:
            return 10
        code = textwrap.dedent("        class _():\n            def _():\n                cls.effect_clues = {\n                    'effect': Clue((cls.effect_time, 'apache_host'), effect_line, 40)\n                }\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testDedentingListComprehension(self):
        if False:
            for i in range(10):
                print('nop')
        unformatted_code = textwrap.dedent('        class Foo():\n            def _pack_results_for_constraint_or():\n                self.param_groups = dict(\n                    (\n                        key + 1, ParamGroup(groups[key], default_converter)\n                    ) for key in six.moves.range(len(groups))\n                )\n\n                for combination in cls._clues_combinations(clues_lists):\n                    if all(\n                        cls._verify_constraint(combination, effect, constraint)\n                        for constraint in constraints\n                    ):\n                        pass\n\n                guessed_dict = dict(\n                    (\n                        key, guessed_pattern_matches[key]\n                    ) for key in six.moves.range(len(guessed_pattern_matches))\n                )\n\n                content = "".join(\n                    itertools.chain(\n                        (first_line_fragment, ), lines_between, (last_line_fragment, )\n                    )\n                )\n\n                rule = Rule(\n                    [self.cause1, self.cause2, self.cause1, self.cause2], self.effect, constraints1,\n                    Rule.LINKAGE_AND\n                )\n\n                assert sorted(log_type.files_to_parse) == [\n                    (\'localhost\', os.path.join(path, \'node_1.log\'), super_parser),\n                    (\'localhost\', os.path.join(path, \'node_2.log\'), super_parser)\n                ]\n    ')
        expected_formatted_code = textwrap.dedent('        class Foo():\n            def _pack_results_for_constraint_or():\n                self.param_groups = dict(\n                    (key + 1, ParamGroup(groups[key], default_converter))\n                    for key in six.moves.range(len(groups))\n                )\n\n                for combination in cls._clues_combinations(clues_lists):\n                    if all(\n                        cls._verify_constraint(combination, effect, constraint)\n                        for constraint in constraints\n                    ):\n                        pass\n\n                guessed_dict = dict(\n                    (key, guessed_pattern_matches[key])\n                    for key in six.moves.range(len(guessed_pattern_matches))\n                )\n\n                content = "".join(\n                    itertools.chain(\n                        (first_line_fragment, ), lines_between, (last_line_fragment, )\n                    )\n                )\n\n                rule = Rule(\n                    [self.cause1, self.cause2, self.cause1, self.cause2], self.effect,\n                    constraints1, Rule.LINKAGE_AND\n                )\n\n                assert sorted(log_type.files_to_parse) == [\n                    (\'localhost\', os.path.join(path, \'node_1.log\'), super_parser),\n                    (\'localhost\', os.path.join(path, \'node_2.log\'), super_parser)\n                ]\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testMustSplitDedenting(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent("        class _():\n            def _():\n                effect_line = FrontInput(\n                    effect_line_offset, line_content,\n                    LineSource('localhost', xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx)\n                )\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testDedentIfConditional(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent("        class _():\n            def _():\n                if True:\n                    if not self.frobbies and (\n                        self.foobars.counters['db.cheeses'] != 1 or\n                        self.foobars.counters['db.marshmellow_skins'] != 1\n                    ):\n                        pass\n    ")
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testDedentSet(self):
        if False:
            for i in range(10):
                print('nop')
        code = textwrap.dedent('        class _():\n            def _():\n                assert set(self.constraint_links.get_links()) == set(\n                    [\n                        (2, 10, 100),\n                        (2, 10, 200),\n                        (2, 20, 100),\n                        (2, 20, 200),\n                    ]\n                )\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        self.assertCodeEqual(code, reformatter.Reformat(llines))

    def testDedentingInnerScope(self):
        if False:
            for i in range(10):
                print('nop')
        code = textwrap.dedent('        class Foo():\n            @classmethod\n            def _pack_results_for_constraint_or(cls, combination, constraints):\n                return cls._create_investigation_result(\n                    (clue for clue in combination if not clue == Verifier.UNMATCHED),\n                    constraints, InvestigationResult.OR\n                )\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(code)
        reformatted_code = reformatter.Reformat(llines)
        self.assertCodeEqual(code, reformatted_code)
        llines = yapf_test_helper.ParseAndUnwrap(reformatted_code)
        reformatted_code = reformatter.Reformat(llines)
        self.assertCodeEqual(code, reformatted_code)

    def testCommentWithNewlinesInPrefix(self):
        if False:
            i = 10
            return i + 15
        unformatted_code = textwrap.dedent('        def foo():\n            if 0:\n                return False\n\n\n            #a deadly comment\n            elif 1:\n                return True\n\n\n        print(foo())\n    ')
        expected_formatted_code = textwrap.dedent('        def foo():\n            if 0:\n                return False\n\n            #a deadly comment\n            elif 1:\n                return True\n\n\n        print(foo())\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))

    def testIfStmtClosingBracket(self):
        if False:
            while True:
                i = 10
        unformatted_code = textwrap.dedent('        if (isinstance(value  , (StopIteration  , StopAsyncIteration  )) and exc.__cause__ is value_asdfasdfasdfasdfsafsafsafdasfasdfs):\n            return False\n    ')
        expected_formatted_code = textwrap.dedent('        if (\n            isinstance(value, (StopIteration, StopAsyncIteration)) and\n            exc.__cause__ is value_asdfasdfasdfasdfsafsafsafdasfasdfs\n        ):\n            return False\n    ')
        llines = yapf_test_helper.ParseAndUnwrap(unformatted_code)
        self.assertCodeEqual(expected_formatted_code, reformatter.Reformat(llines))
if __name__ == '__main__':
    unittest.main()