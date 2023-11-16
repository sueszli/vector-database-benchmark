from libcst.codemod import CodemodTest
from hypothesis.extra import codemods

def test_refactor_function_is_idempotent():
    if False:
        return 10
    before = 'from hypothesis.strategies import complex_numbers\n\ncomplex_numbers(None)\n'
    after = codemods.refactor(before)
    assert before.replace('None', 'min_magnitude=0') == after
    assert codemods.refactor(after) == after

class TestFixComplexMinMagnitude(CodemodTest):
    TRANSFORM = codemods.HypothesisFixComplexMinMagnitude

    def test_noop(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        before = '\n            from hypothesis.strategies import complex_numbers, complex_numbers as cn\n\n            complex_numbers(min_magnitude=1)  # value OK\n            complex_numbers(max_magnitude=None)  # different argument\n\n            class Foo:\n                def complex_numbers(self, **kw): pass\n\n                complex_numbers(min_magnitude=None)  # defined in a different scope\n        '
        self.assertCodemod(before=before, after=before)

    def test_substitution(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        before = '\n            from hypothesis.strategies import complex_numbers, complex_numbers as cn\n\n            complex_numbers(min_magnitude=None)  # simple call to fix\n            complex_numbers(min_magnitude=None, max_magnitude=1)  # plus arg after\n            complex_numbers(allow_infinity=False, min_magnitude=None)  # plus arg before\n            cn(min_magnitude=None)  # imported as alias\n        '
        self.assertCodemod(before=before, after=before.replace('None', '0'))

class TestFixPositionalKeywonlyArgs(CodemodTest):
    TRANSFORM = codemods.HypothesisFixPositionalKeywonlyArgs

    def test_substitution(self) -> None:
        if False:
            print('Hello World!')
        before = '\n            import hypothesis.strategies as st\n\n            st.floats(0, 1, False, False, 32)\n            st.fractions(0, 1, 9)\n        '
        after = '\n            import hypothesis.strategies as st\n\n            st.floats(0, 1, allow_nan=False, allow_infinity=False, width=32)\n            st.fractions(0, 1, max_denominator=9)\n        '
        self.assertCodemod(before=before, after=after)

    def test_noop_with_new_floats_kw(self) -> None:
        if False:
            print('Hello World!')
        before = '\n            import hypothesis.strategies as st\n\n            st.floats(0, 1, False, False, True, 32, False, False)  # allow_subnormal=True\n        '
        self.assertCodemod(before=before, after=before)

    def test_noop_if_unsure(self) -> None:
        if False:
            i = 10
            return i + 15
        before = "\n            import random\n\n            if random.getrandbits(1):\n                from hypothesis import target\n                from hypothesis.strategies import lists as sets\n\n                def fractions(*args):\n                    pass\n\n            else:\n                from hypothesis import target\n                from hypothesis.strategies import fractions, sets\n\n            fractions(0, 1, 9)\n            sets(None, 1)\n            target(0, 'label')\n        "
        after = before.replace("'label'", "label='label'")
        self.assertCodemod(before=before, after=after)

    def test_stateful_rule_noop(self):
        if False:
            while True:
                i = 10
        before = '\n            from hypothesis.stateful import RuleBasedStateMachine, rule\n\n            class MultipleRulesSameFuncMachine(RuleBasedStateMachine):\n                rule1 = rule()(lambda self: None)\n        '
        self.assertCodemod(before=before, after=before)

    def test_kwargs_noop(self):
        if False:
            while True:
                i = 10
        before = '\n            from hypothesis import target\n\n            kwargs = {"observation": 1, "label": "foobar"}\n            target(**kwargs)\n        '
        self.assertCodemod(before=before, after=before)

    def test_noop_with_too_many_arguments_passed(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        before = '\n            import hypothesis.strategies as st\n\n            st.sets(st.integers(), 0, 1, True)\n        '
        self.assertCodemod(before=before, after=before)

class TestHealthcheckAll(CodemodTest):
    TRANSFORM = codemods.HypothesisFixHealthcheckAll

    def test_noop_other_attributes(self):
        if False:
            print('Hello World!')
        before = 'result = Healthcheck.data_too_large'
        self.assertCodemod(before=before, after=before)

    def test_substitution(self) -> None:
        if False:
            i = 10
            return i + 15
        before = 'result = Healthcheck.all()'
        after = 'result = list(Healthcheck)'
        self.assertCodemod(before=before, after=after)

class TestFixCharactersArguments(CodemodTest):
    TRANSFORM = codemods.HypothesisFixCharactersArguments

    def test_substitution(self) -> None:
        if False:
            print('Hello World!')
        for (in_, out) in codemods.HypothesisFixCharactersArguments._replacements.items():
            before = f'\n                import hypothesis.strategies as st\n                st.characters({in_}=...)\n            '
            self.assertCodemod(before=before, after=before.replace(in_, out))

    def test_remove_redundant_exclude_categories(self) -> None:
        if False:
            print('Hello World!')
        args = 'blacklist_categories=OUT, whitelist_categories=IN'
        before = f'\n                import hypothesis.strategies as st\n                st.characters({args})\n            '
        self.assertCodemod(before=before, after=before.replace(args, 'categories=IN'))