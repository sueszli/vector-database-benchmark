import random
import attr
import pytest
from hypothesis import HealthCheck, Phase, Verbosity, assume, given, note, settings, strategies as st
from hypothesis.internal import escalation as esc
from hypothesis.internal.conjecture.data import Status
from hypothesis.internal.conjecture.engine import ConjectureRunner

def setup_module(module):
    if False:
        return 10
    esc.PREVENT_ESCALATION = True

def teardown_module(module):
    if False:
        return 10
    esc.PREVENT_ESCALATION = False

@attr.s()
class Write:
    value = attr.ib()
    child = attr.ib()

@attr.s()
class Branch:
    bits = attr.ib()
    children = attr.ib(default=attr.Factory(dict))

@attr.s()
class Terminal:
    status = attr.ib()
    payload = attr.ib(default=None)
nodes = st.deferred(lambda : terminals | writes | branches)
terminals = st.one_of(st.just(Terminal(Status.VALID)), st.just(Terminal(Status.INVALID)), st.builds(Terminal, status=st.just(Status.INTERESTING), payload=st.integers(0, 10)))
branches = st.builds(Branch, bits=st.integers(1, 64))
writes = st.builds(Write, value=st.binary(min_size=1), child=nodes)
_default_phases = settings.default.phases

def run_language_test_for(root, data, seed):
    if False:
        return 10
    random.seed(seed)

    def test(local_data):
        if False:
            return 10
        node = root
        while not isinstance(node, Terminal):
            if isinstance(node, Write):
                local_data.write(node.value)
                node = node.child
            else:
                assert isinstance(node, Branch)
                c = local_data.draw_bits(node.bits)
                try:
                    node = node.children[c]
                except KeyError:
                    if data is None:
                        return
                    node = node.children.setdefault(c, data.draw(nodes))
        assert isinstance(node, Terminal)
        if node.status == Status.INTERESTING:
            local_data.mark_interesting(node.payload)
        elif node.status == Status.INVALID:
            local_data.mark_invalid()
    runner = ConjectureRunner(test, settings=settings(max_examples=1, database=None, suppress_health_check=list(HealthCheck), verbosity=Verbosity.quiet, phases=_default_phases))
    try:
        runner.run()
    finally:
        if data is not None:
            note(root)
    assume(runner.interesting_examples)

@settings(suppress_health_check=list(HealthCheck), deadline=None, phases=set(settings.default.phases) - {Phase.shrink})
@given(st.data())
def test_explore_an_arbitrary_language(data):
    if False:
        return 10
    root = data.draw(writes | branches)
    seed = data.draw(st.integers(0, 2 ** 64 - 1))
    run_language_test_for(root, data, seed)

@pytest.mark.parametrize('seed, language', [])
def test_run_specific_example(seed, language):
    if False:
        return 10
    "This test recreates individual languages generated with the main test.\n\n    These are typically manually pruned down a bit - e.g. it's\n    OK to remove VALID nodes because KeyError is treated as if it lead to one\n    in this test (but not in the @given test).\n\n    These tests are likely to be fairly fragile with respect to changes in the\n    underlying engine. Feel free to delete examples if they start failing after\n    a change.\n    "
    run_language_test_for(language, None, seed)