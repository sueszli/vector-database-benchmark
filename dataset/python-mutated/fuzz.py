"""Property-based tests for Black.

By Zac Hatfield-Dodds, based on my Hypothesmith tool for source code
generation.  You can run this file with `python`, `pytest`, or (soon)
a coverage-guided fuzzer I'm working on.
"""
import re
import hypothesmith
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
import black
from blib2to3.pgen2.tokenize import TokenError

@settings(max_examples=1000, derandomize=True, deadline=None, suppress_health_check=list(HealthCheck))
@given(src_contents=hypothesmith.from_grammar() | hypothesmith.from_node(), mode=st.builds(black.FileMode, line_length=st.just(88) | st.integers(0, 200), string_normalization=st.booleans(), preview=st.booleans(), is_pyi=st.booleans(), magic_trailing_comma=st.booleans()))
def test_idempotent_any_syntatically_valid_python(src_contents: str, mode: black.FileMode) -> None:
    if False:
        return 10
    compile(src_contents, '<string>', 'exec')
    try:
        dst_contents = black.format_str(src_contents, mode=mode)
    except black.InvalidInput:
        return
    except TokenError as e:
        if e.args[0] == 'EOF in multi-line statement' and re.search('\\\\($|\\r?\\n)', src_contents) is not None:
            return
        raise
    black.assert_equivalent(src_contents, dst_contents)
    black.assert_stable(src_contents, dst_contents, mode=mode)
if __name__ == '__main__':
    test_idempotent_any_syntatically_valid_python()
    try:
        import sys
        import atheris
    except ImportError:
        pass
    else:
        test = test_idempotent_any_syntatically_valid_python
        atheris.Setup(sys.argv, test.hypothesis.fuzz_one_input)
        atheris.Fuzz()