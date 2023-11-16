SHOULD_NOT_IMPORT_NUMPY = '\nimport sys\nfrom hypothesis import given, strategies as st\n\n@given(st.integers() | st.floats() | st.sampled_from(["a", "b"]))\ndef test_no_numpy_import(x):\n    assert "numpy" not in sys.modules\n'

def test_hypothesis_is_not_the_first_to_import_numpy(testdir):
    if False:
        while True:
            i = 10
    result = testdir.runpytest(testdir.makepyfile(SHOULD_NOT_IMPORT_NUMPY))
    result.assert_outcomes(passed=1, failed=0)
try:
    from hypothesis.extra.numpy import *
    star_import_works = True
except AttributeError:
    star_import_works = False

def test_wildcard_import():
    if False:
        for i in range(10):
            print('nop')
    assert star_import_works