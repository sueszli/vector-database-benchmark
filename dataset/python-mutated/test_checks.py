TEST_DECORATORS_ALONE = '\nimport hypothesis\nfrom hypothesis.strategies import composite, none\n\n@composite\ndef test_composite_is_not_a_test(draw):\n    # This strategy will be instantiated, but no draws == no calls.\n    return draw(none())\n\n@hypothesis.seed(0)\ndef test_seed_without_given_fails():\n    pass\n\n@hypothesis.example(x=None)\ndef test_example_without_given_fails():\n    pass\n\n@hypothesis.reproduce_failure(hypothesis.__version__, b"AA==")\ndef test_repro_without_given_fails():\n    pass\n'

def test_decorators_without_given_should_fail(testdir):
    if False:
        while True:
            i = 10
    script = testdir.makepyfile(TEST_DECORATORS_ALONE)
    result = testdir.runpytest(script)
    result.assert_outcomes(failed=4)
    assert 'pytest_runtest_call' not in '\n'.join(result.outlines)