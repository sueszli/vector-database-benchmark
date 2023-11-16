import re
import pytest
pytest_plugins = 'pytester'
TEST_SUITE = '\nfrom hypothesis import given, settings, assume\nimport hypothesis.strategies as st\n\n\nfirst = None\n\n@settings(database=None)\n@given(st.integers())\ndef test_fails_once(some_int):\n    assume(abs(some_int) > 1000)\n    global first\n    if first is None:\n        first = some_int\n\n    assert some_int != first\n'
CONTAINS_SEED_INSTRUCTION = re.compile('--hypothesis-seed=\\d+', re.MULTILINE)

@pytest.mark.parametrize('seed', [0, 42, 'foo'])
def test_runs_repeatably_when_seed_is_set(seed, testdir):
    if False:
        return 10
    script = testdir.makepyfile(TEST_SUITE)
    results = [testdir.runpytest(script, '--verbose', '--strict-markers', f'--hypothesis-seed={seed}', '-rN') for _ in range(2)]
    for r in results:
        for l in r.stdout.lines:
            assert '--hypothesis-seed' not in l
    failure_lines = [l for r in results for l in r.stdout.lines if 'some_int=' in l]
    assert len(failure_lines) == 2
    assert failure_lines[0] == failure_lines[1]
HEALTH_CHECK_FAILURE = '\nimport os\n\nfrom hypothesis import given, strategies as st, assume, reject\n\nRECORD_EXAMPLES = <file>\n\nif os.path.exists(RECORD_EXAMPLES):\n    target = None\n    with open(RECORD_EXAMPLES, "r", encoding="utf-8") as i:\n        seen = set(map(int, i.read().strip().split("\\n")))\nelse:\n    target = open(RECORD_EXAMPLES, "w", encoding="utf-8")\n\n@given(st.integers())\ndef test_failure(i):\n    if target is None:\n        assume(i not in seen)\n    else:\n        target.write(f"{i}\\n")\n        reject()\n'

def test_repeats_healthcheck_when_following_seed_instruction(testdir, tmpdir):
    if False:
        print('Hello World!')
    health_check_test = HEALTH_CHECK_FAILURE.replace('<file>', repr(str(tmpdir.join('seen'))))
    script = testdir.makepyfile(health_check_test)
    initial = testdir.runpytest(script, '--verbose', '--strict-markers')
    match = CONTAINS_SEED_INSTRUCTION.search('\n'.join(initial.stdout.lines))
    initial_output = '\n'.join(initial.stdout.lines)
    match = CONTAINS_SEED_INSTRUCTION.search(initial_output)
    assert match is not None
    rerun = testdir.runpytest(script, '--verbose', '--strict-markers', match.group(0))
    rerun_output = '\n'.join(rerun.stdout.lines)
    assert 'FailedHealthCheck' in rerun_output
    assert '--hypothesis-seed' not in rerun_output
    rerun2 = testdir.runpytest(script, '--verbose', '--strict-markers', '--hypothesis-seed=10')
    rerun2_output = '\n'.join(rerun2.stdout.lines)
    assert 'FailedHealthCheck' not in rerun2_output