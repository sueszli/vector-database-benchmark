import subprocess
import sys
from hypothesis import core as core

def test_is_running_under_pytest():
    if False:
        while True:
            i = 10
    assert core.running_under_pytest
FILE_TO_RUN = '\nimport hypothesis.core as core\nassert not core.running_under_pytest\n'

def test_is_not_running_under_pytest(tmp_path):
    if False:
        return 10
    pyfile = tmp_path.joinpath('test.py')
    pyfile.write_text(FILE_TO_RUN, encoding='utf-8')
    subprocess.check_call([sys.executable, str(pyfile)])
DOES_NOT_IMPORT_HYPOTHESIS = '\nimport sys\n\ndef test_pytest_plugin_does_not_import_hypothesis():\n    assert "hypothesis" not in sys.modules\n'

def test_plugin_does_not_import_pytest(testdir):
    if False:
        while True:
            i = 10
    testdir.makepyfile(DOES_NOT_IMPORT_HYPOTHESIS)
    testdir.runpytest_subprocess().assert_outcomes(passed=1)