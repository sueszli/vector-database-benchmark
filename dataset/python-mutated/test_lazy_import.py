import os
import subprocess
import sys
SHOULD_NOT_IMPORT_TEST_RUNNERS = '\nimport sys\nimport unittest\nfrom hypothesis import given, strategies as st\n\nclass TestDoesNotImportRunners(unittest.TestCase):\n    strat = st.integers() | st.floats() | st.sampled_from(["a", "b"])\n\n    @given(strat)\n    def test_does_not_import_unittest2(self, x):\n        assert "unittest2" not in sys.modules\n\n    @given(strat)\n    def test_does_not_import_nose(self, x):\n        assert "nose" not in sys.modules\n\n    @given(strat)\n    def test_does_not_import_pytest(self, x):\n        assert "pytest" not in sys.modules\n\nif __name__ == \'__main__\':\n    unittest.main()\n'

def test_hypothesis_does_not_import_test_runners(tmp_path):
    if False:
        print('Hello World!')
    fname = tmp_path / 'test.py'
    fname.write_text(SHOULD_NOT_IMPORT_TEST_RUNNERS, encoding='utf-8')
    subprocess.check_call([sys.executable, str(fname)], env={**os.environ, 'HYPOTHESIS_NO_PLUGINS': '1'})