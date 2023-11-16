import os
import quickstart_analyzeiampolicy
PROJECT = os.environ['GOOGLE_CLOUD_PROJECT']

def test_analyze_iam_policy(capsys):
    if False:
        return 10
    quickstart_analyzeiampolicy.analyze_iam_policy(PROJECT)
    (out, _) = capsys.readouterr()
    assert 'fully_explored: true' in out