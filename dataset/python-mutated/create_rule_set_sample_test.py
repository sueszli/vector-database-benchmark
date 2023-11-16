import os
from contentwarehouse.snippets import create_rule_set_sample
from contentwarehouse.snippets import test_utilities
import pytest
project_id = os.environ['GOOGLE_CLOUD_PROJECT']
location = 'us'

def test_create_rule_set(capsys: pytest.CaptureFixture) -> None:
    if False:
        print('Hello World!')
    project_number = test_utilities.get_project_number(project_id)
    create_rule_set_sample.create_rule_set(project_number=project_number, location=location)
    (out, _) = capsys.readouterr()
    assert 'Rule Set Created' in out
    assert 'Rule Sets' in out