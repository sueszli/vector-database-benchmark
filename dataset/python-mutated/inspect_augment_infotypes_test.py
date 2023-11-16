import os
import inspect_augment_infotypes as inspect_content
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_inspect_string_augment_infotype(capsys: pytest.CaptureFixture) -> None:
    if False:
        while True:
            i = 10
    inspect_content.inspect_string_augment_infotype(GCLOUD_PROJECT, "The patient's name is Quasimodo", 'PERSON_NAME', ['quasimodo'])
    (out, _) = capsys.readouterr()
    assert 'Quote: Quasimodo' in out
    assert 'Info type: PERSON_NAME' in out