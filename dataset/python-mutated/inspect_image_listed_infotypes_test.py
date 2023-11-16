import os
import inspect_image_listed_infotypes as inspect_content
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
RESOURCE_DIRECTORY = os.path.join(os.path.dirname(__file__), '../resources')

def test_inspect_image_file_listed_infotypes(capsys: pytest.CaptureFixture) -> None:
    if False:
        while True:
            i = 10
    test_filepath = os.path.join(RESOURCE_DIRECTORY, 'test.png')
    inspect_content.inspect_image_file_listed_infotypes(GCLOUD_PROJECT, test_filepath, ['EMAIL_ADDRESS'])
    (out, _) = capsys.readouterr()
    assert 'Info type: EMAIL_ADDRESS' in out