import os
import shutil
import tempfile
from typing import Iterator, TextIO
import pytest
import redact_image as redact
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
RESOURCE_DIRECTORY = os.path.join(os.path.dirname(__file__), '../resources')

@pytest.fixture(scope='module')
def tempdir() -> Iterator[TextIO]:
    if False:
        print('Hello World!')
    tempdir = tempfile.mkdtemp()
    yield tempdir
    shutil.rmtree(tempdir)

def test_redact_image_file(tempdir: TextIO, capsys: pytest.CaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    test_filepath = os.path.join(RESOURCE_DIRECTORY, 'test.png')
    output_filepath = os.path.join(tempdir, 'redacted.png')
    redact.redact_image(GCLOUD_PROJECT, test_filepath, output_filepath, ['FIRST_NAME', 'EMAIL_ADDRESS'])
    (out, _) = capsys.readouterr()
    assert output_filepath in out