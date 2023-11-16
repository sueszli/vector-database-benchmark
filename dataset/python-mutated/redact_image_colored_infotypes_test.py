import os
import shutil
import tempfile
from typing import Iterator, TextIO
import pytest
import redact_image_colored_infotypes as redact
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
RESOURCE_DIRECTORY = os.path.join(os.path.dirname(__file__), '../resources')

@pytest.fixture(scope='module')
def tempdir() -> Iterator[TextIO]:
    if False:
        i = 10
        return i + 15
    tempdir = tempfile.mkdtemp()
    yield tempdir
    shutil.rmtree(tempdir)

def test_redact_image_with_colored_info_types(tempdir: TextIO, capsys: pytest.CaptureFixture) -> None:
    if False:
        while True:
            i = 10
    test_filepath = os.path.join(RESOURCE_DIRECTORY, 'test.png')
    output_filepath = os.path.join(tempdir, 'redacted.png')
    redact.redact_image_with_colored_info_types(GCLOUD_PROJECT, test_filepath, output_filepath)
    (out, _) = capsys.readouterr()
    assert output_filepath in out