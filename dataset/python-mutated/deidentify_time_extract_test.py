import os
import shutil
import tempfile
from typing import Iterator, TextIO
import deidentify_time_extract as deid
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
CSV_FILE = os.path.join(os.path.dirname(__file__), '../resources/dates.csv')
DATE_FIELDS = ['birth_date', 'register_date']

@pytest.fixture(scope='module')
def tempdir() -> Iterator[TextIO]:
    if False:
        return 10
    tempdir = tempfile.mkdtemp()
    yield tempdir
    shutil.rmtree(tempdir)

def test_deidentify_with_time_extract(tempdir: TextIO, capsys: pytest.CaptureFixture) -> None:
    if False:
        while True:
            i = 10
    output_filepath = os.path.join(str(tempdir), 'year-extracted.csv')
    deid.deidentify_with_time_extract(GCLOUD_PROJECT, input_csv_file=CSV_FILE, output_csv_file=output_filepath, date_fields=DATE_FIELDS)
    (out, _) = capsys.readouterr()
    assert 'Successful' in out