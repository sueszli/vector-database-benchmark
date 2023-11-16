import os
import shutil
import tempfile
from typing import Iterator, TextIO
import deidentify_date_shift as deid
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
WRAPPED_KEY = 'CiQAz0hX4+go8fJwn80Fr8pVImwx+tmZdqU7JL+7TN/S5JxBU9gSSQDhFHpFVyuzJps0YH9ls480mU+JLG7jI/0lL04i6XJRWqmI6gUSZRUtECYcLH5gXK4SXHlLrotx7Chxz/4z7SIpXFOBY61z0/U='
KEY_NAME = f'projects/{GCLOUD_PROJECT}/locations/global/keyRings/dlp-test/cryptoKeys/dlp-test'
CSV_FILE = os.path.join(os.path.dirname(__file__), '../resources/dates.csv')
DATE_SHIFTED_AMOUNT = 30
DATE_FIELDS = ['birth_date', 'register_date']
CSV_CONTEXT_FIELD = 'name'

@pytest.fixture(scope='module')
def tempdir() -> Iterator[TextIO]:
    if False:
        print('Hello World!')
    tempdir = tempfile.mkdtemp()
    yield tempdir
    shutil.rmtree(tempdir)

def test_deidentify_with_date_shift(tempdir: TextIO, capsys: pytest.CaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    output_filepath = os.path.join(tempdir, 'dates-shifted.csv')
    deid.deidentify_with_date_shift(GCLOUD_PROJECT, input_csv_file=CSV_FILE, output_csv_file=output_filepath, lower_bound_days=DATE_SHIFTED_AMOUNT, upper_bound_days=DATE_SHIFTED_AMOUNT, date_fields=DATE_FIELDS)
    (out, _) = capsys.readouterr()
    assert 'Successful' in out

def test_deidentify_with_date_shift_using_context_field(tempdir: TextIO, capsys: pytest.CaptureFixture) -> None:
    if False:
        return 10
    output_filepath = os.path.join(tempdir, 'dates-shifted.csv')
    deid.deidentify_with_date_shift(GCLOUD_PROJECT, input_csv_file=CSV_FILE, output_csv_file=output_filepath, lower_bound_days=DATE_SHIFTED_AMOUNT, upper_bound_days=DATE_SHIFTED_AMOUNT, date_fields=DATE_FIELDS, context_field_id=CSV_CONTEXT_FIELD, wrapped_key=WRAPPED_KEY, key_name=KEY_NAME)
    (out, _) = capsys.readouterr()
    assert 'Successful' in out