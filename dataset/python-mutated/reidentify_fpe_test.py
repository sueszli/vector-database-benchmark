import os
import pytest
import reidentify_fpe as reid
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
WRAPPED_KEY = 'CiQAz0hX4+go8fJwn80Fr8pVImwx+tmZdqU7JL+7TN/S5JxBU9gSSQDhFHpFVyuzJps0YH9ls480mU+JLG7jI/0lL04i6XJRWqmI6gUSZRUtECYcLH5gXK4SXHlLrotx7Chxz/4z7SIpXFOBY61z0/U='
KEY_NAME = f'projects/{GCLOUD_PROJECT}/locations/global/keyRings/dlp-test/cryptoKeys/dlp-test'
SURROGATE_TYPE = 'SSN_TOKEN'

def test_reidentify_with_fpe(capsys: pytest.CaptureFixture) -> None:
    if False:
        print('Hello World!')
    labeled_fpe_string = 'My SSN is SSN_TOKEN(9):731997681'
    reid.reidentify_with_fpe(GCLOUD_PROJECT, labeled_fpe_string, surrogate_type=SURROGATE_TYPE, wrapped_key=WRAPPED_KEY, key_name=KEY_NAME, alphabet='NUMERIC')
    (out, _) = capsys.readouterr()
    assert '731997681' not in out