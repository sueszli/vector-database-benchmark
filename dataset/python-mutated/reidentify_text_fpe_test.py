import os
import pytest
import reidentify_text_fpe as reid
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
WRAPPED_KEY = 'CiQAz0hX4+go8fJwn80Fr8pVImwx+tmZdqU7JL+7TN/S5JxBU9gSSQDhFHpFVyuzJps0YH9ls480mU+JLG7jI/0lL04i6XJRWqmI6gUSZRUtECYcLH5gXK4SXHlLrotx7Chxz/4z7SIpXFOBY61z0/U='
KEY_NAME = f'projects/{GCLOUD_PROJECT}/locations/global/keyRings/dlp-test/cryptoKeys/dlp-test'

def test_reidentify_text_with_fpe(capsys: pytest.CaptureFixture) -> None:
    if False:
        i = 10
        return i + 15
    labeled_fpe_string = 'My phone number is PHONE_NUMBER(10):9617256398'
    reid.reidentify_text_with_fpe(GCLOUD_PROJECT, labeled_fpe_string, wrapped_key=WRAPPED_KEY, key_name=KEY_NAME)
    (out, _) = capsys.readouterr()
    assert 'PHONE_NUMBER' not in out
    assert '9617256398' not in out