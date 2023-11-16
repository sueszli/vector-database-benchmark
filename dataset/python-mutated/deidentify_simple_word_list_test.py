import os
import deidentify_simple_word_list as deid
import pytest
GCLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')

def test_deidentify_with_simple_word_list(capsys: pytest.CaptureFixture) -> None:
    if False:
        for i in range(10):
            print('nop')
    deid.deidentify_with_simple_word_list(GCLOUD_PROJECT, 'Patient was seen in RM-YELLOW then transferred to rm green.', 'CUSTOM_ROOM_ID', ['RM-GREEN', 'RM-YELLOW', 'RM-ORANGE'])
    (out, _) = capsys.readouterr()
    assert 'Patient was seen in [CUSTOM_ROOM_ID] then transferred to [CUSTOM_ROOM_ID]' in out

def test_deidentify_with_simple_word_list_ignores_insensitive_data(capsys: pytest.CaptureFixture) -> None:
    if False:
        while True:
            i = 10
    deid.deidentify_with_simple_word_list(GCLOUD_PROJECT, 'Patient was seen in RM-RED then transferred to rm green', 'CUSTOM_ROOM_ID', ['RM-GREEN', 'RM-YELLOW', 'RM-ORANGE'])
    (out, _) = capsys.readouterr()
    assert 'Patient was seen in RM-RED then transferred to [CUSTOM_ROOM_ID]' in out