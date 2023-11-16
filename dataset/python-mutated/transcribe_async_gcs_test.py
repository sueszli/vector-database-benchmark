import os
import re
from google.api_core.retry import Retry
import transcribe_async_gcs
import transcribe_diarization_gcs_beta
import transcribe_multilanguage_gcs_beta
import transcribe_word_level_confidence_gcs_beta
RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')
BUCKET = 'cloud-samples-data'
GCS_AUDIO_PATH = 'gs://' + BUCKET + '/speech/brooklyn_bridge.flac'
GCS_DIARIZATION_AUDIO_PATH = 'gs://' + BUCKET + '/speech/commercial_mono.wav'
GCS_MUTLILANGUAGE_PATH = 'gs://' + BUCKET + '/speech/Google_Gnome.wav'

@Retry()
def test_transcribe_gcs() -> None:
    if False:
        while True:
            i = 10
    transcript = transcribe_async_gcs.transcribe_gcs(GCS_AUDIO_PATH)
    assert re.search('how old is the Brooklyn Bridge', transcript, re.DOTALL | re.I)

def test_transcribe_diarization_gcs_beta() -> None:
    if False:
        i = 10
        return i + 15
    is_completed = transcribe_diarization_gcs_beta.transcribe_diarization_gcs_beta(GCS_DIARIZATION_AUDIO_PATH)
    assert is_completed

def test_transcribe_multilanguage_gcs_bets() -> None:
    if False:
        print('Hello World!')
    transcript = transcribe_multilanguage_gcs_beta.transcribe_file_with_multilanguage_gcs(GCS_MUTLILANGUAGE_PATH)
    assert re.search('Transcript: OK Google', transcript)

def test_transcribe_word_level_confidence_gcs_beta() -> None:
    if False:
        print('Hello World!')
    transcript = transcribe_word_level_confidence_gcs_beta.transcribe_file_with_word_level_confidence(GCS_AUDIO_PATH)
    assert re.search('Transcript: how old is the Brooklyn Bridge', transcript)
    assert re.search('First Word and Confidence: \\(how', transcript)