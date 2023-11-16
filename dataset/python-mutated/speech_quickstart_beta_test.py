from google.api_core.retry import Retry
import speech_quickstart_beta

@Retry()
def test_quickstart_beta() -> None:
    if False:
        return 10
    response = speech_quickstart_beta.sample_recognize('gs://cloud-samples-data/speech/brooklyn_bridge.mp3')
    assert 'brooklyn' in response.results[0].alternatives[0].transcript.lower()