from google.api_core.retry import Retry
import speech_adaptation_beta

@Retry()
def test_adaptation_beta() -> None:
    if False:
        print('Hello World!')
    response = speech_adaptation_beta.sample_recognize('gs://cloud-samples-data/speech/brooklyn_bridge.mp3', 'Brooklyn Bridge')
    assert 'brooklyn' in response.results[0].alternatives[0].transcript.lower()