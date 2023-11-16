import os
import re
from google.api_core.retry import Retry
from google.cloud.speech_v2.types import cloud_speech
import pytest
import transcribe_streaming_voice_activity_events
_RESOURCES = os.path.join(os.path.dirname(__file__), 'resources')

@Retry()
def test_transcribe_streaming_voice_activity_events(capsys: pytest.CaptureFixture) -> None:
    if False:
        return 10
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    responses = transcribe_streaming_voice_activity_events.transcribe_streaming_voice_activity_events(project_id, os.path.join(_RESOURCES, 'audio.wav'))
    transcript = ''
    for response in responses:
        for result in response.results:
            transcript += result.alternatives[0].transcript
    assert responses[0].speech_event_type == cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_BEGIN
    assert re.search('how old is the Brooklyn Bridge', transcript, re.DOTALL | re.I)