import os
import uuid
import pytest
import conversation_management
import conversation_profile_management
import participant_management
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT')
AUDIO_FILE_PATH = '{0}/resources/book_a_room.wav'.format(os.path.realpath(os.path.dirname(__file__)))

@pytest.fixture
def conversation_profile_display_name():
    if False:
        i = 10
        return i + 15
    return f'sample_conversation_profile_{uuid.uuid4()}'

@pytest.fixture
def conversation_profile_id(conversation_profile_display_name):
    if False:
        while True:
            i = 10
    response = conversation_profile_management.create_conversation_profile_article_faq(project_id=PROJECT_ID, display_name=conversation_profile_display_name)
    conversation_profile_id = response.name.split('conversationProfiles/')[1].rstrip()
    yield conversation_profile_id
    conversation_profile_management.delete_conversation_profile(PROJECT_ID, conversation_profile_id)

@pytest.fixture
def conversation_id(conversation_profile_id):
    if False:
        while True:
            i = 10
    response = conversation_management.create_conversation(project_id=PROJECT_ID, conversation_profile_id=conversation_profile_id)
    conversation_id = response.name.split('conversations/')[1].rstrip()
    yield conversation_id
    conversation_management.complete_conversation(project_id=PROJECT_ID, conversation_id=conversation_id)

@pytest.fixture
def participant_id(conversation_id):
    if False:
        while True:
            i = 10
    response = participant_management.create_participant(project_id=PROJECT_ID, conversation_id=conversation_id, role='END_USER')
    participant_id = response.name.split('participants/')[1].rstrip()
    yield participant_id

def test_analyze_content_audio(capsys, conversation_id, participant_id):
    if False:
        while True:
            i = 10
    results = participant_management.analyze_content_audio(conversation_id=conversation_id, participant_id=participant_id, audio_file_path=AUDIO_FILE_PATH)
    out = ' '.join([result.message.content for result in results]).lower()
    assert 'book a room' in out

def test_analyze_content_audio_stream(capsys, conversation_id, participant_id):
    if False:
        for i in range(10):
            print('nop')

    class stream_generator:

        def __init__(self, audio_file_path):
            if False:
                print('Hello World!')
            self.audio_file_path = audio_file_path

        def generator(self):
            if False:
                i = 10
                return i + 15
            with open(self.audio_file_path, 'rb') as audio_file:
                while True:
                    chunk = audio_file.read(4096)
                    if not chunk:
                        break
                    yield chunk
    results = participant_management.analyze_content_audio_stream(conversation_id=conversation_id, participant_id=participant_id, sample_rate_herz=16000, stream=stream_generator(AUDIO_FILE_PATH), language_code='en-US', timeout=300)
    out = ' '.join([result.message.content for result in results]).lower()
    assert 'book a room' in out