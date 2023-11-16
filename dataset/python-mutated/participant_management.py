"""Dialogflow API Python sample showing how to manage Participants.
"""
import google.auth
from google.cloud import dialogflow_v2beta1 as dialogflow
ROLES = ['HUMAN_AGENT', 'AUTOMATED_AGENT', 'END_USER']

def create_participant(project_id: str, conversation_id: str, role: str):
    if False:
        print('Hello World!')
    'Creates a participant in a given conversation.\n\n    Args:\n        project_id: The GCP project linked with the conversation profile.\n        conversation_id: Id of the conversation.\n        participant: participant to be created.'
    client = dialogflow.ParticipantsClient()
    conversation_path = dialogflow.ConversationsClient.conversation_path(project_id, conversation_id)
    if role in ROLES:
        response = client.create_participant(parent=conversation_path, participant={'role': role}, timeout=600)
        print('Participant Created.')
        print(f'Role: {response.role}')
        print(f'Name: {response.name}')
        return response

def analyze_content_text(project_id: str, conversation_id: str, participant_id: str, text: str):
    if False:
        return 10
    'Analyze text message content from a participant.\n\n    Args:\n        project_id: The GCP project linked with the conversation profile.\n        conversation_id: Id of the conversation.\n        participant_id: Id of the participant.\n        text: the text message that participant typed.'
    client = dialogflow.ParticipantsClient()
    participant_path = client.participant_path(project_id, conversation_id, participant_id)
    text_input = {'text': text, 'language_code': 'en-US'}
    response = client.analyze_content(participant=participant_path, text_input=text_input)
    print('AnalyzeContent Response:')
    print(f'Reply Text: {response.reply_text}')
    for suggestion_result in response.human_agent_suggestion_results:
        if suggestion_result.error is not None:
            print(f'Error: {suggestion_result.error.message}')
        if suggestion_result.suggest_articles_response:
            for answer in suggestion_result.suggest_articles_response.article_answers:
                print(f'Article Suggestion Answer: {answer.title}')
                print(f'Answer Record: {answer.answer_record}')
        if suggestion_result.suggest_faq_answers_response:
            for answer in suggestion_result.suggest_faq_answers_response.faq_answers:
                print(f'Faq Answer: {answer.answer}')
                print(f'Answer Record: {answer.answer_record}')
        if suggestion_result.suggest_smart_replies_response:
            for answer in suggestion_result.suggest_smart_replies_response.smart_reply_answers:
                print(f'Smart Reply: {answer.reply}')
                print(f'Answer Record: {answer.answer_record}')
    for suggestion_result in response.end_user_suggestion_results:
        if suggestion_result.error:
            print(f'Error: {suggestion_result.error.message}')
        if suggestion_result.suggest_articles_response:
            for answer in suggestion_result.suggest_articles_response.article_answers:
                print(f'Article Suggestion Answer: {answer.title}')
                print(f'Answer Record: {answer.answer_record}')
        if suggestion_result.suggest_faq_answers_response:
            for answer in suggestion_result.suggest_faq_answers_response.faq_answers:
                print(f'Faq Answer: {answer.answer}')
                print(f'Answer Record: {answer.answer_record}')
        if suggestion_result.suggest_smart_replies_response:
            for answer in suggestion_result.suggest_smart_replies_response.smart_reply_answers:
                print(f'Smart Reply: {answer.reply}')
                print(f'Answer Record: {answer.answer_record}')
    return response

def analyze_content_audio(conversation_id: str, participant_id: str, audio_file_path: str):
    if False:
        print('Hello World!')
    'Analyze audio content for END_USER with audio files.\n\n    Args:\n        conversation_id: Id of the conversation.\n        participant_id: Id of the participant.\n        audio_file_path: audio file in wav/mp3 format contains utterances of END_USER.\n    '
    (credentials, project_id) = google.auth.default()
    client = dialogflow.ParticipantsClient(credentials=credentials)
    participant_path = client.participant_path(project_id, conversation_id, participant_id)
    audio_encoding = dialogflow.AudioEncoding.AUDIO_ENCODING_LINEAR_16
    sample_rate_hertz = 16000

    def request_generator(audio_config, audio_file_path):
        if False:
            for i in range(10):
                print('nop')
        yield dialogflow.StreamingAnalyzeContentRequest(participant=participant_path, audio_config=audio_config)
        with open(audio_file_path, 'rb') as audio_file:
            while True:
                chunk = audio_file.read(4096)
                if not chunk:
                    break
                yield dialogflow.StreamingAnalyzeContentRequest(input_audio=chunk)
    audio_config = dialogflow.InputAudioConfig(audio_encoding=audio_encoding, language_code='en-US', sample_rate_hertz=sample_rate_hertz, single_utterance=True, model='phone_call', model_variant='USE_ENHANCED')
    requests = request_generator(audio_config, audio_file_path)
    responses = client.streaming_analyze_content(requests=requests)
    results = [response for response in responses]
    print('=' * 20)
    for result in results:
        print(f'Transcript: "{result.message.content}".')
    print('=' * 20)
    return results

def analyze_content_audio_stream(conversation_id: str, participant_id: str, sample_rate_herz: int, stream, timeout: int, language_code: str, single_utterance=False):
    if False:
        print('Hello World!')
    'Stream audio streams to Dialogflow and receive transcripts and\n    suggestions.\n\n    Args:\n        conversation_id: Id of the conversation.\n        participant_id: Id of the participant.\n        sample_rate_herz: herz rate of the sample.\n        stream: the stream to process. It should have generator() method to\n          yield input_audio.\n        timeout: the timeout of one stream.\n        language_code: the language code of the audio. Example: en-US\n        single_utterance: whether to use single_utterance.\n    '
    (credentials, project_id) = google.auth.default()
    client = dialogflow.ParticipantsClient(credentials=credentials)
    participant_name = client.participant_path(project_id, conversation_id, participant_id)
    audio_config = dialogflow.types.audio_config.InputAudioConfig(audio_encoding=dialogflow.types.audio_config.AudioEncoding.AUDIO_ENCODING_LINEAR_16, sample_rate_hertz=sample_rate_herz, language_code=language_code, single_utterance=single_utterance)

    def gen_requests(participant_name, audio_config, stream):
        if False:
            print('Hello World!')
        'Generates requests for streaming.'
        audio_generator = stream.generator()
        yield dialogflow.types.participant.StreamingAnalyzeContentRequest(participant=participant_name, audio_config=audio_config)
        for content in audio_generator:
            yield dialogflow.types.participant.StreamingAnalyzeContentRequest(input_audio=content)
    return client.streaming_analyze_content(gen_requests(participant_name, audio_config, stream), timeout=timeout)