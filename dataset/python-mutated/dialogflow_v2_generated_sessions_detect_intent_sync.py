from google.cloud import dialogflow_v2

def sample_detect_intent():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2.SessionsClient()
    query_input = dialogflow_v2.QueryInput()
    query_input.audio_config.audio_encoding = 'AUDIO_ENCODING_SPEEX_WITH_HEADER_BYTE'
    query_input.audio_config.sample_rate_hertz = 1817
    query_input.audio_config.language_code = 'language_code_value'
    request = dialogflow_v2.DetectIntentRequest(session='session_value', query_input=query_input)
    response = client.detect_intent(request=request)
    print(response)