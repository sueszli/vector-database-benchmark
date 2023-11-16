from google.cloud import dialogflow_v2beta1

def sample_streaming_analyze_content():
    if False:
        i = 10
        return i + 15
    client = dialogflow_v2beta1.ParticipantsClient()
    audio_config = dialogflow_v2beta1.InputAudioConfig()
    audio_config.audio_encoding = 'AUDIO_ENCODING_SPEEX_WITH_HEADER_BYTE'
    audio_config.sample_rate_hertz = 1817
    audio_config.language_code = 'language_code_value'
    request = dialogflow_v2beta1.StreamingAnalyzeContentRequest(audio_config=audio_config, input_audio=b'input_audio_blob', participant='participant_value')
    requests = [request]

    def request_generator():
        if False:
            return 10
        for request in requests:
            yield request
    stream = client.streaming_analyze_content(requests=request_generator())
    for response in stream:
        print(response)