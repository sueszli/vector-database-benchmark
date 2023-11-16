from google.cloud import dialogflow_v2

async def sample_streaming_analyze_content():
    client = dialogflow_v2.ParticipantsAsyncClient()
    audio_config = dialogflow_v2.InputAudioConfig()
    audio_config.audio_encoding = 'AUDIO_ENCODING_SPEEX_WITH_HEADER_BYTE'
    audio_config.sample_rate_hertz = 1817
    audio_config.language_code = 'language_code_value'
    request = dialogflow_v2.StreamingAnalyzeContentRequest(audio_config=audio_config, input_audio=b'input_audio_blob', participant='participant_value')
    requests = [request]

    def request_generator():
        if False:
            for i in range(10):
                print('nop')
        for request in requests:
            yield request
    stream = await client.streaming_analyze_content(requests=request_generator())
    async for response in stream:
        print(response)