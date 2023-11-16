from google.cloud import mediatranslation_v1beta1

async def sample_streaming_translate_speech():
    client = mediatranslation_v1beta1.SpeechTranslationServiceAsyncClient()
    streaming_config = mediatranslation_v1beta1.StreamingTranslateSpeechConfig()
    streaming_config.audio_config.audio_encoding = 'audio_encoding_value'
    streaming_config.audio_config.source_language_code = 'source_language_code_value'
    streaming_config.audio_config.target_language_code = 'target_language_code_value'
    request = mediatranslation_v1beta1.StreamingTranslateSpeechRequest(streaming_config=streaming_config)
    requests = [request]

    def request_generator():
        if False:
            i = 10
            return i + 15
        for request in requests:
            yield request
    stream = await client.streaming_translate_speech(requests=request_generator())
    async for response in stream:
        print(response)