from google.cloud import speech_v1p1beta1 as speech

def transcribe_with_model_adaptation(project_id: str, location: str, storage_uri: str, custom_class_id: str, phrase_set_id: str) -> str:
    if False:
        while True:
            i = 10
    'Create`PhraseSet` and `CustomClasses` to create custom lists of similar\n    items that are likely to occur in your input data.\n\n    Args:\n        project_id: The GCP project ID.\n        location: The GCS location of the input audio.\n        storage_uri: The Cloud Storage URI of the input audio.\n        custom_class_id: The ID of the custom class to create\n\n    Returns:\n        The transcript of the input audio.\n    '
    adaptation_client = speech.AdaptationClient()
    parent = f'projects/{project_id}/locations/{location}'
    adaptation_client.create_custom_class({'parent': parent, 'custom_class_id': custom_class_id, 'custom_class': {'items': [{'value': 'sushido'}, {'value': 'altura'}, {'value': 'taneda'}]}})
    custom_class_name = f'projects/{project_id}/locations/{location}/customClasses/{custom_class_id}'
    phrase_set_response = adaptation_client.create_phrase_set({'parent': parent, 'phrase_set_id': phrase_set_id, 'phrase_set': {'boost': 10, 'phrases': [{'value': f'Visit restaurants like ${{{custom_class_name}}}'}]}})
    phrase_set_name = phrase_set_response.name
    speech_adaptation = speech.SpeechAdaptation(phrase_set_references=[phrase_set_name])
    config = speech.RecognitionConfig(encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, sample_rate_hertz=24000, language_code='en-US', adaptation=speech_adaptation)
    audio = speech.RecognitionAudio(uri=storage_uri)
    speech_client = speech.SpeechClient()
    response = speech_client.recognize(config=config, audio=audio)
    for result in response.results:
        print(f'Transcript: {result.alternatives[0].transcript}')
    return response.results[0].alternatives[0].transcript