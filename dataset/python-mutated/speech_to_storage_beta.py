from google.cloud import speech
from google.cloud import storage
from google.cloud.speech_v1 import types

def export_transcript_to_storage_beta(input_storage_uri, output_storage_uri, encoding, sample_rate_hertz, language_code, bucket_name, object_name):
    if False:
        return 10
    audio = speech.RecognitionAudio(uri=input_storage_uri)
    output_config = speech.TranscriptOutputConfig(gcs_uri=output_storage_uri)
    config = speech.RecognitionConfig(encoding=encoding, sample_rate_hertz=sample_rate_hertz, language_code=language_code)
    request = speech.LongRunningRecognizeRequest(audio=audio, config=config, output_config=output_config)
    speech_client = speech.SpeechClient()
    storage_client = storage.Client()
    operation = speech_client.long_running_recognize(request=request)
    print('Waiting for operation to complete...')
    operation.result(timeout=90)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.get_blob(object_name)
    results_bytes = blob.download_as_bytes()
    storage_transcript = types.LongRunningRecognizeResponse.from_json(results_bytes, ignore_unknown_fields=True)
    for result in storage_transcript.results:
        print(f'Transcript: {result.alternatives[0].transcript}')
        print(f'Confidence: {result.alternatives[0].confidence}')
    return storage_transcript.results