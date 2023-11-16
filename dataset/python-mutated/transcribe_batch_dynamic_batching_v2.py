import argparse
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

def transcribe_batch_dynamic_batching_v2(project_id: str, gcs_uri: str) -> cloud_speech.BatchRecognizeResults:
    if False:
        while True:
            i = 10
    'Transcribes audio from a Google Cloud Storage URI.\n\n    Args:\n        project_id: The Google Cloud project ID.\n        gcs_uri: The Google Cloud Storage URI.\n\n    Returns:\n        The RecognizeResponse.\n    '
    client = SpeechClient()
    config = cloud_speech.RecognitionConfig(auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(), language_codes=['en-US'], model='long')
    file_metadata = cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri)
    request = cloud_speech.BatchRecognizeRequest(recognizer=f'projects/{project_id}/locations/global/recognizers/_', config=config, files=[file_metadata], recognition_output_config=cloud_speech.RecognitionOutputConfig(inline_response_config=cloud_speech.InlineOutputConfig()), processing_strategy=cloud_speech.BatchRecognizeRequest.ProcessingStrategy.DYNAMIC_BATCHING)
    operation = client.batch_recognize(request=request)
    print('Waiting for operation to complete...')
    response = operation.result(timeout=120)
    for result in response.results[gcs_uri].transcript.results:
        print(f'Transcript: {result.alternatives[0].transcript}')
    return response.results[gcs_uri].transcript
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='GCP Project ID')
    parser.add_argument('gcs_uri', help='URI to GCS file')
    args = parser.parse_args()
    transcribe_batch_dynamic_batching_v2(args.project_id, args.gcs_uri)