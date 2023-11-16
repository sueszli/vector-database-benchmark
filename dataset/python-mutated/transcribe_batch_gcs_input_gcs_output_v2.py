import argparse
import re
from google.cloud import storage
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

def transcribe_batch_gcs_input_gcs_output_v2(project_id: str, gcs_uri: str, gcs_output_path: str) -> cloud_speech.BatchRecognizeResults:
    if False:
        for i in range(10):
            print('nop')
    'Transcribes audio from a Google Cloud Storage URI.\n\n    Args:\n        project_id: The Google Cloud project ID.\n        gcs_uri: The Google Cloud Storage URI.\n        gcs_output_path: The Cloud Storage URI to which to write the transcript.\n\n    Returns:\n        The BatchRecognizeResults message.\n    '
    client = SpeechClient()
    config = cloud_speech.RecognitionConfig(auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(), language_codes=['en-US'], model='long')
    file_metadata = cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri)
    request = cloud_speech.BatchRecognizeRequest(recognizer=f'projects/{project_id}/locations/global/recognizers/_', config=config, files=[file_metadata], recognition_output_config=cloud_speech.RecognitionOutputConfig(gcs_output_config=cloud_speech.GcsOutputConfig(uri=gcs_output_path)))
    operation = client.batch_recognize(request=request)
    print('Waiting for operation to complete...')
    response = operation.result(timeout=120)
    file_results = response.results[gcs_uri]
    print(f'Operation finished. Fetching results from {file_results.uri}...')
    (output_bucket, output_object) = re.match('gs://([^/]+)/(.*)', file_results.uri).group(1, 2)
    storage_client = storage.Client()
    bucket = storage_client.bucket(output_bucket)
    blob = bucket.blob(output_object)
    results_bytes = blob.download_as_bytes()
    batch_recognize_results = cloud_speech.BatchRecognizeResults.from_json(results_bytes, ignore_unknown_fields=True)
    for result in batch_recognize_results.results:
        print(f'Transcript: {result.alternatives[0].transcript}')
    return batch_recognize_results
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='GCP Project ID')
    parser.add_argument('gcs_uri', help='URI to GCS file')
    parser.add_argument('gcs_output_path', help='GCS URI to which to write the transcript')
    args = parser.parse_args()
    transcribe_batch_gcs_input_gcs_output_v2(args.project_id, args.gcs_uri, args.gcs_output_path)