import argparse
import re
from typing import List
from google.cloud import storage
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

def transcribe_batch_multiple_files_v2(project_id: str, gcs_uris: List[str], gcs_output_path: str) -> cloud_speech.BatchRecognizeResponse:
    if False:
        return 10
    'Transcribes audio from a Google Cloud Storage URI.\n\n    Args:\n        project_id: The Google Cloud project ID.\n        gcs_uris: The Google Cloud Storage URIs to transcribe.\n        gcs_output_path: The Cloud Storage URI to which to write the transcript.\n\n    Returns:\n        The BatchRecognizeResponse message.\n    '
    client = SpeechClient()
    config = cloud_speech.RecognitionConfig(auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(), language_codes=['en-US'], model='long')
    files = [cloud_speech.BatchRecognizeFileMetadata(uri=uri) for uri in gcs_uris]
    request = cloud_speech.BatchRecognizeRequest(recognizer=f'projects/{project_id}/locations/global/recognizers/_', config=config, files=files, recognition_output_config=cloud_speech.RecognitionOutputConfig(gcs_output_config=cloud_speech.GcsOutputConfig(uri=gcs_output_path)))
    operation = client.batch_recognize(request=request)
    print('Waiting for operation to complete...')
    response = operation.result(timeout=120)
    print('Operation finished. Fetching results from:')
    for uri in gcs_uris:
        file_results = response.results[uri]
        print(f'  {file_results.uri}...')
        (output_bucket, output_object) = re.match('gs://([^/]+)/(.*)', file_results.uri).group(1, 2)
        storage_client = storage.Client()
        bucket = storage_client.bucket(output_bucket)
        blob = bucket.blob(output_object)
        results_bytes = blob.download_as_bytes()
        batch_recognize_results = cloud_speech.BatchRecognizeResults.from_json(results_bytes, ignore_unknown_fields=True)
        for result in batch_recognize_results.results:
            print(f'     Transcript: {result.alternatives[0].transcript}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='GCP Project ID')
    parser.add_argument('gcs_uri', nargs='+', help='URI to GCS file')
    parser.add_argument('gcs_output_path', help='GCS URI to which to write the transcript')
    args = parser.parse_args()
    transcribe_batch_multiple_files_v2(args.project_id, args.gcs_uri, args.gcs_output_path)