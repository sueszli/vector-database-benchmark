import argparse
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

def enable_cmek(project_id: str, kms_key_name: str) -> cloud_speech.RecognizeResponse:
    if False:
        return 10
    'Enable CMEK in a project and region.'
    client = SpeechClient()
    request = cloud_speech.UpdateConfigRequest(config=cloud_speech.Config(name=f'projects/{project_id}/locations/global/config', kms_key_name=kms_key_name), update_mask={'paths': ['kms_key_name']})
    response = client.update_config(request=request)
    print(f'Updated KMS key: {response.kms_key_name}')
    return response
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('project_id', help='GCP Project ID')
    parser.add_argument('kms_key_name', help='Resource path of a KMS key')
    args = parser.parse_args()
    enable_cmek(args.project_id, args.kms_key_name)