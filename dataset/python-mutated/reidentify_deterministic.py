"""Uses of the Data Loss Prevention API for deidentifying sensitive data."""
from __future__ import annotations
import argparse
import base64
import google.cloud.dlp

def reidentify_with_deterministic(project: str, input_str: str, surrogate_type: str=None, key_name: str=None, wrapped_key: str=None) -> None:
    if False:
        print('Hello World!')
    'Re-identifies content that was previously de-identified through deterministic encryption.\n    Args:\n        project: The Google Cloud project ID to use as a parent resource.\n        input_str: The string to be re-identified. Provide the entire token. Example:\n            EMAIL_ADDRESS_TOKEN(52):AVAx2eIEnIQP5jbNEr2j9wLOAd5m4kpSBR/0jjjGdAOmryzZbE/q\n        surrogate_type: The name of the surrogate custom infoType used\n            during the encryption process.\n        key_name: The name of the Cloud KMS key used to encrypt ("wrap") the\n            AES-256 key. Example:\n            keyName = \'projects/YOUR_GCLOUD_PROJECT/locations/YOUR_LOCATION/\n            keyRings/YOUR_KEYRING_NAME/cryptoKeys/YOUR_KEY_NAME\'\n        wrapped_key: The encrypted ("wrapped") AES-256 key previously used to encrypt the content.\n            This key must have been encrypted using the Cloud KMS key specified by key_name.\n    Returns:\n        None; the response from the API is printed to the terminal.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    parent = f'projects/{project}/locations/global'
    wrapped_key = base64.b64decode(wrapped_key)
    reidentify_config = {'info_type_transformations': {'transformations': [{'primitive_transformation': {'crypto_deterministic_config': {'crypto_key': {'kms_wrapped': {'wrapped_key': wrapped_key, 'crypto_key_name': key_name}}, 'surrogate_info_type': {'name': surrogate_type}}}}]}}
    inspect_config = {'custom_info_types': [{'info_type': {'name': surrogate_type}, 'surrogate_type': {}}]}
    item = {'value': input_str}
    response = dlp.reidentify_content(request={'parent': parent, 'reidentify_config': reidentify_config, 'inspect_config': inspect_config, 'item': item})
    print(response.item.value)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('input_str', help='The string to re-identify.')
    parser.add_argument('surrogate_type', help="The name of the surrogate custom info type to use. Only necessary if you want to reverse the de-identification process. Can be essentially any arbitrary string, as long as it doesn't appear in your dataset otherwise.")
    parser.add_argument('key_name', help="The name of the Cloud KMS key used to encrypt ('wrap') the AES-256 key. Example: key_name = 'projects/YOUR_GCLOUD_PROJECT/locations/YOUR_LOCATION/keyRings/YOUR_KEYRING_NAME/cryptoKeys/YOUR_KEY_NAME'")
    parser.add_argument('wrapped_key', help="The encrypted ('wrapped') AES-256 key to use. This key should be encrypted using the Cloud KMS key specified by key_name.")
    args = parser.parse_args()
    reidentify_with_deterministic(args.project, args.input_str, args.surrogate_type, args.key_name, args.wrapped_key)