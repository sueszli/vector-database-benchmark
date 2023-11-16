"""Uses of the Data Loss Prevention API for deidentifying sensitive data."""
from __future__ import annotations
import argparse
import base64
from typing import List
import google.cloud.dlp

def deidentify_with_deterministic(project: str, input_str: str, info_types: List[str], surrogate_type: str=None, key_name: str=None, wrapped_key: str=None) -> None:
    if False:
        print('Hello World!')
    "Deidentifies sensitive data in a string using deterministic encryption.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        input_str: The string to deidentify (will be treated as text).\n        info_types: A list of strings representing info types to look for.\n        surrogate_type: The name of the surrogate custom info type to use. Only\n            necessary if you want to reverse the deidentification process. Can\n            be essentially any arbitrary string, as long as it doesn't appear\n            in your dataset otherwise.\n        key_name: The name of the Cloud KMS key used to encrypt ('wrap') the\n            AES-256 key. Example:\n            key_name = 'projects/YOUR_GCLOUD_PROJECT/locations/YOUR_LOCATION/\n            keyRings/YOUR_KEYRING_NAME/cryptoKeys/YOUR_KEY_NAME'\n        wrapped_key: The encrypted ('wrapped') AES-256 key to use. This key\n            should be encrypted using the Cloud KMS key specified by key_name.\n    Returns:\n        None; the response from the API is printed to the terminal.\n    "
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    parent = f'projects/{project}/locations/global'
    wrapped_key = base64.b64decode(wrapped_key)
    crypto_replace_deterministic_config = {'crypto_key': {'kms_wrapped': {'wrapped_key': wrapped_key, 'crypto_key_name': key_name}}}
    if surrogate_type:
        crypto_replace_deterministic_config['surrogate_info_type'] = {'name': surrogate_type}
    inspect_config = {'info_types': [{'name': info_type} for info_type in info_types]}
    deidentify_config = {'info_type_transformations': {'transformations': [{'primitive_transformation': {'crypto_deterministic_config': crypto_replace_deterministic_config}}]}}
    item = {'value': input_str}
    response = dlp.deidentify_content(request={'parent': parent, 'deidentify_config': deidentify_config, 'inspect_config': inspect_config, 'item': item})
    print(response.item.value)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('input_str', help='The string to de-identify.')
    parser.add_argument('--info_types', action='append', help='Strings representing info types to look for. A full list of info categories and types is available from the API. Examples include "FIRST_NAME", "LAST_NAME", "EMAIL_ADDRESS". ')
    parser.add_argument('surrogate_type', help="The name of the surrogate custom info type to use. Only necessary if you want to reverse the de-identification process. Can be essentially any arbitrary string, as long as it doesn't appear in your dataset otherwise.")
    parser.add_argument('key_name', help="The name of the Cloud KMS key used to encrypt ('wrap') the AES-256 key. Example: key_name = 'projects/YOUR_GCLOUD_PROJECT/locations/YOUR_LOCATION/keyRings/YOUR_KEYRING_NAME/cryptoKeys/YOUR_KEY_NAME'")
    parser.add_argument('wrapped_key', help="The encrypted ('wrapped') AES-256 key to use. This key should be encrypted using the Cloud KMS key specified by key_name.")
    args = parser.parse_args()
    deidentify_with_deterministic(args.project, args.input_str, args.info_types, args.surrogate_type, args.key_name, args.wrapped_key)