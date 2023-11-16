"""Uses of the Data Loss Prevention API for deidentifying sensitive data."""
from __future__ import annotations
import argparse
import base64
import google.cloud.dlp

def reidentify_text_with_fpe(project: str, input_str: str, key_name: str=None, wrapped_key: str=None) -> None:
    if False:
        return 10
    "\n    Uses the Data Loss Prevention API to re-identify sensitive data in a\n    string using Format Preserving Encryption (FPE).\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        input_str: The string to re-identify (will be treated as text).\n        key_name: The name of the Cloud KMS key used to encrypt ('wrap') the\n            AES-256 key. Example:\n            key_name = 'projects/YOUR_GCLOUD_PROJECT/locations/YOUR_LOCATION/\n            keyRings/YOUR_KEYRING_NAME/cryptoKeys/YOUR_KEY_NAME'\n        wrapped_key: The encrypted ('wrapped') AES-256 key to use. This key\n            should be encrypted using the Cloud KMS key specified by key_name.\n    "
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    wrapped_key = base64.b64decode(wrapped_key)
    surrogate_info_type = {'name': 'PHONE_NUMBER'}
    crypto_replace_ffx_fpe_config = {'crypto_key': {'kms_wrapped': {'wrapped_key': wrapped_key, 'crypto_key_name': key_name}}, 'common_alphabet': 'NUMERIC', 'surrogate_info_type': surrogate_info_type}
    reidentify_config = {'info_type_transformations': {'transformations': [{'primitive_transformation': {'crypto_replace_ffx_fpe_config': crypto_replace_ffx_fpe_config}, 'info_types': [surrogate_info_type]}]}}
    inspect_config = {'custom_info_types': [{'info_type': surrogate_info_type, 'surrogate_type': {}}]}
    item = {'value': input_str}
    parent = f'projects/{project}/locations/global'
    response = dlp.reidentify_content(request={'parent': parent, 'reidentify_config': reidentify_config, 'inspect_config': inspect_config, 'item': item})
    print(f'Text after re-identification: {response.item.value}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('item', help="The string to deidentify. Example: string = 'My SSN is 372819127'")
    parser.add_argument('key_name', help="The name of the Cloud KMS key used to encrypt ('wrap') the AES-256 key. Example: key_name = 'projects/YOUR_GCLOUD_PROJECT/locations/YOUR_LOCATION/keyRings/YOUR_KEYRING_NAME/cryptoKeys/YOUR_KEY_NAME'")
    parser.add_argument('wrapped_key', help="The encrypted ('wrapped') AES-256 key to use. This key should be encrypted using the Cloud KMS key specified by key_name.")
    args = parser.parse_args()
    reidentify_text_with_fpe(args.project, args.item, wrapped_key=args.wrapped_key, key_name=args.key_name)