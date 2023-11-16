"""Uses of the Data Loss Prevention API for deidentifying sensitive data."""
from __future__ import annotations
import argparse
import base64
import google.cloud.dlp

def reidentify_with_fpe(project: str, input_str: str, alphabet: str=None, surrogate_type: str=None, key_name: str=None, wrapped_key: str=None) -> None:
    if False:
        i = 10
        return i + 15
    "Uses the Data Loss Prevention API to reidentify sensitive data in a\n    string that was encrypted by Format Preserving Encryption (FPE).\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        input_str: The string to deidentify (will be treated as text).\n        alphabet: The set of characters to replace sensitive ones with. For\n            more information, see https://cloud.google.com/dlp/docs/reference/\n            rest/v2beta2/organizations.deidentifyTemplates#ffxcommonnativealphabet\n        surrogate_type: The name of the surrogate custom info type to used\n            during the encryption process.\n        key_name: The name of the Cloud KMS key used to encrypt ('wrap') the\n            AES-256 key. Example:\n            keyName = 'projects/YOUR_GCLOUD_PROJECT/locations/YOUR_LOCATION/\n            keyRings/YOUR_KEYRING_NAME/cryptoKeys/YOUR_KEY_NAME'\n        wrapped_key: The encrypted ('wrapped') AES-256 key to use. This key\n            should be encrypted using the Cloud KMS key specified by key_name.\n    Returns:\n        None; the response from the API is printed to the terminal.\n    "
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    parent = f'projects/{project}/locations/global'
    wrapped_key = base64.b64decode(wrapped_key)
    reidentify_config = {'info_type_transformations': {'transformations': [{'primitive_transformation': {'crypto_replace_ffx_fpe_config': {'crypto_key': {'kms_wrapped': {'wrapped_key': wrapped_key, 'crypto_key_name': key_name}}, 'common_alphabet': alphabet, 'surrogate_info_type': {'name': surrogate_type}}}}]}}
    inspect_config = {'custom_info_types': [{'info_type': {'name': surrogate_type}, 'surrogate_type': {}}]}
    item = {'value': input_str}
    response = dlp.reidentify_content(request={'parent': parent, 'reidentify_config': reidentify_config, 'inspect_config': inspect_config, 'item': item})
    print(response.item.value)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('item', help="The string to deidentify. Example: string = 'My SSN is 372819127'")
    parser.add_argument('surrogate_type', help="The name of the surrogate custom info type to use. Only necessary if you want to reverse the deidentification process. Can be essentially any arbitrary string, as long as it doesn't appear in your dataset otherwise.")
    parser.add_argument('key_name', help="The name of the Cloud KMS key used to encrypt ('wrap') the AES-256 key. Example: key_name = 'projects/YOUR_GCLOUD_PROJECT/locations/YOUR_LOCATION/keyRings/YOUR_KEYRING_NAME/cryptoKeys/YOUR_KEY_NAME'")
    parser.add_argument('wrapped_key', help="The encrypted ('wrapped') AES-256 key to use. This key should be encrypted using the Cloud KMS key specified by key_name.")
    parser.add_argument('-a', '--alphabet', default='ALPHA_NUMERIC', help='The set of characters to replace sensitive ones with. Commonly used subsets of the alphabet include "NUMERIC", "HEXADECIMAL", "UPPER_CASE_ALPHA_NUMERIC", "ALPHA_NUMERIC", "FFX_COMMON_NATIVE_ALPHABET_UNSPECIFIED"')
    args = parser.parse_args()
    reidentify_with_fpe(args.project, args.item, surrogate_type=args.surrogate_type, wrapped_key=args.wrapped_key, key_name=args.key_name, alphabet=args.alphabet)