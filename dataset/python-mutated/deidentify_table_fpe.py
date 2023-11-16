"""Uses of the Data Loss Prevention API for de-identifying sensitive data
contained in table."""
from __future__ import annotations
import argparse
import base64
from typing import List
import google.cloud.dlp

def deidentify_table_with_fpe(project: str, table_header: List[str], table_rows: List[List[str]], deid_field_names: List[str], key_name: str=None, wrapped_key: bytes=None, alphabet: str=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    "Uses the Data Loss Prevention API to de-identify sensitive data in a\n      table while maintaining format.\n\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        table_header: List of strings representing table field names.\n        table_rows: List of rows representing table data.\n        deid_field_names: A list of fields in table to de-identify.\n        key_name: The name of the Cloud KMS key used to encrypt ('wrap') the\n            AES-256 key. Example:\n            key_name = 'projects/YOUR_GCLOUD_PROJECT/locations/YOUR_LOCATION/\n            keyRings/YOUR_KEYRING_NAME/cryptoKeys/YOUR_KEY_NAME'\n        wrapped_key: The decrypted ('wrapped', in bytes) AES-256 key to use. This key\n            should be encrypted using the Cloud KMS key specified by key_name.\n        alphabet: The set of characters to replace sensitive ones with. For\n            more information, see https://cloud.google.com/dlp/docs/reference/\n            rest/v2/projects.deidentifyTemplates#ffxcommonnativealphabet\n    "
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    headers = [{'name': val} for val in table_header]
    rows = []
    for row in table_rows:
        rows.append({'values': [{'string_value': cell_val} for cell_val in row]})
    table = {'headers': headers, 'rows': rows}
    item = {'table': table}
    deid_field_names = [{'name': _i} for _i in deid_field_names]
    crypto_replace_ffx_fpe_config = {'crypto_key': {'kms_wrapped': {'wrapped_key': wrapped_key, 'crypto_key_name': key_name}}, 'common_alphabet': alphabet}
    deidentify_config = {'record_transformations': {'field_transformations': [{'primitive_transformation': {'crypto_replace_ffx_fpe_config': crypto_replace_ffx_fpe_config}, 'fields': deid_field_names}]}}
    parent = f'projects/{project}/locations/global'
    response = dlp.deidentify_content(request={'parent': parent, 'deidentify_config': deidentify_config, 'item': item})
    print(f'Table after de-identification: {response.item.table}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('table_header', help='List of strings representing table field names.')
    parser.add_argument('table_rows', help='List of rows representing table data')
    parser.add_argument('deid_field_names', help='A list of fields in table to de-identify.')
    parser.add_argument('key_name', help="The name of the Cloud KMS key used to encrypt ('wrap') the AES-256 key. Example: key_name = 'projects/YOUR_GCLOUD_PROJECT/locations/YOUR_LOCATION/keyRings/YOUR_KEYRING_NAME/cryptoKeys/YOUR_KEY_NAME'")
    parser.add_argument('wrapped_key', help="The encrypted ('wrapped') AES-256 key to use. This key should be encrypted using the Cloud KMS key specified by key_name.")
    parser.add_argument('-a', '--alphabet', default='ALPHA_NUMERIC', help='The set of characters to replace sensitive ones with. Commonly used subsets of the alphabet include "NUMERIC", "HEXADECIMAL", "UPPER_CASE_ALPHA_NUMERIC", "ALPHA_NUMERIC", "FFX_COMMON_NATIVE_ALPHABET_UNSPECIFIED"')
    args = parser.parse_args()
    deidentify_table_with_fpe(args.project, args.table_header, args.table_rows, args.deid_field_names, wrapped_key=base64.b64decode(args.wrapped_key), key_name=args.key_name, alphabet=args.alphabet)