"""Uses of the Data Loss Prevention API for de-identifying sensitive data
contained in table."""
from __future__ import annotations
import argparse
from typing import Dict, List, Union
import google.cloud.dlp

def deidentify_table_with_crypto_hash(project: str, table_data: Dict[str, Union[List[str], List[List[str]]]], info_types: List[str], transient_key_name: str) -> None:
    if False:
        while True:
            i = 10
    'Uses the Data Loss Prevention API to de-identify sensitive data\n    in a table using a cryptographic hash transformation.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        table_data: Dictionary representing table data.\n        info_types: A list of strings representing info types to look for.\n            A full list of info type categories can be fetched from the API.\n        transient_key_name: Name of the transient crypto key used for encryption.\n            The scope of this key is a single API call. It is generated for\n            the transformation and then discarded.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    headers = [{'name': val} for val in table_data['header']]
    rows = []
    for row in table_data['rows']:
        rows.append({'values': [{'string_value': cell_val} for cell_val in row]})
    table = {'headers': headers, 'rows': rows}
    item = {'table': table}
    info_types = [{'name': info_type} for info_type in info_types]
    crypto_hash_config = {'crypto_key': {'transient': {'name': transient_key_name}}}
    inspect_config = {'info_types': info_types}
    deidentify_config = {'info_type_transformations': {'transformations': [{'info_types': info_types, 'primitive_transformation': {'crypto_hash_config': crypto_hash_config}}]}}
    parent = f'projects/{project}/locations/global'
    response = dlp.deidentify_content(request={'parent': parent, 'deidentify_config': deidentify_config, 'inspect_config': inspect_config, 'item': item})
    print(f'Table after de-identification: {response.item.table}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('table_data', help='Dictionary representing table data')
    parser.add_argument('--info_types', action='append', help='Strings representing infoTypes to look for. A full list of info categories and types is available from the API. Examples include "FIRST_NAME", "LAST_NAME", "EMAIL_ADDRESS". ')
    parser.add_argument('transient_key_name', help='Name of the transient crypto key used for encryption.')
    args = parser.parse_args()
    deidentify_table_with_crypto_hash(args.project, args.table_data, args.info_types, args.transient_key_name)