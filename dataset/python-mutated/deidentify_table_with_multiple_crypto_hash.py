"""Uses of the Data Loss Prevention API for de-identifying sensitive data
contained in table."""
from __future__ import annotations
import argparse
from typing import Dict, List, Union
import google.cloud.dlp

def deidentify_table_with_multiple_crypto_hash(project: str, table_data: Dict[str, Union[List[str], List[List[str]]]], info_types: List[str], transient_key_name_1: str, transient_key_name_2: str, deid_fields_1: List[str], deid_fields_2: List[str]) -> None:
    if False:
        return 10
    'Uses the Data Loss Prevention API to de-identify sensitive data\n    in table using multiple transient cryptographic hash keys.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        table_data: Dictionary representing table data.\n        info_types: A list of strings representing info types to look for.\n            A full list of info type categories can be fetched from the API.\n        transient_key_name_1: Name of the first transient crypto key used\n            for encryption. The scope of this key is a single API call.\n            It is generated for the transformation and then discarded.\n        transient_key_name_2: Name of the second transient crypto key used\n            for encryption. The scope of this key is a single API call.\n            It is generated for the transformation and then discarded.\n        deid_fields_1: List of column names in table to de-identify using\n            transient_key_name_1.\n        deid_fields_2: List of column names in table to de-identify using\n            transient_key_name_2.\n\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    headers = [{'name': val} for val in table_data['header']]
    rows = []
    for row in table_data['rows']:
        rows.append({'values': [{'string_value': cell_val} for cell_val in row]})
    table = {'headers': headers, 'rows': rows}
    item = {'table': table}
    info_types = [{'name': info_type} for info_type in info_types]
    crypto_hash_config_1 = {'crypto_key': {'transient': {'name': transient_key_name_1}}}
    crypto_hash_config_2 = {'crypto_key': {'transient': {'name': transient_key_name_2}}}
    deid_fields_1 = [{'name': field} for field in deid_fields_1]
    deid_fields_2 = [{'name': field} for field in deid_fields_2]
    inspect_config = {'info_types': info_types}
    deidentify_config = {'record_transformations': {'field_transformations': [{'fields': deid_fields_1, 'primitive_transformation': {'crypto_hash_config': crypto_hash_config_1}}, {'fields': deid_fields_2, 'info_type_transformations': {'transformations': [{'info_types': info_types, 'primitive_transformation': {'crypto_hash_config': crypto_hash_config_2}}]}}]}}
    parent = f'projects/{project}/locations/global'
    response = dlp.deidentify_content(request={'parent': parent, 'deidentify_config': deidentify_config, 'inspect_config': inspect_config, 'item': item})
    print(f'Table after de-identification: {response.item.table}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('table_data', help='Dictionary representing table data')
    parser.add_argument('--info_types', action='append', help='Strings representing infoTypes to look for. A full list of info categories and types is available from the API. Examples include "FIRST_NAME", "LAST_NAME", "EMAIL_ADDRESS". ')
    parser.add_argument('transient_key_name_1', help='Name of the first transient crypto key used for encryption.')
    parser.add_argument('transient_key_name_2', help='Name of the second transient crypto key used for encryption.')
    parser.add_argument('deid_fields_1', help='List of column names in table to de-identify using transient_key_name_1.')
    parser.add_argument('deid_fields_2', help='List of column names in table to de-identify using transient_key_name_2.')
    args = parser.parse_args()
    deidentify_table_with_multiple_crypto_hash(args.project, args.table_data, args.info_types, args.transient_key_name_1, args.transient_key_name_2, args.deid_fields_1, args.deid_fields_2)