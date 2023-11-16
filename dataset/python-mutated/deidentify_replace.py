"""Uses of the Data Loss Prevention API for deidentifying sensitive data."""
from __future__ import annotations
import argparse
from typing import List
import google.cloud.dlp

def deidentify_with_replace(project: str, input_str: str, info_types: List[str], replacement_str: str='REPLACEMENT_STR') -> None:
    if False:
        for i in range(10):
            print('nop')
    'Uses the Data Loss Prevention API to deidentify sensitive data in a\n    string by replacing matched input values with a value you specify.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        input_str: The string to deidentify (will be treated as text).\n        info_types: A list of strings representing info types to look for.\n        replacement_str: The string to replace all values that match given\n            info types.\n    Returns:\n        None; the response from the API is printed to the terminal.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    parent = f'projects/{project}/locations/global'
    inspect_config = {'info_types': [{'name': info_type} for info_type in info_types]}
    deidentify_config = {'info_type_transformations': {'transformations': [{'primitive_transformation': {'replace_config': {'new_value': {'string_value': replacement_str}}}}]}}
    item = {'value': input_str}
    response = dlp.deidentify_content(request={'parent': parent, 'deidentify_config': deidentify_config, 'inspect_config': inspect_config, 'item': item})
    print(response.item.value)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--info_types', nargs='+', help='Strings representing info types to look for. A full list of info categories and types is available from the API. Examples include "FIRST_NAME", "LAST_NAME", "EMAIL_ADDRESS". If unspecified, the three above examples will be used.', default=['FIRST_NAME', 'LAST_NAME', 'EMAIL_ADDRESS'])
    parser.add_argument('project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('item', help='The string to deidentify.')
    parser.add_argument('replacement_str', help='The string to replace all matched values with.')
    args = parser.parse_args()
    deidentify_with_replace(args.project, args.item, args.info_types, replacement_str=args.replacement_str)