"""Uses of the Data Loss Prevention API for deidentifying sensitive data."""
from __future__ import annotations
import argparse
from typing import List
import google.cloud.dlp

def deidentify_with_replace_infotype(project: str, item: str, info_types: List[str]) -> None:
    if False:
        return 10
    'Uses the Data Loss Prevention API to deidentify sensitive data in a\n    string by replacing it with the info type.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        item: The string to deidentify (will be treated as text).\n        info_types: A list of strings representing info types to look for.\n            A full list of info type categories can be fetched from the API.\n    Returns:\n        None; the response from the API is printed to the terminal.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    parent = f'projects/{project}/locations/global'
    inspect_config = {'info_types': [{'name': info_type} for info_type in info_types]}
    deidentify_config = {'info_type_transformations': {'transformations': [{'primitive_transformation': {'replace_with_info_type_config': {}}}]}}
    response = dlp.deidentify_content(request={'parent': parent, 'deidentify_config': deidentify_config, 'inspect_config': inspect_config, 'item': {'value': item}})
    print(response.item.value)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--info_types', action='append', help='Strings representing info types to look for. A full list of info categories and types is available from the API. Examples include "FIRST_NAME", "LAST_NAME", "EMAIL_ADDRESS". If unspecified, the three above examples will be used.', default=['FIRST_NAME', 'LAST_NAME', 'EMAIL_ADDRESS'])
    parser.add_argument('project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('item', help="The string to deidentify.Example: 'My credit card is 4242 4242 4242 4242'")
    args = parser.parse_args()
    deidentify_with_replace_infotype(args.project, item=args.item, info_types=args.info_types)