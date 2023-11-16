"""Uses of the Data Loss Prevention API for deidentifying sensitive data."""
from __future__ import annotations
import argparse
from typing import List
import google.cloud.dlp

def deidentify_with_mask(project: str, input_str: str, info_types: List[str], masking_character: str=None, number_to_mask: int=0) -> None:
    if False:
        i = 10
        return i + 15
    'Uses the Data Loss Prevention API to deidentify sensitive data in a\n    string by masking it with a character.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        input_str: The string to deidentify (will be treated as text).\n        info_types: A list of strings representing info types to look for.\n            A full list of info type categories can be fetched from the API.\n        masking_character: The character to mask matching sensitive data with.\n        number_to_mask: The maximum number of sensitive characters to mask in\n            a match. If omitted or set to zero, the API will default to no\n            maximum.\n    Returns:\n        None; the response from the API is printed to the terminal.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    parent = f'projects/{project}/locations/global'
    inspect_config = {'info_types': [{'name': info_type} for info_type in info_types]}
    deidentify_config = {'info_type_transformations': {'transformations': [{'primitive_transformation': {'character_mask_config': {'masking_character': masking_character, 'number_to_mask': number_to_mask}}}]}}
    item = {'value': input_str}
    response = dlp.deidentify_content(request={'parent': parent, 'deidentify_config': deidentify_config, 'inspect_config': inspect_config, 'item': item})
    print(response.item.value)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--info_types', nargs='+', help='Strings representing info types to look for. A full list of info categories and types is available from the API. Examples include "FIRST_NAME", "LAST_NAME", "EMAIL_ADDRESS". If unspecified, the three above examples will be used.', default=['FIRST_NAME', 'LAST_NAME', 'EMAIL_ADDRESS'])
    parser.add_argument('project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('item', help='The string to deidentify.')
    parser.add_argument('-n', '--number_to_mask', type=int, default=0, help='The maximum number of sensitive characters to mask in a match. If omitted the request or set to 0, the API will mask any mathcing characters.')
    parser.add_argument('-m', '--masking_character', help='The character to mask matching sensitive data with.')
    args = parser.parse_args()
    deidentify_with_mask(args.project, args.item, args.info_types, masking_character=args.masking_character, number_to_mask=args.number_to_mask)