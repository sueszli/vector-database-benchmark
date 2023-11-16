"""Uses of the Data Loss Prevention API for deidentifying sensitive data."""
from __future__ import annotations
import argparse
from typing import List
import google.cloud.dlp

def deindentify_with_dictionary_replacement(project: str, input_str: str, info_types: List[str], word_list: List[str]) -> None:
    if False:
        i = 10
        return i + 15
    'Uses the Data Loss Prevention API to de-identify sensitive data in a\n    string by replacing each piece of detected sensitive data with a value\n    that Cloud DLP randomly selects from a list of words that you provide.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        input_str: The string to deidentify (will be treated as text).\n        info_types: A list of strings representing infoTypes to look for.\n        word_list: List of words or phrases to search for in the data.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    info_types = [{'name': info_type} for info_type in info_types]
    deidentify_config = {'info_type_transformations': {'transformations': [{'info_types': info_types, 'primitive_transformation': {'replace_dictionary_config': {'word_list': {'words': word_list}}}}]}}
    item = {'value': input_str}
    parent = f'projects/{project}/locations/global'
    response = dlp.deidentify_content(request={'parent': parent, 'deidentify_config': deidentify_config, 'inspect_config': {'info_types': info_types}, 'item': item})
    print(f'De-identified Content: {response.item.value}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('--info_types', action='append', help='Strings representing infoTypes to look for. A full list of info categories and types is available from the API. Examples include "FIRST_NAME", "LAST_NAME", "EMAIL_ADDRESS". ')
    parser.add_argument('input_str', help='The string to de-identify.')
    parser.add_argument('word_list', help='List of words or phrases to search for in the data.')
    args = parser.parse_args()
    deindentify_with_dictionary_replacement(args.project, args.input_str, args.info_types, args.word_list)