"""Uses of the Data Loss Prevention API for deidentifying sensitive data."""
from __future__ import annotations
import argparse
import google.cloud.dlp

def deidentify_with_simple_word_list(project: str, input_str: str, custom_info_type_name: str, word_list: list[str]) -> None:
    if False:
        print('Hello World!')
    'Uses the Data Loss Prevention API to de-identify sensitive data in a\n      string by matching against custom word list.\n\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        input_str: The string to deidentify (will be treated as text).\n        custom_info_type_name: The name of the custom info type to use.\n        word_list: The list of strings to match against.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    word_list = {'words': word_list}
    custom_info_types = [{'info_type': {'name': custom_info_type_name}, 'dictionary': {'word_list': word_list}}]
    inspect_config = {'custom_info_types': custom_info_types}
    deidentify_config = {'info_type_transformations': {'transformations': [{'primitive_transformation': {'replace_with_info_type_config': {}}}]}}
    item = {'value': input_str}
    parent = f'projects/{project}/locations/global'
    response = dlp.deidentify_content(request={'parent': parent, 'deidentify_config': deidentify_config, 'inspect_config': inspect_config, 'item': item})
    print(f'De-identified Content: {response.item.value}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('input_str', help='The string to deidentify.')
    parser.add_argument('custom_info_type_name', help='The name of the custom info type to use.')
    parser.add_argument('word_list', help='The list of strings to match against.')
    args = parser.parse_args()
    deidentify_with_simple_word_list(args.project, args.input_str, args.custom_info_type_name, args.word_list)