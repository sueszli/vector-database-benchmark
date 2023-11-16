"""Uses of the Data Loss Prevention API for deidentifying sensitive data."""
from __future__ import annotations
import argparse
from typing import List
import google.cloud.dlp

def deidentify_with_exception_list(project: str, content_string: str, info_types: List[str], exception_list: List[str]) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Uses the Data Loss Prevention API to de-identify sensitive data in a\n      string but ignore matches against custom list.\n\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        content_string: The string to deidentify (will be treated as text).\n        info_types: A list of strings representing info types to look for.\n            A full list of info type categories can be fetched from the API.\n        exception_list: The list of strings to ignore matches on.\n\n    Returns:\n          None; the response from the API is printed to the terminal.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    info_types = [{'name': info_type} for info_type in info_types]
    rule_set = [{'info_types': info_types, 'rules': [{'exclusion_rule': {'dictionary': {'word_list': {'words': exception_list}}, 'matching_type': google.cloud.dlp_v2.MatchingType.MATCHING_TYPE_FULL_MATCH}}]}]
    inspect_config = {'info_types': info_types, 'rule_set': rule_set}
    deidentify_config = {'info_type_transformations': {'transformations': [{'primitive_transformation': {'replace_with_info_type_config': {}}}]}}
    item = {'value': content_string}
    parent = f'projects/{project}/locations/global'
    response = dlp.deidentify_content(request={'parent': parent, 'deidentify_config': deidentify_config, 'inspect_config': inspect_config, 'item': item})
    print(response.item.value)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('content_string', help='The string to de-identify.')
    parser.add_argument('--info_types', nargs='+', help='Strings representing info types to look for. A full list of info categories and types is available from the API. Examples include "FIRST_NAME", "LAST_NAME", "EMAIL_ADDRESS". ')
    parser.add_argument('exception_list', help='The list of strings to ignore matches against.')
    args = parser.parse_args()
    deidentify_with_exception_list(args.project, args.content_string, args.info_types, args.exception_list)