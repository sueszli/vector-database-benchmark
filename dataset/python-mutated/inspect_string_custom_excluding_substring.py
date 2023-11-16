"""Custom infoType snippets.

This file contains sample code that uses the Data Loss Prevention API to create
custom infoType detectors to refine scan results.
"""
from typing import List
import google.cloud.dlp

def inspect_string_custom_excluding_substring(project: str, content_string: str, exclusion_list: List[str]=['jimmy']) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Inspects the provided text with a custom detector, avoiding matches on specific tokens\n\n    Uses the Data Loss Prevention API to omit matches on a custom detector\n    if they include tokens in the specified exclusion list.\n\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        content_string: The string to inspect.\n        exclusion_list: The list of strings to ignore matches on\n\n    Returns:\n        None; the response from the API is printed to the terminal.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    custom_info_types = [{'info_type': {'name': 'CUSTOM_NAME_DETECTOR'}, 'regex': {'pattern': '[A-Z][a-z]{1,15}, [A-Z][a-z]{1,15}'}}]
    rule_set = [{'info_types': [{'name': 'CUSTOM_NAME_DETECTOR'}], 'rules': [{'exclusion_rule': {'dictionary': {'word_list': {'words': exclusion_list}}, 'matching_type': google.cloud.dlp_v2.MatchingType.MATCHING_TYPE_PARTIAL_MATCH}}]}]
    inspect_config = {'custom_info_types': custom_info_types, 'rule_set': rule_set, 'include_quote': True}
    item = {'value': content_string}
    parent = f'projects/{project}'
    response = dlp.inspect_content(request={'parent': parent, 'inspect_config': inspect_config, 'item': item})
    if response.result.findings:
        for finding in response.result.findings:
            print(f'Quote: {finding.quote}')
            print(f'Info type: {finding.info_type.name}')
            print(f'Likelihood: {finding.likelihood}')
    else:
        print('No findings.')