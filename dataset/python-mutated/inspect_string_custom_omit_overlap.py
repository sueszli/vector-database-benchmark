"""Custom infoType snippets.

This file contains sample code that uses the Data Loss Prevention API to create
custom infoType detectors to refine scan results.
"""
import google.cloud.dlp

def inspect_string_custom_omit_overlap(project: str, content_string: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Matches PERSON_NAME and a custom detector,\n    but if they overlap only matches the custom detector\n\n    Uses the Data Loss Prevention API to omit matches on a built-in detector\n    if they overlap with matches from a custom detector\n\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        content_string: The string to inspect.\n\n    Returns:\n        None; the response from the API is printed to the terminal.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    custom_info_types = [{'info_type': {'name': 'VIP_DETECTOR'}, 'regex': {'pattern': 'Larry Page|Sergey Brin'}, 'exclusion_type': google.cloud.dlp_v2.CustomInfoType.ExclusionType.EXCLUSION_TYPE_EXCLUDE}]
    rule_set = [{'info_types': [{'name': 'PERSON_NAME'}], 'rules': [{'exclusion_rule': {'exclude_info_types': {'info_types': [{'name': 'VIP_DETECTOR'}]}, 'matching_type': google.cloud.dlp_v2.MatchingType.MATCHING_TYPE_FULL_MATCH}}]}]
    inspect_config = {'info_types': [{'name': 'PERSON_NAME'}], 'custom_info_types': custom_info_types, 'rule_set': rule_set, 'include_quote': True}
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