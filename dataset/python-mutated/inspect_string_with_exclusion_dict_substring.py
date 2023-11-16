"""Custom infoType snippets.

This file contains sample code that uses the Data Loss Prevention API to create
custom infoType detectors to refine scan results.
"""
from typing import List
import google.cloud.dlp

def inspect_string_with_exclusion_dict_substring(project: str, content_string: str, exclusion_list: List[str]=['TEST']) -> None:
    if False:
        return 10
    'Inspects the provided text, avoiding matches that contain excluded tokens\n\n    Uses the Data Loss Prevention API to omit matches if they include tokens\n    in the specified exclusion list.\n\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        content_string: The string to inspect.\n        exclusion_list: The list of strings to ignore partial matches on\n\n    Returns:\n        None; the response from the API is printed to the terminal.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    info_types_to_locate = [{'name': 'EMAIL_ADDRESS'}, {'name': 'DOMAIN_NAME'}]
    rule_set = [{'info_types': info_types_to_locate, 'rules': [{'exclusion_rule': {'dictionary': {'word_list': {'words': exclusion_list}}, 'matching_type': google.cloud.dlp_v2.MatchingType.MATCHING_TYPE_PARTIAL_MATCH}}]}]
    inspect_config = {'info_types': info_types_to_locate, 'rule_set': rule_set, 'include_quote': True}
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