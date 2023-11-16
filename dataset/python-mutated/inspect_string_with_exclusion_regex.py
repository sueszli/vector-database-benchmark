"""Custom infoType snippets.

This file contains sample code that uses the Data Loss Prevention API to create
custom infoType detectors to refine scan results.
"""
import google.cloud.dlp

def inspect_string_with_exclusion_regex(project: str, content_string: str, exclusion_regex: str='.+@example.com') -> None:
    if False:
        i = 10
        return i + 15
    'Inspects the provided text, avoiding matches specified in the exclusion regex\n\n    Uses the Data Loss Prevention API to omit matches on EMAIL_ADDRESS if they match\n    the specified exclusion regex.\n\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        content_string: The string to inspect.\n        exclusion_regex: The regular expression to exclude matches on\n\n    Returns:\n        None; the response from the API is printed to the terminal.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    info_types_to_locate = [{'name': 'EMAIL_ADDRESS'}]
    rule_set = [{'info_types': info_types_to_locate, 'rules': [{'exclusion_rule': {'regex': {'pattern': exclusion_regex}, 'matching_type': google.cloud.dlp_v2.MatchingType.MATCHING_TYPE_FULL_MATCH}}]}]
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