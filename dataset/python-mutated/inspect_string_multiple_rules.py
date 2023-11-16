"""Custom infoType snippets.

This file contains sample code that uses the Data Loss Prevention API to create
custom infoType detectors to refine scan results.
"""
import google.cloud.dlp

def inspect_string_multiple_rules(project: str, content_string: str) -> None:
    if False:
        return 10
    'Uses the Data Loss Prevention API to modify likelihood for matches on\n       PERSON_NAME combining multiple hotword and exclusion rules.\n\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        content_string: The string to inspect.\n\n    Returns:\n        None; the response from the API is printed to the terminal.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    patient_rule = {'hotword_regex': {'pattern': 'patient'}, 'proximity': {'window_before': 10}, 'likelihood_adjustment': {'fixed_likelihood': google.cloud.dlp_v2.Likelihood.VERY_LIKELY}}
    doctor_rule = {'hotword_regex': {'pattern': 'doctor'}, 'proximity': {'window_before': 10}, 'likelihood_adjustment': {'fixed_likelihood': google.cloud.dlp_v2.Likelihood.UNLIKELY}}
    quasimodo_rule = {'dictionary': {'word_list': {'words': ['quasimodo']}}, 'matching_type': google.cloud.dlp_v2.MatchingType.MATCHING_TYPE_PARTIAL_MATCH}
    redacted_rule = {'regex': {'pattern': 'REDACTED'}, 'matching_type': google.cloud.dlp_v2.MatchingType.MATCHING_TYPE_PARTIAL_MATCH}
    rule_set = [{'info_types': [{'name': 'PERSON_NAME'}], 'rules': [{'hotword_rule': patient_rule}, {'hotword_rule': doctor_rule}, {'exclusion_rule': quasimodo_rule}, {'exclusion_rule': redacted_rule}]}]
    inspect_config = {'info_types': [{'name': 'PERSON_NAME'}], 'rule_set': rule_set, 'include_quote': True}
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