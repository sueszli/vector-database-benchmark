"""Custom infoType snippets.

This file contains sample code that uses the Data Loss Prevention API to create
custom infoType detectors to refine scan results.
"""
import google.cloud.dlp

def inspect_string_w_custom_hotword(project: str, content_string: str, custom_hotword: str='patient') -> None:
    if False:
        for i in range(10):
            print('nop')
    'Uses the Data Loss Prevention API increase likelihood for matches on\n       PERSON_NAME if the user specified custom hot-word is present. Only\n       includes findings with the increased likelihood by setting a minimum\n       likelihood threshold of VERY_LIKELY.\n\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        content_string: The string to inspect.\n        custom_hotword: The custom hot-word used for likelihood boosting.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    hotword_rule = {'hotword_regex': {'pattern': custom_hotword}, 'likelihood_adjustment': {'fixed_likelihood': google.cloud.dlp_v2.Likelihood.VERY_LIKELY}, 'proximity': {'window_before': 50}}
    rule_set = [{'info_types': [{'name': 'PERSON_NAME'}], 'rules': [{'hotword_rule': hotword_rule}]}]
    inspect_config = {'rule_set': rule_set, 'min_likelihood': google.cloud.dlp_v2.Likelihood.VERY_LIKELY, 'include_quote': True}
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