"""Custom infoType snippets.

This file contains sample code that uses the Data Loss Prevention API to create
custom infoType detectors to refine scan results.
"""
import google.cloud.dlp

def inspect_data_w_custom_hotwords(project: str, content_string: str) -> None:
    if False:
        while True:
            i = 10
    'Uses the Data Loss Prevention API to analyze string with medical record\n       number custom regex detector, with custom hotwords rules to boost finding\n       certainty under some circumstances.\n\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        content_string: The string to inspect.\n\n    Returns:\n        None; the response from the API is printed to the terminal.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    custom_info_types = [{'info_type': {'name': 'C_MRN'}, 'regex': {'pattern': '[1-9]{3}-[1-9]{1}-[1-9]{5}'}, 'likelihood': google.cloud.dlp_v2.Likelihood.POSSIBLE}]
    hotword_rule = {'hotword_regex': {'pattern': '(?i)(mrn|medical)(?-i)'}, 'likelihood_adjustment': {'fixed_likelihood': google.cloud.dlp_v2.Likelihood.VERY_LIKELY}, 'proximity': {'window_before': 10}}
    rule_set = [{'info_types': [{'name': 'C_MRN'}], 'rules': [{'hotword_rule': hotword_rule}]}]
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