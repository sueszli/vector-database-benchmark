"""Custom infoType snippets.

This file contains sample code that uses the Data Loss Prevention API to create
custom infoType detectors to refine scan results.
"""
import google.cloud.dlp

def inspect_data_with_custom_regex_detector(project: str, content_string: str) -> None:
    if False:
        while True:
            i = 10
    'Uses the Data Loss Prevention API to analyze string with medical record\n       number custom regex detector\n\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        content_string: The string to inspect.\n\n    Returns:\n        None; the response from the API is printed to the terminal.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    custom_info_types = [{'info_type': {'name': 'C_MRN'}, 'regex': {'pattern': '[1-9]{3}-[1-9]{1}-[1-9]{5}'}, 'likelihood': google.cloud.dlp_v2.Likelihood.POSSIBLE}]
    inspect_config = {'custom_info_types': custom_info_types, 'include_quote': True}
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