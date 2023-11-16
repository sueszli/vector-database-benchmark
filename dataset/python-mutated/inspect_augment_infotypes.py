"""Sample app that uses the Data Loss Prevention API to inspect a string, a
local file or a file on Google Cloud Storage."""
import argparse
from typing import List
import google.cloud.dlp

def inspect_string_augment_infotype(project: str, input_str: str, info_type: str, word_list: List[str]) -> None:
    if False:
        while True:
            i = 10
    'Uses the Data Loss Prevention API to augment built-in infoType\n    detector and inspect the content string with augmented infoType.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        input_str: The string to inspect using augmented infoType\n            (will be treated as text).\n        info_type: A string representing built-in infoType to augment.\n            A full list of infoType categories can be fetched from the API.\n        word_list: List of words or phrases to be added to extend the behaviour\n            of built-in infoType.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    custom_info_types = [{'info_type': {'name': info_type}, 'dictionary': {'word_list': {'words': word_list}}}]
    inspect_config = {'custom_info_types': custom_info_types, 'include_quote': True}
    item = {'value': input_str}
    parent = f'projects/{project}'
    response = dlp.inspect_content(request={'parent': parent, 'inspect_config': inspect_config, 'item': item})
    if response.result.findings:
        for finding in response.result.findings:
            print(f'Quote: {finding.quote}')
            print(f'Info type: {finding.info_type.name}')
            print(f'Likelihood: {finding.likelihood} \n')
    else:
        print('No findings.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('input_str', help='The string to inspect.')
    parser.add_argument('--info_type', help='A String representing info types to look for. A full list of info categories and types is available from the API. Examples include "FIRST_NAME", "LAST_NAME", "EMAIL_ADDRESS". ')
    parser.add_argument('--word_list', help='List of words or phrases to be added to extend the behaviour of built-in infoType.')
    args = parser.parse_args()
    inspect_string_augment_infotype(args.project, args.input_str, args.info_type, args.word_list)