"""Sample app that uses the Data Loss Prevention API to inspect a string, a
local file or a file on Google Cloud Storage."""
import argparse
import google.cloud.dlp

def inspect_phone_number(project: str, content_string: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Uses the Data Loss Prevention API to analyze strings for protected data.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        content_string: The string to inspect phone number from.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    info_types = [{'name': 'PHONE_NUMBER'}]
    inspect_config = {'info_types': info_types, 'include_quote': True}
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
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('content_string', help='The string to inspect phone number from.')
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    args = parser.parse_args()
    inspect_phone_number(args.project, args.content_string)