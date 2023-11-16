"""Sample app that uses the Data Loss Prevention API to inspect a string, a
local file or a file on Google Cloud Storage."""
import argparse
import google.cloud.dlp

def inspect_image_file(project: str, filename: str, include_quote: bool=True) -> None:
    if False:
        print('Hello World!')
    'Uses the Data Loss Prevention API to analyze strings for\n    protected data in image file.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        filename: The path to the file to inspect.\n        include_quote: Boolean for whether to display a quote of the detected\n            information in the results.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    info_types = ['PHONE_NUMBER', 'EMAIL_ADDRESS', 'CREDIT_CARD_NUMBER']
    info_types = [{'name': info_type} for info_type in info_types]
    inspect_config = {'info_types': info_types, 'include_quote': include_quote}
    with open(filename, mode='rb') as f:
        byte_item = {'type_': 'IMAGE', 'data': f.read()}
    parent = f'projects/{project}/locations/global'
    response = dlp.inspect_content(request={'parent': parent, 'inspect_config': inspect_config, 'item': {'byte_item': byte_item}})
    if response.result.findings:
        for finding in response.result.findings:
            print(f'Quote: {finding.quote}')
            print(f'Info type: {finding.info_type.name}')
            print(f'Likelihood: {finding.likelihood}')
    else:
        print('No findings.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('filename', help='The path to the file to inspect.')
    parser.add_argument('--include_quote', help='A Boolean for whether to display a quote of the detectedinformation in the results.', default=True)
    args = parser.parse_args()
    inspect_image_file(args.project, args.filename, include_quote=args.include_quote)