"""Sample app that uses the Data Loss Prevention API to inspect a string, a
local file or a file on Google Cloud Storage."""
import argparse
from typing import List
import google.cloud.dlp

def inspect_image_file_listed_infotypes(project: str, filename: str, info_types: List[str], include_quote: bool=True) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Uses the Data Loss Prevention API to analyze strings in an image for\n    data matching the given infoTypes.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        filename: The path of the image file to inspect.\n        info_types:  A list of strings representing infoTypes to look for.\n            A full list of info type categories can be fetched from the API.\n        include_quote: Boolean for whether to display a matching snippet of\n            the detected information in the results.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    info_types = [{'name': info_type} for info_type in info_types]
    inspect_config = {'info_types': info_types, 'include_quote': include_quote}
    with open(filename, mode='rb') as f:
        byte_item = {'type_': 'IMAGE', 'data': f.read()}
    parent = f'projects/{project}'
    response = dlp.inspect_content(request={'parent': parent, 'inspect_config': inspect_config, 'item': {'byte_item': byte_item}})
    if response.result.findings:
        for finding in response.result.findings:
            print(f'Info type: {finding.info_type.name}')
            if include_quote:
                print(f'Quote: {finding.quote}')
            print(f'Likelihood: {finding.likelihood} \n')
    else:
        print('No findings.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('filename', help='The path to the file to inspect.')
    parser.add_argument('--info_types', nargs='+', help='Strings representing info types to look for. A full list of info categories and types is available from the API. Examples include "FIRST_NAME", "LAST_NAME", "EMAIL_ADDRESS". ')
    parser.add_argument('--include_quote', help='A Boolean for whether to display a quote of the detectedinformation in the results.', default=True)
    args = parser.parse_args()
    inspect_image_file_listed_infotypes(args.project, args.filename, args.info_types, include_quote=args.include_quote)