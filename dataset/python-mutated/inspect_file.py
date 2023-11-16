"""Sample app that uses the Data Loss Prevention API to inspect a string, a
local file or a file on Google Cloud Storage."""
import argparse
import mimetypes
from typing import List
from typing import Optional
import google.cloud.dlp

def inspect_file(project: str, filename: str, info_types: List[str], min_likelihood: str=None, custom_dictionaries: List[str]=None, custom_regexes: List[str]=None, max_findings: Optional[int]=None, include_quote: bool=True, mime_type: str=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    "Uses the Data Loss Prevention API to analyze a file for protected data.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        filename: The path to the file to inspect.\n        info_types: A list of strings representing info types to look for.\n            A full list of info type categories can be fetched from the API.\n        min_likelihood: A string representing the minimum likelihood threshold\n            that constitutes a match. One of: 'LIKELIHOOD_UNSPECIFIED',\n            'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY'.\n        max_findings: The maximum number of findings to report; 0 = no maximum.\n        include_quote: Boolean for whether to display a quote of the detected\n            information in the results.\n        mime_type: The MIME type of the file. If not specified, the type is\n            inferred via the Python standard library's mimetypes module.\n    Returns:\n        None; the response from the API is printed to the terminal.\n    "
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    if not info_types:
        info_types = ['FIRST_NAME', 'LAST_NAME', 'EMAIL_ADDRESS']
    info_types = [{'name': info_type} for info_type in info_types]
    if custom_dictionaries is None:
        custom_dictionaries = []
    dictionaries = [{'info_type': {'name': f'CUSTOM_DICTIONARY_{i}'}, 'dictionary': {'word_list': {'words': custom_dict.split(',')}}} for (i, custom_dict) in enumerate(custom_dictionaries)]
    if custom_regexes is None:
        custom_regexes = []
    regexes = [{'info_type': {'name': f'CUSTOM_REGEX_{i}'}, 'regex': {'pattern': custom_regex}} for (i, custom_regex) in enumerate(custom_regexes)]
    custom_info_types = dictionaries + regexes
    inspect_config = {'info_types': info_types, 'custom_info_types': custom_info_types, 'min_likelihood': min_likelihood, 'include_quote': include_quote, 'limits': {'max_findings_per_request': max_findings}}
    if mime_type is None:
        mime_guess = mimetypes.MimeTypes().guess_type(filename)
        mime_type = mime_guess[0]
    supported_content_types = {None: 0, 'image/jpeg': 1, 'image/bmp': 2, 'image/png': 3, 'image/svg': 4, 'text/plain': 5}
    content_type_index = supported_content_types.get(mime_type, 0)
    with open(filename, mode='rb') as f:
        item = {'byte_item': {'type_': content_type_index, 'data': f.read()}}
    parent = f'projects/{project}'
    response = dlp.inspect_content(request={'parent': parent, 'inspect_config': inspect_config, 'item': item})
    if response.result.findings:
        for finding in response.result.findings:
            try:
                print(f'Quote: {finding.quote}')
            except AttributeError:
                pass
            print(f'Info type: {finding.info_type.name}')
            print(f'Likelihood: {finding.likelihood}')
    else:
        print('No findings.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='The path to the file to inspect.')
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('--info_types', action='append', help='Strings representing info types to look for. A full list of info categories and types is available from the API. Examples include "FIRST_NAME", "LAST_NAME", "EMAIL_ADDRESS". If unspecified, the three above examples will be used.', default=['FIRST_NAME', 'LAST_NAME', 'EMAIL_ADDRESS'])
    parser.add_argument('--custom_dictionaries', action='append', help='Strings representing comma-delimited lists of dictionary words to search for as custom info types. Each string is a comma delimited list of words representing a distinct dictionary.', default=None)
    parser.add_argument('--custom_regexes', action='append', help='Strings representing regex patterns to search for as custom  info types.', default=None)
    parser.add_argument('--min_likelihood', choices=['LIKELIHOOD_UNSPECIFIED', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY'], help='A string representing the minimum likelihood threshold that constitutes a match.')
    parser.add_argument('--max_findings', type=int, help='The maximum number of findings to report; 0 = no maximum.')
    parser.add_argument('--include_quote', type=bool, help='A boolean for whether to display a quote of the detected information in the results.', default=True)
    parser.add_argument('--mime_type', help="The MIME type of the file. If not specified, the type is inferred via the Python standard library's mimetypes module.")
    args = parser.parse_args()
    inspect_file(args.project, args.filename, args.info_types, custom_dictionaries=args.custom_dictionaries, custom_regexes=args.custom_regexes, min_likelihood=args.min_likelihood, max_findings=args.max_findings, include_quote=args.include_quote, mime_type=args.mime_type)