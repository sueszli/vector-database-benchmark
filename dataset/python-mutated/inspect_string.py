"""Sample app that uses the Data Loss Prevention API to inspect a string, a
local file or a file on Google Cloud Storage."""
import argparse
from typing import List
import google.cloud.dlp

def inspect_string(project: str, content_string: str, info_types: List[str], custom_dictionaries: List[str]=None, custom_regexes: List[str]=None, min_likelihood: str=None, max_findings: str=None, include_quote: bool=True) -> None:
    if False:
        while True:
            i = 10
    "Uses the Data Loss Prevention API to analyze strings for protected data.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        content_string: The string to inspect.\n        info_types: A list of strings representing info types to look for.\n            A full list of info type categories can be fetched from the API.\n        min_likelihood: A string representing the minimum likelihood threshold\n            that constitutes a match. One of: 'LIKELIHOOD_UNSPECIFIED',\n            'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY'.\n        max_findings: The maximum number of findings to report; 0 = no maximum.\n        include_quote: Boolean for whether to display a quote of the detected\n            information in the results.\n    Returns:\n        None; the response from the API is printed to the terminal.\n    "
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    info_types = [{'name': info_type} for info_type in info_types]
    if custom_dictionaries is None:
        custom_dictionaries = []
    dictionaries = [{'info_type': {'name': f'CUSTOM_DICTIONARY_{i}'}, 'dictionary': {'word_list': {'words': custom_dict.split(',')}}} for (i, custom_dict) in enumerate(custom_dictionaries)]
    if custom_regexes is None:
        custom_regexes = []
    regexes = [{'info_type': {'name': f'CUSTOM_REGEX_{i}'}, 'regex': {'pattern': custom_regex}} for (i, custom_regex) in enumerate(custom_regexes)]
    custom_info_types = dictionaries + regexes
    inspect_config = {'info_types': info_types, 'custom_info_types': custom_info_types, 'min_likelihood': min_likelihood, 'include_quote': include_quote, 'limits': {'max_findings_per_request': max_findings}}
    item = {'value': content_string}
    parent = f'projects/{project}'
    response = dlp.inspect_content(request={'parent': parent, 'inspect_config': inspect_config, 'item': item})
    if response.result.findings:
        for finding in response.result.findings:
            try:
                if finding.quote:
                    print(f'Quote: {finding.quote}')
            except AttributeError:
                pass
            print(f'Info type: {finding.info_type.name}')
            print(f'Likelihood: {finding.likelihood}')
    else:
        print('No findings.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('item', help='The string to inspect.')
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('--info_types', nargs='+', help='Strings representing info types to look for. A full list of info categories and types is available from the API. Examples include "FIRST_NAME", "LAST_NAME", "EMAIL_ADDRESS". If unspecified, the three above examples will be used.', default=['FIRST_NAME', 'LAST_NAME', 'EMAIL_ADDRESS'])
    parser.add_argument('--custom_dictionaries', action='append', help='Strings representing comma-delimited lists of dictionary words to search for as custom info types. Each string is a comma delimited list of words representing a distinct dictionary.', default=None)
    parser.add_argument('--custom_regexes', action='append', help='Strings representing regex patterns to search for as custom  info types.', default=None)
    parser.add_argument('--min_likelihood', choices=['LIKELIHOOD_UNSPECIFIED', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY'], help='A string representing the minimum likelihood threshold that constitutes a match.')
    parser.add_argument('--max_findings', type=int, help='The maximum number of findings to report; 0 = no maximum.')
    parser.add_argument('--include_quote', type=bool, help='A boolean for whether to display a quote of the detected information in the results.', default=True)
    args = parser.parse_args()
    inspect_string(args.project, args.item, args.info_types, custom_dictionaries=args.custom_dictionaries, custom_regexes=args.custom_regexes, min_likelihood=args.min_likelihood, max_findings=args.max_findings, include_quote=args.include_quote)