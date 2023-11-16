"""Sample app that uses the Data Loss Prevention API to inspect a string, a
local file or a file on Google Cloud Storage."""
import argparse
from typing import List
import google.cloud.dlp

def inspect_column_values_w_custom_hotwords(project: str, table_header: List[str], table_rows: List[List[str]], info_types: List[str], custom_hotword: str) -> None:
    if False:
        print('Hello World!')
    'Uses the Data Loss Prevention API to inspect table data using built-in\n    infoType detectors, excluding columns that match a custom hot-word.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        table_header: List of strings representing table field names.\n        table_rows: List of rows representing table values.\n        info_types: The infoType for which hot-word rule is applied.\n        custom_hotword: The custom regular expression used for likelihood boosting.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    headers = [{'name': val} for val in table_header]
    rows = []
    for row in table_rows:
        rows.append({'values': [{'string_value': cell_val} for cell_val in row]})
    table = {'headers': headers, 'rows': rows}
    item = {'table': table}
    info_types = [{'name': info_type} for info_type in info_types]
    hotword_rule = {'hotword_regex': {'pattern': custom_hotword}, 'likelihood_adjustment': {'fixed_likelihood': google.cloud.dlp_v2.Likelihood.VERY_UNLIKELY}, 'proximity': {'window_before': 1}}
    rule_set = [{'info_types': info_types, 'rules': [{'hotword_rule': hotword_rule}]}]
    inspect_config = {'info_types': info_types, 'rule_set': rule_set, 'min_likelihood': google.cloud.dlp_v2.Likelihood.POSSIBLE, 'include_quote': True}
    parent = f'projects/{project}/locations/global'
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
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('--table_header', help='List of strings representing table field names.Example include \'[\'Fake_Email_Address\', \'Real_Email_Address]\'. The method can be used to exclude matches from entire column"Fake_Email_Address".')
    parser.add_argument('--table_rows', help='List of rows representing table values.Example: "[["example1@example.org", "test1@example.com],["example2@example.org", "test2@example.com]]"')
    parser.add_argument('--info_types', action='append', help='Strings representing info types to look for. A full list of info categories and types is available from the API. Examples include "FIRST_NAME", "LAST_NAME", "EMAIL_ADDRESS". ')
    parser.add_argument('custom_hotword', help='The custom regular expression used for likelihood boosting.')
    args = parser.parse_args()
    inspect_column_values_w_custom_hotwords(args.project, args.table_header, args.table_rows, args.info_types, args.custom_hotword)