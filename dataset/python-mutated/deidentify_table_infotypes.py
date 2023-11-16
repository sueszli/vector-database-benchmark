"""Uses of the Data Loss Prevention API for de-identifying sensitive data
contained in table."""
from __future__ import annotations
import argparse
from typing import Dict, List, Union
import google.cloud.dlp

def deidentify_table_replace_with_info_types(project: str, table_data: Dict[str, Union[List[str], List[List[str]]]], info_types: List[str], deid_content_list: List[str]) -> None:
    if False:
        i = 10
        return i + 15
    ' Uses the Data Loss Prevention API to de-identify sensitive data in a\n      table by replacing them with info type.\n\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        table_data: Json string representing table data.\n        info_types: A list of strings representing info types to look for.\n            A full list of info type categories can be fetched from the API.\n        deid_content_list: A list of fields in table to de-identify\n\n    Returns:\n        None; the response from the API is printed to the terminal.\n\n    Example:\n    >> $ python deidentify_table_infotypes.py     \'{\n        "header": ["name", "email", "phone number"],\n        "rows": [\n            ["Robert Frost", "robertfrost@example.com", "4232342345"],\n            ["John Doe", "johndoe@example.com", "4253458383"]\n        ]\n    }\'     ["PERSON_NAME"] ["name"]\n    >> \'{\n            "header": ["name", "email", "phone number"],\n            "rows": [\n                ["[PERSON_NAME]", "robertfrost@example.com", "4232342345"],\n                ["[PERSON_NAME]", "johndoe@example.com", "4253458383"]\n            ]\n        }\'\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    headers = [{'name': val} for val in table_data['header']]
    rows = []
    for row in table_data['rows']:
        rows.append({'values': [{'string_value': cell_val} for cell_val in row]})
    table = {'headers': headers, 'rows': rows}
    item = {'table': table}
    deid_content_list = [{'name': _i} for _i in deid_content_list]
    inspect_config = {'info_types': [{'name': info_type} for info_type in info_types]}
    deidentify_config = {'record_transformations': {'field_transformations': [{'info_type_transformations': {'transformations': [{'primitive_transformation': {'replace_with_info_type_config': {}}}]}, 'fields': deid_content_list}]}}
    parent = f'projects/{project}/locations/global'
    response = dlp.deidentify_content(request={'parent': parent, 'deidentify_config': deidentify_config, 'item': item, 'inspect_config': inspect_config})
    print(f'Table after de-identification: {response.item.table}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('table_data', help='Json string representing a table.')
    parser.add_argument('--info_types', action='append', help='Strings representing info types to look for. A full list of info categories and types is available from the API. Examples include "FIRST_NAME", "LAST_NAME", "EMAIL_ADDRESS". ')
    parser.add_argument('deid_content_list', help='A list of fields in table to de-identify.')
    args = parser.parse_args()
    deidentify_table_replace_with_info_types(args.project, args.table_data, args.info_types, args.deid_content_list)