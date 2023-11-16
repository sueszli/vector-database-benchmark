"""Uses of the Data Loss Prevention API for de-identifying sensitive data
contained in table."""
from __future__ import annotations
import argparse
from typing import Dict, List, Union
import google.cloud.dlp
from google.cloud.dlp_v2 import types

def deidentify_table_condition_replace_with_info_types(project: str, table_data: Dict[str, Union[List[str], List[List[str]]]], deid_content_list: List[str], info_types: List[str], condition_field: str=None, condition_operator: str=None, condition_value: int=None) -> types.dlp.Table:
    if False:
        print('Hello World!')
    'Uses the Data Loss Prevention API to de-identify sensitive data in a\n    table by replacing them with info-types based on a condition.\n    Args:\n       project: The Google Cloud project id to use as a parent resource.\n       table_data: Json string representing table data.\n       deid_content_list: A list of fields in table to de-identify.\n       info_types: A list of strings representing info types to look for.\n           A full list of info categories and types is available from the API.\n           Examples include "FIRST_NAME", "LAST_NAME", "EMAIL_ADDRESS". \'\n       condition_field: A table field within the record this condition is evaluated against.\n       condition_operator: Operator used to compare the field or infoType to the value. One of:\n           RELATIONAL_OPERATOR_UNSPECIFIED, EQUAL_TO, NOT_EQUAL_TO, GREATER_THAN, LESS_THAN, GREATER_THAN_OR_EQUALS,\n           LESS_THAN_OR_EQUALS, EXISTS.\n       condition_value: Value to compare against. [Mandatory, except for ``EXISTS`` tests.].\n\n    Returns:\n       De-identified table is returned;\n       the response from the API is also printed to the terminal.\n\n    Example:\n    >> $ python deidentify_table_condition_infotypes.py     \'{"header": ["email", "phone number", "age"],\n    "rows": [["robertfrost@example.com", "4232342345", "45"],\n    ["johndoe@example.com", "4253458383", "63"]]}\' ["email"]     ["EMAIL_ADDRESS"] "age" "GREATER_THAN" 50\n    >> \'{"header": ["email", "phone number", "age"],\n        "rows": [["robertfrost@example.com", "4232342345", "45"],\n        ["[EMAIL_ADDRESS]", "4253458383", "63"]]}\'\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    headers = [{'name': val} for val in table_data['header']]
    rows = []
    for row in table_data['rows']:
        rows.append({'values': [{'string_value': cell_val} for cell_val in row]})
    table = {'headers': headers, 'rows': rows}
    item = {'table': table}
    deid_field_list = [{'name': _i} for _i in deid_content_list]
    inspect_config = {'info_types': [{'name': info_type} for info_type in info_types]}
    condition = [{'field': {'name': condition_field}, 'operator': condition_operator, 'value': {'integer_value': condition_value}}]
    deidentify_config = {'record_transformations': {'field_transformations': [{'info_type_transformations': {'transformations': [{'primitive_transformation': {'replace_with_info_type_config': {}}}]}, 'fields': deid_field_list, 'condition': {'expressions': {'conditions': {'conditions': condition}}}}]}}
    parent = f'projects/{project}/locations/global'
    response = dlp.deidentify_content(request={'parent': parent, 'deidentify_config': deidentify_config, 'item': item, 'inspect_config': inspect_config})
    print(f'Table after de-identification: {response.item.table}')
    return response.item.table
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('table_data', help='Json string representing table data')
    parser.add_argument('deid_content_list', help='A list of fields in table to de-identify.')
    parser.add_argument('--info_types', nargs='+', help='Strings representing info types to look for. A full list of info categories and types is available from the API. Examples include "FIRST_NAME", "LAST_NAME", "EMAIL_ADDRESS". ')
    parser.add_argument('--condition_field', help='A table Field within the record this condition is evaluated against.')
    parser.add_argument('--condition_operator', help='Operator used to compare the field or infoType to the value. One of: RELATIONAL_OPERATOR_UNSPECIFIED, EQUAL_TO, NOT_EQUAL_TO, GREATER_THAN, LESS_THAN, GREATER_THAN_OR_EQUALS, LESS_THAN_OR_EQUALS, EXISTS.')
    parser.add_argument('--condition_value', help='Value to compare against. [Mandatory, except for ``EXISTS`` tests.].')
    args = parser.parse_args()
    deidentify_table_condition_replace_with_info_types(args.project, args.table_data, args.deid_content_list, args.info_types, condition_field=args.condition_field, condition_operator=args.condition_operator, condition_value=args.condition_value)