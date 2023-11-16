"""Uses of the Data Loss Prevention API for de-identifying sensitive data
contained in table."""
from __future__ import annotations
import argparse
from typing import Dict, List, Union
import google.cloud.dlp

def deidentify_table_suppress_row(project: str, table_data: Dict[str, Union[List[str], List[List[str]]]], condition_field: str, condition_operator: str, condition_value: int) -> None:
    if False:
        i = 10
        return i + 15
    ' Uses the Data Loss Prevention API to de-identify sensitive data in a\n      table by suppressing entire row/s based on a condition.\n\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        table_data: Dictionary representing table data.\n        condition_field: A table field within the record this condition is evaluated against.\n        condition_operator: Operator used to compare the field or infoType to the value. One of:\n            RELATIONAL_OPERATOR_UNSPECIFIED, EQUAL_TO, NOT_EQUAL_TO, GREATER_THAN, LESS_THAN, GREATER_THAN_OR_EQUALS,\n            LESS_THAN_OR_EQUALS, EXISTS.\n        condition_value: Value to compare against. [Mandatory, except for ``EXISTS`` tests.].\n\n    Example:\n\n    >> $ python deidentify_table_row_suppress.py     \'{"header": ["email", "phone number", "age"],\n    "rows": [["robertfrost@example.com", "4232342345", "35"],\n    ["johndoe@example.com", "4253458383", "64"]]}\'     "age" "GREATER_THAN" 50\n    >> \'{"header": ["email", "phone number", "age"],\n        "rows": [["robertfrost@example.com", "4232342345", "35", "21"]]}\'\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    headers = [{'name': val} for val in table_data['header']]
    rows = []
    for row in table_data['rows']:
        rows.append({'values': [{'string_value': cell_val} for cell_val in row]})
    table = {'headers': headers, 'rows': rows}
    item = {'table': table}
    condition = [{'field': {'name': condition_field}, 'operator': condition_operator, 'value': {'integer_value': condition_value}}]
    deidentify_config = {'record_transformations': {'record_suppressions': [{'condition': {'expressions': {'conditions': {'conditions': condition}}}}]}}
    parent = f'projects/{project}/locations/global'
    response = dlp.deidentify_content(request={'parent': parent, 'deidentify_config': deidentify_config, 'item': item})
    print(f'Table after de-identification: {response.item.table}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('table_data', help='Json string representing table data')
    parser.add_argument('--condition_field', help='A table Field within the record this condition is evaluated against.')
    parser.add_argument('--condition_operator', help='Operator used to compare the field or infoType to the value. One of: RELATIONAL_OPERATOR_UNSPECIFIED, EQUAL_TO, NOT_EQUAL_TO, GREATER_THAN, LESS_THAN, GREATER_THAN_OR_EQUALS, LESS_THAN_OR_EQUALS, EXISTS.')
    parser.add_argument('--condition_value', help='Value to compare against. [Mandatory, except for ``EXISTS`` tests.].')
    args = parser.parse_args()
    deidentify_table_suppress_row(args.project, args.table_data, condition_field=args.condition_field, condition_operator=args.condition_operator, condition_value=args.condition_value)