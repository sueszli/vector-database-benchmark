"""Uses of the Data Loss Prevention API for de-identifying sensitive data
contained in table."""
from __future__ import annotations
import argparse
import google.cloud.dlp

def deidentify_table_primitive_bucketing(project: str) -> None:
    if False:
        i = 10
        return i + 15
    'Uses the Data Loss Prevention API to de-identify sensitive data in\n    a table by replacing them with generalized bucket labels.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    parent = f'projects/{project}/locations/global'
    table_to_deid = {'header': ['age', 'patient', 'happiness_score'], 'rows': [['101', 'Charles Dickens', '95'], ['22', 'Jane Austen', '21'], ['90', 'Mark Twain', '75']]}
    headers = [{'name': val} for val in table_to_deid['header']]
    rows = []
    for row in table_to_deid['rows']:
        rows.append({'values': [{'string_value': cell_val} for cell_val in row]})
    table = {'headers': headers, 'rows': rows}
    item = {'table': table}
    buckets_config = [{'min_': {'integer_value': 0}, 'max_': {'integer_value': 25}, 'replacement_value': {'string_value': 'Low'}}, {'min_': {'integer_value': 25}, 'max_': {'integer_value': 75}, 'replacement_value': {'string_value': 'Medium'}}, {'min_': {'integer_value': 75}, 'max_': {'integer_value': 100}, 'replacement_value': {'string_value': 'High'}}]
    deidentify_config = {'record_transformations': {'field_transformations': [{'fields': [{'name': 'happiness_score'}], 'primitive_transformation': {'bucketing_config': {'buckets': buckets_config}}}]}}
    response = dlp.deidentify_content(request={'parent': parent, 'deidentify_config': deidentify_config, 'item': item})
    print(f'Table after de-identification: {response.item.table}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    args = parser.parse_args()
    deidentify_table_primitive_bucketing(args.project)