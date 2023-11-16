"""Uses of the Data Loss Prevention API for de-identifying sensitive data
contained in table."""
from __future__ import annotations
import argparse
from typing import Dict, List, Union
import google.cloud.dlp
from google.cloud.dlp_v2 import types

def deidentify_table_bucketing(project: str, table_data: Dict[str, Union[List[str], List[List[str]]]], deid_content_list: List[str], bucket_size: int, bucketing_lower_bound: int, bucketing_upper_bound: int) -> types.dlp.Table:
    if False:
        i = 10
        return i + 15
    'Uses the Data Loss Prevention API to de-identify sensitive data in a\n    table by replacing them with fixed size bucket ranges.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        table_data: Dictionary representing table data.\n        deid_content_list: A list of fields in table to de-identify.\n        bucket_size: Size of each bucket for fixed sized bucketing\n            (except for minimum and maximum buckets). So if ``bucketing_lower_bound`` = 10,\n            ``bucketing_upper_bound`` = 89, and ``bucket_size`` = 10, then the\n            following buckets would be used: -10, 10-20, 20-30, 30-40,\n            40-50, 50-60, 60-70, 70-80, 80-89, 89+.\n       bucketing_lower_bound: Lower bound value of buckets.\n       bucketing_upper_bound:  Upper bound value of buckets.\n\n    Returns:\n       De-identified table is returned;\n       the response from the API is also printed to the terminal.\n\n    Example:\n    >> $ python deidentify_table_bucketing.py         \'{"header": ["email", "phone number", "age"],\n        "rows": [["robertfrost@example.com", "4232342345", "35"],\n        ["johndoe@example.com", "4253458383", "68"]]}\'         ["age"] 10 0 100\n        >>  \'{"header": ["email", "phone number", "age"],\n            "rows": [["robertfrost@example.com", "4232342345", "30:40"],\n            ["johndoe@example.com", "4253458383", "60:70"]]}\'\n    '
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    parent = f'projects/{project}/locations/global'
    headers = [{'name': val} for val in table_data['header']]
    rows = []
    for row in table_data['rows']:
        rows.append({'values': [{'string_value': cell_val} for cell_val in row]})
    table = {'headers': headers, 'rows': rows}
    item = {'table': table}
    fixed_size_bucketing_config = {'bucket_size': bucket_size, 'lower_bound': {'integer_value': bucketing_lower_bound}, 'upper_bound': {'integer_value': bucketing_upper_bound}}
    deid_content_list = [{'name': _i} for _i in deid_content_list]
    deidentify_config = {'record_transformations': {'field_transformations': [{'fields': deid_content_list, 'primitive_transformation': {'fixed_size_bucketing_config': fixed_size_bucketing_config}}]}}
    response = dlp.deidentify_content(request={'parent': parent, 'deidentify_config': deidentify_config, 'item': item})
    print(f'Table after de-identification: {response.item.table}')
    return response.item.table
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('--table_data', help='Json string representing table data')
    parser.add_argument('--deid_content_list', help='A list of fields in table to de-identify.')
    parser.add_argument('--bucket_size', help='Size of each bucket for fixed sized bucketing.')
    parser.add_argument('--bucketing_lower_bound', help='Lower bound value of buckets.')
    parser.add_argument('--bucketing_upper_bound', help='Upper bound value of buckets.')
    args = parser.parse_args()
    deidentify_table_bucketing(args.project, args.table_data, args.deid_content_list, args.bucket_size, args.bucketing_lower_bound, args.bucketing_upper_bound)