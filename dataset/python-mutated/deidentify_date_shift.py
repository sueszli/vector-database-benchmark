"""Uses of the Data Loss Prevention API for de-identifying sensitive data."""
from __future__ import annotations
import argparse
import base64
import csv
from datetime import datetime
from typing import List
import google.cloud.dlp
from google.cloud.dlp_v2 import types

def deidentify_with_date_shift(project: str, input_csv_file: str=None, output_csv_file: str=None, date_fields: List[str]=None, lower_bound_days: int=None, upper_bound_days: int=None, context_field_id: str=None, wrapped_key: str=None, key_name: str=None) -> None:
    if False:
        while True:
            i = 10
    "Uses the Data Loss Prevention API to deidentify dates in a CSV file by\n        pseudorandomly shifting them.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        input_csv_file: The path to the CSV file to deidentify. The first row\n            of the file must specify column names, and all other rows must\n            contain valid values.\n        output_csv_file: The path to save the date-shifted CSV file.\n        date_fields: The list of (date) fields in the CSV file to date shift.\n            Example: ['birth_date', 'register_date']\n        lower_bound_days: The maximum number of days to shift a date backward\n        upper_bound_days: The maximum number of days to shift a date forward\n        context_field_id: (Optional) The column to determine date shift amount\n            based on. If this is not specified, a random shift amount will be\n            used for every row. If this is specified, then 'wrappedKey' and\n            'keyName' must also be set. Example:\n            contextFieldId = [{ 'name': 'user_id' }]\n        key_name: (Optional) The name of the Cloud KMS key used to encrypt\n            ('wrap') the AES-256 key. Example:\n            key_name = 'projects/YOUR_GCLOUD_PROJECT/locations/YOUR_LOCATION/\n            keyRings/YOUR_KEYRING_NAME/cryptoKeys/YOUR_KEY_NAME'\n        wrapped_key: (Optional) The encrypted ('wrapped') AES-256 key to use.\n            This key should be encrypted using the Cloud KMS key specified by\n            key_name.\n    Returns:\n        None; the response from the API is printed to the terminal.\n    "
    dlp = google.cloud.dlp_v2.DlpServiceClient()
    parent = f'projects/{project}/locations/global'

    def map_fields(field: str) -> dict:
        if False:
            for i in range(10):
                print('nop')
        return {'name': field}
    if date_fields:
        date_fields = map(map_fields, date_fields)
    else:
        date_fields = []
    f = []
    with open(input_csv_file) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            f.append(row)

    def map_headers(header: str) -> dict:
        if False:
            return 10
        return {'name': header}

    def map_data(value: str) -> dict:
        if False:
            return 10
        try:
            date = datetime.strptime(value, '%m/%d/%Y')
            return {'date_value': {'year': date.year, 'month': date.month, 'day': date.day}}
        except ValueError:
            return {'string_value': value}

    def map_rows(row: str) -> dict:
        if False:
            for i in range(10):
                print('nop')
        return {'values': map(map_data, row)}
    csv_headers = map(map_headers, f[0])
    csv_rows = map(map_rows, f[1:])
    table_item = {'table': {'headers': csv_headers, 'rows': csv_rows}}
    date_shift_config = {'lower_bound_days': lower_bound_days, 'upper_bound_days': upper_bound_days}
    if context_field_id and key_name and wrapped_key:
        date_shift_config['context'] = {'name': context_field_id}
        date_shift_config['crypto_key'] = {'kms_wrapped': {'wrapped_key': base64.b64decode(wrapped_key), 'crypto_key_name': key_name}}
    elif context_field_id or key_name or wrapped_key:
        raise ValueError('You must set either ALL or NONE of\n        [context_field_id, key_name, wrapped_key]!')
    deidentify_config = {'record_transformations': {'field_transformations': [{'fields': date_fields, 'primitive_transformation': {'date_shift_config': date_shift_config}}]}}

    def write_header(header: types.storage.FieldId) -> str:
        if False:
            while True:
                i = 10
        return header.name

    def write_data(data: types.storage.Value) -> str:
        if False:
            i = 10
            return i + 15
        return data.string_value or '{}/{}/{}'.format(data.date_value.month, data.date_value.day, data.date_value.year)
    response = dlp.deidentify_content(request={'parent': parent, 'deidentify_config': deidentify_config, 'item': table_item})
    with open(output_csv_file, 'w') as csvfile:
        write_file = csv.writer(csvfile, delimiter=',')
        write_file.writerow(map(write_header, response.item.table.headers))
        for row in response.item.table.rows:
            write_file.writerow(map(write_data, row.values))
    print(f'Successfully saved date-shift output to {output_csv_file}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('input_csv_file', help='The path to the CSV file to deidentify. The first row of the file must specify column names, and all other rows must contain valid values.')
    parser.add_argument('output_csv_file', help='The path to save the date-shifted CSV file.')
    parser.add_argument('lower_bound_days', type=int, help='The maximum number of days to shift a date backward')
    parser.add_argument('upper_bound_days', type=int, help='The maximum number of days to shift a date forward')
    parser.add_argument('date_fields', nargs='+', help="The list of date fields in the CSV file to date shift. Example: ['birth_date', 'register_date']")
    parser.add_argument('--context_field_id', help="(Optional) The column to determine date shift amount based on. If this is not specified, a random shift amount will be used for every row. If this is specified, then 'wrappedKey' and 'keyName' must also be set.")
    parser.add_argument('--key_name', help="(Optional) The name of the Cloud KMS key used to encrypt ('wrap') the AES-256 key. Example: key_name = 'projects/YOUR_GCLOUD_PROJECT/locations/YOUR_LOCATION/keyRings/YOUR_KEYRING_NAME/cryptoKeys/YOUR_KEY_NAME'")
    parser.add_argument('--wrapped_key', help="(Optional) The encrypted ('wrapped') AES-256 key to use. This key should be encrypted using the Cloud KMS key specified bykey_name.")
    args = parser.parse_args()
    deidentify_with_date_shift(args.project, input_csv_file=args.input_csv_file, output_csv_file=args.output_csv_file, lower_bound_days=args.lower_bound_days, upper_bound_days=args.upper_bound_days, date_fields=args.date_fields, context_field_id=args.context_field_id, wrapped_key=args.wrapped_key, key_name=args.key_name)