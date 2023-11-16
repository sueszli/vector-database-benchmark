"""Uses of the Data Loss Prevention API for deidentifying sensitive data."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
from typing import List
import google.cloud.dlp

def deidentify_with_time_extract(project: str, date_fields: List[str], input_csv_file: str, output_csv_file: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    "Uses the Data Loss Prevention API to deidentify dates in a CSV file through\n     time part extraction.\n    Args:\n        project: The Google Cloud project id to use as a parent resource.\n        date_fields: A list of (date) fields in CSV file to de-identify\n            through time extraction. Example: ['birth_date', 'register_date'].\n            Date values in format: mm/DD/YYYY are considered as part of this\n            sample.\n        input_csv_file: The path to the CSV file to deidentify. The first row\n            of the file must specify column names, and all other rows must\n            contain valid values.\n        output_csv_file: The output file path to save the time extracted data.\n    "
    dlp = google.cloud.dlp_v2.DlpServiceClient()

    def map_fields(field):
        if False:
            for i in range(10):
                print('nop')
        return {'name': field}
    if date_fields:
        date_fields = map(map_fields, date_fields)
    else:
        date_fields = []
    csv_lines = []
    with open(input_csv_file) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            csv_lines.append(row)

    def map_headers(header):
        if False:
            i = 10
            return i + 15
        return {'name': header}

    def map_data(value):
        if False:
            for i in range(10):
                print('nop')
        try:
            date = datetime.strptime(value, '%m/%d/%Y')
            return {'date_value': {'year': date.year, 'month': date.month, 'day': date.day}}
        except ValueError:
            return {'string_value': value}

    def map_rows(row):
        if False:
            return 10
        return {'values': map(map_data, row)}
    csv_headers = map(map_headers, csv_lines[0])
    csv_rows = map(map_rows, csv_lines[1:])
    table = {'headers': csv_headers, 'rows': csv_rows}
    item = {'table': table}
    deidentify_config = {'record_transformations': {'field_transformations': [{'primitive_transformation': {'time_part_config': {'part_to_extract': 'YEAR'}}, 'fields': date_fields}]}}

    def write_header(header):
        if False:
            print('Hello World!')
        return header.name

    def write_data(data):
        if False:
            for i in range(10):
                print('nop')
        return data.string_value or '{}/{}/{}'.format(data.date_value.month, data.date_value.day, data.date_value.year)
    parent = f'projects/{project}/locations/global'
    response = dlp.deidentify_content(request={'parent': parent, 'deidentify_config': deidentify_config, 'item': item})
    print(f'Table after de-identification: {response.item.table}')
    with open(output_csv_file, 'w') as csvfile:
        write_file = csv.writer(csvfile, delimiter=',')
        write_file.writerow(map(write_header, response.item.table.headers))
        for row in response.item.table.rows:
            write_file.writerow(map(write_data, row.values))
    print(f'Successfully saved date-extracted output to {output_csv_file}')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', help='The Google Cloud project id to use as a parent resource.')
    parser.add_argument('input_csv_file', help='The path to the CSV file to deidentify. The first row of the file must specify column names, and all other rows must contain valid values.')
    parser.add_argument('date_fields', nargs='+', help="The list of date fields in the CSV file to de-identify. Example: ['birth_date', 'register_date']")
    parser.add_argument('output_csv_file', help='The path to save the time-extracted data.')
    args = parser.parse_args()
    deidentify_with_time_extract(args.project, date_fields=args.date_fields, input_csv_file=args.input_csv_file, output_csv_file=args.output_csv_file)