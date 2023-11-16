"""A workflow using BigQuery sources and sinks.

The workflow will read from a table that has the 'month' and 'tornado' fields as
part of the table schema (other additional fields are ignored). The 'month'
field is a number represented as a string (e.g., '23') and the 'tornado' field
is a boolean field.

The workflow will compute the number of tornadoes in each month and output
the results to a table (created if needed) with the following schema:

- month: number
- tornado_count: number

This example uses the default behavior for BigQuery source and sinks that
represents table rows as plain Python dictionaries.
"""
import argparse
import logging
import apache_beam as beam

def count_tornadoes(input_data):
    if False:
        return 10
    "Workflow computing the number of tornadoes for each month that had one.\n\n  Args:\n    input_data: a PCollection of dictionaries representing table rows. Each\n      dictionary will have a 'month' and a 'tornado' key as described in the\n      module comment.\n\n  Returns:\n    A PCollection of dictionaries containing 'month' and 'tornado_count' keys.\n    Months without tornadoes are skipped.\n  "
    return input_data | 'months with tornadoes' >> beam.FlatMap(lambda row: [(int(row['month']), 1)] if row['tornado'] else []) | 'monthly count' >> beam.CombinePerKey(sum) | 'format' >> beam.Map(lambda k_v: {'month': k_v[0], 'tornado_count': k_v[1]})

def run(argv=None):
    if False:
        i = 10
        return i + 15
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='apache-beam-testing.samples.weather_stations', help='Input BigQuery table to process specified as: PROJECT:DATASET.TABLE or DATASET.TABLE.')
    parser.add_argument('--output', required=True, help='Output BigQuery table for results specified as: PROJECT:DATASET.TABLE or DATASET.TABLE.')
    parser.add_argument('--gcs_location', required=False, help='GCS Location to store files to load data into Bigquery')
    (known_args, pipeline_args) = parser.parse_known_args(argv)
    with beam.Pipeline(argv=pipeline_args) as p:
        rows = p | 'read' >> beam.io.ReadFromBigQuery(table=known_args.input)
        counts = count_tornadoes(rows)
        counts | 'Write' >> beam.io.WriteToBigQuery(known_args.output, schema='month:INTEGER, tornado_count:INTEGER', create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED, write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()