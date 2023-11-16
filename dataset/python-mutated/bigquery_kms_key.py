import argparse

def run(output_bigquery_table, kms_key, beam_args):
    if False:
        i = 10
        return i + 15
    import apache_beam as beam
    query = '\n        SELECT latitude,longitude,acq_date,acq_time,bright_ti4,confidence\n        FROM `bigquery-public-data.nasa_wildfire.past_week`\n        LIMIT 10\n    '
    schema = {'fields': [{'name': 'latitude', 'type': 'FLOAT'}, {'name': 'longitude', 'type': 'FLOAT'}, {'name': 'acq_date', 'type': 'DATE'}, {'name': 'acq_time', 'type': 'TIME'}, {'name': 'bright_ti4', 'type': 'FLOAT'}, {'name': 'confidence', 'type': 'STRING'}]}
    options = beam.options.pipeline_options.PipelineOptions(beam_args)
    with beam.Pipeline(options=options) as pipeline:
        pipeline | 'Read from BigQuery with KMS key' >> beam.io.Read(beam.io.BigQuerySource(query=query, use_standard_sql=True, kms_key=kms_key)) | 'Write to BigQuery with KMS key' >> beam.io.WriteToBigQuery(output_bigquery_table, schema=schema, write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE, kms_key=kms_key)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kms_key', required=True, help='Cloud Key Management Service key name')
    parser.add_argument('--output_bigquery_table', required=True, help="Output BigQuery table in the format 'PROJECT:DATASET.TABLE'")
    (args, beam_args) = parser.parse_known_args()
    run(args.output_bigquery_table, args.kms_key, beam_args)