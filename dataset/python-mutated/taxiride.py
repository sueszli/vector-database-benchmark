"""Pipelines that use the DataFrame API to process NYC taxiride CSV data."""
from __future__ import absolute_import
import argparse
import logging
import apache_beam as beam
from apache_beam.dataframe.io import read_csv
from apache_beam.options.pipeline_options import PipelineOptions
ZONE_LOOKUP_PATH = 'gs://apache-beam-samples/nyc_taxi/misc/taxi+_zone_lookup.csv'

def run_aggregation_pipeline(pipeline, input_path, output_path):
    if False:
        return 10
    with pipeline as p:
        rides = p | read_csv(input_path)
        agg = rides.groupby('DOLocationID').passenger_count.sum()
        agg.to_csv(output_path)

def run_enrich_pipeline(pipeline, input_path, output_path, zone_lookup_path=ZONE_LOOKUP_PATH):
    if False:
        i = 10
        return i + 15
    'Enrich taxi ride data with zone lookup table and perform a grouped\n  aggregation.'
    with pipeline as p:
        rides = p | 'Read taxi rides' >> read_csv(input_path)
        zones = p | 'Read zone lookup' >> read_csv(zone_lookup_path)
        rides = rides.merge(zones.set_index('LocationID').Borough, right_index=True, left_on='DOLocationID', how='left')
        agg = rides.groupby('Borough').passenger_count.sum()
        agg.to_csv(output_path)

def run(argv=None):
    if False:
        for i in range(10):
            print('nop')
    'Main entry point.'
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', dest='input', default='gs://apache-beam-samples/nyc_taxi/misc/sample.csv', help='Input file to process.')
    parser.add_argument('--output', dest='output', required=True, help='Output file to write results to.')
    parser.add_argument('--zone_lookup', dest='zone_lookup_path', default=ZONE_LOOKUP_PATH, help='Location for taxi zone lookup CSV.')
    parser.add_argument('--pipeline', dest='pipeline', default='location_id_agg', help='Choice of pipeline to run. Must be one of (location_id_agg, borough_enrich).')
    (known_args, pipeline_args) = parser.parse_known_args(argv)
    pipeline = beam.Pipeline(options=PipelineOptions(pipeline_args))
    if known_args.pipeline == 'location_id_agg':
        run_aggregation_pipeline(pipeline, known_args.input, known_args.output)
    elif known_args.pipeline == 'borough_enrich':
        run_enrich_pipeline(pipeline, known_args.input, known_args.output, known_args.zone_lookup_path)
    else:
        raise ValueError(f"Unrecognized value for --pipeline: {known_args.pipeline!r}. Must be one of ('location_id_agg', 'borough_enrich')")
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()