"""An example that processes streaming NYC Taxi data with SqlTransform.

This example reads from the PubSub NYC Taxi stream described in
https://github.com/googlecodelabs/cloud-dataflow-nyc-taxi-tycoon, aggregates
the data in 15s windows using SqlTransform, and writes the output to
a user-defined PubSub topic.

A Java version supported by Beam must be installed locally to run this pipeline.
Additionally, Docker must also be available to run this pipeline locally.
"""
import json
import logging
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.transforms.sql import SqlTransform

def run(output_topic, pipeline_args):
    if False:
        return 10
    pipeline_options = PipelineOptions(pipeline_args, save_main_session=True, streaming=True)
    with beam.Pipeline(options=pipeline_options) as pipeline:
        _ = pipeline | beam.io.ReadFromPubSub(topic='projects/pubsub-public-data/topics/taxirides-realtime', timestamp_attribute='ts').with_output_types(bytes) | 'Parse JSON payload' >> beam.Map(json.loads) | 'Create beam Row' >> beam.Map(lambda x: beam.Row(ride_status=str(x['ride_status']), passenger_count=int(x['passenger_count']))) | '15s fixed windows' >> beam.WindowInto(beam.window.FixedWindows(15)) | SqlTransform("\n             SELECT\n               ride_status,\n               COUNT(*) AS num_rides,\n               SUM(passenger_count) AS total_passengers\n             FROM PCOLLECTION\n             WHERE NOT ride_status = 'enroute'\n             GROUP BY ride_status") | 'Assemble Dictionary' >> beam.Map(lambda row, window=beam.DoFn.WindowParam: {'ride_status': row.ride_status, 'num_rides': row.num_rides, 'total_passengers': row.total_passengers, 'window_start': window.start.to_rfc3339(), 'window_end': window.end.to_rfc3339()}) | 'Convert to JSON' >> beam.Map(json.dumps) | 'UTF-8 encode' >> beam.Map(lambda s: s.encode('utf-8')) | beam.io.WriteToPubSub(topic=output_topic)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_topic', dest='output_topic', required=True, help='Cloud PubSub topic to write to (e.g. projects/my-project/topics/my-topic), must be created prior to running the pipeline.')
    (known_args, pipeline_args) = parser.parse_known_args()
    run(known_args.output_topic, pipeline_args)