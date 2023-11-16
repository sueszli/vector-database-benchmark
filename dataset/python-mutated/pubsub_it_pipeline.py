"""
Test pipeline for use by pubsub_integration_test.
"""
import argparse
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import StandardOptions

def run_pipeline(argv, with_attributes, id_label, timestamp_attribute):
    if False:
        while True:
            i = 10
    'Build and run the pipeline.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_topic', required=True, help='Output PubSub topic of the form "projects/<PROJECT>/topic/<TOPIC>".')
    parser.add_argument('--input_subscription', required=True, help='Input PubSub subscription of the form "projects/<PROJECT>/subscriptions/<SUBSCRIPTION>."')
    (known_args, pipeline_args) = parser.parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(StandardOptions).streaming = True
    p = beam.Pipeline(options=pipeline_options)
    runner_name = type(p.runner).__name__
    if runner_name == 'TestDirectRunner':
        messages = p | beam.io.ReadFromPubSub(subscription=known_args.input_subscription, with_attributes=with_attributes, timestamp_attribute=timestamp_attribute)
    else:
        messages = p | beam.io.ReadFromPubSub(subscription=known_args.input_subscription, id_label=id_label, with_attributes=with_attributes, timestamp_attribute=timestamp_attribute)

    def add_attribute(msg, timestamp=beam.DoFn.TimestampParam):
        if False:
            print('Hello World!')
        msg.data += b'-seen'
        msg.attributes['processed'] = 'IT'
        if timestamp_attribute in msg.attributes:
            msg.attributes[timestamp_attribute + '_out'] = timestamp.to_rfc3339()
        return msg

    def modify_data(data):
        if False:
            while True:
                i = 10
        return data + b'-seen'
    if with_attributes:
        output = messages | 'add_attribute' >> beam.Map(add_attribute)
    else:
        output = messages | 'modify_data' >> beam.Map(modify_data)
    if runner_name == 'TestDirectRunner':
        _ = output | beam.io.WriteToPubSub(known_args.output_topic, with_attributes=with_attributes)
    else:
        _ = output | beam.io.WriteToPubSub(known_args.output_topic, id_label=id_label, with_attributes=with_attributes, timestamp_attribute=timestamp_attribute)
    result = p.run()
    result.wait_until_finish()