"""A streaming word-counting workflow.
"""
import argparse
import logging
import apache_beam as beam
from apache_beam.examples.wordcount_with_metrics import WordExtractingDoFn
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.options.pipeline_options import StandardOptions
from apache_beam.transforms import window

def run(argv=None, save_main_session=True):
    if False:
        while True:
            i = 10
    'Build and run the pipeline.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_topic', required=True, help='Output PubSub topic of the form "projects/<PROJECT>/topics/<TOPIC>".')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input_topic', help='Input PubSub topic of the form "projects/<PROJECT>/topics/<TOPIC>".')
    group.add_argument('--input_subscription', help='Input PubSub subscription of the form "projects/<PROJECT>/subscriptions/<SUBSCRIPTION>."')
    (known_args, pipeline_args) = parser.parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
    pipeline_options.view_as(StandardOptions).streaming = True
    with beam.Pipeline(options=pipeline_options) as p:
        if known_args.input_subscription:
            messages = p | beam.io.ReadFromPubSub(subscription=known_args.input_subscription).with_output_types(bytes)
        else:
            messages = p | beam.io.ReadFromPubSub(topic=known_args.input_topic).with_output_types(bytes)
        lines = messages | 'decode' >> beam.Map(lambda x: x.decode('utf-8'))

        def count_ones(word_ones):
            if False:
                return 10
            (word, ones) = word_ones
            return (word, sum(ones))
        counts = lines | 'split' >> beam.ParDo(WordExtractingDoFn()).with_output_types(str) | 'pair_with_one' >> beam.Map(lambda x: (x, 1)) | beam.WindowInto(window.FixedWindows(15, 0)) | 'group' >> beam.GroupByKey() | 'count' >> beam.Map(count_ones)

        def format_result(word_count):
            if False:
                for i in range(10):
                    print('nop')
            (word, count) = word_count
            return '%s: %d' % (word, count)
        output = counts | 'format' >> beam.Map(format_result) | 'encode' >> beam.Map(lambda x: x.encode('utf-8')).with_output_types(bytes)
        output | beam.io.WriteToPubSub(known_args.output_topic)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()