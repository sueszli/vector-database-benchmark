"""A streaming word-counting workflow.

Important: streaming pipeline support in Python Dataflow is in development
and is not yet available for use.
"""
import argparse
import logging
import apache_beam as beam
from apache_beam.transforms import window
TABLE_SCHEMA = 'word:STRING, count:INTEGER, window_start:TIMESTAMP, window_end:TIMESTAMP'

def find_words(element):
    if False:
        for i in range(10):
            print('nop')
    import re
    return re.findall("[A-Za-z\\']+", element)

class FormatDoFn(beam.DoFn):

    def process(self, element, window=beam.DoFn.WindowParam):
        if False:
            return 10
        ts_format = '%Y-%m-%d %H:%M:%S.%f UTC'
        window_start = window.start.to_utc_datetime().strftime(ts_format)
        window_end = window.end.to_utc_datetime().strftime(ts_format)
        return [{'word': element[0], 'count': element[1], 'window_start': window_start, 'window_end': window_end}]

def main(argv=None):
    if False:
        print('Hello World!')
    'Build and run the pipeline.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_topic', required=True, help='Input PubSub topic of the form "/topics/<PROJECT>/<TOPIC>".')
    parser.add_argument('--output_table', required=True, help='Output BigQuery table for results specified as: PROJECT:DATASET.TABLE or DATASET.TABLE.')
    (known_args, pipeline_args) = parser.parse_known_args(argv)
    with beam.Pipeline(argv=pipeline_args) as p:
        lines = p | beam.io.ReadFromPubSub(known_args.input_topic)

        def count_ones(word_ones):
            if False:
                return 10
            (word, ones) = word_ones
            return (word, sum(ones))
        transformed = lines | 'Split' >> beam.FlatMap(find_words).with_output_types(str) | 'PairWithOne' >> beam.Map(lambda x: (x, 1)) | beam.WindowInto(window.FixedWindows(2 * 60, 0)) | 'Group' >> beam.GroupByKey() | 'Count' >> beam.Map(count_ones) | 'Format' >> beam.ParDo(FormatDoFn())
        transformed | 'Write' >> beam.io.WriteToBigQuery(known_args.output_table, schema=TABLE_SCHEMA, create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED, write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    main()