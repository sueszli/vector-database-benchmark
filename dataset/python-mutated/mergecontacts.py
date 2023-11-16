"""Merge phone, email, and mailing address information.

A Dataflow pipeline that merges phone, email, and address information associated
with the same names. Each input "database" is a tab-delimited text file pairing
names with one phone number/email address/mailing address; multiple entries
associated with the same name are allowed. Outputs are a tab-delimited text file
with the merged information and another file containing some simple statistics.
See mergecontacts_test.py for example inputs and outputs.

A demonstration of:
  CoGroupByKey
  Non-linear pipelines (i.e., pipelines with branches)
"""
import argparse
import logging
import re
import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to

def run(argv=None, assert_results=None, save_main_session=True):
    if False:
        return 10
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_email', required=True, help='Email database, with each line formatted as "name<TAB>email".')
    parser.add_argument('--input_phone', required=True, help='Phonebook, with each line formatted as "name<TAB>phone number".')
    parser.add_argument('--input_snailmail', required=True, help='Address database, with each line formatted as "name<TAB>address".')
    parser.add_argument('--output_tsv', required=True, help='Tab-delimited output file.')
    parser.add_argument('--output_stats', required=True, help='Output file for statistics about the input.')
    (known_args, pipeline_args) = parser.parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
    with beam.Pipeline(options=pipeline_options) as p:

        def read_kv_textfile(label, textfile):
            if False:
                while True:
                    i = 10
            return p | 'Read: %s' % label >> ReadFromText(textfile) | 'Backslash: %s' % label >> beam.Map(lambda x: re.sub('\\\\', '\\\\\\\\', x)) | 'EscapeQuotes: %s' % label >> beam.Map(lambda x: re.sub('"', '\\"', x)) | 'Split: %s' % label >> beam.Map(lambda x: re.split('\\t+', x, 1))
        email = read_kv_textfile('email', known_args.input_email)
        phone = read_kv_textfile('phone', known_args.input_phone)
        snailmail = read_kv_textfile('snailmail', known_args.input_snailmail)
        grouped = (email, phone, snailmail) | 'group_by_name' >> beam.CoGroupByKey()

        def format_as_tsv(name_email_phone_snailmail):
            if False:
                print('Hello World!')
            (name, (email, phone, snailmail)) = name_email_phone_snailmail
            return '\t'.join(['"%s"' % name, '"%s"' % ','.join(email), '"%s"' % ','.join(phone), '"%s"' % next(iter(snailmail), '')])
        tsv_lines = grouped | beam.Map(format_as_tsv)

        def without_email(name_email_phone_snailmail):
            if False:
                for i in range(10):
                    print('nop')
            (_, (email, _, _)) = name_email_phone_snailmail
            return not next(iter(email), None)

        def without_phones(name_email_phone_snailmail):
            if False:
                print('Hello World!')
            (_, (_, phone, _)) = name_email_phone_snailmail
            return not next(iter(phone), None)

        def without_address(name_email_phone_snailmail):
            if False:
                return 10
            (_, (_, _, snailmail)) = name_email_phone_snailmail
            return not next(iter(snailmail), None)
        luddites = grouped | beam.Filter(without_email)
        writers = grouped | beam.Filter(without_phones)
        nomads = grouped | beam.Filter(without_address)
        num_luddites = luddites | 'Luddites' >> beam.combiners.Count.Globally()
        num_writers = writers | 'Writers' >> beam.combiners.Count.Globally()
        num_nomads = nomads | 'Nomads' >> beam.combiners.Count.Globally()
        tsv_lines | 'WriteTsv' >> WriteToText(known_args.output_tsv)
        if assert_results is not None:
            (expected_luddites, expected_writers, expected_nomads) = assert_results
            assert_that(num_luddites, equal_to([expected_luddites]), label='assert:luddites')
            assert_that(num_writers, equal_to([expected_writers]), label='assert:writers')
            assert_that(num_nomads, equal_to([expected_nomads]), label='assert:nomads')
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()