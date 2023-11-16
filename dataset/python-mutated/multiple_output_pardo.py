"""A workflow demonstrating a DoFn with multiple outputs.

DoFns may produce multiple outputs. Outputs that are not the default ("main")
output are marked with a tag at output time and later the same tag will be used
to get the corresponding result (a PCollection) for that output.

This is a slightly modified version of the basic wordcount example. In this
example words are divided into 2 buckets as shorts words (3 characters in length
or less) and words (all other words). There will be 3 output files:::

  [OUTPUT]-chars        :   Character count for the input.
  [OUTPUT]-short-words  :   Word count for short words only.
  [OUTPUT]-words        :   Word count for all other words.

To execute this pipeline locally, specify a local output file or output prefix
on GCS:::

  --output [YOUR_LOCAL_FILE | gs://YOUR_OUTPUT_PREFIX]

To execute this pipeline using the Google Cloud Dataflow service, specify
pipeline configuration:::

  --project YOUR_PROJECT_ID
  --region GCE_REGION
  --staging_location gs://YOUR_STAGING_DIRECTORY
  --temp_location gs://YOUR_TEMP_DIRECTORY
  --job_name YOUR_JOB_NAME
  --runner DataflowRunner

and an output prefix on GCS:::

  --output gs://YOUR_OUTPUT_PREFIX
"""
import argparse
import logging
import re
import apache_beam as beam
from apache_beam import pvalue
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

class SplitLinesToWordsFn(beam.DoFn):
    """A transform to split a line of text into individual words.

  This transform will have 3 outputs:
    - main output: all words that are longer than 3 characters.
    - short words output: all other words.
    - character count output: Number of characters in each processed line.
  """
    OUTPUT_TAG_SHORT_WORDS = 'tag_short_words'
    OUTPUT_TAG_CHARACTER_COUNT = 'tag_character_count'

    def process(self, element):
        if False:
            return 10
        'Receives a single element (a line) and produces words and character\n    counts.\n\n    Important things to note here:\n      - For a single element you may produce multiple main outputs:\n        words of a single line.\n      - For that same input you may produce multiple outputs, potentially\n        across multiple PCollections\n      - Outputs may have different types (count) or may share the same type\n        (words) as with the main output.\n\n    Args:\n      element: processing element.\n\n    Yields:\n      words as main output, short words as tagged output, line character count\n      as tagged output.\n    '
        yield pvalue.TaggedOutput(self.OUTPUT_TAG_CHARACTER_COUNT, len(element))
        words = re.findall("[A-Za-z\\']+", element)
        for word in words:
            if len(word) <= 3:
                yield pvalue.TaggedOutput(self.OUTPUT_TAG_SHORT_WORDS, word)
            else:
                yield word

class CountWords(beam.PTransform):
    """A transform to count the occurrences of each word.

  A PTransform that converts a PCollection containing words into a PCollection
  of "word: count" strings.
  """

    def expand(self, pcoll):
        if False:
            print('Hello World!')

        def count_ones(word_ones):
            if False:
                i = 10
                return i + 15
            (word, ones) = word_ones
            return (word, sum(ones))

        def format_result(word_count):
            if False:
                print('Hello World!')
            (word, count) = word_count
            return '%s: %s' % (word, count)
        return pcoll | 'pair_with_one' >> beam.Map(lambda x: (x, 1)) | 'group' >> beam.GroupByKey() | 'count' >> beam.Map(count_ones) | 'format' >> beam.Map(format_result)

def run(argv=None, save_main_session=True):
    if False:
        return 10
    'Runs the workflow counting the long words and short words separately.'
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='gs://dataflow-samples/shakespeare/kinglear.txt', help='Input file to process.')
    parser.add_argument('--output', required=True, help='Output prefix for files to write results to.')
    (known_args, pipeline_args) = parser.parse_known_args(argv)
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
    with beam.Pipeline(options=pipeline_options) as p:
        lines = p | ReadFromText(known_args.input)
        split_lines_result = lines | beam.ParDo(SplitLinesToWordsFn()).with_outputs(SplitLinesToWordsFn.OUTPUT_TAG_SHORT_WORDS, SplitLinesToWordsFn.OUTPUT_TAG_CHARACTER_COUNT, main='words')
        (words, _, _) = split_lines_result
        short_words = split_lines_result[SplitLinesToWordsFn.OUTPUT_TAG_SHORT_WORDS]
        character_count = split_lines_result.tag_character_count
        character_count | 'pair_with_key' >> beam.Map(lambda x: ('chars_temp_key', x)) | beam.GroupByKey() | 'count chars' >> beam.Map(lambda char_counts: sum(char_counts[1])) | 'write chars' >> WriteToText(known_args.output + '-chars')
        short_words | 'count short words' >> CountWords() | 'write short words' >> WriteToText(known_args.output + '-short-words')
        words | 'count words' >> CountWords() | 'write words' >> WriteToText(known_args.output + '-words')
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()