import apache_beam as beam
from apache_beam.examples.snippets.snippets import SnippetUtils
from apache_beam.metrics import Metrics
from apache_beam.testing.test_pipeline import TestPipeline

def examples_wordcount_debugging(renames):
    if False:
        return 10
    'DebuggingWordCount example snippets.'
    import re
    import logging

    class FilterTextFn(beam.DoFn):
        """A DoFn that filters for a specific key based on a regular expression."""

        def __init__(self, pattern):
            if False:
                for i in range(10):
                    print('nop')
            self.pattern = pattern
            self.matched_words = Metrics.counter(self.__class__, 'matched_words')
            self.umatched_words = Metrics.counter(self.__class__, 'umatched_words')

        def process(self, element):
            if False:
                i = 10
                return i + 15
            (word, _) = element
            if re.match(self.pattern, word):
                logging.info('Matched %s', word)
                self.matched_words.inc()
                yield element
            else:
                logging.debug('Did not match %s', word)
                self.umatched_words.inc()
    with TestPipeline() as pipeline:
        filtered_words = pipeline | beam.io.ReadFromText('gs://dataflow-samples/shakespeare/kinglear.txt') | 'ExtractWords' >> beam.FlatMap(lambda x: re.findall("[A-Za-z\\']+", x)) | beam.combiners.Count.PerElement() | 'FilterText' >> beam.ParDo(FilterTextFn('Flourish|stomach'))
        beam.testing.util.assert_that(filtered_words, beam.testing.util.equal_to([('Flourish', 3), ('stomach', 1)]))

        def format_result(word_count):
            if False:
                i = 10
                return i + 15
            (word, count) = word_count
            return '%s: %s' % (word, count)
        output = filtered_words | 'format' >> beam.Map(format_result) | 'Write' >> beam.io.WriteToText('output.txt')
        pipeline.visit(SnippetUtils.RenameFiles(renames))
if __name__ == '__main__':
    import glob
    examples_wordcount_debugging(None)
    for file_name in glob.glob('output.txt*'):
        with open(file_name) as f:
            print(f.read())