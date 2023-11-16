def examples_wordcount_wordcount():
    if False:
        while True:
            i = 10
    'WordCount example snippets.'
    import re
    import apache_beam as beam
    from apache_beam.options.pipeline_options import PipelineOptions
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', default='gs://dataflow-samples/shakespeare/kinglear.txt', help='The file path for the input text to process.')
    parser.add_argument('--output-path', required=True, help='The path prefix for output files.')
    (args, beam_args) = parser.parse_known_args()
    beam_options = PipelineOptions(beam_args)
    with beam.Pipeline(options=beam_options) as pipeline:
        lines = pipeline | beam.io.ReadFromText(args.input_file)

        @beam.ptransform_fn
        def CountWords(pcoll):
            if False:
                for i in range(10):
                    print('nop')
            return pcoll | 'ExtractWords' >> beam.FlatMap(lambda x: re.findall("[A-Za-z\\']+", x)) | beam.combiners.Count.PerElement()
        counts = lines | CountWords()

        class FormatAsTextFn(beam.DoFn):

            def process(self, element):
                if False:
                    while True:
                        i = 10
                (word, count) = element
                yield ('%s: %s' % (word, count))
        formatted = counts | beam.ParDo(FormatAsTextFn())
        formatted | beam.io.WriteToText(args.output_path)
if __name__ == '__main__':
    examples_wordcount_wordcount()