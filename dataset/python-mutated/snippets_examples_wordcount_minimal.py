def examples_wordcount_minimal():
    if False:
        i = 10
        return i + 15
    'MinimalWordCount example snippets.'
    import re
    import apache_beam as beam
    from apache_beam.options.pipeline_options import PipelineOptions
    input_file = 'gs://dataflow-samples/shakespeare/kinglear.txt'
    output_path = 'gs://my-bucket/counts.txt'
    beam_options = PipelineOptions(runner='DataflowRunner', project='my-project-id', job_name='unique-job-name', temp_location='gs://my-bucket/temp')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file')
    parser.add_argument('--output-path')
    (args, beam_args) = parser.parse_known_args()
    input_file = args.input_file
    output_path = args.output_path
    beam_options = PipelineOptions(beam_args)
    pipeline = beam.Pipeline(options=beam_options)
    pipeline | beam.io.ReadFromText(input_file) | 'ExtractWords' >> beam.FlatMap(lambda x: re.findall("[A-Za-z\\']+", x)) | beam.combiners.Count.PerElement() | beam.MapTuple(lambda word, count: '%s: %s' % (word, count)) | beam.io.WriteToText(output_path)
    result = pipeline.run()
    result.wait_until_finish()
if __name__ == '__main__':
    examples_wordcount_minimal()