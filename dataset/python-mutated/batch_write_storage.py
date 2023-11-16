import apache_beam as beam
from apache_beam.io.textio import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions

def write_to_cloud_storage(argv=None):
    if False:
        for i in range(10):
            print('nop')

    class MyOptions(PipelineOptions):

        @classmethod
        def _add_argparse_args(cls, parser):
            if False:
                print('Hello World!')
            parser.add_argument('--output', required=True)
    wordsList = ['1', '2', '3', '4']
    options = MyOptions()
    with beam.Pipeline(options=options) as pipeline:
        pipeline | 'Create elements' >> beam.Create(wordsList) | 'Write Files' >> WriteToText(options.output, file_name_suffix='.txt')
if __name__ == '__main__':
    write_to_cloud_storage()