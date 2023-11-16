import logging
import re
import typing
import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.transforms.external import ImplicitSchemaPayloadBuilder
from apache_beam.options.pipeline_options import PipelineOptions
'A Python multi-language pipeline that counts words.\n\nThis pipeline reads an input text file and counts the words using the Java SDK\ntransform `Count.perElement()`.\n\nExample commands for executing the program:\n\nDirectRunner:\n$ python javacount.py --runner DirectRunner --environment_type=DOCKER --input <INPUT FILE> --output output --expansion_service_port <PORT>\n\nDataflowRunner:\n$ python javacount.py       --runner DataflowRunner       --temp_location $TEMP_LOCATION       --project $GCP_PROJECT       --region $GCP_REGION       --job_name $JOB_NAME       --num_workers $NUM_WORKERS       --input "gs://dataflow-samples/shakespeare/kinglear.txt"       --output "gs://$GCS_BUCKET/javacount/output"       --expansion_service_port <PORT>\n'

class WordExtractingDoFn(beam.DoFn):

    def process(self, element):
        if False:
            print('Hello World!')
        return re.findall("[\\w\\']+", element, re.UNICODE)

def run(input_path, output_path, expansion_service_port, pipeline_args):
    if False:
        return 10
    pipeline_options = PipelineOptions(pipeline_args)
    with beam.Pipeline(options=pipeline_options) as p:
        lines = p | 'Read' >> ReadFromText(input_path).with_output_types(str)
        words = lines | 'Split' >> beam.ParDo(WordExtractingDoFn()).with_output_types(str)
        java_output = words | 'JavaCount' >> beam.ExternalTransform('beam:transform:org.apache.beam:javacount:v1', None, 'localhost:%s' % expansion_service_port)

        def format(kv):
            if False:
                return 10
            (key, value) = kv
            return '%s:%s' % (key, value)
        output = java_output | 'Format' >> beam.Map(format)
        output | 'Write' >> WriteToText(output_path)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='input', required=True, help='Input file')
    parser.add_argument('--output', dest='output', required=True, help='Output file')
    parser.add_argument('--expansion_service_port', dest='expansion_service_port', required=True, help='Expansion service port')
    (known_args, pipeline_args) = parser.parse_known_args()
    run(known_args.input, known_args.output, known_args.expansion_service_port, pipeline_args)