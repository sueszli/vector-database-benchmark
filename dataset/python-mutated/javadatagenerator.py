import logging
import numpy as np
import re
import typing
import apache_beam as beam
from apache_beam.io import ReadFromText
from apache_beam.io import WriteToText
from apache_beam.transforms.external import ImplicitSchemaPayloadBuilder
from apache_beam.transforms.external import JavaExternalTransform
from apache_beam.options.pipeline_options import PipelineOptions
'A Python multi-language pipeline that produces a set of strings generated from Java.\n\nThis example uses the `JavaExternalTransform` API, hence the corresponding Java transform does not\nhave to be specifically registered with an expansion service.\n\nExample commands for executing the program:\n\nDirectRunner:\n$ python javadatagenerator.py --runner DirectRunner --environment_type=DOCKER --output output --expansion_service_port <PORT>\n\nDataflowRunner:\n$ python javadatagenerator.py       --runner DataflowRunner       --temp_location $TEMP_LOCATION       --project $GCP_PROJECT       --region $GCP_REGION       --job_name $JOB_NAME       --num_workers $NUM_WORKERS       --output "gs://$GCS_BUCKET/javadatagenerator/output"       --expansion_service_port <PORT>\n'

def run(output_path, expansion_service_port, pipeline_args):
    if False:
        return 10
    pipeline_options = PipelineOptions(pipeline_args)
    with beam.Pipeline(options=pipeline_options) as p:
        DataConfig = typing.NamedTuple('DataConfig', [('prefix', str), ('length', int), ('suffix', str)])
        data_config = DataConfig(prefix='start', length=20, suffix='end')
        java_transform = JavaExternalTransform('org.apache.beam.examples.multilanguage.JavaDataGenerator', expansion_service='localhost:%s' % expansion_service_port).create(np.int32(100)).withDataConfig(data_config)
        data = p | 'Generate' >> java_transform
        data | 'Write' >> WriteToText(output_path)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', dest='output', required=True, help='Output file')
    parser.add_argument('--expansion_service_port', dest='expansion_service_port', required=True, help='Expansion service port')
    (known_args, pipeline_args) = parser.parse_known_args()
    run(known_args.output, known_args.expansion_service_port, pipeline_args)