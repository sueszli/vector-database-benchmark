import unittest
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.runners.dataflow import dataflow_job_service
from apache_beam.runners.portability import local_job_service
try:
    from apache_beam.runners.dataflow.internal import apiclient
except ImportError:
    apiclient = None

@unittest.skipIf(apiclient is None, 'GCP dependencies are not installed')
class DirectPipelineResultTest(unittest.TestCase):

    def test_dry_run(self):
        if False:
            return 10
        job_servicer = local_job_service.LocalJobServicer(None, beam_job_type=dataflow_job_service.DataflowBeamJob)
        port = job_servicer.start_grpc_server(0)
        try:
            options = PipelineOptions(runner='PortableRunner', job_endpoint=f'localhost:{port}', project='some_project', temp_location='gs://bucket/dir', region='us-central1', dry_run=True)
            with beam.Pipeline(options=options) as p:
                _ = p | beam.Create([1, 2, 3]) | beam.Map(lambda x: x * x)
        finally:
            job_servicer.stop()
if __name__ == '__main__':
    unittest.main()