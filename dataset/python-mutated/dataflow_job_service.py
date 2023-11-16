import argparse
import logging
import sys
from apache_beam.runners.dataflow import dataflow_runner
from apache_beam.runners.portability import local_job_service
from apache_beam.runners.portability import local_job_service_main
from apache_beam.runners.portability import portable_runner

class DataflowBeamJob(local_job_service.BeamJob):
    """A representation of a single Beam job to be run on the Dataflow runner.
  """

    def _invoke_runner(self):
        if False:
            while True:
                i = 10
        'Actually calls Dataflow and waits for completion.\n    '
        runner = dataflow_runner.DataflowRunner()
        self.result = runner.run_pipeline(None, self.pipeline_options(), self._pipeline_proto)
        dataflow_runner.DataflowRunner.poll_for_job_completion(runner, self.result, None, lambda dataflow_state: self.set_state(portable_runner.PipelineResult.pipeline_state_to_runner_api_state(self.result.api_jobstate_to_pipeline_state(dataflow_state))))
        return self.result

    def cancel(self):
        if False:
            while True:
                i = 10
        if not self.is_terminal_state(self.state):
            self.result.cancel()

def run(argv, beam_job_type=DataflowBeamJob):
    if False:
        i = 10
        return i + 15
    if argv[0] == __file__:
        argv = argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', '--job_port', type=int, default=0, help='port on which to serve the job api')
    parser.add_argument('--staging_dir')
    options = parser.parse_args(argv)
    job_servicer = local_job_service.LocalJobServicer(options.staging_dir, beam_job_type=beam_job_type)
    port = job_servicer.start_grpc_server(options.port)
    try:
        local_job_service_main.serve('Listening for beam jobs on port %d.' % port, job_servicer)
    finally:
        job_servicer.stop()
if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    run(sys.argv)