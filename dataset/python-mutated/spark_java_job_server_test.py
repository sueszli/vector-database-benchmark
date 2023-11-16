import logging
import unittest
from apache_beam.options import pipeline_options
from apache_beam.runners.portability.spark_runner import SparkRunner

class SparkTestPipelineOptions(pipeline_options.PipelineOptions):

    def view_as(self, cls):
        if False:
            while True:
                i = 10
        assert cls is pipeline_options.SparkRunnerOptions or cls is pipeline_options.JobServerOptions
        return super().view_as(cls)

class SparkJavaJobServerTest(unittest.TestCase):

    def test_job_server_cache(self):
        if False:
            return 10
        job_server1 = SparkRunner().default_job_server(SparkTestPipelineOptions(['--sdk_worker_parallelism=1']))
        job_server2 = SparkRunner().default_job_server(SparkTestPipelineOptions(['--sdk_worker_parallelism=2']))
        self.assertIs(job_server2, job_server1)
        job_server3 = SparkRunner().default_job_server(SparkTestPipelineOptions(['--job_port=1234']))
        self.assertIsNot(job_server3, job_server1)
        job_server4 = SparkRunner().default_job_server(SparkTestPipelineOptions(['--spark_master_url=spark://localhost:5678']))
        self.assertIsNot(job_server4, job_server1)
        self.assertIsNot(job_server4, job_server3)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()