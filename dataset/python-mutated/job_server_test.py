import logging
import unittest
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.runners.portability.job_server import JavaJarJobServer

class JavaJarJobServerStub(JavaJarJobServer):

    def java_arguments(self, job_port, artifact_port, expansion_port, artifacts_dir):
        if False:
            for i in range(10):
                print('nop')
        return ['--artifacts-dir', artifacts_dir, '--job-port', job_port, '--artifact-port', artifact_port, '--expansion-port', expansion_port]

    def path_to_jar(self):
        if False:
            print('Hello World!')
        return '/path/to/jar'

    @staticmethod
    def local_jar(url):
        if False:
            return 10
        return url

class JavaJarJobServerTest(unittest.TestCase):

    def test_subprocess_cmd_and_endpoint(self):
        if False:
            print('Hello World!')
        pipeline_options = PipelineOptions(['--job_port=8099', '--artifact_port=8098', '--expansion_port=8097', '--artifacts_dir=/path/to/artifacts/', '--job_server_java_launcher=/path/to/java', '--job_server_jvm_properties=-Dsome.property=value'])
        job_server = JavaJarJobServerStub(pipeline_options)
        (subprocess_cmd, endpoint) = job_server.subprocess_cmd_and_endpoint()
        self.assertEqual(subprocess_cmd, ['/path/to/java', '-jar', '-Dsome.property=value', '/path/to/jar', '--artifacts-dir', '/path/to/artifacts/', '--job-port', 8099, '--artifact-port', 8098, '--expansion-port', 8097])
        self.assertEqual(endpoint, 'localhost:8099')
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()