import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import grpc
from concurrent import futures
from apache_beam.portability.api.org.apache.beam.model.fn_execution.v1.beam_provision_api_pb2 import ProvisionInfo, GetProvisionInfoResponse
from apache_beam.portability.api.org.apache.beam.model.fn_execution.v1.beam_provision_api_pb2_grpc import ProvisionServiceServicer, add_ProvisionServiceServicer_to_server
from google.protobuf import json_format
from pyflink.java_gateway import get_gateway
from pyflink.pyflink_gateway_server import on_windows
from pyflink.testing.test_case_utils import PyFlinkTestCase

class PythonBootTests(PyFlinkTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        provision_info = json_format.Parse('{"retrievalToken": "test_token"}', ProvisionInfo())
        response = GetProvisionInfoResponse(info=provision_info)

        def get_unused_port():
            if False:
                print('Hello World!')
            sock = socket.socket()
            sock.bind(('', 0))
            port = sock.getsockname()[1]
            sock.close()
            return port

        class ProvisionService(ProvisionServiceServicer):

            def GetProvisionInfo(self, request, context):
                if False:
                    return 10
                return response

        def start_test_provision_server():
            if False:
                i = 10
                return i + 15
            server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
            add_ProvisionServiceServicer_to_server(ProvisionService(), server)
            port = get_unused_port()
            server.add_insecure_port('[::]:' + str(port))
            server.start()
            return (server, port)
        (self.provision_server, self.provision_port) = start_test_provision_server()
        self.env = dict(os.environ)
        self.env['python'] = sys.executable
        self.env['FLINK_BOOT_TESTING'] = '1'
        self.env['BOOT_LOG_DIR'] = os.path.join(self.env['FLINK_HOME'], 'log')
        self.tmp_dir = tempfile.mkdtemp(str(time.time()), dir=self.tempdir)
        pyflink_package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        runner_script = 'pyflink-udf-runner.bat' if on_windows() else 'pyflink-udf-runner.sh'
        self.runner_path = os.path.join(pyflink_package_dir, 'bin', runner_script)

    def run_boot_py(self):
        if False:
            print('Hello World!')
        args = [self.runner_path, '--id', '1', '--logging_endpoint', 'localhost:0000', '--artifact_endpoint', 'whatever', '--provision_endpoint', 'localhost:%d' % self.provision_port, '--control_endpoint', 'localhost:0000', '--semi_persist_dir', self.tmp_dir]
        return subprocess.call(args, env=self.env)

    def test_python_boot(self):
        if False:
            print('Hello World!')
        exit_code = self.run_boot_py()
        self.assertTrue(exit_code == 0, 'the boot.py exited with non-zero code.')

    @unittest.skipIf(on_windows(), "'subprocess.check_output' in Windows always return empty string, skip this test.")
    def test_param_validation(self):
        if False:
            return 10
        args = [self.runner_path]
        exit_message = subprocess.check_output(args, env=self.env).decode('utf-8')
        self.assertIn('No id provided.', exit_message)
        args = [self.runner_path, '--id', '1']
        exit_message = subprocess.check_output(args, env=self.env).decode('utf-8')
        self.assertIn('No provision endpoint provided.', exit_message)

    def test_set_working_directory(self):
        if False:
            print('Hello World!')
        JProcessPythonEnvironmentManager = get_gateway().jvm.org.apache.flink.python.env.process.ProcessPythonEnvironmentManager
        output_file = os.path.join(self.tmp_dir, 'output.txt')
        pyflink_dir = os.path.join(self.tmp_dir, 'pyflink')
        os.mkdir(pyflink_dir)
        open(os.path.join(pyflink_dir, '__init__.py'), 'a').close()
        fn_execution_dir = os.path.join(pyflink_dir, 'fn_execution')
        os.mkdir(fn_execution_dir)
        open(os.path.join(fn_execution_dir, '__init__.py'), 'a').close()
        beam_dir = os.path.join(fn_execution_dir, 'beam')
        os.mkdir(beam_dir)
        open(os.path.join(beam_dir, '__init__.py'), 'a').close()
        with open(os.path.join(beam_dir, 'beam_boot.py'), 'w') as f:
            f.write("import os\nwith open(r'%s', 'w') as f:\n    f.write(os.getcwd())" % output_file)
        self.env[JProcessPythonEnvironmentManager.PYTHON_WORKING_DIR] = self.tmp_dir
        self.env['python'] = sys.executable
        args = [self.runner_path]
        subprocess.check_output(args, env=self.env)
        process_cwd = None
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                process_cwd = f.read()
        self.assertEqual(os.path.realpath(self.tmp_dir), process_cwd, 'setting working directory variable is not work!')

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.provision_server.stop(0)
        try:
            if self.tmp_dir is not None:
                shutil.rmtree(self.tmp_dir)
        except:
            pass