import os
import shutil
from os import path
from apps.dummy.dummyenvironment import DummyTaskEnvironment
from golem.core.common import get_golem_path
from golem.tools.ci import ci_skip
from .test_docker_job import TestDockerJob

@ci_skip
class TestDummyTaskDockerJob(TestDockerJob):
    """Tests for Docker image golem/base"""

    def _get_test_repository(self):
        if False:
            i = 10
            return i + 15
        return 'golemfactory/dummy'

    def _get_test_tag(self):
        if False:
            for i in range(10):
                print('nop')
        return DummyTaskEnvironment.DOCKER_TAG

    def test_dummytask_job(self):
        if False:
            print('Hello World!')
        os.mkdir(os.path.join(self.resources_dir, 'data'))
        os.mkdir(os.path.join(self.resources_dir, 'code'))
        data_dir = path.join(get_golem_path(), 'apps', 'dummy', 'test_data')
        for f in os.listdir(data_dir):
            task_file = path.join(data_dir, f)
            if path.isfile(task_file) or path.isdir(task_file):
                shutil.copy(task_file, path.join(self.resources_dir, 'data', f))
        code_dir = path.join(get_golem_path(), 'apps', 'dummy', 'resources', 'code_dir')
        for f in os.listdir(code_dir):
            task_file = path.join(code_dir, f)
            if (path.isfile(task_file) or path.isdir(task_file)) and os.path.basename(task_file) != '__pycache__':
                shutil.copy(task_file, path.join(self.resources_dir, 'code', f))
        params = {'data_files': ['in.data'], 'subtask_data': '00110011', 'subtask_data_size': 8, 'difficulty': 10, 'result_size': 256, 'result_file': 'out.result'}
        with self._create_test_job(script='/golem/scripts/job.py', params=params) as job:
            job.start()
            exit_code = job.wait()
            self.assertEqual(exit_code, 0)
        out_files = os.listdir(self.output_dir)
        self.assertTrue(any((f.endswith('.result') and 'out' in f for f in out_files)))