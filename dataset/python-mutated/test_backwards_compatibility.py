import logging
import pytest
import sys
import os
import subprocess
import uuid
from contextlib import contextmanager
from ray.job_submission import JobSubmissionClient, JobStatus
from ray._private.test_utils import wait_for_condition
logger = logging.getLogger(__name__)

@contextmanager
def conda_env(env_name):
    if False:
        while True:
            i = 10
    os.environ['JOB_COMPATIBILITY_TEST_TEMP_ENV'] = env_name
    try:
        yield
    finally:
        del os.environ['JOB_COMPATIBILITY_TEST_TEMP_ENV']
        subprocess.run(f'conda env remove -y --name {env_name}', shell=True, stdout=subprocess.PIPE)

def _compatibility_script_path(file_name: str) -> str:
    if False:
        i = 10
        return i + 15
    return os.path.join(os.path.dirname(__file__), 'backwards_compatibility_scripts', file_name)

class TestBackwardsCompatibility:

    def test_cli(self):
        if False:
            return 10
        "\n        Test that the current commit's CLI works with old server-side Ray versions.\n\n        1) Create a new conda environment with old ray version X installed;\n            inherits same env as current conda envionment except ray version\n        2) (Server) Start head node and dashboard with old ray version X\n        3) (Client) Use current commit's CLI code to do sample job submission flow\n        4) Deactivate the new conda environment and back to original place\n        "
        env_name = f'jobs-backwards-compatibility-{uuid.uuid4().hex}'
        with conda_env(env_name):
            shell_cmd = f"{_compatibility_script_path('test_backwards_compatibility.sh')}"
            try:
                subprocess.check_output(shell_cmd, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                logger.error(str(e))
                logger.error(e.stdout.decode())
                raise e

@pytest.mark.skipif(os.environ.get('JOB_COMPATIBILITY_TEST_TEMP_ENV') is None, reason='This test is only meant to be run from the test_backwards_compatibility.sh shell script.')
def test_error_message():
    if False:
        print('Hello World!')
    '\n    Check that we get a good error message when running against an old server version.\n    '
    client = JobSubmissionClient('http://127.0.0.1:8265')
    job_id = client.submit_job(entrypoint="echo 'hello world'")
    wait_for_condition(lambda : client.get_job_status(job_id) == JobStatus.SUCCEEDED)
    for unsupported_submit_kwargs in [{'entrypoint_num_cpus': 1}, {'entrypoint_num_gpus': 1}, {'entrypoint_resources': {'custom': 1}}]:
        with pytest.raises(Exception, match='Ray version 2.0.1 is running on the cluster. `entrypoint_num_cpus`, `entrypoint_num_gpus`, and `entrypoint_resources` kwargs are not supported on the Ray cluster. Please ensure the cluster is running Ray 2.2 or higher.'):
            client.submit_job(entrypoint='echo hello', **unsupported_submit_kwargs)
    with pytest.raises(Exception, match='Ray version 2.0.1 is running on the cluster. `entrypoint_memory` kwarg is not supported on the Ray cluster. Please ensure the cluster is running Ray 2.8 or higher.'):
        client.submit_job(entrypoint='echo hello', entrypoint_memory=4)
    assert True
if __name__ == '__main__':
    sys.exit(pytest.main(['-v', __file__]))