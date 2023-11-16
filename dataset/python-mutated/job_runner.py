import json
import os
import tempfile
from typing import TYPE_CHECKING, Any, Dict, Optional
from ray_release.cluster_manager.cluster_manager import ClusterManager
from ray_release.command_runner.command_runner import CommandRunner
from ray_release.exception import ClusterNodesWaitTimeout, CommandError, CommandTimeout, LocalEnvSetupError, LogsError, FetchResultError
from ray_release.file_manager.file_manager import FileManager
from ray_release.job_manager import JobManager
from ray_release.logger import logger
from ray_release.util import format_link, get_anyscale_sdk
from ray_release.wheels import install_matching_ray_locally
if TYPE_CHECKING:
    from anyscale.sdk.anyscale_client.sdk import AnyscaleSDK

class JobRunner(CommandRunner):

    def __init__(self, cluster_manager: ClusterManager, file_manager: FileManager, working_dir: str, sdk: Optional['AnyscaleSDK']=None, artifact_path: Optional[str]=None):
        if False:
            return 10
        super(JobRunner, self).__init__(cluster_manager=cluster_manager, file_manager=file_manager, working_dir=working_dir)
        self.sdk = sdk or get_anyscale_sdk()
        self.job_manager = JobManager(cluster_manager)
        self.last_command_scd_id = None

    def prepare_local_env(self, ray_wheels_url: Optional[str]=None):
        if False:
            i = 10
            return i + 15
        if not os.environ.get('BUILDKITE'):
            return
        try:
            install_matching_ray_locally(ray_wheels_url or os.environ.get('RAY_WHEELS', None))
        except Exception as e:
            raise LocalEnvSetupError(f'Error setting up local environment: {e}') from e

    def _copy_script_to_working_dir(self, script_name):
        if False:
            print('Hello World!')
        script = os.path.join(os.path.dirname(__file__), f'_{script_name}')
        if os.path.exists(script_name):
            os.unlink(script_name)
        os.link(script, script_name)

    def prepare_remote_env(self):
        if False:
            while True:
                i = 10
        self._copy_script_to_working_dir('wait_cluster.py')
        self._copy_script_to_working_dir('prometheus_metrics.py')

    def wait_for_nodes(self, num_nodes: int, timeout: float=900):
        if False:
            return 10
        try:
            self.run_prepare_command(f'python wait_cluster.py {num_nodes} {timeout}', timeout=timeout + 30)
        except (CommandError, CommandTimeout) as e:
            raise ClusterNodesWaitTimeout(f'Not all {num_nodes} nodes came up within {timeout} seconds.') from e

    def save_metrics(self, start_time: float, timeout: float=900):
        if False:
            i = 10
            return i + 15
        self.run_prepare_command(f'python prometheus_metrics.py {start_time}', timeout=timeout)

    def run_command(self, command: str, env: Optional[Dict]=None, timeout: float=3600.0, raise_on_timeout: bool=True) -> float:
        if False:
            print('Hello World!')
        full_env = self.get_full_command_env(env)
        if full_env:
            env_str = ' '.join((f'{k}={v}' for (k, v) in full_env.items())) + ' '
        else:
            env_str = ''
        full_command = f'{env_str}{command}'
        logger.info(f'Running command in cluster {self.cluster_manager.cluster_name}: {full_command}')
        logger.info(f'Link to cluster: {format_link(self.cluster_manager.get_cluster_url())}')
        (status_code, time_taken) = self.job_manager.run_and_wait(full_command, full_env, working_dir='.', timeout=int(timeout))
        if status_code != 0:
            raise CommandError(f'Command returned non-success status: {status_code}')
        return time_taken

    def get_last_logs_ex(self, scd_id: Optional[str]=None):
        if False:
            while True:
                i = 10
        try:
            return self.job_manager.get_last_logs()
        except Exception as e:
            raise LogsError(f'Could not get last logs: {e}') from e

    def _fetch_json(self, path: str) -> Dict[str, Any]:
        if False:
            return 10
        try:
            tmpfile = tempfile.mkstemp(suffix='.json')[1]
            logger.info(tmpfile)
            self.file_manager.download(path, tmpfile)
            with open(tmpfile, 'rt') as f:
                data = json.load(f)
            os.unlink(tmpfile)
            return data
        except Exception as e:
            raise FetchResultError(f'Could not fetch results from session: {e}') from e

    def fetch_results(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return self._fetch_json(self._RESULT_OUTPUT_JSON)

    def fetch_metrics(self) -> Dict[str, Any]:
        if False:
            return 10
        return self._fetch_json(self._METRICS_OUTPUT_JSON)

    def fetch_artifact(self):
        if False:
            print('Hello World!')
        raise NotImplementedError