"""Run cloud checkpointing tests.

This script provides utilities and end to end tests for cloud checkpointing.

Generally the flow is as follows:

A Tune run is started in a separate process. It is terminated after some
time. It is then restarted for another period of time.

We also ensure that checkpoints are properly deleted.

The Tune run is kicked off in _tune_script.py. Trials write a checkpoint
every 2 iterations, and take 5 seconds per iteration.

More details on the expected results can be found in the scenario descriptions.
"""
import argparse
import csv
import io
import tarfile
from dataclasses import dataclass
import json
import os
import pickle
import platform
import re
import shutil
import signal
import subprocess
import tempfile
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import ray
from ray.train._internal.storage import StorageContext
from ray.tune.execution.experiment_state import _find_newest_experiment_checkpoint
from ray.tune.result import _get_defaults_results_dir
from ray.tune.utils.serialization import TuneFunctionDecoder
TUNE_SCRIPT = os.path.join(os.path.dirname(__file__), '_tune_script.py')
ARTIFACT_FILENAME = 'artifact.txt'
CHECKPOINT_DATA_FILENAME = 'dict_checkpoint.pkl'

class ExperimentStateCheckpoint:

    def __init__(self, dir: str, runner_data: Dict[str, Any], trials: List['TrialStub']):
        if False:
            for i in range(10):
                print('nop')
        self.dir = dir
        self.runner_data = runner_data
        self.trials = trials

class ExperimentDirCheckpoint:

    def __init__(self, dir: str, trial_to_cps: Dict['TrialStub', 'TrialCheckpointData']):
        if False:
            i = 10
            return i + 15
        self.dir = dir
        self.trial_to_cps = trial_to_cps

class TrialStub:

    def __init__(self, trainable_name: str, trial_id: str, status: str, config: Dict[str, Any], experiment_tag: str, last_result: Dict[str, Any], relative_logdir: str, storage: StorageContext, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.trainable_name = trainable_name
        self.trial_id = trial_id
        self.status = status
        self.config = config
        self.storage = storage
        self.experiment_tag = experiment_tag
        self.last_result = last_result
        self.relative_logdir = relative_logdir

    @property
    def hostname(self):
        if False:
            i = 10
            return i + 15
        return self.last_result.get('hostname')

    @property
    def node_ip(self):
        if False:
            print('Hello World!')
        return self.last_result.get('node_ip')

    @property
    def dirname(self):
        if False:
            i = 10
            return i + 15
        return os.path.basename(self.relative_logdir)

    @property
    def was_on_driver_node(self):
        if False:
            return 10
        return self.hostname == platform.node()

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash(self.trial_id)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'<TrialStub trial_id={self.trial_id}>'

@dataclass
class TrialCheckpointData:
    params: Dict[str, Any]
    results: List[Dict[str, Any]]
    progress: List[Dict[str, Any]]
    checkpoints: List[Tuple[str, Dict[Any, Any]]]
    num_skipped: int
    artifact_data: str

def delete_file_if_exists(filename: str):
    if False:
        return 10
    if os.path.exists(filename):
        os.remove(filename)

def cleanup_driver_experiment_dir(experiment_name: str):
    if False:
        for i in range(10):
            print('nop')
    experiment_dir = os.path.join(os.path.expanduser('~/ray_results'), experiment_name)
    if os.path.exists(experiment_dir):
        print('Removing existing experiment dir:', experiment_dir)
        shutil.rmtree(experiment_dir)

def cleanup_remote_node_experiment_dir(experiment_name: str):
    if False:
        return 10
    experiment_dir = os.path.join(os.path.expanduser('~/ray_results'), experiment_name)

    @ray.remote
    def _remove_on_remove_node(path: str):
        if False:
            print('Hello World!')
        return shutil.rmtree(path, ignore_errors=True)
    futures = []
    for node in ray.nodes():
        if not node['Alive']:
            continue
        hostname = node['NodeManagerHostname']
        ip = node['NodeManagerAddress']
        if hostname == platform.node():
            continue
        rfn = _remove_on_remove_node.options(resources={f'node:{ip}': 0.01})
        futures.append(rfn.remote(experiment_dir))
    ray.get(futures)

def wait_for_nodes(num_nodes: int, timeout: float=300.0, feedback_interval: float=10.0):
    if False:
        i = 10
        return i + 15
    start = time.time()
    max_time = start + timeout
    next_feedback = start + feedback_interval
    curr_nodes = len(ray.nodes())
    while curr_nodes < num_nodes:
        now = time.time()
        if now >= max_time:
            raise RuntimeError(f'Maximum wait time reached, but only {curr_nodes}/{num_nodes} nodes came up. Aborting.')
        if now >= next_feedback:
            passed = now - start
            print(f'Waiting for more nodes to come up: {curr_nodes}/{num_nodes} ({passed:.0f} seconds passed)')
            next_feedback = now + feedback_interval
        time.sleep(5)
        curr_nodes = len(ray.nodes())

def start_run(no_syncer: bool, storage_path: Optional[str]=None, experiment_name: str='cloud_test', indicator_file: str='/tmp/tune_cloud_indicator') -> subprocess.Popen:
    if False:
        print('Hello World!')
    args = []
    if no_syncer:
        args.append('--no-syncer')
    if storage_path:
        args.extend(['--storage-path', storage_path])
    if experiment_name:
        args.extend(['--experiment-name', experiment_name])
    if indicator_file:
        args.extend(['--indicator-file', indicator_file])
    env = os.environ.copy()
    env['TUNE_RESULT_BUFFER_LENGTH'] = '1'
    env['TUNE_GLOBAL_CHECKPOINT_S'] = '10'
    tune_script = os.environ.get('OVERWRITE_TUNE_SCRIPT', TUNE_SCRIPT)
    full_command = ['python', tune_script] + args
    print(f"Running command: {' '.join(full_command)}")
    process = subprocess.Popen(full_command, env=env)
    return process

def wait_for_run_or_raise(process: subprocess.Popen, indicator_file: str, timeout: int=30):
    if False:
        i = 10
        return i + 15
    print(f'Waiting up to {timeout} seconds until trials have been started (indicated by existence of `{indicator_file}`)')
    timeout = time.monotonic() + timeout
    while process.poll() is None and time.monotonic() < timeout and (not os.path.exists(indicator_file)):
        time.sleep(1)
    if not os.path.exists(indicator_file):
        process.terminate()
        raise RuntimeError(f"Indicator file `{indicator_file}` still doesn't exist, indicating that trials have not been started. Please check the process output.")
    print('Process started, trials are running')

def send_signal_after_wait(process: subprocess.Popen, signal: int, wait: int=30):
    if False:
        return 10
    print(f'Waiting {wait} seconds until sending signal {signal} to process {process.pid}')
    time.sleep(wait)
    if process.poll() is not None:
        raise RuntimeError(f"Process {process.pid} already terminated. This usually means that some of the trials ERRORed (e.g. because they couldn't be restored. Try re-running this test to see if this fixes the issue.")
    print(f'Sending signal {signal} to process {process.pid}')
    process.send_signal(signal)

def wait_until_process_terminated(process: subprocess.Popen, timeout: int=60):
    if False:
        return 10
    print(f'Waiting up to {timeout} seconds until process {process.pid} terminates')
    timeout = time.monotonic() + timeout
    while process.poll() is None and time.monotonic() < timeout:
        time.sleep(1)
    if process.poll() is None:
        process.terminate()
        print(f'Warning: Process {process.pid} did not terminate within timeout, terminating forcefully instead.')
    else:
        print(f'Process {process.pid} terminated gracefully.')

def run_tune_script_for_time(run_time: int, experiment_name: str, indicator_file: str, no_syncer: bool, storage_path: Optional[str], run_start_timeout: int=30):
    if False:
        print('Hello World!')
    process = start_run(no_syncer=no_syncer, storage_path=storage_path, experiment_name=experiment_name, indicator_file=indicator_file)
    try:
        wait_for_run_or_raise(process, indicator_file=indicator_file, timeout=run_start_timeout)
        send_signal_after_wait(process, signal=signal.SIGUSR1, wait=run_time)
        wait_until_process_terminated(process, timeout=45)
    finally:
        process.terminate()

def run_resume_flow(experiment_name: str, indicator_file: str, no_syncer: bool, storage_path: Optional[str], first_run_time: int=33, second_run_time: int=33, run_start_timeout: int=30, before_experiments_callback: Optional[Callable[[], None]]=None, between_experiments_callback: Optional[Callable[[], None]]=None, after_experiments_callback: Optional[Callable[[], None]]=None):
    if False:
        i = 10
        return i + 15
    'Run full flow, i.e.\n\n    - Clean up existing experiment dir\n    - Call before experiment callback\n    - Run tune script for `first_run_time` seconds\n    - Call between experiment callback\n    - Run tune script for another `second_run_time` seconds\n    - Call after experiment callback\n    '
    cleanup_driver_experiment_dir(experiment_name)
    cleanup_remote_node_experiment_dir(experiment_name)
    if before_experiments_callback:
        print('Before experiments: Invoking callback')
        before_experiments_callback()
        print('Before experiments: Callback completed')
    delete_file_if_exists(indicator_file)
    run_tune_script_for_time(run_time=first_run_time, experiment_name=experiment_name, indicator_file=indicator_file, no_syncer=no_syncer, storage_path=storage_path, run_start_timeout=run_start_timeout)
    if between_experiments_callback:
        print('Between experiments: Invoking callback')
        between_experiments_callback()
        print('Between experiments: Callback completed')
    delete_file_if_exists(indicator_file)
    run_tune_script_for_time(run_time=second_run_time, experiment_name=experiment_name, indicator_file=indicator_file, no_syncer=no_syncer, storage_path=storage_path)
    if after_experiments_callback:
        print('After experiments: Invoking callback')
        after_experiments_callback()
        print('After experiments: Callback completed')

def fetch_remote_directory_content(node_ip: str, remote_dir: str, local_dir: str):
    if False:
        i = 10
        return i + 15

    def _pack(dir: str):
        if False:
            print('Hello World!')
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode='w:gz', format=tarfile.PAX_FORMAT) as tar:
            tar.add(dir, arcname='')
        return stream.getvalue()

    def _unpack(stream: str, dir: str):
        if False:
            print('Hello World!')
        with tarfile.open(fileobj=io.BytesIO(stream)) as tar:
            tar.extractall(dir)
    try:
        packed = ray.get(ray.remote(resources={f'node:{node_ip}': 0.01})(_pack).remote(remote_dir))
        _unpack(packed, local_dir)
    except Exception as e:
        print(f'Warning: Could not fetch remote directory contents. Message: {str(e)}')

def send_local_file_to_remote_file(local_path: str, remote_path: str, ip: str):
    if False:
        for i in range(10):
            print('nop')

    def _write(stream: bytes, path: str):
        if False:
            while True:
                i = 10
        with open(path, 'wb') as f:
            f.write(stream)
    with open(local_path, 'rb') as f:
        stream = f.read()
    _remote_write = ray.remote(resources={f'node:{ip}': 0.01})(_write)
    return ray.get(_remote_write.remote(stream, remote_path))

def fetch_remote_file_to_local_file(remote_path: str, ip: str, local_path: str):
    if False:
        return 10

    def _read(path: str):
        if False:
            return 10
        with open(path, 'rb') as f:
            return f.read()
    _remote_read = ray.remote(resources={f'node:{ip}': 0.01})(_read)
    stream = ray.get(_remote_read.remote(remote_path))
    with open(local_path, 'wb') as f:
        f.write(stream)

def fetch_trial_node_dirs_to_tmp_dir(trials: List[TrialStub]) -> Dict[TrialStub, str]:
    if False:
        while True:
            i = 10
    dirmap = {}
    for trial in trials:
        tmpdir = tempfile.mkdtemp(prefix='tune_cloud_test')
        if trial.was_on_driver_node:
            shutil.rmtree(tmpdir)
            shutil.copytree(trial.storage.experiment_local_path, tmpdir)
            print('Copied local node experiment dir', trial.storage.experiment_local_path, 'to', tmpdir, 'for trial', trial.trial_id)
        else:
            fetch_remote_directory_content(trial.node_ip, remote_dir=trial.storage.experiment_local_path, local_dir=tmpdir)
        dirmap[trial] = tmpdir
    return dirmap

def clear_bucket_contents(bucket: str):
    if False:
        for i in range(10):
            print('nop')
    if bucket.startswith('s3://'):
        print('Clearing bucket contents:', bucket)
        subprocess.check_call(['aws', 's3', 'rm', '--recursive', '--quiet', bucket])
    elif bucket.startswith('gs://'):
        print('Clearing bucket contents:', bucket)
        try:
            subprocess.check_call(['gsutil', '-m', 'rm', '-f', '-r', bucket])
        except subprocess.CalledProcessError:
            pass
    else:
        raise ValueError(f'Invalid bucket URL: {bucket}')

def fetch_bucket_contents_to_tmp_dir(bucket: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    tmpdir = tempfile.mkdtemp(prefix='tune_cloud_test')
    subfolder = None
    if bucket.startswith('s3://'):
        subprocess.check_call(['aws', 's3', 'cp', '--recursive', '--quiet', bucket, tmpdir])
    elif bucket.startswith('gs://'):
        try:
            subprocess.check_call(['gsutil', '-m', 'cp', '-r', bucket, tmpdir])
        except subprocess.CalledProcessError as e:
            if len(os.listdir(tmpdir)) == 0:
                raise RuntimeError(f'Local dir {tmpdir} empty after trying to fetch bucket data.') from e
        pattern = re.compile('gs://[^/]+/(.+)')
        subfolder = re.match(pattern, bucket).group(1).split('/')[-1]
    else:
        raise ValueError(f'Invalid bucket URL: {bucket}')
    if subfolder:
        tmpdir = os.path.join(tmpdir, subfolder)
    print('Copied bucket data from', bucket, 'to', tmpdir)
    return tmpdir

def load_experiment_checkpoint_from_state_file(experiment_dir: str) -> ExperimentStateCheckpoint:
    if False:
        while True:
            i = 10
    newest_ckpt_path = _find_newest_experiment_checkpoint(experiment_dir)
    with open(newest_ckpt_path, 'r') as f:
        runner_state = json.load(f, cls=TuneFunctionDecoder)
    trials = []
    for (trial_cp_str, trial_runtime_str) in runner_state['trial_data']:
        trial_state = json.loads(trial_cp_str, cls=TuneFunctionDecoder)
        runtime = json.loads(trial_runtime_str, cls=TuneFunctionDecoder)
        trial_state.update(runtime)
        trial = TrialStub(**trial_state)
        trials.append(trial)
    runner_data = runner_state['runner_data']
    return ExperimentStateCheckpoint(experiment_dir, runner_data, trials)

def load_experiment_checkpoint_from_dir(trials: Iterable[TrialStub], experiment_dir: str) -> ExperimentDirCheckpoint:
    if False:
        i = 10
        return i + 15
    trial_to_cps = {}
    for f in sorted(os.listdir(experiment_dir)):
        full_path = os.path.join(experiment_dir, f)
        if os.path.isdir(full_path):
            trial_stub = None
            for trial in trials:
                if trial.dirname == f:
                    trial_stub = trial
                    break
            if not trial_stub:
                raise RuntimeError(f'Trial with dirname {f} not found.')
            trial_checkpoint_data = load_trial_checkpoint_data(full_path)
            trial_to_cps[trial_stub] = trial_checkpoint_data
    return ExperimentDirCheckpoint(experiment_dir, trial_to_cps)

def load_trial_checkpoint_data(trial_dir: str) -> TrialCheckpointData:
    if False:
        i = 10
        return i + 15
    params_file = os.path.join(trial_dir, 'params.json')
    if os.path.exists(params_file):
        with open(params_file, 'rt') as f:
            params = json.load(f)
    else:
        params = {}
    result_file = os.path.join(trial_dir, 'result.json')
    if os.path.exists(result_file):
        results = []
        with open(result_file, 'rt') as f:
            for line in f.readlines():
                results.append(json.loads(line))
    else:
        results = []
    progress_file = os.path.join(trial_dir, 'progress.csv')
    if os.path.exists(progress_file):
        with open(progress_file, 'rt') as f:
            reader = csv.DictReader(f)
            progress = list(reader)
    else:
        progress = []
    checkpoints = []
    num_skipped = 0
    for cp_dir in sorted(os.listdir(trial_dir)):
        if not cp_dir.startswith('checkpoint_'):
            continue
        cp_full_dir = os.path.join(trial_dir, cp_dir)
        json_path = os.path.join(cp_full_dir, CHECKPOINT_DATA_FILENAME)
        if os.path.exists(json_path):
            with open(json_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
        else:
            continue
        checkpoints.append((cp_dir, checkpoint_data))
    trial_artifact_path = os.path.join(trial_dir, ARTIFACT_FILENAME)
    artifact_data = None
    if os.path.exists(trial_artifact_path):
        with open(trial_artifact_path, 'r') as f:
            artifact_data = f.read()
    return TrialCheckpointData(params=params, results=results, progress=progress, checkpoints=checkpoints, num_skipped=num_skipped, artifact_data=artifact_data)

def load_data_from_trial_exp_checkpoints(trial_to_exp_dir: Dict[TrialStub, str]) -> Dict[TrialStub, ExperimentDirCheckpoint]:
    if False:
        for i in range(10):
            print('nop')
    trial_to_checkpoint_data = {}
    for (trial, dirname) in trial_to_exp_dir.items():
        trial_to_checkpoint_data[trial] = load_experiment_checkpoint_from_dir(trial_to_exp_dir.keys(), dirname)
    return trial_to_checkpoint_data

def get_experiment_and_trial_data(experiment_name: str) -> Tuple[ExperimentStateCheckpoint, ExperimentDirCheckpoint, Dict[TrialStub, ExperimentDirCheckpoint]]:
    if False:
        while True:
            i = 10
    experiment_dir = assert_experiment_dir_exists(experiment_name=experiment_name)
    experiment_state = load_experiment_checkpoint_from_state_file(experiment_dir=experiment_dir)
    assert_experiment_checkpoint_validity(experiment_state)
    driver_dir_cp = load_experiment_checkpoint_from_dir(experiment_state.trials, experiment_dir)
    trial_to_exp_dir = fetch_trial_node_dirs_to_tmp_dir(experiment_state.trials)
    trial_exp_checkpoint_data = load_data_from_trial_exp_checkpoints(trial_to_exp_dir)
    return (experiment_state, driver_dir_cp, trial_exp_checkpoint_data)

def get_bucket_data(bucket: str, experiment_name: str) -> Tuple[ExperimentStateCheckpoint, ExperimentDirCheckpoint]:
    if False:
        print('Hello World!')
    local_bucket_dir = fetch_bucket_contents_to_tmp_dir(bucket)
    local_experiment_dir = os.path.join(local_bucket_dir, experiment_name)
    bucket_state_cp = load_experiment_checkpoint_from_state_file(local_experiment_dir)
    bucket_dir_cp = load_experiment_checkpoint_from_dir(bucket_state_cp.trials, local_experiment_dir)
    return (bucket_state_cp, bucket_dir_cp)

def assert_experiment_dir_exists(experiment_name: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    experiment_dir = os.path.join(_get_defaults_results_dir(), experiment_name)
    if not os.path.exists(experiment_dir):
        raise RuntimeError(f'Check failed: Experiment dir {experiment_dir} does not exist.')
    return experiment_dir

def assert_experiment_checkpoint_validity(experiment_state: ExperimentStateCheckpoint):
    if False:
        return 10
    assert len(experiment_state.trials) == 4, 'Not all trials have been created.'

def assert_min_num_trials(trials: Iterable[TrialStub], on_driver: int, on_worker: int) -> Tuple[int, int]:
    if False:
        return 10
    num_trials_on_driver = len([trial for trial in trials if trial.was_on_driver_node])
    num_trials_not_on_driver = len(trials) - num_trials_on_driver
    assert num_trials_on_driver >= on_driver, f'Not enough trials were scheduled on the driver node ({num_trials_on_driver} < {on_driver}).'
    assert num_trials_not_on_driver >= on_worker, f'Not enough trials were scheduled on remote nodes.({num_trials_on_driver} < {on_worker}).'
    return (num_trials_on_driver, len(trials) - num_trials_on_driver)

def assert_checkpoint_count(experiment_dir_cp: ExperimentDirCheckpoint, for_driver_trial: int, for_worker_trial: int, max_additional: int=0):
    if False:
        return 10
    for (trial, trial_cp) in experiment_dir_cp.trial_to_cps.items():
        cps = len(trial_cp.checkpoints)
        num_skipped = trial_cp.num_skipped
        if trial.was_on_driver_node:
            assert cps >= for_driver_trial and cps <= for_driver_trial + max_additional, f'Trial {trial.trial_id} was on driver, but did not observe the expected amount of checkpoints ({cps} != {for_driver_trial}, skipped={num_skipped}, max_additional={max_additional}). Directory: {experiment_dir_cp.dir}'
        else:
            assert cps >= for_worker_trial and cps <= for_worker_trial + max_additional, f'Trial {trial.trial_id} was not on the driver, but did not observe the expected amount of checkpoints ({cps} != {for_worker_trial}, skipped={num_skipped}, max_additional={max_additional}). Directory: {experiment_dir_cp.dir}'

def assert_artifact_existence_and_validity(experiment_dir_cp: ExperimentDirCheckpoint, exists_for_driver_trials: bool, exists_for_worker_trials: bool, skip_validation: bool=False):
    if False:
        i = 10
        return i + 15
    for (trial, trial_cp) in experiment_dir_cp.trial_to_cps.items():
        artifact_data = trial_cp.artifact_data
        artifact_exists = artifact_data is not None
        if trial.was_on_driver_node:
            assert exists_for_driver_trials == artifact_exists, f"Trial {{trial.trial_id}} was ON THE DRIVER, where the artifact SHOULD {('' if exists_for_driver_trials else 'NOT')} exist, but found that it DOES {('' if artifact_exists else 'NOT')} exist.\nDirectory: {experiment_dir_cp.dir}"
        else:
            assert exists_for_worker_trials == artifact_exists, f"Trial {{trial.trial_id}} was NOT ON THE DRIVER, where the artifact SHOULD {('' if exists_for_driver_trials else 'NOT')} exist, but found that it DOES {('' if artifact_exists else 'NOT')} exist.\nDirectory: {experiment_dir_cp.dir}"
        if not artifact_exists or skip_validation:
            continue
        artifact_data_list = artifact_data.split(',')[:-1]
        artifact_iter = len(artifact_data_list)
        checkpoint_iters = sorted([checkpoint_data['internal_iter'] for (_, checkpoint_data) in trial_cp.checkpoints], reverse=True)
        top_two = checkpoint_iters[:2]
        print(f'\nGot artifact_iter = {artifact_iter}, and top 2 checkpoint iters were {top_two}')
        trial_id = trial.config['id']
        assert all((id == str(trial_id) for id in artifact_data_list)), f'The artifact data should contain only {trial_id}: {artifact_data_list}'
        assert artifact_iter >= min(top_two), f'The artifact data is not synced with respect to the latest checkpoint! Expected the artifact to contain at least {min(top_two)} iterations of data, but only got {artifact_iter}.'

def assert_trial_progressed_training(trial: TrialStub):
    if False:
        return 10
    assert trial.last_result['training_iteration'] > trial.last_result['iterations_since_restore'], f"Trial {trial.trial_id} had a checkpoint but did not continue on resume (training iteration: {trial.last_result['training_iteration']} <={trial.last_result['iterations_since_restore']}). This probably means the checkpoint has not been synced to the node correctly."

def test_durable_upload(bucket: str):
    if False:
        for i in range(10):
            print('nop')
    '\n    Sync trial and experiment checkpoints to cloud, so:\n\n        storage_path="s3://"\n\n    Expected results after first checkpoint:\n\n        - 4 trials are running\n        - At least one trial ran on the head node\n        - At least one trial ran remotely\n        - Driver has NO trial checkpoints from head node trial\n          (since they\'re uploaded directly to storage instead)\n        - Driver has trial artifacts from head node trial\n        - Driver has NO trial checkpoints from remote node trials\n        - Driver has NO trial artifacts from remote node trials\n        - Remote trial dirs only have data for one trial\n        - Remote trial dirs have NO checkpoints for node-local trials\n          (since they\'re uploaded directly to storage instead)\n        - Remote trial dirs have trial artifacts for node-local trials\n        - Cloud checkpoint is valid\n        - Cloud checkpoint has checkpoints from ALL trials\n        - Cloud checkpoint has artifacts from ALL trials (NOT IMPLEMENTED)\n\n    Then, remote checkpoint directories are cleaned up.\n\n    Expected results after second checkpoint:\n\n        - 4 trials are running\n        - All trials progressed with training\n        - Cloud checkpoint is valid\n        - Cloud checkpoint has checkpoints from all trials\n        - Cloud checkpoint has updated synced artifacts for all trials (NOT IMPLEMENTED)\n\n    '
    if not bucket:
        raise ValueError('The `durable_upload` test requires a `--bucket` argument to be set.')
    experiment_name = 'cloud_durable_upload'
    indicator_file = f'/tmp/{experiment_name}_indicator'

    def before_experiments():
        if False:
            while True:
                i = 10
        clear_bucket_contents(bucket)

    def between_experiments():
        if False:
            for i in range(10):
                print('nop')
        (experiment_state, driver_dir_cp, trial_exp_checkpoint_data) = get_experiment_and_trial_data(experiment_name=experiment_name)
        assert all((trial.status == 'RUNNING' for trial in experiment_state.trials)), 'Not all trials are RUNNING'
        assert_min_num_trials(driver_dir_cp.trial_to_cps.keys(), on_driver=1, on_worker=1)
        assert_checkpoint_count(driver_dir_cp, for_driver_trial=0, for_worker_trial=0, max_additional=0)
        for (trial, exp_dir_cp) in trial_exp_checkpoint_data.items():
            seen = len(exp_dir_cp.trial_to_cps)
            if trial.was_on_driver_node:
                assert seen == 4, f'Trial {trial.trial_id} was on driver, but observed too few trials ({seen}) in experiment dir.'
            else:
                assert seen == 1, f'Trial {trial.trial_id} was not on driver, but observed not exactly 1 trials ({seen}) in experiment dir.'
                assert_checkpoint_count(exp_dir_cp, for_driver_trial=0, for_worker_trial=0, max_additional=0)
        (bucket_state_cp, bucket_dir_cp) = get_bucket_data(bucket, experiment_name)
        assert_experiment_checkpoint_validity(bucket_state_cp)
        assert_checkpoint_count(bucket_dir_cp, for_driver_trial=2, for_worker_trial=2, max_additional=2)
        print('Deleting remote checkpoints before resume')
        cleanup_remote_node_experiment_dir(experiment_name)

    def after_experiments():
        if False:
            return 10
        (experiment_state, _, _) = get_experiment_and_trial_data(experiment_name=experiment_name)
        assert all((trial.status == 'RUNNING' for trial in experiment_state.trials)), 'Not all trials are RUNNING'
        for trial in experiment_state.trials:
            assert_trial_progressed_training(trial)
        (bucket_state_cp, bucket_dir_cp) = get_bucket_data(bucket, experiment_name)
        assert_experiment_checkpoint_validity(bucket_state_cp)
        assert_checkpoint_count(bucket_dir_cp, for_driver_trial=2, for_worker_trial=2, max_additional=2)
    run_time = int(os.getenv('TUNE_RUN_TIME', '180')) or 180
    run_start_timeout = 600 if 'rllib' in os.environ['TUNE_TRAINABLE'] else 30
    run_resume_flow(experiment_name=experiment_name, indicator_file=indicator_file, no_syncer=False, storage_path=bucket, first_run_time=run_time, second_run_time=run_time, run_start_timeout=run_start_timeout, before_experiments_callback=before_experiments, between_experiments_callback=between_experiments, after_experiments_callback=after_experiments)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=['no_sync_down', 'ssh_sync', 'durable_upload'])
    parser.add_argument('--trainable', type=str, default='function')
    parser.add_argument('--bucket', type=str, default=None)
    parser.add_argument('--cpus-per-trial', required=False, default=2, type=int)
    args = parser.parse_args()
    addr = os.environ.get('RAY_ADDRESS', '')
    job_name = os.environ.get('RAY_JOB_NAME', 'client_cloud_test')
    if addr.startswith('anyscale://'):
        uses_ray_client = True
        ray.init(address=addr, job_name=job_name, runtime_env={'working_dir': os.path.abspath(os.path.dirname(__file__))})
    else:
        uses_ray_client = False
        ray.init(address='auto')
    print(f'Running cloud test variant: {args.variant}')
    release_test_out = os.environ.get('TEST_OUTPUT_JSON', '/tmp/release_test_out.json')

    def _run_test(variant: str, trainable: str='function', run_time: int=180, bucket: str='', cpus_per_trial: int=2, overwrite_tune_script: Optional[str]=None) -> Dict:
        if False:
            i = 10
            return i + 15
        start_time = time.monotonic()
        print(f'Running test variant `{variant}` on node {ray.util.get_node_ip_address()} with {cpus_per_trial} CPUs per trial.')
        os.environ['TUNE_TRAINABLE'] = str(trainable)
        os.environ['TUNE_RUN_TIME'] = str(run_time)
        os.environ['TUNE_NUM_CPUS_PER_TRIAL'] = str(cpus_per_trial)
        if overwrite_tune_script:
            os.environ['OVERWRITE_TUNE_SCRIPT'] = overwrite_tune_script
            print(f'The test script has been overwritten with {overwrite_tune_script}')
        if variant == 'durable_upload':
            test_durable_upload(bucket)
        else:
            raise NotImplementedError(f'Unknown variant: {variant}')
        time_taken = time.monotonic() - start_time
        result = {'time_taken': time_taken, 'last_update': time.time()}
        return result
    run_time = 180 if 'rllib' in args.trainable else 90
    bucket = None
    if args.bucket:
        bucket = os.path.join(args.bucket, f'test_{int(time.time())}')
    err = None
    try:
        if not uses_ray_client:
            print('This test will *not* use Ray client.')
            result = _run_test(args.variant, args.trainable, run_time, bucket, args.cpus_per_trial)
        else:
            print('This test will run using Ray client.')
            wait_for_nodes(num_nodes=4, timeout=300.0)

            @ray.remote
            def _get_head_ip():
                if False:
                    return 10
                return ray.util.get_node_ip_address()
            ip = ray.get(_get_head_ip.remote())
            remote_tune_script = '/tmp/_tune_script.py'
            print(f'Sending tune script to remote node {ip} ({remote_tune_script})')
            send_local_file_to_remote_file(TUNE_SCRIPT, remote_tune_script, ip)
            print('Starting remote cloud test using Ray client')
            _run_test_remote = ray.remote(resources={f'node:{ip}': 0.01}, num_cpus=0)(_run_test)
            result = ray.get(_run_test_remote.remote(args.variant, args.trainable, run_time, bucket, args.cpus_per_trial, remote_tune_script))
    except Exception as e:
        err = e
        result = {}
    if bucket:
        try:
            pass
        except Exception as be:
            print(f'Error during cleanup of bucket: {be}')
    with open(release_test_out, 'wt') as f:
        json.dump(result, f)
    if err:
        raise err
    print(f'Test for variant {args.variant} SUCCEEDED')