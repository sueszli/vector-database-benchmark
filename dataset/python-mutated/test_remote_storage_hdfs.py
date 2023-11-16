"""Test remote_storage in a ci environment with real hdfs setup."""
import os
import pytest
from ray import train
from ray.train.base_trainer import TrainingFailedError
from ray.train.data_parallel_trainer import DataParallelTrainer
from ray.train.tests.test_new_persistence import TestConstants, train_fn, _assert_storage_contents, _get_local_inspect_dir, _resume_from_checkpoint

@pytest.fixture
def setup_hdfs():
    if False:
        i = 10
        return i + 15
    'Set env vars required by pyarrow to talk to hdfs correctly.\n\n    Returns hostname and port needed for the hdfs uri.'
    with open('/tmp/hdfs_env', 'r') as f:
        for line in f.readlines():
            line = line.rstrip('\n')
            tokens = line.split('=', maxsplit=1)
            os.environ[tokens[0]] = tokens[1]
    import sys
    sys.path.insert(0, os.path.join(os.environ['HADOOP_HOME'], 'bin'))
    hostname = os.getenv('CONTAINER_ID')
    port = os.getenv('HDFS_PORT')
    yield (hostname, port)

@pytest.mark.skip('TODO(justinvyu): Fix and re-enable this test.')
def test_hdfs_train_checkpointing(tmp_path, monkeypatch, setup_hdfs):
    if False:
        for i in range(10):
            print('nop')
    'See `ray.train.tests.test_new_persistence` for details.'
    LOCAL_CACHE_DIR = tmp_path / 'ray_results'
    monkeypatch.setenv('RAY_AIR_LOCAL_CACHE_DIR', str(LOCAL_CACHE_DIR))
    exp_name = 'trainer_new_persistence'
    no_checkpoint_ranks = [0]
    (hostname, port) = setup_hdfs
    storage_path = f'hdfs://{hostname}:{port}/results/'
    storage_filesystem = None
    checkpoint_config = train.CheckpointConfig(num_to_keep=1, checkpoint_score_attribute=TestConstants.SCORE_KEY, checkpoint_score_order='max')
    trainer = DataParallelTrainer(train_fn, train_loop_config={'in_trainer': True, 'num_iterations': TestConstants.NUM_ITERATIONS, 'fail_iters': [2, 4], 'no_checkpoint_ranks': no_checkpoint_ranks}, scaling_config=train.ScalingConfig(num_workers=TestConstants.NUM_WORKERS), run_config=train.RunConfig(storage_path=storage_path, storage_filesystem=storage_filesystem, name=exp_name, verbose=0, checkpoint_config=checkpoint_config, failure_config=train.FailureConfig(max_failures=1), sync_config=train.SyncConfig(sync_artifacts=True)))
    print('\nStarting initial run.\n')
    with pytest.raises(TrainingFailedError):
        result = trainer.fit()
    print('\nStarting manually restored run.\n')
    restored_trainer = DataParallelTrainer.restore(path=storage_path + exp_name)
    result = restored_trainer.fit()
    with monkeypatch.context() as m:
        m.setenv('RAY_AIR_LOCAL_CACHE_DIR', str(tmp_path / 'resume_from_checkpoint'))
        _resume_from_checkpoint(result.checkpoint, expected_state={'iter': TestConstants.NUM_ITERATIONS - 1})
    (local_inspect_dir, storage_fs_path) = _get_local_inspect_dir(root_local_path=tmp_path, storage_path=storage_path, storage_local_path=LOCAL_CACHE_DIR, storage_filesystem=storage_filesystem)
    print(result)
    trial_fs_path = result.path
    assert trial_fs_path.startswith(storage_fs_path)
    for (checkpoint, _) in result.best_checkpoints:
        assert checkpoint.path.startswith(trial_fs_path)
    _assert_storage_contents(local_inspect_dir, exp_name, checkpoint_config, trainable_name='DataParallelTrainer', test_trainer=True, no_checkpoint_ranks=no_checkpoint_ranks)
if __name__ == '__main__':
    import sys
    sys.exit(pytest.main(['-v', __file__]))