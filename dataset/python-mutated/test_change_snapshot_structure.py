from dagster._core.host_representation import ExternalExecutionPlan
from dagster._core.instance import DagsterInstance, InstanceRef
from dagster._core.snap import create_execution_plan_snapshot_id, create_job_snapshot_id
from dagster._utils import file_relative_path
from dagster._utils.test import copy_directory

def test_run_created_in_0_7_9_snapshot_id_change():
    if False:
        i = 10
        return i + 15
    src_dir = file_relative_path(__file__, 'snapshot_0_7_9_shapshot_id_creation_change/sqlite')
    with copy_directory(src_dir) as test_dir:
        instance = DagsterInstance.from_ref(InstanceRef.from_dir(test_dir))
        old_job_snapshot_id = '88528edde2ed64da3c39cca0da8ba2f7586c1a5d'
        old_execution_plan_snapshot_id = '2246f8e5a10d21e15fbfa3773d7b2d0bc1fa9d3d'
        historical_job = instance.get_historical_job(old_job_snapshot_id)
        job_snapshot = historical_job.job_snapshot
        ep_snapshot = instance.get_execution_plan_snapshot(old_execution_plan_snapshot_id)
        created_snapshot_id = create_job_snapshot_id(job_snapshot)
        assert created_snapshot_id != old_job_snapshot_id
        assert historical_job.computed_job_snapshot_id == created_snapshot_id
        assert historical_job.identifying_job_snapshot_id == old_job_snapshot_id
        assert create_execution_plan_snapshot_id(ep_snapshot) != old_execution_plan_snapshot_id
        assert ExternalExecutionPlan(ep_snapshot)