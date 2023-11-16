from dagster._core.launcher.sync_in_memory_run_launcher import SyncInMemoryRunLauncher
from dagster._core.test_utils import create_run_for_test, instance_for_test

from .repo import get_main_workspace, main_repo_location_name, main_repo_name


def test_sync_run_launcher_from_configurable_class():
    with instance_for_test(
        overrides={
            "run_launcher": {
                "module": "dagster._core.launcher.sync_in_memory_run_launcher",
                "class": "SyncInMemoryRunLauncher",
            }
        },
    ) as instance_no_hijack:
        assert isinstance(instance_no_hijack.run_launcher, SyncInMemoryRunLauncher)


def test_sync_run_launcher_run():
    with instance_for_test(
        overrides={
            "run_launcher": {
                "module": "dagster._core.launcher.sync_in_memory_run_launcher",
                "class": "SyncInMemoryRunLauncher",
            }
        },
    ) as instance:
        with get_main_workspace(instance) as workspace:
            location = workspace.get_code_location(main_repo_location_name())
            external_repo = location.get_repository(main_repo_name())
            external_pipeline = external_repo.get_full_external_job("noop_job")

            run = create_run_for_test(
                instance=instance,
                job_name=external_pipeline.name,
                external_job_origin=external_pipeline.get_external_origin(),
                job_code_origin=external_pipeline.get_python_origin(),
            )

            run = instance.launch_run(run_id=run.run_id, workspace=workspace)

            completed_run = instance.get_run_by_id(run.run_id)
            assert completed_run.is_success
