import os
import re
import time
import docker
import pytest
from dagster._core.storage.dagster_run import DagsterRunStatus, RunsFilter
from dagster._core.test_utils import environ, poll_for_finished_run, poll_for_step_start
from dagster._utils.yaml_utils import merge_yamls
from dagster_docker.docker_run_launcher import DOCKER_CONTAINER_ID_TAG, DOCKER_IMAGE_TAG
from dagster_test.test_project import ReOriginatedExternalJobForTest, find_local_test_image, get_buildkite_registry_config, get_test_project_docker_image, get_test_project_environments_path, get_test_project_recon_job, get_test_project_workspace_and_external_job
from . import IS_BUILDKITE, docker_postgres_instance

@pytest.mark.integration
def test_launch_docker_no_network(aws_env):
    if False:
        for i in range(10):
            print('nop')
    docker_image = get_test_project_docker_image()
    launcher_config = {'env_vars': aws_env}
    if IS_BUILDKITE:
        launcher_config['registry'] = get_buildkite_registry_config()
    else:
        find_local_test_image(docker_image)
    run_config = merge_yamls([os.path.join(get_test_project_environments_path(), 'env.yaml'), os.path.join(get_test_project_environments_path(), 'env_s3.yaml')])
    with docker_postgres_instance(overrides={'run_launcher': {'class': 'DockerRunLauncher', 'module': 'dagster_docker', 'config': launcher_config}}, conn_args={'params': {'connect_timeout': 2}}) as instance:
        recon_job = get_test_project_recon_job('demo_job_s3', docker_image)
        with get_test_project_workspace_and_external_job(instance, 'demo_job_s3', container_image=docker_image) as (workspace, orig_job):
            external_job = ReOriginatedExternalJobForTest(orig_job, container_image=docker_image)
            run = instance.create_run_for_job(job_def=recon_job.get_definition(), run_config=run_config, external_job_origin=external_job.get_external_origin(), job_code_origin=external_job.get_python_origin())
            instance.launch_run(run.run_id, workspace)
            run = instance.get_run_by_id(run.run_id)
            assert run.tags[DOCKER_IMAGE_TAG] == docker_image
            container_id = run.tags[DOCKER_CONTAINER_ID_TAG]
            run = instance.get_run_by_id(run.run_id)
            assert run.status == DagsterRunStatus.STARTING
            assert run.tags[DOCKER_IMAGE_TAG] == docker_image
            client = docker.client.from_env()
            container = None
            try:
                start_time = time.time()
                while True:
                    container = client.containers.get(container_id)
                    if time.time() - start_time > 60:
                        raise Exception('Timed out waiting for container to exit')
                    if container.status == 'exited':
                        break
                    time.sleep(3)
            finally:
                if container:
                    container.remove(force=True)

@pytest.mark.integration
def test_launch_docker_image_on_job_config(aws_env):
    if False:
        return 10
    docker_image = get_test_project_docker_image()
    launcher_config = {'env_vars': aws_env + ['DOCKER_LAUNCHER_NETWORK'], 'network': {'env': 'DOCKER_LAUNCHER_NETWORK'}, 'container_kwargs': {'auto_remove': True}}
    if IS_BUILDKITE:
        launcher_config['registry'] = get_buildkite_registry_config()
    else:
        find_local_test_image(docker_image)
    run_config = merge_yamls([os.path.join(get_test_project_environments_path(), 'env.yaml'), os.path.join(get_test_project_environments_path(), 'env_s3.yaml')])
    with environ({'DOCKER_LAUNCHER_NETWORK': 'container:test-postgres-db-docker'}):
        with docker_postgres_instance(overrides={'run_launcher': {'class': 'DockerRunLauncher', 'module': 'dagster_docker', 'config': launcher_config}}) as instance:
            recon_job = get_test_project_recon_job('demo_job_s3', docker_image)
            with get_test_project_workspace_and_external_job(instance, 'demo_job_s3', container_image=docker_image) as (workspace, orig_job):
                external_job = ReOriginatedExternalJobForTest(orig_job, container_image=docker_image)
                run = instance.create_run_for_job(job_def=recon_job.get_definition(), run_config=run_config, external_job_origin=external_job.get_external_origin(), job_code_origin=external_job.get_python_origin())
                instance.launch_run(run.run_id, workspace)
                poll_for_finished_run(instance, run.run_id, timeout=60)
                run = instance.get_run_by_id(run.run_id)
                assert run.status == DagsterRunStatus.SUCCESS
                assert run.tags[DOCKER_IMAGE_TAG] == docker_image

def _check_event_log_contains(event_log, expected_type_and_message):
    if False:
        while True:
            i = 10
    types_and_messages = [(e.dagster_event.event_type_value, e.message) for e in event_log if e.is_dagster_event]
    for (expected_event_type, expected_message_fragment) in expected_type_and_message:
        assert any((event_type == expected_event_type and expected_message_fragment in message for (event_type, message) in types_and_messages))

@pytest.mark.integration
def test_terminate_launched_docker_run(aws_env):
    if False:
        while True:
            i = 10
    docker_image = get_test_project_docker_image()
    launcher_config = {'env_vars': aws_env, 'network': 'container:test-postgres-db-docker'}
    if IS_BUILDKITE:
        launcher_config['registry'] = get_buildkite_registry_config()
    else:
        find_local_test_image(docker_image)
    run_config = merge_yamls([os.path.join(get_test_project_environments_path(), 'env_s3.yaml')])
    with docker_postgres_instance(overrides={'run_launcher': {'class': 'DockerRunLauncher', 'module': 'dagster_docker', 'config': launcher_config}}) as instance:
        recon_job = get_test_project_recon_job('hanging_job', docker_image)
        with get_test_project_workspace_and_external_job(instance, 'hanging_job', container_image=docker_image) as (workspace, orig_job):
            external_job = ReOriginatedExternalJobForTest(orig_job, container_image=docker_image)
            run = instance.create_run_for_job(job_def=recon_job.get_definition(), run_config=run_config, external_job_origin=external_job.get_external_origin(), job_code_origin=external_job.get_python_origin())
            run_id = run.run_id
            instance.launch_run(run_id, workspace)
            poll_for_step_start(instance, run_id)
            assert instance.run_launcher.terminate(run_id)
            terminated_run = poll_for_finished_run(instance, run_id, timeout=30)
            terminated_run = instance.get_run_by_id(run_id)
            assert terminated_run.status == DagsterRunStatus.CANCELED
            run_logs = instance.all_logs(run_id)
            _check_event_log_contains(run_logs, [('PIPELINE_CANCELING', 'Sending run termination request'), ('STEP_FAILURE', 'Execution of step "hanging_op" failed.'), ('PIPELINE_CANCELED', 'Execution of run for "hanging_job" canceled.'), ('ENGINE_EVENT', 'Process for run exited')])

@pytest.mark.integration
def test_launch_docker_invalid_image(aws_env):
    if False:
        return 10
    docker_image = '_invalid_format_image'
    launcher_config = {'env_vars': aws_env, 'network': 'container:test-postgres-db-docker', 'image': docker_image}
    if IS_BUILDKITE:
        launcher_config['registry'] = get_buildkite_registry_config()
    run_config = merge_yamls([os.path.join(get_test_project_environments_path(), 'env.yaml'), os.path.join(get_test_project_environments_path(), 'env_s3.yaml')])
    with docker_postgres_instance(overrides={'run_launcher': {'class': 'DockerRunLauncher', 'module': 'dagster_docker', 'config': launcher_config}}) as instance:
        recon_job = get_test_project_recon_job('demo_job_s3')
        with get_test_project_workspace_and_external_job(instance, 'demo_job_s3') as (workspace, orig_job):
            external_job = ReOriginatedExternalJobForTest(orig_job)
            run = instance.create_run_for_job(job_def=recon_job.get_definition(), run_config=run_config, external_job_origin=external_job.get_external_origin(), job_code_origin=external_job.get_python_origin())
            with pytest.raises(Exception, match=re.escape('Docker image name _invalid_format_image is not correctly formatted')):
                instance.launch_run(run.run_id, workspace)

@pytest.mark.integration
def test_launch_docker_image_on_instance_config(aws_env):
    if False:
        return 10
    docker_image = get_test_project_docker_image()
    launcher_config = {'env_vars': aws_env, 'network': 'container:test-postgres-db-docker', 'image': docker_image}
    _test_launch(docker_image, launcher_config)

@pytest.mark.integration
def test_launch_docker_image_multiple_networks(aws_env):
    if False:
        i = 10
        return i + 15
    docker_image = get_test_project_docker_image()
    launcher_config = {'env_vars': aws_env, 'networks': ['container:test-postgres-db-docker', 'postgres'], 'image': docker_image}
    _test_launch(docker_image, launcher_config)

@pytest.mark.integration
def test_launch_docker_config_on_container_context(aws_env):
    if False:
        while True:
            i = 10
    docker_image = get_test_project_docker_image()
    launcher_config = {}
    _test_launch(docker_image, launcher_config, container_image=docker_image, container_context={'docker': {'env_vars': aws_env, 'networks': ['container:test-postgres-db-docker', 'postgres']}})

@pytest.mark.integration
def test_cant_combine_network_and_networks(aws_env):
    if False:
        for i in range(10):
            print('nop')
    docker_image = get_test_project_docker_image()
    launcher_config = {'env_vars': aws_env, 'network': 'container:test-postgres-db-docker', 'networks': ['postgres'], 'image': docker_image}
    with pytest.raises(Exception, match='cannot set both `network` and `networks`'):
        with docker_postgres_instance(overrides={'run_launcher': {'class': 'DockerRunLauncher', 'module': 'dagster_docker', 'config': launcher_config}}) as instance:
            print(instance.run_launcher)

@pytest.mark.integration
def test_terminate(aws_env):
    if False:
        while True:
            i = 10
    docker_image = get_test_project_docker_image()
    launcher_config = {'env_vars': aws_env, 'network': 'container:test-postgres-db-docker', 'image': docker_image}
    _test_launch(docker_image, launcher_config, terminate=True)

def _test_launch(docker_image, launcher_config, terminate=False, container_image=None, container_context=None):
    if False:
        while True:
            i = 10
    if IS_BUILDKITE:
        launcher_config['registry'] = get_buildkite_registry_config()
    else:
        find_local_test_image(docker_image)
    run_config = merge_yamls([os.path.join(get_test_project_environments_path(), 'env.yaml'), os.path.join(get_test_project_environments_path(), 'env_s3.yaml')])
    with docker_postgres_instance(overrides={'run_launcher': {'class': 'DockerRunLauncher', 'module': 'dagster_docker', 'config': launcher_config}}) as instance:
        recon_job = get_test_project_recon_job('demo_job_s3', container_image=container_image, container_context=container_context)
        with get_test_project_workspace_and_external_job(instance, 'demo_job_s3', container_image=container_image) as (workspace, orig_job):
            external_job = ReOriginatedExternalJobForTest(orig_job)
            run = instance.create_run_for_job(job_def=recon_job.get_definition(), run_config=run_config, external_job_origin=external_job.get_external_origin(), job_code_origin=recon_job.get_python_origin())
            instance.launch_run(run.run_id, workspace)
            if not terminate:
                poll_for_finished_run(instance, run.run_id, timeout=60)
                assert instance.get_run_by_id(run.run_id).status == DagsterRunStatus.SUCCESS
            else:
                start_time = time.time()
                filters = RunsFilter(run_ids=[run.run_id], statuses=[DagsterRunStatus.STARTED])
                while True:
                    runs = instance.get_runs(filters, limit=1)
                    if runs:
                        break
                    else:
                        time.sleep(0.1)
                        if time.time() - start_time > 60:
                            raise Exception('Timed out waiting for run to start')
                launcher = instance.run_launcher
                assert launcher.terminate(run.run_id)
                poll_for_finished_run(instance, run.run_id, timeout=60)
                assert instance.get_run_by_id(run.run_id).status == DagsterRunStatus.CANCELED