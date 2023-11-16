import os
import pytest
from dagster._core.execution.api import execute_job
from dagster._core.test_utils import environ
from dagster._utils.merger import merge_dicts
from dagster._utils.yaml_utils import merge_yamls
from dagster_test.test_project import find_local_test_image, get_buildkite_registry_config, get_test_project_docker_image, get_test_project_environments_path, get_test_project_recon_job
from . import IS_BUILDKITE, docker_postgres_instance

@pytest.mark.integration
def test_docker_executor(aws_env):
    if False:
        for i in range(10):
            print('nop')
    'Note that this test relies on having AWS credentials in the environment.'
    executor_config = {'execution': {'config': {'networks': ['container:test-postgres-db-docker'], 'env_vars': aws_env}}}
    docker_image = get_test_project_docker_image()
    if IS_BUILDKITE:
        executor_config['execution']['config']['registry'] = get_buildkite_registry_config()
    else:
        find_local_test_image(docker_image)
    run_config = merge_dicts(merge_yamls([os.path.join(get_test_project_environments_path(), 'env.yaml'), os.path.join(get_test_project_environments_path(), 'env_s3.yaml')]), executor_config)
    with environ({'DOCKER_LAUNCHER_NETWORK': 'container:test-postgres-db-docker'}):
        with docker_postgres_instance() as instance:
            recon_job = get_test_project_recon_job('demo_job_docker', docker_image)
            with execute_job(recon_job, run_config=run_config, instance=instance) as result:
                assert result.success

@pytest.mark.integration
def test_docker_executor_check_step_health(aws_env):
    if False:
        return 10
    executor_config = {'execution': {'config': {'networks': ['container:test-postgres-db-docker'], 'env_vars': aws_env}}}
    docker_image = get_test_project_docker_image()
    if IS_BUILDKITE:
        executor_config['execution']['config']['registry'] = get_buildkite_registry_config()
    else:
        find_local_test_image(docker_image)
    run_config = merge_dicts(merge_yamls([os.path.join(get_test_project_environments_path(), 'env.yaml'), os.path.join(get_test_project_environments_path(), 'env_s3.yaml')]), executor_config)
    run_config['ops']['multiply_the_word']['config']['should_segfault'] = True
    with environ({'DOCKER_LAUNCHER_NETWORK': 'container:test-postgres-db-docker'}):
        with docker_postgres_instance() as instance:
            recon_job = get_test_project_recon_job('demo_job_docker', docker_image)
            with execute_job(recon_job, run_config=run_config, instance=instance) as result:
                assert not result.success

@pytest.mark.integration
def test_docker_executor_config_on_container_context(aws_env):
    if False:
        return 10
    'Note that this test relies on having AWS credentials in the environment.'
    executor_config = {'execution': {'config': {}}}
    docker_image = get_test_project_docker_image()
    if IS_BUILDKITE:
        executor_config['execution']['config']['registry'] = get_buildkite_registry_config()
    else:
        find_local_test_image(docker_image)
    run_config = merge_dicts(merge_yamls([os.path.join(get_test_project_environments_path(), 'env.yaml'), os.path.join(get_test_project_environments_path(), 'env_s3.yaml')]), executor_config)
    with environ({'DOCKER_LAUNCHER_NETWORK': 'container:test-postgres-db-docker'}):
        with docker_postgres_instance() as instance:
            recon_job = get_test_project_recon_job('demo_job_docker', docker_image, container_context={'docker': {'networks': ['container:test-postgres-db-docker'], 'env_vars': aws_env}})
            with execute_job(recon_job, run_config=run_config, instance=instance) as result:
                assert result.success

@pytest.mark.integration
def test_docker_executor_retries(aws_env):
    if False:
        for i in range(10):
            print('nop')
    'Note that this test relies on having AWS credentials in the environment.'
    executor_config = {'execution': {'config': {'networks': ['container:test-postgres-db-docker'], 'env_vars': aws_env, 'retries': {'enabled': {}}}}}
    docker_image = get_test_project_docker_image()
    if IS_BUILDKITE:
        executor_config['execution']['config']['registry'] = get_buildkite_registry_config()
    else:
        find_local_test_image(docker_image)
    run_config = merge_dicts(merge_yamls([os.path.join(get_test_project_environments_path(), 'env_s3.yaml')]), executor_config)
    with environ({'DOCKER_LAUNCHER_NETWORK': 'container:test-postgres-db-docker'}):
        with docker_postgres_instance() as instance:
            recon_job = get_test_project_recon_job('step_retries_job_docker', docker_image)
            with execute_job(recon_job, run_config=run_config, instance=instance) as result:
                assert result.success