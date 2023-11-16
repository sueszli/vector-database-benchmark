import os
import sys
from typing import List, Tuple, Optional
import yaml
import click
from ci.ray_ci.container import _DOCKER_ECR_REPO
from ci.ray_ci.builder_container import BuilderContainer, DEFAULT_BUILD_TYPE, DEFAULT_PYTHON_VERSION, DEFAULT_ARCHITECTURE
from ci.ray_ci.tester_container import TesterContainer
from ci.ray_ci.utils import docker_login
CUDA_COPYRIGHT = '\n==========\n== CUDA ==\n==========\n\nCUDA Version 11.8.0\n\nContainer image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n\nThis container image and its contents are governed by the NVIDIA Deep Learning Container License.\nBy pulling and using the container, you accept the terms and conditions of this license:\nhttps://developer.nvidia.com/ngc/nvidia-deep-learning-container-license\n\nA copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.\n'
DEFAULT_EXCEPT_TAGS = {'manual'}
bazel_workspace_dir = os.environ.get('BUILD_WORKSPACE_DIRECTORY', '')

@click.command()
@click.argument('targets', required=True, type=str, nargs=-1)
@click.argument('team', required=True, type=str, nargs=1)
@click.option('--workers', default=1, type=int, help='Number of concurrent test jobs to run.')
@click.option('--worker-id', default=0, type=int, help='Index of the concurrent shard to run.')
@click.option('--parallelism-per-worker', default=1, type=int, help='Number of concurrent test jobs to run per worker.')
@click.option('--except-tags', default='', type=str, help='Except tests with the given tags.')
@click.option('--only-tags', default='', type=str, help='Only include tests with the given tags.')
@click.option('--run-flaky-tests', is_flag=True, show_default=True, default=False, help='Run flaky tests.')
@click.option('--skip-ray-installation', is_flag=True, show_default=True, default=False, help='Skip ray installation.')
@click.option('--build-only', is_flag=True, show_default=True, default=False, help='Build ray only, skip running tests.')
@click.option('--gpus', default=0, type=int, help='Number of GPUs to use for the test.')
@click.option('--test-env', multiple=True, type=str, help='Environment variables to set for the test.')
@click.option('--test-arg', type=str, help='Arguments to pass to the test.')
@click.option('--build-name', type=str, help='Name of the build used to run tests')
@click.option('--build-type', type=click.Choice(['optimized', 'debug', 'asan', 'wheel', 'clang', 'asan-clang', 'ubsan', 'tsan-clang', 'java']), default='optimized')
def main(targets: List[str], team: str, workers: int, worker_id: int, parallelism_per_worker: int, except_tags: str, only_tags: str, run_flaky_tests: bool, skip_ray_installation: bool, build_only: bool, gpus: int, test_env: Tuple[str], test_arg: Optional[str], build_name: Optional[str], build_type: Optional[str]) -> None:
    if False:
        return 10
    if not bazel_workspace_dir:
        raise Exception('Please use `bazelisk run //ci/ray_ci`')
    os.chdir(bazel_workspace_dir)
    docker_login(_DOCKER_ECR_REPO.split('/')[0])
    if build_type == 'wheel':
        BuilderContainer(DEFAULT_PYTHON_VERSION, DEFAULT_BUILD_TYPE, DEFAULT_ARCHITECTURE).run()
    container = _get_container(team, workers, worker_id, parallelism_per_worker, gpus, test_env=list(test_env), build_name=build_name, build_type=build_type, skip_ray_installation=skip_ray_installation)
    if build_only:
        sys.exit(0)
    test_targets = _get_test_targets(container, targets, team, except_tags=_add_default_except_tags(except_tags), only_tags=only_tags, get_flaky_tests=run_flaky_tests)
    success = container.run_tests(test_targets, test_arg)
    sys.exit(0 if success else 1)

def _add_default_except_tags(except_tags: str) -> str:
    if False:
        while True:
            i = 10
    final_except_tags = set(DEFAULT_EXCEPT_TAGS)
    if except_tags:
        final_except_tags.update(except_tags.split(','))
    return ','.join(final_except_tags)

def _get_container(team: str, workers: int, worker_id: int, parallelism_per_worker: int, gpus: int, test_env: Optional[List[str]]=None, build_name: Optional[str]=None, build_type: Optional[str]=None, skip_ray_installation: bool=False) -> TesterContainer:
    if False:
        for i in range(10):
            print('nop')
    shard_count = workers * parallelism_per_worker
    shard_start = worker_id * parallelism_per_worker
    shard_end = (worker_id + 1) * parallelism_per_worker
    return TesterContainer(build_name or f'{team}build', test_envs=test_env, shard_count=shard_count, shard_ids=list(range(shard_start, shard_end)), gpus=gpus, skip_ray_installation=skip_ray_installation, build_type=build_type)

def _get_tag_matcher(tag: str) -> str:
    if False:
        print('Hello World!')
    '\n    Return a regular expression that matches the given bazel tag. This is required for\n    an exact tag match because bazel query uses regex to match tags.\n\n    The word boundary is escaped twice because it is used in a python string and then\n    used again as a string in bazel query.\n    '
    return f'\\\\b{tag}\\\\b'

def _get_all_test_query(targets: List[str], team: str, except_tags: Optional[str]=None, only_tags: Optional[str]=None) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Get all test targets that are owned by a particular team, except those that\n    have the given tags\n    '
    test_query = ' union '.join([f'tests({target})' for target in targets])
    query = f"attr(tags, '{_get_tag_matcher(f'team:{team}')}', {test_query})"
    if only_tags:
        only_query = ' union '.join([f"attr(tags, '{_get_tag_matcher(t)}', {test_query})" for t in only_tags.split(',')])
        query = f'{query} intersect ({only_query})'
    if except_tags:
        except_query = ' union '.join([f"attr(tags, '{_get_tag_matcher(t)}', {test_query})" for t in except_tags.split(',')])
        query = f'{query} except ({except_query})'
    return query

def _get_test_targets(container: TesterContainer, targets: str, team: str, except_tags: Optional[str]='', only_tags: Optional[str]='', yaml_dir: Optional[str]=None, get_flaky_tests: bool=False) -> List[str]:
    if False:
        i = 10
        return i + 15
    '\n    Get test targets that are owned by a particular team\n    '
    query = _get_all_test_query(targets, team, except_tags, only_tags)
    test_targets = set(container.run_script_with_output([f'bazel query "{query}"']).decode('utf-8').replace(CUDA_COPYRIGHT, '').strip().split('\n'))
    flaky_tests = set(_get_flaky_test_targets(team, yaml_dir))
    if get_flaky_tests:
        return list(flaky_tests.intersection(test_targets))
    return list(test_targets.difference(flaky_tests))

def _get_flaky_test_targets(team: str, yaml_dir: Optional[str]=None) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Get all test targets that are flaky\n    '
    if not yaml_dir:
        yaml_dir = os.path.join(bazel_workspace_dir, 'ci/ray_ci')
    with open(f'{yaml_dir}/{team}.tests.yml', 'rb') as f:
        flaky_tests = yaml.safe_load(f)['flaky_tests']
    return flaky_tests