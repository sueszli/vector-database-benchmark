import datetime
import io
import json
import functools
import glob
import itertools
import os
import platform
import shutil
import subprocess
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import click
import docker
print = functools.partial(print, file=sys.stderr, flush=True)
DOCKER_USERNAME = 'raytravisbot'
DOCKER_CLIENT = docker.from_env()
PYTHON_WHL_VERSION = 'cp3'
ADDITIONAL_PLATFORMS = ['aarch64']
DOCKER_HUB_REPO = 'rayproject'
DOCKER_HUB_DESCRIPTION = {'base-deps': f'Internal Image, refer to https://hub.docker.com/r/{DOCKER_HUB_REPO}/ray', 'ray-deps': f'Internal Image, refer to https://hub.docker.com/r/{DOCKER_HUB_REPO}/ray', 'ray': 'Official Docker Images for Ray, the distributed computing API.', 'ray-ml': 'Developer ready Docker Image for Ray.', 'ray-worker-container': 'Internal Image for CI test'}
PY_MATRIX = {'py38': '3.8', 'py39': '3.9', 'py310': '3.10', 'py311': '3.11'}
ML_IMAGES_PY_VERSIONS = {'py38', 'py39', 'py310'}
BASE_IMAGES = {'cu121': 'nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04', 'cu118': 'nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04', 'cu117': 'nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04', 'cu116': 'nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04', 'cu115': 'nvidia/cuda:11.5.2-cudnn8-devel-ubuntu20.04', 'cpu': 'ubuntu:focal'}
CUDA_FULL = {'cu121': 'CUDA 12.1', 'cu118': 'CUDA 11.8', 'cu117': 'CUDA 11.7', 'cu116': 'CUDA 11.6', 'cu115': 'CUDA 11.5'}
ML_CUDA_VERSION = 'cu118'
DEFAULT_PYTHON_VERSION = 'py38'
IMAGE_NAMES = list(DOCKER_HUB_DESCRIPTION.keys())
RELEASE_PR_PIPELINE_ID = 'd912884a-5198-497d-9ac3-178420500b6e'

def _with_suffix(tag: str, suffix: Optional[str]=None) -> str:
    if False:
        print('Hello World!')
    if suffix:
        return tag + '-' + suffix
    return tag

def _get_branch() -> str:
    if False:
        for i in range(10):
            print('nop')
    branch = os.environ.get('TRAVIS_BRANCH') or os.environ.get('BUILDKITE_BRANCH')
    if not branch:
        print('Branch not found!')
        print(os.environ)
        print('Environment is above ^^')
        return ''
    return branch
_BRANCH = _get_branch()

def _release_build(branch: Optional[str]=None) -> bool:
    if False:
        while True:
            i = 10
    if not branch:
        branch = _BRANCH
    return branch and branch.startswith('releases/')

def _valid_branch(branch: Optional[str]=None) -> bool:
    if False:
        print('Hello World!')
    if not branch:
        branch = _BRANCH
    return branch == 'master' or _release_build(branch)

def _get_curr_dir() -> str:
    if False:
        while True:
            i = 10
    return os.path.dirname(os.path.realpath(__file__))

def _get_root_dir() -> str:
    if False:
        while True:
            i = 10
    return os.path.join(_get_curr_dir(), '../../')

def _get_commit_sha() -> str:
    if False:
        while True:
            i = 10
    sha = os.environ.get('TRAVIS_COMMIT') or os.environ.get('BUILDKITE_COMMIT') or ''
    if len(sha) < 6:
        print('INVALID SHA FOUND')
        return 'ERROR'
    return sha[:6]
_COMMIT_SHA = _get_commit_sha()

def _configure_human_version():
    if False:
        i = 10
        return i + 15
    global _BRANCH
    global _COMMIT_SHA
    _BRANCH = input("Provide a 'branch name'. For releases, it should be `releases/x.x.x`")
    _COMMIT_SHA = input('Provide a SHA (used for tag value)')

def _get_wheel_name(minor_version_number) -> str:
    if False:
        while True:
            i = 10
    if minor_version_number:
        matches = [file for file in glob.glob(f'{_get_root_dir()}/.whl/ray-*{PYTHON_WHL_VERSION}{minor_version_number}*-manylinux*') if '+' not in file]
        assert len(matches) == 1, f"Found ({len(matches)}) matches for 'ray-*{PYTHON_WHL_VERSION}{minor_version_number}*-manylinux*' instead of 1.\nwheel matches: {matches}"
        return os.path.basename(matches[0])
    else:
        matches = glob.glob(f'{_get_root_dir()}/.whl/*{PYTHON_WHL_VERSION}*-manylinux*')
        return [os.path.basename(i) for i in matches]

def _check_if_docker_files_modified():
    if False:
        return 10
    stdout = subprocess.check_output([sys.executable, f'{_get_curr_dir()}/../pipeline/determine_tests_to_run.py', '--output=json'])
    affected_env_var_list = json.loads(stdout)
    affected = 'RAY_CI_DOCKER_AFFECTED' in affected_env_var_list or 'RAY_CI_PYTHON_DEPENDENCIES_AFFECTED' in affected_env_var_list
    print(f'Docker affected: {affected}')
    return affected

def _build_docker_image(image_name: str, py_version: str, image_type: str, suffix: Optional[str]=None, no_cache=True):
    if False:
        return 10
    'Builds Docker image with the provided info.\n\n    image_name: The name of the image to build. Must be one of\n        IMAGE_NAMES.\n    py_version: The Python version to build the image for.\n        Must be one of PY_MATRIX.keys()\n    image_type: The image type to build. Must be one of\n        BASE_IMAGES.keys()\n    suffix: Suffix to add to the tags (e.g. "aarch64" for "ray:sha256-aarch64")\n    no_cache: If True, don\'t use caching when building the image.\n    '
    if image_name not in IMAGE_NAMES:
        raise ValueError(f'The provided image name {image_name} is not recognized. Image names must be one of {IMAGE_NAMES}')
    if py_version not in PY_MATRIX.keys():
        raise ValueError(f'The provided python version {py_version} is not recognized. Python version must be one of {PY_MATRIX.keys()}')
    if image_type not in BASE_IMAGES.keys():
        raise ValueError(f'The provided CUDA version {image_type} is not recognized. CUDA version must be one of {BASE_IMAGES.keys()}')
    build_args = {}
    build_args['PYTHON_VERSION'] = PY_MATRIX[py_version]
    assert py_version[:3] == 'py3'
    python_minor_version = py_version[3:]
    constraints_file = 'requirements_compiled.txt'
    build_args['CONSTRAINTS_FILE'] = constraints_file
    if platform.processor() in ADDITIONAL_PLATFORMS:
        build_args['HOSTTYPE'] = platform.processor()
    device_tag = f'{image_type}'
    if image_name == 'base-deps':
        base_image = BASE_IMAGES[image_type]
    else:
        base_image = _with_suffix(f'-{py_version}-{device_tag}', suffix=suffix)
    if image_name != 'ray-worker-container':
        build_args['BASE_IMAGE'] = base_image
    if image_name in ['ray', 'ray-deps', 'ray-worker-container']:
        wheel = _get_wheel_name(python_minor_version)
        build_args['WHEEL_PATH'] = f'.whl/{wheel}'
        build_args['FIND_LINKS_PATH'] = '.whl'
    tagged_name = f'{DOCKER_HUB_REPO}/{image_name}:nightly-{py_version}-{device_tag}'
    tagged_name = _with_suffix(tagged_name, suffix=suffix)
    for i in range(2):
        cleanup = DOCKER_CLIENT.containers.prune().get('SpaceReclaimed')
        if cleanup is not None:
            print(f'Cleaned up {cleanup / 2 ** 20}MB')
        labels = {'image-name': image_name, 'python-version': PY_MATRIX[py_version], 'ray-commit': _COMMIT_SHA}
        if image_type in CUDA_FULL:
            labels['cuda-version'] = CUDA_FULL[image_type]
        output = DOCKER_CLIENT.api.build(path=os.path.join(_get_root_dir(), 'docker', image_name), tag=tagged_name, nocache=no_cache, labels=labels, buildargs=build_args)
        cmd_output = []
        try:
            start = datetime.datetime.now()
            current_iter = start
            for line in output:
                cmd_output.append(line.decode('utf-8'))
                if datetime.datetime.now() - current_iter >= datetime.timedelta(minutes=5):
                    current_iter = datetime.datetime.now()
                    elapsed = datetime.datetime.now() - start
                    print(f'Still building {tagged_name} after {elapsed.seconds} seconds')
                    if elapsed >= datetime.timedelta(minutes=15):
                        print('Additional build output:')
                        print(*cmd_output, sep='\n')
                        cmd_output = []
        except Exception as e:
            print(f'FAILURE with error {e}')
        if len(DOCKER_CLIENT.api.images(tagged_name)) == 0:
            print(f'ERROR building: {tagged_name}. Output below:')
            print(*cmd_output, sep='\n')
            if i == 1:
                raise Exception('FAILED TO BUILD IMAGE')
            print('TRYING AGAIN')
        else:
            break
    print('BUILT: ', tagged_name)
    return tagged_name

def _extract_files_from_docker(docker_image: str, files: Dict[str, str]):
    if False:
        return 10
    'Extract files from docker container image and save to local disk.\n\n    ``files`` is a dict mapping from paths inside the docker container to\n    local paths on the host system.\n    '
    container = DOCKER_CLIENT.containers.create(docker_image)
    for (container_path, local_path) in files.items():
        (stream, stat) = container.get_archive(f'{container_path}')
        local_path = Path(local_path)
        local_path.parent.mkdir(exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(b''.join((d for d in stream)))) as tar:
            with open(local_path, 'wb') as f:
                for r in tar.extractfile(os.path.basename(container_path)):
                    f.write(r)
    container.remove()

def extract_image_infos(images: List[str], target_dir: str):
    if False:
        for i in range(10):
            print('nop')
    for image in images:
        image_basename = image.replace('rayproject/', '')
        _extract_files_from_docker(image, {'/home/ray/pip-freeze.txt': f'{target_dir}/{image_basename}_pip-freeze.txt'})

def copy_wheels(human_build):
    if False:
        while True:
            i = 10
    if human_build:
        print('Please download images using:\n`pip download --python-version <py_version> ray==<ray_version>')
    root_dir = _get_root_dir()
    wheels = _get_wheel_name(None)
    for wheel in wheels:
        source = os.path.join(root_dir, '.whl', wheel)
        ray_dst = os.path.join(root_dir, 'docker/ray/.whl/')
        ray_dep_dst = os.path.join(root_dir, 'docker/ray-deps/.whl/')
        ray_worker_container_dst = os.path.join(root_dir, 'docker/ray-worker-container/.whl/')
        os.makedirs(ray_dst, exist_ok=True)
        shutil.copy(source, ray_dst)
        os.makedirs(ray_dep_dst, exist_ok=True)
        shutil.copy(source, ray_dep_dst)
        os.makedirs(ray_worker_container_dst, exist_ok=True)
        shutil.copy(source, ray_worker_container_dst)

def check_staleness(repository, tag):
    if False:
        return 10
    DOCKER_CLIENT.api.pull(repository=repository, tag=tag)
    age = DOCKER_CLIENT.api.inspect_image(f'{repository}:{tag}')['Created']
    short_date = datetime.datetime.strptime(age.split('T')[0], '%Y-%m-%d')
    is_stale = datetime.datetime.now() - short_date > datetime.timedelta(days=14)
    return is_stale

def build_for_all_versions(image_name, py_versions, image_types, suffix, **kwargs) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'Builds the given Docker image for all Python & CUDA versions'
    tagged_names = []
    for py_version in py_versions:
        for image_type in image_types:
            tagged_name = _build_docker_image(image_name, py_version=py_version, image_type=image_type, suffix=suffix, **kwargs)
            tagged_names.append(tagged_name)
    return tagged_names

def build_base_images(py_versions, image_types, suffix):
    if False:
        for i in range(10):
            print('nop')
    build_for_all_versions('base-deps', py_versions, image_types, suffix=suffix, no_cache=False)
    build_for_all_versions('ray-deps', py_versions, image_types, suffix=suffix, no_cache=False)

def build_or_pull_base_images(py_versions: List[str], image_types: List[str], rebuild_base_images: bool=True, suffix: Optional[str]=None) -> bool:
    if False:
        return 10
    'Returns images to tag and build.'
    repositories = [f'{DOCKER_HUB_REPO}/base-deps', f'{DOCKER_HUB_REPO}/ray-deps']
    tags = [f'nightly-{py_version}-{image_type}' for (py_version, image_type) in itertools.product(py_versions, image_types)]
    try:
        is_stale = check_staleness(repositories[0], tags[0])
        for repository in repositories:
            for tag in tags:
                DOCKER_CLIENT.api.pull(repository=repository, tag=tag)
    except Exception as e:
        print(e)
        is_stale = True
    if rebuild_base_images or _release_build() or is_stale:
        build_base_images(py_versions, image_types, suffix=suffix)
        return True
    else:
        print('Just pulling images!')
        return False

def prep_ray_base():
    if False:
        return 10
    root_dir = _get_root_dir()
    requirements_files = ['python/requirements_compiled.txt']
    for requirement_file in requirements_files:
        shutil.copy(os.path.join(root_dir, requirement_file), os.path.join(root_dir, 'docker/ray/'))

def prep_ray_ml():
    if False:
        for i in range(10):
            print('nop')
    root_dir = _get_root_dir()
    requirements_files = ['python/requirements.txt', 'python/requirements_compiled.txt']
    ml_requirements_files = ['python/requirements/docker/ray-docker-requirements.txt', 'python/requirements/ml/core-requirements.txt', 'python/requirements/ml/data-requirements.txt', 'python/requirements/ml/dl-gpu-requirements.txt', 'python/requirements/ml/dl-cpu-requirements.txt', 'python/requirements/ml/tune-requirements.txt', 'python/requirements/ml/tune-test-requirements.txt', 'python/requirements/ml/rllib-requirements.txt', 'python/requirements/ml/rllib-test-requirements.txt', 'python/requirements/ml/train-requirements.txt', 'python/requirements/ml/train-test-requirements.txt']
    ignore_requirements = ['python/requirements/compat/requirements_legacy_compat.txt', 'python/requirements/ml/data-test-requirements.txt']
    files_on_disk = glob.glob(f'{root_dir}/python/**/*-requirements.txt', recursive=True)
    for file_on_disk in files_on_disk:
        rel = os.path.relpath(file_on_disk, start=root_dir)
        print(rel)
        if not rel.startswith('python/requirements/ml'):
            continue
        elif rel not in ml_requirements_files and rel not in ignore_requirements:
            raise RuntimeError(f'A new requirements file was found in the repository, but it has not been added to `build-docker-images.py` (and the `ray-ml/Dockerfile`): {rel}')
    for requirement_file in requirements_files + ml_requirements_files:
        shutil.copy(os.path.join(root_dir, requirement_file), os.path.join(root_dir, 'docker/ray-ml/'))

def _get_docker_creds() -> Tuple[str, str]:
    if False:
        for i in range(10):
            print('nop')
    docker_password = os.environ.get('DOCKER_PASSWORD')
    assert docker_password, 'DOCKER_PASSWORD not set.'
    return (DOCKER_USERNAME, docker_password)

def _docker_push(image, tag):
    if False:
        while True:
            i = 10
    print(f'PUSHING: {image}:{tag}, result:')
    i = 0
    for progress_line in DOCKER_CLIENT.api.push(image, tag=tag, stream=True):
        if i % 100 == 0:
            print(progress_line)

def _tag_and_push(full_image_name: str, old_tag: str, new_tag: str, merge_build: bool=False, release_pr_build: bool=False):
    if False:
        for i in range(10):
            print('nop')
    if 'nightly' in new_tag and (_release_build() or release_pr_build):
        return
    if old_tag != new_tag:
        DOCKER_CLIENT.api.tag(image=f'{full_image_name}:{old_tag}', repository=full_image_name, tag=new_tag)
    if not merge_build and (not release_pr_build):
        print('Not pushing build. Otherwise we would have pushed to: {full_image_name}:{new_tag}')
    else:
        _docker_push(full_image_name, new_tag)

def _create_new_tags(all_tags, old_str, new_str):
    if False:
        i = 10
        return i + 15
    new_tags = []
    for full_tag in all_tags:
        new_tag = full_tag.replace(old_str, new_str)
        new_tags.append(new_tag)
    return new_tags

def create_image_tags(image_name: str, py_versions: List[str], image_types: List[str], specific_tag: Optional[str]=None, version: str='nightly', suffix: Optional[str]=None):
    if False:
        while True:
            i = 10
    tag_mapping = defaultdict(list)
    for py_name in py_versions:
        for image_type in image_types:
            if image_name == 'ray-ml':
                if image_type not in [ML_CUDA_VERSION, 'cpu']:
                    print(f'ML Docker image is not built for the following device type: {image_type}')
                    continue
                if py_name not in ML_IMAGES_PY_VERSIONS:
                    print(f'ML Docker iamge is not build for the following python version: {py_name}')
                    continue
            tag = _with_suffix(f'{version}-{py_name}-{image_type}', suffix=suffix)
            tag_mapping[tag].append(tag)
    for old_tag in tag_mapping.keys():
        if 'cpu' in old_tag and image_name != 'ray-ml':
            new_tags = _create_new_tags(tag_mapping[old_tag], old_str='-cpu', new_str='')
            tag_mapping[old_tag].extend(new_tags)
        elif ML_CUDA_VERSION in old_tag:
            new_tags = _create_new_tags(tag_mapping[old_tag], old_str=f'-{ML_CUDA_VERSION}', new_str='-gpu')
            tag_mapping[old_tag].extend(new_tags)
            if image_name == 'ray-ml':
                new_tags = _create_new_tags(tag_mapping[old_tag], old_str=f'-{ML_CUDA_VERSION}', new_str='')
                tag_mapping[old_tag].extend(new_tags)
    for old_tag in tag_mapping.keys():
        if DEFAULT_PYTHON_VERSION in old_tag:
            new_tags = _create_new_tags(tag_mapping[old_tag], old_str=f'-{DEFAULT_PYTHON_VERSION}', new_str='')
            tag_mapping[old_tag].extend(new_tags)
    if specific_tag:
        for old_tag in tag_mapping.keys():
            new_tags = _create_new_tags(tag_mapping[old_tag], old_str=version, new_str=specific_tag)
            tag_mapping[old_tag].extend(new_tags)
    return tag_mapping

def push_and_tag_images(py_versions: List[str], image_types: List[str], merge_build: bool=False, release_pr_build: bool=False, image_list: Optional[List[str]]=None, suffix: Optional[str]=None):
    if False:
        i = 10
        return i + 15
    date_tag = datetime.datetime.now().strftime('%Y-%m-%d')
    sha_tag = _COMMIT_SHA
    if _release_build():
        release_name = _BRANCH[len('releases/'):]
        date_tag = release_name + '.' + date_tag
        sha_tag = release_name + '.' + sha_tag
    if release_pr_build:
        pr = f"pr-{os.environ['BUILDKITE_PULL_REQUEST']}"
        date_tag = pr + '.' + date_tag
        sha_tag = pr + '.' + sha_tag
    for image_name in image_list:
        full_image_name = f'rayproject/{image_name}'
        tag_mapping = create_image_tags(image_name=image_name, py_versions=py_versions, image_types=image_types, specific_tag=date_tag if '-deps' in image_name else sha_tag, version='nightly', suffix=suffix)
        print(f'These tags will be created for {image_name}: ', tag_mapping)
        for old_tag in tag_mapping.keys():
            if DEFAULT_PYTHON_VERSION in old_tag:
                if '-cpu' in old_tag:
                    assert _with_suffix('nightly-cpu', suffix=suffix) in tag_mapping[old_tag]
                    if '-deps' in image_name:
                        assert _with_suffix('nightly', suffix=suffix) in tag_mapping[old_tag]
                        assert _with_suffix(f'{date_tag}-cpu', suffix=suffix) in tag_mapping[old_tag]
                        assert _with_suffix(f'{date_tag}', suffix=suffix) in tag_mapping[old_tag]
                    elif image_name == 'ray':
                        assert _with_suffix('nightly', suffix=suffix) in tag_mapping[old_tag]
                        assert _with_suffix(f'{sha_tag}-cpu', suffix=suffix) in tag_mapping[old_tag]
                        assert _with_suffix(f'{sha_tag}', suffix=suffix) in tag_mapping[old_tag]
                    elif image_name == 'ray-ml':
                        assert _with_suffix(f'{sha_tag}-cpu', suffix=suffix) in tag_mapping[old_tag]
                    else:
                        raise RuntimeError(f'Invalid image name: {image_name}')
                elif ML_CUDA_VERSION in old_tag:
                    assert _with_suffix('nightly-gpu', suffix=suffix) in tag_mapping[old_tag]
                    if '-deps' in image_name:
                        assert _with_suffix(f'{date_tag}-gpu', suffix=suffix) in tag_mapping[old_tag]
                    elif image_name == 'ray':
                        assert _with_suffix(f'{sha_tag}-gpu', suffix=suffix) in tag_mapping[old_tag]
                    elif image_name == 'ray-ml':
                        assert _with_suffix('nightly', suffix=suffix) in tag_mapping[old_tag]
                        assert _with_suffix(f'{sha_tag}', suffix=suffix) in tag_mapping[old_tag]
                        assert _with_suffix(f'{sha_tag}-gpu', suffix=suffix) in tag_mapping[old_tag]
                    else:
                        raise RuntimeError(f'Invalid image name: {image_name}')
        for old_tag in tag_mapping.keys():
            for new_tag in tag_mapping[old_tag]:
                _tag_and_push(full_image_name, old_tag=old_tag, new_tag=new_tag, merge_build=merge_build, release_pr_build=release_pr_build)

def push_readmes(merge_build: bool):
    if False:
        while True:
            i = 10
    if not merge_build:
        print('Not pushing README because this is a PR build.')
        return
    (username, password) = _get_docker_creds()
    for (image, tag_line) in DOCKER_HUB_DESCRIPTION.items():
        environment = {'DOCKER_USER': username, 'DOCKER_PASS': password, 'PUSHRM_FILE': f'/myvol/docker/{image}/README.md', 'PUSHRM_DEBUG': 1, 'PUSHRM_SHORT': tag_line}
        cmd_string = f'{DOCKER_HUB_REPO}/{image}'
        print(DOCKER_CLIENT.containers.run('chko/docker-pushrm:1', command=cmd_string, volumes={os.path.abspath(_get_root_dir()): {'bind': '/myvol', 'mode': 'rw'}}, environment=environment, remove=True, detach=False, stderr=True, stdout=True, tty=False))
MERGE = 'MERGE'
HUMAN = 'HUMAN'
PR = 'PR'
RELEASE_PR = 'RELEASE_PR'
BUILDKITE = 'BUILDKITE'
LOCAL = 'LOCAL'
BUILD_TYPES = [MERGE, HUMAN, PR, RELEASE_PR, BUILDKITE, LOCAL]

@click.command()
@click.option('--py-versions', '-V', default=['py38'], type=click.Choice(list(PY_MATRIX.keys())), multiple=True, help='Which python versions to build. Must be in (py38, py39, py310, py311)')
@click.option('--device-types', '-T', default=[], type=click.Choice(list(BASE_IMAGES.keys())), multiple=True, help='Which device types (CPU/CUDA versions) to build images for. If not specified, images will be built for all device types.')
@click.option('--build-type', type=click.Choice(BUILD_TYPES), required=True, help='Whether to bypass checking if docker is affected')
@click.option('--suffix', type=click.Choice(ADDITIONAL_PLATFORMS), help='Suffix to append to the build tags')
@click.option('--build-base/--no-build-base', default=True, help='Whether to build base-deps & ray-deps')
@click.option('--only-build-worker-container/--no-only-build-worker-container', default=False, help='Whether only to build ray-worker-container')
def main(py_versions: Tuple[str], device_types: Tuple[str], build_type: str, suffix: Optional[str]=None, build_base: bool=True, only_build_worker_container: bool=False):
    if False:
        i = 10
        return i + 15
    py_versions = list(py_versions) if isinstance(py_versions, (list, tuple)) else [py_versions]
    image_types = list(device_types) if isinstance(device_types, (list, tuple)) else list(BASE_IMAGES.keys())
    assert set(list(CUDA_FULL.keys()) + ['cpu']) == set(BASE_IMAGES.keys())
    py_version_file = os.path.join(_get_root_dir(), 'docker/retag-lambda', 'python_versions.txt')
    with open(py_version_file) as f:
        py_file_versions = f.read().splitlines()
        assert set(PY_MATRIX.keys()) == set(py_file_versions), (PY_MATRIX.keys(), py_file_versions)
    cuda_version_file = os.path.join(_get_root_dir(), 'docker/retag-lambda', 'cuda_versions.txt')
    with open(cuda_version_file) as f:
        cuda_file_versions = f.read().splitlines()
        assert set(BASE_IMAGES.keys()) == set(cuda_file_versions + ['cpu']), (BASE_IMAGES.keys(), cuda_file_versions + ['cpu'])
    print('Building the following python versions: ', [PY_MATRIX[py_version] for py_version in py_versions])
    print('Building images for the following devices: ', image_types)
    print('Building base images: ', build_base)
    is_buildkite = build_type == BUILDKITE
    is_local = build_type == LOCAL
    if build_type == BUILDKITE:
        if os.environ.get('BUILDKITE_PULL_REQUEST', '') == 'false':
            build_type = MERGE
        elif os.environ.get('BUILDKITE_PIPELINE_ID', '') == RELEASE_PR_PIPELINE_ID:
            build_type = RELEASE_PR
        else:
            build_type = PR
    if build_type == HUMAN:
        _configure_human_version()
    if build_type in {HUMAN, MERGE, BUILDKITE, LOCAL, RELEASE_PR} or _check_if_docker_files_modified() or only_build_worker_container:
        is_merge = build_type == MERGE
        if is_merge and (not is_buildkite) and (not is_local):
            (username, password) = _get_docker_creds()
            DOCKER_CLIENT.api.login(username=username, password=password)
        copy_wheels(build_type == HUMAN)
        is_base_images_built = build_or_pull_base_images(py_versions, image_types, build_base, suffix=suffix)
        if only_build_worker_container:
            build_for_all_versions('ray-worker-container', py_versions, image_types, suffix=suffix)
        else:
            prep_ray_base()
            all_tagged_images = []
            all_tagged_images += build_for_all_versions('ray', py_versions, image_types, suffix=suffix)
            images_to_tag_and_push = []
            if is_base_images_built:
                images_to_tag_and_push += ['base-deps', 'ray-deps']
            images_to_tag_and_push += ['ray']
            if platform.processor() not in ADDITIONAL_PLATFORMS:
                ml_image_types = [image_type for image_type in image_types if image_type in [ML_CUDA_VERSION, 'cpu']]
            else:
                ml_image_types = []
            ml_py_versions = [py_version for py_version in py_versions if py_version in ML_IMAGES_PY_VERSIONS]
            if len(ml_image_types) > 0:
                prep_ray_ml()
                all_tagged_images += build_for_all_versions('ray-ml', ml_py_versions, image_types=ml_image_types, suffix=suffix)
                images_to_tag_and_push += ['ray-ml']
            if is_buildkite:
                extract_image_infos(all_tagged_images, target_dir='/artifact-mount/.image-info')
            if build_type in {MERGE, RELEASE_PR}:
                valid_branch = _valid_branch()
                if not valid_branch and is_merge:
                    print(f'Invalid Branch found: {_get_branch()}')
                push_and_tag_images(py_versions, image_types, merge_build=valid_branch and is_merge, release_pr_build=build_type == RELEASE_PR, image_list=images_to_tag_and_push, suffix=suffix)

def _fix_docker_images(image: str='ray-ml', version: str='nightly', repo: str=DOCKER_HUB_REPO):
    if False:
        print('Hello World!')
    'Print commands to manually update docker images post-release.\n\n    This function prints commands that can be run to add new layers to\n    fix docker images post-release, e.g. when dependencies have to be fixed\n    or public keys expired.\n\n    The commands can be copied/pasted and executed in a shell.\n\n    Example:\n        FIX_IMAGE=ray-ml FIX_VERSION=2.3.0 python build-docker-images.py\n\n    '
    tags = create_image_tags(image_name=image, py_versions=list(PY_MATRIX.keys()), image_types=list(BASE_IMAGES.keys()), specific_tag=None, version=version, suffix=None)
    print(dict(tags))
    for base_tag in tags:
        base_image = f'{repo}/{image}:{base_tag}'
        print(f'docker pull {base_image}')
    pinned_base_image = {}
    for base_tag in tags:
        base_image = f'{repo}/{image}:{base_tag}'
        pinned_image = f'pinned/{image}:{base_tag}'
        pinned_base_image[base_image] = pinned_image
        print(f'docker tag {base_image} {pinned_image}')
    for base_tag in tags:
        base_image = f'{repo}/{image}:{base_tag}'
        pinned_image = pinned_base_image[base_image]
        print(f'docker build --build-arg BASE_IMAGE={pinned_image} -t {base_image} .')
        for subtag in tags[base_tag]:
            if subtag == base_tag:
                continue
            target_image = f'{repo}/{image}:{subtag}'
            print(f'docker tag {base_image} {target_image}')
    print(f'docker push --all-tags {repo}/{image}')
if __name__ == '__main__':
    fix_image = os.environ.get('FIX_IMAGE')
    if not fix_image:
        main()
    else:
        _fix_docker_images(fix_image, os.environ.get('FIX_VERSION'))