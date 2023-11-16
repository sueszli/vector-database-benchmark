import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import requests
from celery.contrib.abortable import AbortableAsyncResult
from _orchest.internals import config as _config
from _orchest.internals.utils import copytree, rmtree
from app import models
from app import utils as app_utils
from app.connections import k8s_core_api
from app.core import image_utils
from app.core.sio_streamed_task import SioStreamedTask
from config import CONFIG_CLASS
__ENV_BUILD_FULL_LOGS_DIRECTORY = '/tmp/environment_image_builds_logs'
_logger = app_utils.get_logger()

def update_environment_image_build_status(session: requests.sessions.Session, project_uuid: str, environment_uuid: str, image_tag: str, status: str, cluster_node: Optional[str]=None) -> Any:
    if False:
        print('Hello World!')
    'Update environment build status.'
    data = {'status': status}
    if cluster_node is not None:
        data['cluster_node'] = cluster_node
    if data['status'] == 'STARTED':
        data['started_time'] = datetime.utcnow().isoformat()
    elif data['status'] in ['SUCCESS', 'FAILURE']:
        data['finished_time'] = datetime.utcnow().isoformat()
    url = f'{CONFIG_CLASS.ORCHEST_API_ADDRESS}/environment-builds/{project_uuid}/{environment_uuid}/{image_tag}'
    with session.put(url, json=data) as response:
        return response.json()

def write_environment_dockerfile(base_image, task_uuid, project_uuid, env_uuid, work_dir, bash_script, path):
    if False:
        while True:
            i = 10
    "Write a custom dockerfile with the given specifications.\n\n    ! The dockerfile is written in a way that the layer where the user\n    setup script is run is effectively cached when possible, i.e.  we\n    don't disrupt the caching capability by using task dependent\n    information like the task_uuid in that layer. We make use of the\n    task_uuid in a layer that is created at the end so that each image\n    has a unique digest, which helps reducing complexity when it comes\n    to deleting images from the registry.\n\n    This dockerfile is built in an ad-hoc way to later be able to only\n    log messages related to the user script. Note that the produced\n    dockerfile will make it so that the entire context is copied.\n\n    Args:\n        base_image: Base image of the docker file.\n        task_uuid: Used to create a layer that is unique for this\n            particular image, this way the registry digest of the image\n            will be unique.\n        project_uuid:\n        env_uuid:\n        work_dir: Working directory.\n        bash_script: Script to run in a RUN command.\n        path: Where to save the file.\n\n    Returns:\n\n    "
    statements = []
    custom_registry_prefix = 'registry:'
    if base_image.startswith(custom_registry_prefix):
        full_basename = base_image[len(custom_registry_prefix):]
    else:
        full_basename = f'docker.io/{base_image}'
    statements.append(f'FROM {full_basename}')
    statements.append(f'LABEL _orchest_project_uuid={project_uuid}')
    statements.append(f'LABEL _orchest_environment_uuid={env_uuid}')
    statements.append(f"WORKDIR {os.path.join('/', work_dir)}")
    statements.append('COPY . .')
    ps = ['chown -R :$(id -g) . > /dev/null 2>&1 ', "find . -type d -not -perm -g+rwxs -exec chmod g+rwxs '{}' + > /dev/null 2>&1 ", "find . -type f -not -perm -g+rwx -exec chmod g+rwx '{}' + > /dev/null 2>&1 ", 'chmod g+rwx . > /dev/null 2>&1 ']
    sps = ['sudo ' + s for s in ps]
    ps = ' && '.join(ps)
    sps = ' && '.join(sps)
    ps_fail_msg = 'The base image must have USER root or "sudo" must be installed, "find" must also be installed.'
    rm_statement = f'&& (if [ $(id -u) = 0 ]; then rm {bash_script}; else sudo rm {bash_script}; fi)'
    flag = CONFIG_CLASS.BUILD_IMAGE_LOG_FLAG
    error_flag = CONFIG_CLASS.BUILD_IMAGE_ERROR_FLAG
    statements.append(f'RUN ((if [ $(id -u) = 0 ]; then {ps}; else {sps}; fi) || ! echo "{ps_fail_msg}") && bash < {bash_script} && echo {flag} {rm_statement} || (echo {error_flag} && PRODUCE_AN_ERROR)')
    write_task_uuid = f"{{sudo}} mkdir -p /orchest && echo '{task_uuid}' | {{sudo}} tee /orchest/task_{task_uuid}.txt"
    non_sudo_write_task_uuid = write_task_uuid.format(sudo='')
    sudo_write_task_uuid = write_task_uuid.format(sudo='sudo')
    write_task_uuid_fail_msg = 'The base image must have USER root or "sudo" must be installed, "tee" must also be installed.'
    statements.append(f"RUN ((if [ $(id -u) = 0 ]; then {non_sudo_write_task_uuid}; else {sudo_write_task_uuid}; fi) || ! echo '{write_task_uuid_fail_msg}') ")
    statements = '\n'.join(statements)
    with open(path, 'w') as dockerfile:
        dockerfile.write(statements)

def check_environment_correctness(project_uuid, environment_uuid, project_path):
    if False:
        while True:
            i = 10
    'A series of sanity checks that needs to be passed.\n\n    Args:\n        project_uuid:\n        environment_uuid:\n        project_path:\n\n    Returns:\n\n    Raises:\n        OSError if the project path is missing, if the environment\n            within the project cannot be found, if the environment\n            properties.json cannot be found or if the user bash script\n            cannot be found.\n        ValueError if project_uuid, environment_uuid, base_image are\n            incorrect or missing.\n\n    '
    if not os.path.exists(project_path):
        raise OSError(f'Project path {project_path} does not exist')
    environment_path = os.path.join(project_path, f'.orchest/environments/{environment_uuid}')
    if not os.path.exists(environment_path):
        raise OSError(f'Environment path {environment_path} does not exist')
    environment_properties = os.path.join(environment_path, 'properties.json')
    if not os.path.isfile(environment_properties):
        raise OSError('Environment properties file (properties.json) not found')
    environment_user_script = os.path.join(environment_path, _config.ENV_SETUP_SCRIPT_FILE_NAME)
    if not os.path.isfile(environment_user_script):
        raise OSError(f'Environment user script ({_config.ENV_SETUP_SCRIPT_FILE_NAME}) not found')
    with open(environment_properties) as json_file:
        environment_properties = json.load(json_file)
        if 'base_image' not in environment_properties:
            raise ValueError('base_image not found in environment properties.json')
        if 'uuid' not in environment_properties:
            raise ValueError('uuid not found in environment properties.json')
        if environment_properties['uuid'] != environment_uuid:
            raise ValueError(f"The environment properties environment uuid {environment_properties['uuid']} differs {environment_uuid}")

def prepare_build_context(task_uuid, project_uuid, environment_uuid, project_path):
    if False:
        while True:
            i = 10
    'Prepares the build context for a given environment.\n\n    Prepares the build context by taking a snapshot of the project\n    directory, and using this snapshot as a context in which the ad-hoc\n    docker file will be placed. This dockerfile is built in a way to\n    respect the environment properties (base image, user bash script,\n    etc.) while also allowing to log only the messages that are related\n    to the user script while building the image.\n\n    Args:\n        task_uuid:\n        project_uuid:\n        environment_uuid:\n        project_path:\n\n    Returns:\n        Dictionary containing build context details.\n\n    Raises:\n        See the check_environment_correctness_function\n    '
    env_builds_dir = _config.USERDIR_ENV_IMG_BUILDS
    Path(env_builds_dir).mkdir(parents=True, exist_ok=True)
    snapshot_path = f'{env_builds_dir}/{task_uuid}'
    if os.path.isdir(snapshot_path):
        rmtree(snapshot_path)
    try:
        userdir_project_path = os.path.join(_config.USERDIR_PROJECTS, project_path)
        copytree(userdir_project_path, snapshot_path, use_gitignore=True)
    except OSError as e:
        _logger.error(e)
        proj = models.Project.query.filter_by(uuid=project_uuid).one()
        userdir_project_path = os.path.join(_config.USERDIR_PROJECTS, proj.name)
        copytree(userdir_project_path, snapshot_path, use_gitignore=True)
    check_environment_correctness(project_uuid, environment_uuid, snapshot_path)
    environment_path = os.path.join(snapshot_path, f'.orchest/environments/{environment_uuid}')
    with open(os.path.join(environment_path, 'properties.json')) as json_file:
        environment_properties = json.load(json_file)
        base_image: str = environment_properties['base_image']
        if 'orchest/' in base_image:
            if ':' not in base_image.split('orchest/')[1]:
                base_image = f'{base_image}:{CONFIG_CLASS.ORCHEST_VERSION}'
    bash_script_name = f'.orchest-reserved-env-setup-script-{project_uuid}-{environment_uuid}.sh'
    snapshot_setup_script_path = os.path.join(snapshot_path, bash_script_name)
    os.system('cp "%s" "%s"' % (os.path.join(environment_path, _config.ENV_SETUP_SCRIPT_FILE_NAME), snapshot_setup_script_path))
    dockerfile_name = f'.orchest-reserved-env-dockerfile-{project_uuid}-{environment_uuid}'
    write_environment_dockerfile(base_image, task_uuid, project_uuid, environment_uuid, _config.PROJECT_DIR, bash_script_name, os.path.join(snapshot_path, dockerfile_name))
    with open(os.path.join(snapshot_path, '.dockerignore'), 'w') as docker_ignore:
        docker_ignore.write('.dockerignore\n')
        docker_ignore.write('.orchest\n')
        docker_ignore.write(f'{dockerfile_name}\n')
    return {'snapshot_path': snapshot_path, 'base_image': base_image, 'dockerfile_path': dockerfile_name}

def build_environment_image_task(task_uuid: str, project_uuid: str, environment_uuid: str, image_tag: str, project_path: str):
    if False:
        for i in range(10):
            print('nop')
    'Function called by the celery task to build an environment.\n\n    Builds an environment (image) given the arguments, the logs produced\n    by the user provided script are forwarded to a SocketIO server and\n    namespace defined in the orchest internals config.\n\n    Args:\n        task_uuid:\n        project_uuid:\n        environment_uuid:\n        image_tag:\n        project_path:\n\n    Returns:\n\n    '
    with requests.sessions.Session() as session:
        try:
            update_environment_image_build_status(session, project_uuid, environment_uuid, image_tag, 'STARTED')
            build_context = prepare_build_context(task_uuid, project_uuid, environment_uuid, project_path)
            image_name = _config.ENVIRONMENT_IMAGE_NAME.format(project_uuid=project_uuid, environment_uuid=environment_uuid)
            if not os.path.exists(__ENV_BUILD_FULL_LOGS_DIRECTORY):
                os.mkdir(__ENV_BUILD_FULL_LOGS_DIRECTORY)
            complete_logs_path = os.path.join(__ENV_BUILD_FULL_LOGS_DIRECTORY, image_name)
            status = SioStreamedTask.run(task_lambda=lambda user_logs_fo: image_utils.build_image(task_uuid, image_name, image_tag, build_context, user_logs_fo, complete_logs_path), identity=f'{project_uuid}-{environment_uuid}', server=_config.ORCHEST_SOCKETIO_SERVER_ADDRESS, namespace=_config.ORCHEST_SOCKETIO_ENV_IMG_BUILDING_NAMESPACE, abort_lambda=lambda : AbortableAsyncResult(task_uuid).is_aborted())
            rmtree(build_context['snapshot_path'])
            pod_name = image_utils.image_build_task_to_pod_name(task_uuid)
            pod = k8s_core_api.read_namespaced_pod(name=pod_name, namespace=_config.ORCHEST_NAMESPACE)
            update_environment_image_build_status(session, project_uuid, environment_uuid, image_tag, status, pod.spec.node_name)
        except Exception as e:
            _logger.error(e)
            update_environment_image_build_status(session, project_uuid, environment_uuid, image_tag, 'FAILURE')
            raise e
        finally:
            k8s_core_api.delete_namespaced_pod(image_utils.image_build_task_to_pod_name(task_uuid), _config.ORCHEST_NAMESPACE)
    return 'SUCCESS'