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

__ENV_BUILD_FULL_LOGS_DIRECTORY = "/tmp/environment_image_builds_logs"

_logger = app_utils.get_logger()


def update_environment_image_build_status(
    session: requests.sessions.Session,
    project_uuid: str,
    environment_uuid: str,
    image_tag: str,
    status: str,
    cluster_node: Optional[str] = None,
) -> Any:
    """Update environment build status."""
    data = {"status": status}
    if cluster_node is not None:
        data["cluster_node"] = cluster_node

    if data["status"] == "STARTED":
        data["started_time"] = datetime.utcnow().isoformat()
    elif data["status"] in ["SUCCESS", "FAILURE"]:
        data["finished_time"] = datetime.utcnow().isoformat()

    url = (
        f"{CONFIG_CLASS.ORCHEST_API_ADDRESS}/environment-builds/"
        f"{project_uuid}/{environment_uuid}/{image_tag}"
    )

    with session.put(url, json=data) as response:
        return response.json()


def write_environment_dockerfile(
    base_image, task_uuid, project_uuid, env_uuid, work_dir, bash_script, path
):
    """Write a custom dockerfile with the given specifications.

    ! The dockerfile is written in a way that the layer where the user
    setup script is run is effectively cached when possible, i.e.  we
    don't disrupt the caching capability by using task dependent
    information like the task_uuid in that layer. We make use of the
    task_uuid in a layer that is created at the end so that each image
    has a unique digest, which helps reducing complexity when it comes
    to deleting images from the registry.

    This dockerfile is built in an ad-hoc way to later be able to only
    log messages related to the user script. Note that the produced
    dockerfile will make it so that the entire context is copied.

    Args:
        base_image: Base image of the docker file.
        task_uuid: Used to create a layer that is unique for this
            particular image, this way the registry digest of the image
            will be unique.
        project_uuid:
        env_uuid:
        work_dir: Working directory.
        bash_script: Script to run in a RUN command.
        path: Where to save the file.

    Returns:

    """
    statements = []

    # Users can override using the docker.io repo by
    # prefixing the base_image with 'registry:'
    # E.g. 'python' (assumed docker.io)
    # E.g. 'registry:quay.io/python/python:3.9' (uses quay.io registry)
    custom_registry_prefix = "registry:"
    if base_image.startswith(custom_registry_prefix):
        full_basename = base_image[len(custom_registry_prefix) :]
    else:
        full_basename = f"docker.io/{base_image}"

    statements.append(f"FROM {full_basename}")
    statements.append(f"LABEL _orchest_project_uuid={project_uuid}")
    statements.append(f"LABEL _orchest_environment_uuid={env_uuid}")
    statements.append(f'WORKDIR {os.path.join("/", work_dir)}')

    # Copy the entire context, that is, given the current use case, that
    # we are copying the project directory (from the snapshot) into the
    # image that is to be built, this allows the user defined script
    # defined through orchest to make use of files that are part of its
    # project, e.g. a requirements.txt or other scripts.
    statements.append("COPY . .")

    # Permission statements.
    ps = [
        "chown -R :$(id -g) . > /dev/null 2>&1 ",
        "find . -type d -not -perm -g+rwxs -exec chmod g+rwxs '{}' + > /dev/null 2>&1 ",
        "find . -type f -not -perm -g+rwx -exec chmod g+rwx '{}' + > /dev/null 2>&1 ",
        "chmod g+rwx . > /dev/null 2>&1 ",
    ]
    # sudo'd permission statements.
    sps = ["sudo " + s for s in ps]
    ps = " && ".join(ps)
    sps = " && ".join(sps)
    ps_fail_msg = (
        'The base image must have USER root or "sudo" must be installed, "find" '
        "must also be installed."
    )

    # Note: commands are concatenated with && because this way an
    # exit_code != 0 will bubble up and cause the build to fail, as it
    # should. The bash script is removed so that the user won't be able
    # to see it after the build is done.
    rm_statement = (
        f"&& (if [ $(id -u) = 0 ]; then rm {bash_script}; else "
        f"sudo rm {bash_script}; fi)"
    )

    flag = CONFIG_CLASS.BUILD_IMAGE_LOG_FLAG
    error_flag = CONFIG_CLASS.BUILD_IMAGE_ERROR_FLAG
    statements.append(
        # The ! in front of echo is there so that the script will fail
        # since the statements in the "if" have failed, the echo is a
        # way of injecting the help message.
        f"RUN ((if [ $(id -u) = 0 ]; then {ps}; else {sps}; fi) "
        f'|| ! echo "{ps_fail_msg}") '
        f"&& bash < {bash_script} "
        # Needed to inject the rm statement this way, black was
        # introducing an error.
        f"&& echo {flag} {rm_statement} "
        # The || <error flag> allows to avoid builder errors logs making
        # into it the user logs and tell us that there has been an
        # error.
        f"|| (echo {error_flag} && PRODUCE_AN_ERROR)"
    )

    # Make it so that the digest of the produced image is unique.
    write_task_uuid = (
        "{sudo} mkdir -p /orchest && "
        f"echo '{task_uuid}' | {{sudo}} tee /orchest/task_{task_uuid}.txt"
    )
    non_sudo_write_task_uuid = write_task_uuid.format(sudo="")
    sudo_write_task_uuid = write_task_uuid.format(sudo="sudo")
    write_task_uuid_fail_msg = (
        'The base image must have USER root or "sudo" must be installed, "tee" '
        "must also be installed."
    )
    statements.append(
        (
            f"RUN ((if [ $(id -u) = 0 ]; then {non_sudo_write_task_uuid}; else "
            f"{sudo_write_task_uuid}; fi) || ! echo '{write_task_uuid_fail_msg}') "
        )
    )
    statements = "\n".join(statements)

    with open(path, "w") as dockerfile:
        dockerfile.write(statements)


def check_environment_correctness(project_uuid, environment_uuid, project_path):
    """A series of sanity checks that needs to be passed.

    Args:
        project_uuid:
        environment_uuid:
        project_path:

    Returns:

    Raises:
        OSError if the project path is missing, if the environment
            within the project cannot be found, if the environment
            properties.json cannot be found or if the user bash script
            cannot be found.
        ValueError if project_uuid, environment_uuid, base_image are
            incorrect or missing.

    """
    if not os.path.exists(project_path):
        raise OSError(f"Project path {project_path} does not exist")

    environment_path = os.path.join(
        project_path, f".orchest/environments/{environment_uuid}"
    )
    if not os.path.exists(environment_path):
        raise OSError(f"Environment path {environment_path} does not exist")

    environment_properties = os.path.join(environment_path, "properties.json")
    if not os.path.isfile(environment_properties):
        raise OSError("Environment properties file (properties.json) not found")

    environment_user_script = os.path.join(
        environment_path, _config.ENV_SETUP_SCRIPT_FILE_NAME
    )
    if not os.path.isfile(environment_user_script):
        raise OSError(
            f"Environment user script ({_config.ENV_SETUP_SCRIPT_FILE_NAME}) not found"
        )

    with open(environment_properties) as json_file:
        environment_properties = json.load(json_file)

        if "base_image" not in environment_properties:
            raise ValueError("base_image not found in environment properties.json")

        if "uuid" not in environment_properties:
            raise ValueError("uuid not found in environment properties.json")

        if environment_properties["uuid"] != environment_uuid:
            raise ValueError(
                f"The environment properties environment "
                f"uuid {environment_properties['uuid']} differs {environment_uuid}"
            )


def prepare_build_context(task_uuid, project_uuid, environment_uuid, project_path):
    """Prepares the build context for a given environment.

    Prepares the build context by taking a snapshot of the project
    directory, and using this snapshot as a context in which the ad-hoc
    docker file will be placed. This dockerfile is built in a way to
    respect the environment properties (base image, user bash script,
    etc.) while also allowing to log only the messages that are related
    to the user script while building the image.

    Args:
        task_uuid:
        project_uuid:
        environment_uuid:
        project_path:

    Returns:
        Dictionary containing build context details.

    Raises:
        See the check_environment_correctness_function
    """

    env_builds_dir = _config.USERDIR_ENV_IMG_BUILDS
    # K8S_TODO: remove this?
    Path(env_builds_dir).mkdir(parents=True, exist_ok=True)
    # Make a snapshot of the project state, used for the context.
    snapshot_path = f"{env_builds_dir}/{task_uuid}"
    if os.path.isdir(snapshot_path):
        rmtree(snapshot_path)

    try:
        # The project path we receive is relative to the projects
        # directory.
        userdir_project_path = os.path.join(_config.USERDIR_PROJECTS, project_path)
        copytree(userdir_project_path, snapshot_path, use_gitignore=True)
    except OSError as e:
        # This is a temporary band-aid to the fact that, currently, a
        # project rename can happen while builds are queued. We use the
        # fact that the orchest-webserver updates (PUTs) the orchest-api
        # project name on a project rename and the fact that the project
        # name currently coincides with the project path w.r.t. the
        # projects directory.
        # Note: this solution only covers 1 rename, i.e. it only tries
        # once.
        _logger.error(e)
        proj = models.Project.query.filter_by(uuid=project_uuid).one()
        userdir_project_path = os.path.join(_config.USERDIR_PROJECTS, proj.name)
        copytree(userdir_project_path, snapshot_path, use_gitignore=True)

    # Sanity checks, if not respected exception will be raised.
    check_environment_correctness(project_uuid, environment_uuid, snapshot_path)

    # Take the environment from the snapshot.
    environment_path = os.path.join(
        snapshot_path, f".orchest/environments/{environment_uuid}"
    )

    # Build the docker file and move it to the context.
    with open(os.path.join(environment_path, "properties.json")) as json_file:
        environment_properties = json.load(json_file)
        base_image: str = environment_properties["base_image"]
        # Workaround for common.tsx not using the orchest version.
        if "orchest/" in base_image:
            if ":" not in base_image.split("orchest/")[1]:
                base_image = f"{base_image}:{CONFIG_CLASS.ORCHEST_VERSION}"

    # Use the task_uuid to avoid clashing with user stuff.
    bash_script_name = (
        f".orchest-reserved-env-setup-script-{project_uuid}-{environment_uuid}.sh"
    )
    snapshot_setup_script_path = os.path.join(snapshot_path, bash_script_name)
    # Move the startup script to the context.
    os.system(
        'cp "%s" "%s"'
        % (
            os.path.join(environment_path, _config.ENV_SETUP_SCRIPT_FILE_NAME),
            snapshot_setup_script_path,
        )
    )

    dockerfile_name = (
        f".orchest-reserved-env-dockerfile-{project_uuid}-{environment_uuid}"
    )

    write_environment_dockerfile(
        base_image,
        task_uuid,
        project_uuid,
        environment_uuid,
        _config.PROJECT_DIR,
        bash_script_name,
        os.path.join(snapshot_path, dockerfile_name),
    )

    # Hide stuff from the user.
    with open(os.path.join(snapshot_path, ".dockerignore"), "w") as docker_ignore:
        docker_ignore.write(".dockerignore\n")
        docker_ignore.write(".orchest\n")
        docker_ignore.write(f"{dockerfile_name}\n")

    return {
        "snapshot_path": snapshot_path,
        "base_image": base_image,
        "dockerfile_path": dockerfile_name,
    }


def build_environment_image_task(
    task_uuid: str,
    project_uuid: str,
    environment_uuid: str,
    image_tag: str,
    project_path: str,
):
    """Function called by the celery task to build an environment.

    Builds an environment (image) given the arguments, the logs produced
    by the user provided script are forwarded to a SocketIO server and
    namespace defined in the orchest internals config.

    Args:
        task_uuid:
        project_uuid:
        environment_uuid:
        image_tag:
        project_path:

    Returns:

    """
    with requests.sessions.Session() as session:

        try:
            update_environment_image_build_status(
                session, project_uuid, environment_uuid, image_tag, "STARTED"
            )

            # Prepare the project snapshot with the correctly placed
            # dockerfile, scripts, etc.
            build_context = prepare_build_context(
                task_uuid, project_uuid, environment_uuid, project_path
            )

            # Use the agreed upon pattern for the image name.
            image_name = _config.ENVIRONMENT_IMAGE_NAME.format(
                project_uuid=project_uuid, environment_uuid=environment_uuid
            )

            if not os.path.exists(__ENV_BUILD_FULL_LOGS_DIRECTORY):
                os.mkdir(__ENV_BUILD_FULL_LOGS_DIRECTORY)
            # place the logs in the celery container
            complete_logs_path = os.path.join(
                __ENV_BUILD_FULL_LOGS_DIRECTORY, image_name
            )

            status = SioStreamedTask.run(
                # What we are actually running/doing in this task,
                task_lambda=lambda user_logs_fo: image_utils.build_image(
                    task_uuid,
                    image_name,
                    image_tag,
                    build_context,
                    user_logs_fo,
                    complete_logs_path,
                ),
                identity=f"{project_uuid}-{environment_uuid}",
                server=_config.ORCHEST_SOCKETIO_SERVER_ADDRESS,
                namespace=_config.ORCHEST_SOCKETIO_ENV_IMG_BUILDING_NAMESPACE,
                # note: using task.is_aborted() could be an option but
                # it was giving some issues related to
                # multithreading/processing, moreover, also just passing
                # the task_uuid to this function is less information to
                # rely on, which is good.
                abort_lambda=lambda: AbortableAsyncResult(task_uuid).is_aborted(),
            )

            # Cleanup.
            rmtree(build_context["snapshot_path"])

            pod_name = image_utils.image_build_task_to_pod_name(task_uuid)
            pod = k8s_core_api.read_namespaced_pod(
                name=pod_name, namespace=_config.ORCHEST_NAMESPACE
            )
            update_environment_image_build_status(
                session,
                project_uuid,
                environment_uuid,
                image_tag,
                status,
                pod.spec.node_name,
            )

        # Catch all exceptions because we need to make sure to set the
        # build state to failed.
        except Exception as e:
            _logger.error(e)
            update_environment_image_build_status(
                session, project_uuid, environment_uuid, image_tag, "FAILURE"
            )
            raise e
        finally:
            # The task was successful or aborted, cleanup the pod.
            k8s_core_api.delete_namespaced_pod(
                image_utils.image_build_task_to_pod_name(task_uuid),
                _config.ORCHEST_NAMESPACE,
            )

    # The status of the Celery task is SUCCESS since it has finished
    # running. Not related to the actual state of the build, e.g.
    # FAILURE.
    return "SUCCESS"
