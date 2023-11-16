"""
Represents Lambda Build Containers.
"""
import json
import logging
import os
import pathlib
from typing import List
from uuid import uuid4
from samcli.commands._utils.experimental import get_enabled_experimental_flags
from samcli.lib.utils.lambda_builders import patch_runtime
from samcli.local.docker.container import Container
LOG = logging.getLogger(__name__)

class LambdaBuildContainer(Container):
    """
    Class to manage Build containers that are capable of building AWS Lambda functions.
    This container mounts necessary folders, issues a command to the Lambda Builder CLI,
    and if the build was successful, copies back artifacts to the host filesystem
    """
    _IMAGE_URI_PREFIX = 'public.ecr.aws/sam/build'
    _IMAGE_TAG = 'latest'
    _BUILDERS_EXECUTABLE = 'lambda-builders'

    def __init__(self, protocol_version, language, dependency_manager, application_framework, source_dir, manifest_path, runtime, architecture, specified_workflow=None, optimizations=None, options=None, executable_search_paths=None, log_level=None, mode=None, env_vars=None, image=None, is_building_layer=False, build_in_source=None, mount_with_write: bool=False, build_dir=None):
        if False:
            return 10
        abs_manifest_path = pathlib.Path(manifest_path).resolve()
        manifest_file_name = abs_manifest_path.name
        manifest_dir = str(abs_manifest_path.parent)
        source_dir = str(pathlib.Path(source_dir).resolve())
        container_dirs = LambdaBuildContainer.get_container_dirs(source_dir, manifest_dir)
        env_vars = env_vars if env_vars else {}
        executable_search_paths = LambdaBuildContainer._convert_to_container_dirs(host_paths_to_convert=executable_search_paths, host_to_container_path_mapping={source_dir: container_dirs['source_dir'], manifest_dir: container_dirs['manifest_dir']})
        request_json = self._make_request(protocol_version, language, dependency_manager, application_framework, container_dirs, manifest_file_name, runtime, optimizations, options, executable_search_paths, mode, architecture, is_building_layer, build_in_source)
        if image is None:
            runtime_to_get_image = specified_workflow if specified_workflow else runtime
            image = LambdaBuildContainer._get_image(runtime_to_get_image, architecture)
        entry = LambdaBuildContainer._get_entrypoint(request_json)
        cmd: List[str] = []
        mount_mode = 'rw' if mount_with_write else 'ro'
        additional_volumes = {manifest_dir: {'bind': container_dirs['manifest_dir'], 'mode': mount_mode}}
        host_tmp_dir = None
        if mount_with_write and build_dir:
            host_tmp_dir = os.path.join(build_dir, f'tmp-{uuid4().hex}')
            additional_volumes.update({host_tmp_dir: {'bind': container_dirs['base_dir'], 'mode': mount_mode}})
        if log_level:
            env_vars['LAMBDA_BUILDERS_LOG_LEVEL'] = log_level
        super().__init__(image, cmd, container_dirs['source_dir'], source_dir, additional_volumes=additional_volumes, entrypoint=entry, env_vars=env_vars, mount_with_write=mount_with_write, host_tmp_dir=host_tmp_dir)

    @property
    def executable_name(self):
        if False:
            print('Hello World!')
        return LambdaBuildContainer._BUILDERS_EXECUTABLE

    @staticmethod
    def _make_request(protocol_version, language, dependency_manager, application_framework, container_dirs, manifest_file_name, runtime, optimizations, options, executable_search_paths, mode, architecture, is_building_layer, build_in_source):
        if False:
            for i in range(10):
                print('nop')
        runtime = patch_runtime(runtime)
        return json.dumps({'jsonschema': '2.0', 'id': 1, 'method': 'LambdaBuilder.build', 'params': {'__protocol_version': protocol_version, 'capability': {'language': language, 'dependency_manager': dependency_manager, 'application_framework': application_framework}, 'source_dir': container_dirs['source_dir'], 'artifacts_dir': container_dirs['artifacts_dir'], 'scratch_dir': container_dirs['scratch_dir'], 'manifest_path': '{}/{}'.format(container_dirs['manifest_dir'], manifest_file_name), 'runtime': runtime, 'optimizations': optimizations, 'options': options, 'executable_search_paths': executable_search_paths, 'mode': mode, 'architecture': architecture, 'is_building_layer': is_building_layer, 'experimental_flags': get_enabled_experimental_flags(), 'build_in_source': build_in_source}})

    @staticmethod
    def _get_entrypoint(request_json):
        if False:
            while True:
                i = 10
        return [LambdaBuildContainer._BUILDERS_EXECUTABLE, request_json]

    @staticmethod
    def get_container_dirs(source_dir, manifest_dir):
        if False:
            i = 10
            return i + 15
        '\n        Provides paths to directories within the container that is required by the builder\n\n        Parameters\n        ----------\n        source_dir : str\n            Path to the function source code\n\n        manifest_dir : str\n            Path to the directory containing manifest\n\n        Returns\n        -------\n        dict\n            Contains paths to source, artifacts, scratch & manifest directories\n        '
        base = '/tmp/samcli'
        result = {'base_dir': base, 'source_dir': '{}/source'.format(base), 'artifacts_dir': '{}/artifacts'.format(base), 'scratch_dir': '{}/scratch'.format(base), 'manifest_dir': '{}/manifest'.format(base)}
        if pathlib.PurePath(source_dir) == pathlib.PurePath(manifest_dir):
            result['manifest_dir'] = result['source_dir']
        return result

    @staticmethod
    def _convert_to_container_dirs(host_paths_to_convert, host_to_container_path_mapping):
        if False:
            print('Hello World!')
        '\n        Use this method to convert a list of host paths to a list of equivalent paths within the container\n        where the given host path is mounted. This is necessary when SAM CLI needs to pass path information to\n        the Lambda Builder running within the container.\n\n        If a host path is not mounted within the container, then this method simply passes the path to the result\n        without any changes.\n\n        Ex:\n            [ "/home/foo", "/home/bar", "/home/not/mounted"]  => ["/tmp/source", "/tmp/manifest", "/home/not/mounted"]\n\n        Parameters\n        ----------\n        host_paths_to_convert : list\n            List of paths in host that needs to be converted\n\n        host_to_container_path_mapping : dict\n            Mapping of paths in host to the equivalent paths within the container\n\n        Returns\n        -------\n        list\n            Equivalent paths within the container\n        '
        if not host_paths_to_convert:
            return host_paths_to_convert
        mapping = {str(pathlib.Path(p).resolve()): v for (p, v) in host_to_container_path_mapping.items()}
        result = []
        for original_path in host_paths_to_convert:
            abspath = str(pathlib.Path(original_path).resolve())
            if abspath in mapping:
                result.append(mapping[abspath])
            else:
                result.append(original_path)
                LOG.debug("Cannot convert host path '%s' to its equivalent path within the container. Host path is not mounted within the container", abspath)
        return result

    @staticmethod
    def _get_image(runtime, architecture):
        if False:
            while True:
                i = 10
        "\n        Parameters\n        ----------\n        runtime : str\n            Name of the Lambda runtime\n        architecture : str\n            Architecture type either 'x86_64' or 'arm64\n\n        Returns\n        -------\n        str\n            valid image name\n        "
        return f'{LambdaBuildContainer._IMAGE_URI_PREFIX}-{runtime}:' + LambdaBuildContainer.get_image_tag(architecture)

    @staticmethod
    def get_image_tag(architecture):
        if False:
            while True:
                i = 10
        '\n        Returns the lambda build image tag for an architecture\n\n        Parameters\n        ----------\n        architecture : str\n            Architecture\n\n        Returns\n        -------\n        str\n            Image tag\n        '
        return f'{LambdaBuildContainer._IMAGE_TAG}-{architecture}'