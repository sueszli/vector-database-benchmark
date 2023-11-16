"""
Manages the set of application templates.
"""
import itertools
import json
import logging
import os
from enum import Enum
from pathlib import Path
from subprocess import STDOUT, CalledProcessError, check_output
from typing import Dict, Optional
import requests
from samcli.cli.global_config import GlobalConfig
from samcli.commands.exceptions import AppTemplateUpdateException, UserException
from samcli.commands.init.init_flow_helpers import _get_runtime_from_image
from samcli.lib.utils import configuration
from samcli.lib.utils.git_repo import CloneRepoException, CloneRepoUnstableStateException, GitRepo, ManifestNotFoundException
from samcli.lib.utils.packagetype import IMAGE
from samcli.local.common.runtime_template import RUNTIME_DEP_TEMPLATE_MAPPING, get_local_lambda_images_location, get_local_manifest_path, get_provided_runtime_from_custom_runtime, is_custom_runtime
LOG = logging.getLogger(__name__)
APP_TEMPLATES_REPO_COMMIT = configuration.get_app_template_repo_commit()
MANIFEST_URL = f'https://raw.githubusercontent.com/aws/aws-sam-cli-app-templates/{APP_TEMPLATES_REPO_COMMIT}/manifest-v2.json'
APP_TEMPLATES_REPO_URL = 'https://github.com/aws/aws-sam-cli-app-templates'
APP_TEMPLATES_REPO_NAME = 'aws-sam-cli-app-templates'
APP_TEMPLATES_REPO_NAME_WINDOWS = 'tmpl'

class Status(Enum):
    NOT_FOUND = 404

class InvalidInitTemplateError(UserException):
    pass

class InitTemplates:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._git_repo: GitRepo = GitRepo(url=APP_TEMPLATES_REPO_URL)
        self.manifest_file_name = 'manifest-v2.json'

    def location_from_app_template(self, package_type, runtime, base_image, dependency_manager, app_template):
        if False:
            for i in range(10):
                print('nop')
        options = self.init_options(package_type, runtime, base_image, dependency_manager)
        try:
            template = next((item for item in options if self._check_app_template(item, app_template)))
            if template.get('init_location') is not None:
                return template['init_location']
            if template.get('directory') is not None:
                return os.path.normpath(os.path.join(self._git_repo.local_path, template['directory']))
            raise InvalidInitTemplateError('Invalid template. This should not be possible, please raise an issue.')
        except StopIteration as ex:
            msg = "Can't find application template " + app_template + ' - check valid values in interactive init.'
            raise InvalidInitTemplateError(msg) from ex

    @staticmethod
    def _check_app_template(entry: Dict, app_template: str) -> bool:
        if False:
            while True:
                i = 10
        return bool(entry['appTemplate'] == app_template)

    def init_options(self, package_type, runtime, base_image, dependency_manager):
        if False:
            for i in range(10):
                print('nop')
        self.clone_templates_repo()
        if self._git_repo.local_path is None:
            return self._init_options_from_bundle(package_type, runtime, dependency_manager)
        return self._init_options_from_manifest(package_type, runtime, base_image, dependency_manager)

    def clone_templates_repo(self):
        if False:
            print('Hello World!')
        if not self._git_repo.clone_attempted:
            from platform import system
            shared_dir: Path = GlobalConfig().config_dir
            os_name = system().lower()
            cloned_folder_name = APP_TEMPLATES_REPO_NAME_WINDOWS if os_name == 'windows' else APP_TEMPLATES_REPO_NAME
            if not self._check_upsert_templates(shared_dir, cloned_folder_name):
                return
            try:
                self._git_repo.clone(clone_dir=shared_dir, clone_name=cloned_folder_name, replace_existing=True, commit=APP_TEMPLATES_REPO_COMMIT)
            except CloneRepoUnstableStateException as ex:
                raise AppTemplateUpdateException(str(ex)) from ex
            except (OSError, CloneRepoException):
                LOG.debug('Clone error, attempting to use an old clone from a previous run')
                expected_previous_clone_local_path: Path = shared_dir.joinpath(cloned_folder_name)
                if expected_previous_clone_local_path.exists():
                    self._git_repo.local_path = expected_previous_clone_local_path

    def _check_upsert_templates(self, shared_dir: Path, cloned_folder_name: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Check if the app templates repository should be cloned, or if cloning should be skipped.\n\n        Parameters\n        ----------\n        shared_dir: Path\n            Folder containing the aws-sam-cli shared data\n\n        cloned_folder_name: str\n            Name of the directory into which the app templates will be copied\n\n        Returns\n        -------\n        bool\n            True if the cache should be updated, False otherwise\n\n        '
        cache_dir = Path(shared_dir, cloned_folder_name)
        git_executable = self._git_repo.git_executable()
        command = [git_executable, 'rev-parse', '--verify', 'HEAD']
        try:
            existing_hash = check_output(command, cwd=cache_dir, stderr=STDOUT).decode('utf-8').strip()
        except CalledProcessError as ex:
            LOG.debug(f"Unable to check existing cache hash\n{ex.output.decode('utf-8')}")
            return True
        except (FileNotFoundError, NotADirectoryError):
            LOG.debug('Cache directory does not yet exist, creating one.')
            return True
        self._git_repo.local_path = cache_dir
        return not existing_hash == APP_TEMPLATES_REPO_COMMIT

    def _init_options_from_manifest(self, package_type, runtime, base_image, dependency_manager):
        if False:
            print('Hello World!')
        manifest_path = self.get_manifest_path()
        with open(str(manifest_path)) as fp:
            body = fp.read()
            manifest_body = json.loads(body)
            templates = None
            if base_image:
                templates = manifest_body.get(base_image)
            elif runtime:
                templates = manifest_body.get(runtime)
            if templates is None:
                return self._init_options_from_bundle(package_type, runtime, dependency_manager)
            if dependency_manager is not None:
                templates_by_dep = filter(lambda x: x['dependencyManager'] == dependency_manager, list(templates))
                return list(templates_by_dep)
            return list(templates)

    @staticmethod
    def _init_options_from_bundle(package_type, runtime, dependency_manager):
        if False:
            while True:
                i = 10
        for mapping in list(itertools.chain(*RUNTIME_DEP_TEMPLATE_MAPPING.values())):
            if runtime in mapping['runtimes'] or any([r.startswith(runtime) for r in mapping['runtimes']]):
                if not dependency_manager or dependency_manager == mapping['dependency_manager']:
                    if package_type == IMAGE:
                        mapping['appTemplate'] = 'hello-world-lambda-image'
                        mapping['init_location'] = get_local_lambda_images_location(mapping, runtime)
                    else:
                        mapping['appTemplate'] = 'hello-world'
                    return [mapping]
        msg = 'Lambda Runtime {} and dependency manager {} does not have an available initialization template.'.format(runtime, dependency_manager)
        raise InvalidInitTemplateError(msg)

    def is_dynamic_schemas_template(self, package_type, app_template, runtime, base_image, dependency_manager):
        if False:
            return 10
        '\n        Check if provided template is dynamic template e.g: AWS Schemas template.\n        Currently dynamic templates require different handling e.g: for schema download & merge schema code in sam-app.\n        :param package_type:\n        :param app_template:\n        :param runtime:\n        :param base_image:\n        :param dependency_manager:\n        :return:\n        '
        options = self.init_options(package_type, runtime, base_image, dependency_manager)
        for option in options:
            if option.get('appTemplate') == app_template:
                return option.get('isDynamicTemplate', False)
        return False

    def get_app_template_location(self, template_directory):
        if False:
            i = 10
            return i + 15
        return os.path.normpath(os.path.join(self._git_repo.local_path, template_directory))

    def get_manifest_path(self):
        if False:
            i = 10
            return i + 15
        if self._git_repo.local_path and Path(self._git_repo.local_path, self.manifest_file_name).exists():
            return Path(self._git_repo.local_path, self.manifest_file_name)
        return get_local_manifest_path()

    def get_preprocessed_manifest(self, filter_value: Optional[str]=None, app_template: Optional[str]=None, package_type: Optional[str]=None, dependency_manager: Optional[str]=None) -> dict:
        if False:
            while True:
                i = 10
        '\n        This method get the manifest cloned from the git repo and preprocessed it.\n        Below is the link to manifest:\n        https://github.com/aws/aws-sam-cli-app-templates/blob/master/manifest-v2.json\n        The structure of the manifest is shown below:\n        {\n            "dotnet6": [\n                {\n                    "directory": "dotnet6/hello",\n                    "displayName": "Hello World Example",\n                    "dependencyManager": "cli-package",\n                    "appTemplate": "hello-world",\n                    "packageType": "Zip",\n                    "useCaseName": "Hello World Example"\n                },\n            ]\n        }\n        Parameters\n        ----------\n        filter_value : string, optional\n            This could be a runtime or a base-image, by default None\n        app_template : string, optional\n            Application template generated\n        package_type : string, optional\n            The package type, \'Zip\' or \'Image\', see samcli/lib/utils/packagetype.py\n        dependency_manager : string, optional\n            dependency manager\n        Returns\n        -------\n        [dict]\n            This is preprocessed manifest with the use_case as key\n        '
        manifest_body = self._get_manifest()
        preprocessed_manifest = {'Hello World Example': {}}
        for template_runtime in manifest_body:
            if not filter_value_matches_template_runtime(filter_value, template_runtime):
                LOG.debug('Template runtime %s does not match filter value %s', template_runtime, filter_value)
                continue
            template_list = manifest_body[template_runtime]
            for template in template_list:
                template_package_type = get_template_value('packageType', template)
                use_case_name = get_template_value('useCaseName', template)
                if not (template_package_type or use_case_name) or template_does_not_meet_filter_criteria(app_template, package_type, dependency_manager, template):
                    continue
                runtime = get_runtime(template_package_type, template_runtime)
                if runtime is None:
                    LOG.debug('Unable to infer runtime for template %s, %s', template_package_type, template_runtime)
                    continue
                use_case = preprocessed_manifest.get(use_case_name, {})
                use_case[runtime] = use_case.get(runtime, {})
                use_case[runtime][template_package_type] = use_case[runtime].get(template_package_type, [])
                use_case[runtime][template_package_type].append(template)
                preprocessed_manifest[use_case_name] = use_case
        if not bool(preprocessed_manifest['Hello World Example']):
            del preprocessed_manifest['Hello World Example']
        return preprocessed_manifest

    def _get_manifest(self):
        if False:
            print('Hello World!')
        "\n        In an attempt to reduce initial wait time to achieve an interactive\n        flow <= 10sec, This method first attempts to spools just the manifest file and\n        if the manifest can't be spooled, it attempts to clone the cli template git repo or\n        use local cli template\n        "
        try:
            response = requests.get(MANIFEST_URL, timeout=10)
            body = response.text
            if response.status_code == Status.NOT_FOUND.value:
                LOG.warning('Request to MANIFEST_URL: %s failed, the commit hash in this url maybe invalid, Using manifest.json in the latest commit instead.', MANIFEST_URL)
                raise ManifestNotFoundException()
        except (requests.Timeout, requests.ConnectionError, ManifestNotFoundException):
            LOG.debug('Request to get Manifest failed, attempting to clone the repository')
            self.clone_templates_repo()
            manifest_path = self.get_manifest_path()
            with open(str(manifest_path)) as fp:
                body = fp.read()
        manifest_body = json.loads(body)
        return manifest_body

def get_template_value(value: str, template: dict) -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    if value not in template:
        LOG.debug(f'Template is missing the value for {value} in manifest file. Please raise a github issue.' + f' Template details: {template}')
    return template.get(value)

def get_runtime(package_type: Optional[str], template_runtime: str) -> Optional[str]:
    if False:
        return 10
    if package_type == IMAGE:
        return _get_runtime_from_image(template_runtime)
    return template_runtime

def template_does_not_meet_filter_criteria(app_template: Optional[str], package_type: Optional[str], dependency_manager: Optional[str], template: dict) -> bool:
    if False:
        return 10
    "\n    Parameters\n    ----------\n    app_template : Optional[str]\n        Application template generated\n    package_type : Optional[str]\n        The package type, 'Zip' or 'Image', see samcli/lib/utils/packagetype.py\n    dependency_manager : Optional[str]\n        Dependency manager\n    template : dict\n        key-value pair app template configuration\n\n    Returns\n    -------\n    bool\n        True if template does not meet filter criteria else False\n    "
    return bool(app_template and app_template != template.get('appTemplate') or (package_type and package_type != template.get('packageType')) or (dependency_manager and dependency_manager != template.get('dependencyManager')))

def filter_value_matches_template_runtime(filter_value, template_runtime):
    if False:
        for i in range(10):
            print('nop')
    '\n    Validate if the filter value matches template runtimes from the manifest file\n\n    Parameters\n    ----------\n    filter_value : str\n        Lambda runtime used to filter through data generated from the manifest\n    template_runtime : str\n        Runtime of the template in view\n\n    Returns\n    -------\n    bool\n        True if there is a match else False\n    '
    if not filter_value:
        return True
    if is_custom_runtime(filter_value) and filter_value != get_provided_runtime_from_custom_runtime(template_runtime):
        return False
    if not is_custom_runtime(filter_value) and filter_value != template_runtime:
        return False
    return True