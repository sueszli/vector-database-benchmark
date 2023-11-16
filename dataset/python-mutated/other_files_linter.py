"""Lint checks of other file types."""
from __future__ import annotations
import glob
import json
import os
import re
from core import utils
from typing import Any, Dict, Final, List, Tuple, TypedDict
import yaml
from . import linter_utils
from .. import concurrent_task_utils
MYPY = False
if MYPY:
    from scripts.linters import pre_commit_linter

class ThirdPartyLibDict(TypedDict):
    """Type for the dictionary representation of elements of THIRD_PARTY_LIB."""
    name: str
    dependency_key: str
    dependency_source: str
    type_defs_filename_prefix: str
STRICT_TS_CONFIG_FILE_NAME: Final = 'tsconfig-strict.json'
STRICT_TS_CONFIG_FILEPATH: Final = os.path.join(os.getcwd(), STRICT_TS_CONFIG_FILE_NAME)
WEBPACK_CONFIG_FILE_NAME: Final = 'webpack.common.config.ts'
WEBPACK_CONFIG_FILEPATH: Final = os.path.join(os.getcwd(), WEBPACK_CONFIG_FILE_NAME)
APP_YAML_FILEPATH: Final = os.path.join(os.getcwd(), 'app_dev.yaml')
DEPENDENCIES_JSON_FILE_PATH: Final = os.path.join(os.getcwd(), 'dependencies.json')
PACKAGE_JSON_FILE_PATH: Final = os.path.join(os.getcwd(), 'package.json')
_TYPE_DEFS_FILE_EXTENSION_LENGTH: Final = len('.d.ts')
_DEPENDENCY_SOURCE_DEPENDENCIES_JSON: Final = 'dependencies.json'
_DEPENDENCY_SOURCE_PACKAGE: Final = 'package.json'
WORKFLOWS_DIR: Final = os.path.join(os.getcwd(), '.github', 'workflows')
WORKFLOW_FILENAME_REGEX: Final = '\\.(yaml)|(yml)$'
GIT_COMMIT_HASH_REGEX: Final = '^git\\+https:\\/\\/github\\.com\\/.*#(.*)$'
MERGE_STEP: Final = {'uses': './.github/actions/merge'}
WORKFLOWS_EXEMPT_FROM_MERGE_REQUIREMENT: Final = ('backend_tests.yml', 'develop_commit_notification.yml', 'pending-review-notification.yml', 'revert-web-wiki-updates.yml', 'frontend_tests.yml')
THIRD_PARTY_LIBS: List[ThirdPartyLibDict] = [{'name': 'Guppy', 'dependency_key': 'guppy-dev', 'dependency_source': _DEPENDENCY_SOURCE_PACKAGE, 'type_defs_filename_prefix': 'guppy-defs-'}, {'name': 'Skulpt', 'dependency_key': 'skulpt-dist', 'dependency_source': _DEPENDENCY_SOURCE_PACKAGE, 'type_defs_filename_prefix': 'skulpt-defs-'}, {'name': 'MIDI', 'dependency_key': 'midi', 'dependency_source': _DEPENDENCY_SOURCE_PACKAGE, 'type_defs_filename_prefix': 'midi-defs-'}, {'name': 'Nerdamer', 'dependency_key': 'nerdamer', 'dependency_source': _DEPENDENCY_SOURCE_PACKAGE, 'type_defs_filename_prefix': 'nerdamer-defs-'}]

class CustomLintChecksManager(linter_utils.BaseLinter):
    """Manages other files lint checks."""

    def __init__(self, file_cache: pre_commit_linter.FileCache) -> None:
        if False:
            while True:
                i = 10
        'Constructs a CustomLintChecksManager object.\n\n        Args:\n            file_cache: FileCache. Provides thread-safe access to cached\n                file content.\n        '
        self.file_cache = file_cache

    def check_skip_files_in_app_dev_yaml(self) -> concurrent_task_utils.TaskResult:
        if False:
            while True:
                i = 10
        'Check to ensure that all lines in skip_files in app_dev.yaml\n        reference valid files in the repository.\n        '
        name = 'App dev file'
        failed = False
        error_messages = []
        skip_files_section_found = False
        for (line_num, line) in enumerate(self.file_cache.readlines(APP_YAML_FILEPATH)):
            stripped_line = line.strip()
            if '# Third party files:' in stripped_line:
                skip_files_section_found = True
            if not skip_files_section_found:
                continue
            if not stripped_line or stripped_line[0] == '#':
                continue
            line_in_concern = stripped_line[len('- '):]
            if line_in_concern.endswith('/'):
                line_in_concern = line_in_concern[:-1]
            if not glob.glob(line_in_concern):
                error_message = "%s --> Pattern on line %s doesn't match any file or directory" % (APP_YAML_FILEPATH, line_num + 1)
                error_messages.append(error_message)
                failed = True
        return concurrent_task_utils.TaskResult(name, failed, error_messages, error_messages)

    def check_third_party_libs_type_defs(self) -> concurrent_task_utils.TaskResult:
        if False:
            return 10
        'Checks the type definitions for third party libs\n        are up to date.\n\n        Returns:\n            TaskResult. A TaskResult object representing the result of the lint\n            check.\n        '
        name = 'Third party type defs'
        failed = False
        error_messages = []
        package = json.load(utils.open_file(PACKAGE_JSON_FILE_PATH, 'r'))['dependencies']
        files_in_typings_dir = os.listdir(os.path.join(os.getcwd(), 'typings'))
        for third_party_lib in THIRD_PARTY_LIBS:
            lib_dependency_source = third_party_lib['dependency_source']
            if lib_dependency_source == _DEPENDENCY_SOURCE_PACKAGE:
                lib_version = package[third_party_lib['dependency_key']]
                if lib_version[0] == '^':
                    lib_version = lib_version[1:]
                elif re.search(GIT_COMMIT_HASH_REGEX, lib_version):
                    match = re.search(GIT_COMMIT_HASH_REGEX, lib_version)
                    if match:
                        lib_version = match.group(1)
            prefix_name = third_party_lib['type_defs_filename_prefix']
            files_with_prefix_name = []
            files_with_prefix_name = [file_name for file_name in files_in_typings_dir if file_name.startswith(prefix_name)]
            if len(files_with_prefix_name) > 1:
                error_message = 'There are multiple type definitions for %s in the typings dir.' % third_party_lib['name']
                error_messages.append(error_message)
                failed = True
            elif len(files_with_prefix_name) == 0:
                error_message = 'There are no type definitions for %s in the typings dir.' % third_party_lib['name']
                error_messages.append(error_message)
                failed = True
            else:
                type_defs_filename = files_with_prefix_name[0]
                type_defs_version = type_defs_filename[len(prefix_name):-_TYPE_DEFS_FILE_EXTENSION_LENGTH]
                if lib_version != type_defs_version:
                    error_message = 'Type definitions for %s are not up to date. The current version of %s is %s and the type definitions are for version %s. Please refer typings/README.md for more details.' % (third_party_lib['name'], third_party_lib['name'], lib_version, type_defs_version)
                    error_messages.append(error_message)
                    failed = True
        return concurrent_task_utils.TaskResult(name, failed, error_messages, error_messages)

    def check_webpack_config_file(self) -> concurrent_task_utils.TaskResult:
        if False:
            for i in range(10):
                print('nop')
        'Check to ensure that the instances of HtmlWebpackPlugin in\n        webpack.common.config.ts contains all needed keys.\n\n        Returns:\n            TaskResult. A TaskResult object representing the result of the lint\n            check.\n        '
        name = 'Webpack config file'
        failed = False
        error_messages = []
        plugins_section_found = False
        htmlwebpackplugin_section_found = False
        for (line_num, line) in enumerate(self.file_cache.readlines(WEBPACK_CONFIG_FILEPATH)):
            stripped_line = line.strip()
            if stripped_line.startswith('plugins:'):
                plugins_section_found = True
            if not plugins_section_found:
                continue
            if stripped_line.startswith('new HtmlWebpackPlugin('):
                error_line_num = line_num
                htmlwebpackplugin_section_found = True
                keys = ['chunks', 'filename', 'meta', 'template', 'minify', 'inject']
            elif htmlwebpackplugin_section_found and stripped_line.startswith('}),'):
                htmlwebpackplugin_section_found = False
                if keys:
                    error_message = 'Line %s: The following keys: %s are missing in HtmlWebpackPlugin block in %s' % (error_line_num + 1, ', '.join(keys), WEBPACK_CONFIG_FILE_NAME)
                    error_messages.append(error_message)
                    failed = True
            if htmlwebpackplugin_section_found:
                key = stripped_line.split(':')[0]
                if key in keys:
                    keys.remove(key)
        return concurrent_task_utils.TaskResult(name, failed, error_messages, error_messages)

    def check_github_workflows_use_merge_action(self) -> concurrent_task_utils.TaskResult:
        if False:
            for i in range(10):
                print('nop')
        'Checks that all github actions workflows use the merge action.\n\n        Returns:\n            TaskResult. A TaskResult object describing any workflows\n            that failed to use the merge action.\n        '
        name = 'Github workflows use merge action'
        workflow_paths = {os.path.join(WORKFLOWS_DIR, filename) for filename in os.listdir(WORKFLOWS_DIR) if re.search(WORKFLOW_FILENAME_REGEX, filename) if filename not in WORKFLOWS_EXEMPT_FROM_MERGE_REQUIREMENT}
        errors = []
        for workflow_path in workflow_paths:
            workflow_str = self.file_cache.read(workflow_path)
            workflow_dict = yaml.load(workflow_str, Loader=yaml.Loader)
            errors += self._check_that_workflow_steps_use_merge_action(workflow_dict, workflow_path)
        return concurrent_task_utils.TaskResult(name, bool(errors), errors, errors)

    @staticmethod
    def _check_that_workflow_steps_use_merge_action(workflow_dict: Dict[str, Any], workflow_path: str) -> List[str]:
        if False:
            while True:
                i = 10
        'Check that a workflow uses the merge action.\n\n        Args:\n            workflow_dict: dict. Dictionary representation of the\n                workflow YAML file.\n            workflow_path: str. Path to workflow file.\n\n        Returns:\n            list(str). A list of error messages describing any jobs\n            failing to use the merge action.\n        '
        jobs_without_merge = []
        for (job, job_dict) in workflow_dict['jobs'].items():
            if MERGE_STEP not in job_dict['steps']:
                jobs_without_merge.append(job)
        return ['%s --> Job %s does not use the .github/actions/merge action.' % (workflow_path, job) for job in jobs_without_merge]

    def perform_all_lint_checks(self) -> List[concurrent_task_utils.TaskResult]:
        if False:
            while True:
                i = 10
        'Perform all the lint checks and returns the messages returned by all\n        the checks.\n\n        Returns:\n            list(TaskResult). A list of TaskResult objects representing the\n            results of the lint checks.\n        '
        linter_stdout = []
        linter_stdout.append(self.check_skip_files_in_app_dev_yaml())
        linter_stdout.append(self.check_third_party_libs_type_defs())
        linter_stdout.append(self.check_webpack_config_file())
        linter_stdout.append(self.check_github_workflows_use_merge_action())
        return linter_stdout

def get_linters(file_cache: pre_commit_linter.FileCache) -> Tuple[CustomLintChecksManager, None]:
    if False:
        print('Hello World!')
    'Creates CustomLintChecksManager and returns it.\n\n    Args:\n        file_cache: object(FileCache). Provides thread-safe access to cached\n            file content.\n\n    Returns:\n        tuple(CustomLintChecksManager, None). A 2-tuple of custom and\n        third_party linter objects.\n    '
    custom_linter = CustomLintChecksManager(file_cache)
    return (custom_linter, None)