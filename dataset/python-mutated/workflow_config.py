"""
Contains Builder Workflow Configs for different Runtimes
"""
import os
import logging
from typing import Dict, List, Optional, Tuple, Union, cast
from samcli.lib.build.workflows import CONFIG, PYTHON_PIP_CONFIG, NODEJS_NPM_CONFIG, RUBY_BUNDLER_CONFIG, JAVA_GRADLE_CONFIG, JAVA_KOTLIN_GRADLE_CONFIG, JAVA_MAVEN_CONFIG, DOTNET_CLIPACKAGE_CONFIG, GO_MOD_CONFIG, PROVIDED_MAKE_CONFIG, NODEJS_NPM_ESBUILD_CONFIG, RUST_CARGO_LAMBDA_CONFIG
from samcli.lib.telemetry.event import EventTracker
LOG = logging.getLogger(__name__)

class UnsupportedRuntimeException(Exception):
    pass

class UnsupportedBuilderException(Exception):
    pass
WorkFlowSelector = Union['BasicWorkflowSelector', 'ManifestWorkflowSelector']

def get_selector(selector_list: List[Dict[str, WorkFlowSelector]], identifiers: List[Optional[str]], specified_workflow: Optional[str]=None) -> Optional[WorkFlowSelector]:
    if False:
        i = 10
        return i + 15
    '\n    Determine the correct workflow selector from a list of selectors,\n    series of identifiers and user specified workflow if defined.\n\n    Parameters\n    ----------\n    selector_list list\n        List of dictionaries, where the value of all dictionaries are workflow selectors.\n    identifiers list\n        List of identifiers specified in order of precedence that are to be looked up in selector_list.\n    specified_workflow str\n        User specified workflow for build.\n\n    Returns\n    -------\n    selector(BasicWorkflowSelector)\n        selector object which can specify a workflow configuration that can be passed to `aws-lambda-builders`\n\n    '
    all_selectors: Dict[str, WorkFlowSelector] = dict()
    for selector in selector_list:
        all_selectors = {**all_selectors, **selector}
    if specified_workflow and specified_workflow not in all_selectors:
        raise UnsupportedBuilderException("'{}' does not have a supported builder".format(specified_workflow))
    selectors = [all_selectors.get(identifier) for identifier in identifiers if identifier]
    try:
        return next((_selector for _selector in selectors if _selector))
    except StopIteration:
        pass
    return None

def get_layer_subfolder(build_workflow: str) -> str:
    if False:
        return 10
    subfolders_by_runtime = {'python3.7': 'python', 'python3.8': 'python', 'python3.9': 'python', 'python3.10': 'python', 'python3.11': 'python', 'nodejs4.3': 'nodejs', 'nodejs6.10': 'nodejs', 'nodejs8.10': 'nodejs', 'nodejs12.x': 'nodejs', 'nodejs14.x': 'nodejs', 'nodejs16.x': 'nodejs', 'nodejs18.x': 'nodejs', 'nodejs20.x': 'nodejs', 'ruby2.7': 'ruby/lib', 'ruby3.2': 'ruby/lib', 'java8': 'java', 'java11': 'java', 'java8.al2': 'java', 'java17': 'java', 'dotnet6': 'dotnet', 'makefile': ''}
    if build_workflow not in subfolders_by_runtime:
        raise UnsupportedRuntimeException("'{}' runtime is not supported for layers".format(build_workflow))
    return subfolders_by_runtime[build_workflow]

def get_workflow_config(runtime: Optional[str], code_dir: str, project_dir: str, specified_workflow: Optional[str]=None) -> CONFIG:
    if False:
        print('Hello World!')
    '\n    Get a workflow config that corresponds to the runtime provided. This method examines contents of the project\n    and code directories to determine the most appropriate workflow for the given runtime. Currently the decision is\n    based on the presence of a supported manifest file. For runtimes that have more than one workflow, we choose a\n    workflow by examining ``code_dir`` followed by ``project_dir`` for presence of a supported manifest.\n\n    Parameters\n    ----------\n    runtime str\n        The runtime of the config\n\n    code_dir str\n        Directory where Lambda function code is present\n\n    project_dir str\n        Root of the Serverless application project.\n\n    specified_workflow str\n        Workflow to be used, if directly specified. They are currently scoped to "makefile" and the official runtime\n        identifier names themselves, eg: nodejs14.x. If a workflow is not directly specified,\n        it is calculated by the current method based on the runtime.\n\n    Returns\n    -------\n    namedtuple(Capability)\n        namedtuple that represents the Builder Workflow Config\n    '
    selectors_by_build_method = {'makefile': BasicWorkflowSelector(PROVIDED_MAKE_CONFIG), 'dotnet7': BasicWorkflowSelector(DOTNET_CLIPACKAGE_CONFIG), 'rust-cargolambda': BasicWorkflowSelector(RUST_CARGO_LAMBDA_CONFIG)}
    selectors_by_runtime = {'python3.7': BasicWorkflowSelector(PYTHON_PIP_CONFIG), 'python3.8': BasicWorkflowSelector(PYTHON_PIP_CONFIG), 'python3.9': BasicWorkflowSelector(PYTHON_PIP_CONFIG), 'python3.10': BasicWorkflowSelector(PYTHON_PIP_CONFIG), 'python3.11': BasicWorkflowSelector(PYTHON_PIP_CONFIG), 'nodejs12.x': BasicWorkflowSelector(NODEJS_NPM_CONFIG), 'nodejs14.x': BasicWorkflowSelector(NODEJS_NPM_CONFIG), 'nodejs16.x': BasicWorkflowSelector(NODEJS_NPM_CONFIG), 'nodejs18.x': BasicWorkflowSelector(NODEJS_NPM_CONFIG), 'nodejs20.x': BasicWorkflowSelector(NODEJS_NPM_CONFIG), 'ruby2.7': BasicWorkflowSelector(RUBY_BUNDLER_CONFIG), 'ruby3.2': BasicWorkflowSelector(RUBY_BUNDLER_CONFIG), 'dotnet6': BasicWorkflowSelector(DOTNET_CLIPACKAGE_CONFIG), 'go1.x': BasicWorkflowSelector(GO_MOD_CONFIG), 'java8': ManifestWorkflowSelector([JAVA_GRADLE_CONFIG._replace(executable_search_paths=[code_dir, project_dir]), JAVA_KOTLIN_GRADLE_CONFIG._replace(executable_search_paths=[code_dir, project_dir]), JAVA_MAVEN_CONFIG]), 'java11': ManifestWorkflowSelector([JAVA_GRADLE_CONFIG._replace(executable_search_paths=[code_dir, project_dir]), JAVA_KOTLIN_GRADLE_CONFIG._replace(executable_search_paths=[code_dir, project_dir]), JAVA_MAVEN_CONFIG]), 'java8.al2': ManifestWorkflowSelector([JAVA_GRADLE_CONFIG._replace(executable_search_paths=[code_dir, project_dir]), JAVA_KOTLIN_GRADLE_CONFIG._replace(executable_search_paths=[code_dir, project_dir]), JAVA_MAVEN_CONFIG]), 'java17': ManifestWorkflowSelector([JAVA_GRADLE_CONFIG._replace(executable_search_paths=[code_dir, project_dir]), JAVA_KOTLIN_GRADLE_CONFIG._replace(executable_search_paths=[code_dir, project_dir]), JAVA_MAVEN_CONFIG]), 'provided': BasicWorkflowSelector(PROVIDED_MAKE_CONFIG), 'provided.al2': BasicWorkflowSelector(PROVIDED_MAKE_CONFIG), 'provided.al2023': BasicWorkflowSelector(PROVIDED_MAKE_CONFIG)}
    selectors_by_builder = {'esbuild': BasicWorkflowSelector(NODEJS_NPM_ESBUILD_CONFIG)}
    if runtime and runtime not in selectors_by_runtime:
        raise UnsupportedRuntimeException("'{}' runtime is not supported".format(runtime))
    try:
        selector = get_selector(selector_list=[selectors_by_build_method, selectors_by_runtime, selectors_by_builder], identifiers=[specified_workflow, runtime], specified_workflow=specified_workflow)
        config = cast(WorkFlowSelector, selector).get_config(code_dir, project_dir)
        EventTracker.track_event('BuildWorkflowUsed', f'{config.language}-{config.dependency_manager}')
        return config
    except ValueError as ex:
        raise UnsupportedRuntimeException("Unable to find a supported build workflow for runtime '{}'. Reason: {}".format(runtime, str(ex))) from ex

def supports_specified_workflow(specified_workflow: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a specified workflow, returns whether it is supported in container builds,\n    can be used to overwrite runtime and get docker image or not\n\n    Parameters\n    ----------\n    specified_workflow\n        Workflow specified in the template\n\n    Returns\n    -------\n    bool\n        True, if this workflow is supported, can be used to overwrite runtime and get docker image\n    '
    supported_specified_workflow = ['dotnet7']
    return specified_workflow in supported_specified_workflow

class BasicWorkflowSelector:
    """
    Basic workflow selector that returns the first available configuration in the given list of configurations
    """

    def __init__(self, configs: Union[CONFIG, List[CONFIG]]) -> None:
        if False:
            print('Hello World!')
        if not isinstance(configs, list):
            configs = [configs]
        self.configs: List[CONFIG] = configs

    def get_config(self, code_dir: str, project_dir: str) -> CONFIG:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the first available configuration\n        '
        return self.configs[0]

class ManifestWorkflowSelector(BasicWorkflowSelector):
    """
    Selects a workflow by examining the directories for presence of a supported manifest
    """

    def get_config(self, code_dir: str, project_dir: str) -> CONFIG:
        if False:
            print('Hello World!')
        '\n        Finds a configuration by looking for a manifest in the given directories.\n\n        Returns\n        -------\n        samcli.lib.build.workflow_config.CONFIG\n            A supported configuration if one is found\n\n        Raises\n        ------\n        ValueError\n            If none of the supported manifests files are found\n        '
        search_dirs = [code_dir, project_dir]
        LOG.debug('Looking for a supported build workflow in following directories: %s', search_dirs)
        for config in self.configs:
            if any([self._has_manifest(config, directory) for directory in search_dirs]):
                return config
        raise ValueError("None of the supported manifests '{}' were found in the following paths '{}'".format([config.manifest_name for config in self.configs], search_dirs))

    @staticmethod
    def _has_manifest(config: CONFIG, directory: str) -> bool:
        if False:
            print('Hello World!')
        return os.path.exists(os.path.join(directory, config.manifest_name))