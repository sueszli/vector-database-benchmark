"""
Context object used by build command
"""
import logging
import os
import pathlib
import shutil
from typing import Dict, Optional, List, Tuple
import click
from samcli.commands._utils.constants import DEFAULT_BUILD_DIR
from samcli.commands._utils.experimental import ExperimentalFlag, prompt_experimental
from samcli.commands._utils.template import get_template_data, move_template
from samcli.commands.build.exceptions import InvalidBuildDirException, MissingBuildMethodException
from samcli.commands.build.utils import prompt_user_to_enable_mount_with_write_if_needed, MountMode
from samcli.commands.exceptions import UserException
from samcli.lib.bootstrap.nested_stack.nested_stack_manager import NestedStackManager
from samcli.lib.build.app_builder import ApplicationBuilder, BuildError, UnsupportedBuilderLibraryVersionError, ApplicationBuildResult
from samcli.lib.build.build_graph import DEFAULT_DEPENDENCIES_DIR
from samcli.lib.build.bundler import EsbuildBundlerManager
from samcli.lib.build.exceptions import BuildInsideContainerError, InvalidBuildGraphException
from samcli.lib.build.workflow_config import UnsupportedRuntimeException
from samcli.lib.intrinsic_resolver.intrinsics_symbol_table import IntrinsicsSymbolTable
from samcli.lib.providers.provider import ResourcesToBuildCollector, Stack, Function, LayerVersion
from samcli.lib.providers.sam_api_provider import SamApiProvider
from samcli.lib.providers.sam_function_provider import SamFunctionProvider
from samcli.lib.providers.sam_layer_provider import SamLayerProvider
from samcli.lib.providers.sam_stack_provider import SamLocalStackProvider
from samcli.lib.telemetry.event import EventTracker
from samcli.lib.utils.osutils import BUILD_DIR_PERMISSIONS
from samcli.local.docker.manager import ContainerManager
from samcli.local.lambdafn.exceptions import FunctionNotFound, ResourceNotFound
LOG = logging.getLogger(__name__)

class BuildContext:

    def __init__(self, resource_identifier: Optional[str], template_file: str, base_dir: Optional[str], build_dir: str, cache_dir: str, cached: bool, parallel: bool, mode: Optional[str], manifest_path: Optional[str]=None, clean: bool=False, use_container: bool=False, parameter_overrides: Optional[dict]=None, docker_network: Optional[str]=None, skip_pull_image: bool=False, container_env_var: Optional[dict]=None, container_env_var_file: Optional[str]=None, build_images: Optional[dict]=None, excluded_resources: Optional[Tuple[str, ...]]=None, aws_region: Optional[str]=None, create_auto_dependency_layer: bool=False, stack_name: Optional[str]=None, print_success_message: bool=True, locate_layer_nested: bool=False, hook_name: Optional[str]=None, build_in_source: Optional[bool]=None, mount_with: str=MountMode.READ.value) -> None:
        if False:
            while True:
                i = 10
        "\n        Initialize the class\n\n        Parameters\n        ----------\n        resource_identifier: Optional[str]\n            The unique identifier of the resource\n        template_file: str\n            Path to the template for building\n        base_dir : str\n            Path to a folder. Use this folder as the root to resolve relative source code paths against\n        build_dir : str\n            Path to the directory where we will be storing built artifacts\n        cache_dir : str\n            Path to a the directory where we will be caching built artifacts\n        cached:\n            Optional. Set to True to build each function with cache to improve performance\n        parallel : bool\n            Optional. Set to True to build each function in parallel to improve performance\n        mode : str\n            Optional, name of the build mode to use ex: 'debug'\n        manifest_path : Optional[str]\n            Optional path to manifest file to replace the default one\n        clean: bool\n            Clear the build directory before building\n        use_container: bool\n            Build inside container\n        parameter_overrides: Optional[dict]\n            Optional dictionary of values for SAM template parameters that might want\n            to get substituted within the template\n        docker_network: Optional[str]\n            Docker network to run the container in.\n        skip_pull_image: bool\n            Whether we should pull new Docker container image or not\n        container_env_var: Optional[dict]\n            An optional dictionary of environment variables to pass to the container\n        container_env_var_file: Optional[dict]\n            An optional path to file that contains environment variables to pass to the container\n        build_images: Optional[dict]\n            An optional dictionary of build images to be used for building functions\n        aws_region: Optional[str]\n            Aws region code\n        create_auto_dependency_layer: bool\n            Create auto dependency layer for accelerate feature\n        stack_name: Optional[str]\n            Original stack name, which is used to generate layer name for accelerate feature\n        print_success_message: bool\n            Print successful message\n        locate_layer_nested: bool\n            Locate layer to its actual, worked with nested stack\n        hook_name: Optional[str]\n            Name of the hook package\n        build_in_source: Optional[bool]\n            Set to True to build in the source directory.\n        mount_with:\n            Mount mode of source code directory when building inside container, READ ONLY by default\n        "
        self._resource_identifier = resource_identifier
        self._template_file = template_file
        self._base_dir = base_dir
        self._use_raw_codeuri = bool(self._base_dir)
        self._build_dir = build_dir
        self._cache_dir = cache_dir
        self._parallel = parallel
        self._manifest_path = manifest_path
        self._clean = clean
        self._use_container = use_container
        self._parameter_overrides = parameter_overrides
        self._global_parameter_overrides: Optional[Dict] = None
        if aws_region:
            self._global_parameter_overrides = {IntrinsicsSymbolTable.AWS_REGION: aws_region}
        self._docker_network = docker_network
        self._skip_pull_image = skip_pull_image
        self._mode = mode
        self._cached = cached
        self._container_env_var = container_env_var
        self._container_env_var_file = container_env_var_file
        self._build_images = build_images
        self._exclude = excluded_resources
        self._create_auto_dependency_layer = create_auto_dependency_layer
        self._stack_name = stack_name
        self._print_success_message = print_success_message
        self._function_provider: Optional[SamFunctionProvider] = None
        self._layer_provider: Optional[SamLayerProvider] = None
        self._container_manager: Optional[ContainerManager] = None
        self._stacks: List[Stack] = []
        self._locate_layer_nested = locate_layer_nested
        self._hook_name = hook_name
        self._build_in_source = build_in_source
        self._build_result: Optional[ApplicationBuildResult] = None
        self._mount_with = MountMode(mount_with)

    def __enter__(self) -> 'BuildContext':
        if False:
            for i in range(10):
                print('nop')
        self.set_up()
        return self

    def set_up(self) -> None:
        if False:
            while True:
                i = 10
        'Set up class members used for building\n        This should be called each time before run() if stacks are changed.'
        (self._stacks, remote_stack_full_paths) = SamLocalStackProvider.get_stacks(self._template_file, parameter_overrides=self._parameter_overrides, global_parameter_overrides=self._global_parameter_overrides)
        if remote_stack_full_paths:
            LOG.warning('Below nested stacks(s) specify non-local URL(s), which are unsupported:\n%s\nSkipping building resources inside these nested stacks.', '\n'.join([f'- {full_path}' for full_path in remote_stack_full_paths]))
        self._function_provider = SamFunctionProvider(self.stacks, self._use_raw_codeuri, locate_layer_nested=self._locate_layer_nested)
        self._layer_provider = SamLayerProvider(self.stacks, self._use_raw_codeuri)
        if not self._base_dir:
            self._base_dir = str(pathlib.Path(self._template_file).resolve().parent)
        self._build_dir = self._setup_build_dir(self._build_dir, self._clean)
        if self._cached:
            cache_path = pathlib.Path(self._cache_dir)
            cache_path.mkdir(mode=BUILD_DIR_PERMISSIONS, parents=True, exist_ok=True)
            self._cache_dir = str(cache_path.resolve())
            dependencies_path = pathlib.Path(DEFAULT_DEPENDENCIES_DIR)
            dependencies_path.mkdir(mode=BUILD_DIR_PERMISSIONS, parents=True, exist_ok=True)
        if self._use_container:
            self._container_manager = ContainerManager(docker_network_id=self._docker_network, skip_pull_image=self._skip_pull_image)

    def __exit__(self, *args):
        if False:
            print('Hello World!')
        pass

    def get_resources_to_build(self):
        if False:
            while True:
                i = 10
        return self.resources_to_build

    def run(self):
        if False:
            while True:
                i = 10
        'Runs the building process by creating an ApplicationBuilder.'
        if self._is_sam_template():
            SamApiProvider.check_implicit_api_resource_ids(self.stacks)
        self._stacks = self._handle_build_pre_processing()
        try:
            mount_with_write = False
            if self._use_container:
                if self._mount_with == MountMode.WRITE:
                    mount_with_write = True
                else:
                    mount_with_write = prompt_user_to_enable_mount_with_write_if_needed(self.get_resources_to_build(), self.base_dir)
            builder = ApplicationBuilder(self.get_resources_to_build(), self.build_dir, self.base_dir, self.cache_dir, self.cached, self.is_building_specific_resource, manifest_path_override=self.manifest_path_override, container_manager=self.container_manager, mode=self.mode, parallel=self._parallel, container_env_var=self._container_env_var, container_env_var_file=self._container_env_var_file, build_images=self._build_images, combine_dependencies=not self._create_auto_dependency_layer, build_in_source=self._build_in_source, mount_with_write=mount_with_write)
            self._check_exclude_warning()
            self._check_rust_cargo_experimental_flag()
            for f in self.get_resources_to_build().functions:
                EventTracker.track_event('BuildFunctionRuntime', f.runtime)
            self._build_result = builder.build()
            self._handle_build_post_processing(builder, self._build_result)
            click.secho('\nBuild Succeeded', fg='green')
            root_stack = SamLocalStackProvider.find_root_stack(self.stacks)
            out_template_path = root_stack.get_output_template_path(self.build_dir)
            try:
                build_dir_in_success_message = os.path.relpath(self.build_dir)
                output_template_path_in_success_message = os.path.relpath(out_template_path)
            except ValueError:
                LOG.debug('Failed to retrieve relpath - using the specified path as-is instead')
                build_dir_in_success_message = self.build_dir
                output_template_path_in_success_message = out_template_path
            if self._print_success_message:
                msg = self._gen_success_msg(build_dir_in_success_message, output_template_path_in_success_message, os.path.abspath(self.build_dir) == os.path.abspath(DEFAULT_BUILD_DIR))
                click.secho(msg, fg='yellow')
        except FunctionNotFound as function_not_found_ex:
            raise UserException(str(function_not_found_ex), wrapped_from=function_not_found_ex.__class__.__name__) from function_not_found_ex
        except (UnsupportedRuntimeException, BuildError, BuildInsideContainerError, UnsupportedBuilderLibraryVersionError, InvalidBuildGraphException, ResourceNotFound) as ex:
            click.secho('\nBuild Failed', fg='red')
            deep_wrap = getattr(ex, 'wrapped_from', None)
            wrapped_from = deep_wrap if deep_wrap else ex.__class__.__name__
            raise UserException(str(ex), wrapped_from=wrapped_from) from ex

    def _is_sam_template(self) -> bool:
        if False:
            while True:
                i = 10
        'Check if a given template is a SAM template'
        template_dict = get_template_data(self._template_file)
        template_transforms = template_dict.get('Transform', [])
        if not isinstance(template_transforms, list):
            template_transforms = [template_transforms]
        for template_transform in template_transforms:
            if isinstance(template_transform, str) and template_transform.startswith('AWS::Serverless'):
                return True
        return False

    def _handle_build_pre_processing(self) -> List[Stack]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Pre-modify the stacks as required before invoking the build\n        :return: List of modified stacks\n        '
        stacks = []
        if any((EsbuildBundlerManager(stack).esbuild_configured() for stack in self.stacks)):
            for stack in self.stacks:
                stacks.append(EsbuildBundlerManager(stack).set_sourcemap_metadata_from_env())
            self.function_provider.update(stacks, self._use_raw_codeuri, locate_layer_nested=self._locate_layer_nested)
        return stacks if stacks else self.stacks

    def _handle_build_post_processing(self, builder: ApplicationBuilder, build_result: ApplicationBuildResult) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Add any template modifications necessary before moving the template to build directory\n        :param stack: Stack resources\n        :param template: Current template file\n        :param build_result: Result of the application build\n        :return: Modified template dict\n        '
        artifacts = build_result.artifacts
        stack_output_template_path_by_stack_path = {stack.stack_path: stack.get_output_template_path(self.build_dir) for stack in self.stacks}
        for stack in self.stacks:
            modified_template = builder.update_template(stack, artifacts, stack_output_template_path_by_stack_path)
            output_template_path = stack.get_output_template_path(self.build_dir)
            stack_name = self._stack_name if self._stack_name else ''
            if self._create_auto_dependency_layer:
                LOG.debug('Auto creating dependency layer for each function resource into a nested stack')
                nested_stack_manager = NestedStackManager(stack, stack_name, self.build_dir, modified_template, build_result)
                modified_template = nested_stack_manager.generate_auto_dependency_layer_stack()
            esbuild_manager = EsbuildBundlerManager(stack=stack, template=modified_template, build_dir=self.build_dir)
            if esbuild_manager.esbuild_configured():
                modified_template = esbuild_manager.handle_template_post_processing()
            move_template(stack.location, output_template_path, modified_template)

    def _gen_success_msg(self, artifacts_dir: str, output_template_path: str, is_default_build_dir: bool) -> str:
        if False:
            print('Hello World!')
        '\n        Generates a success message containing some suggested commands to run\n\n        Parameters\n        ----------\n        artifacts_dir: str\n            A string path representing the folder of built artifacts\n        output_template_path: str\n            A string path representing the final template file\n        is_default_build_dir: bool\n            True if the build folder is the folder defined by SAM CLI\n\n        Returns\n        -------\n        str\n            A formatted success message string\n        '
        validate_suggestion = 'Validate SAM template: sam validate'
        invoke_suggestion = 'Invoke Function: sam local invoke'
        sync_suggestion = 'Test Function in the Cloud: sam sync --stack-name {{stack-name}} --watch'
        deploy_suggestion = 'Deploy: sam deploy --guided'
        start_lambda_suggestion = 'Emulate local Lambda functions: sam local start-lambda'
        if not is_default_build_dir and (not self._hook_name):
            invoke_suggestion += ' -t {}'.format(output_template_path)
            deploy_suggestion += ' --template-file {}'.format(output_template_path)
        commands = [validate_suggestion, invoke_suggestion, sync_suggestion, deploy_suggestion]
        if self._hook_name:
            hook_package_flag = f' --hook-name {self._hook_name}'
            start_lambda_suggestion += hook_package_flag
            invoke_suggestion += hook_package_flag
            commands = [invoke_suggestion, start_lambda_suggestion]
        msg = f'\nBuilt Artifacts  : {artifacts_dir}\nBuilt Template   : {output_template_path}\n\nCommands you can use next\n=========================\n'
        msg += '[*] ' + f'{os.linesep}[*] '.join(commands)
        return msg

    @staticmethod
    def _setup_build_dir(build_dir: str, clean: bool) -> str:
        if False:
            return 10
        build_path = pathlib.Path(build_dir)
        if os.path.abspath(str(build_path)) == os.path.abspath(str(pathlib.Path.cwd())):
            exception_message = "Failing build: Running a build with build-dir as current working directory is extremely dangerous since the build-dir contents is first removed. This is no longer supported, please remove the '--build-dir' option from the command to allow the build artifacts to be placed in the directory your template is in."
            raise InvalidBuildDirException(exception_message)
        if build_path.exists() and os.listdir(build_dir) and clean:
            shutil.rmtree(build_dir)
        build_path.mkdir(mode=BUILD_DIR_PERMISSIONS, parents=True, exist_ok=True)
        return str(build_path.resolve())

    @property
    def container_manager(self) -> Optional[ContainerManager]:
        if False:
            print('Hello World!')
        return self._container_manager

    @property
    def function_provider(self) -> SamFunctionProvider:
        if False:
            return 10
        return self._function_provider

    @property
    def layer_provider(self) -> SamLayerProvider:
        if False:
            for i in range(10):
                print('nop')
        return self._layer_provider

    @property
    def build_dir(self) -> str:
        if False:
            while True:
                i = 10
        return self._build_dir

    @property
    def base_dir(self) -> str:
        if False:
            return 10
        return self._base_dir

    @property
    def cache_dir(self) -> str:
        if False:
            print('Hello World!')
        return self._cache_dir

    @property
    def cached(self) -> bool:
        if False:
            print('Hello World!')
        return self._cached

    @property
    def use_container(self) -> bool:
        if False:
            return 10
        return self._use_container

    @property
    def stacks(self) -> List[Stack]:
        if False:
            return 10
        return self._stacks

    @property
    def manifest_path_override(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        if self._manifest_path:
            return os.path.abspath(self._manifest_path)
        return None

    @property
    def mode(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        return self._mode

    @property
    def use_base_dir(self) -> bool:
        if False:
            while True:
                i = 10
        return self._use_raw_codeuri

    @property
    def resources_to_build(self) -> ResourcesToBuildCollector:
        if False:
            i = 10
            return i + 15
        '\n        Function return resources that should be build by current build command. This function considers\n        Lambda Functions and Layers with build method as buildable resources.\n        Returns\n        -------\n        ResourcesToBuildCollector\n        '
        return self.collect_build_resources(self._resource_identifier) if self._resource_identifier else self.collect_all_build_resources()

    @property
    def create_auto_dependency_layer(self) -> bool:
        if False:
            print('Hello World!')
        return self._create_auto_dependency_layer

    @property
    def build_result(self) -> Optional[ApplicationBuildResult]:
        if False:
            return 10
        return self._build_result

    def collect_build_resources(self, resource_identifier: str) -> ResourcesToBuildCollector:
        if False:
            print('Hello World!')
        'Collect a single buildable resource and its dependencies.\n        For a Lambda function, its layers will be included.\n\n        Parameters\n        ----------\n        resource_identifier : str\n            Resource identifier for the resource to be built\n\n        Returns\n        -------\n        ResourcesToBuildCollector\n            ResourcesToBuildCollector containing the buildable resource and its dependencies\n\n        Raises\n        ------\n        ResourceNotFound\n            raises ResourceNotFound is the specified resource cannot be found.\n        '
        result = ResourcesToBuildCollector()
        self._collect_single_function_and_dependent_layers(resource_identifier, result)
        self._collect_single_buildable_layer(resource_identifier, result)
        if not result.functions and (not result.layers):
            all_resources = [f.name for f in self.function_provider.get_all() if not f.inlinecode]
            all_resources.extend([l.name for l in self.layer_provider.get_all()])
            available_resource_message = f'{resource_identifier} not found. Possible options in your template: {all_resources}'
            LOG.info(available_resource_message)
            raise ResourceNotFound(f"Unable to find a function or layer with name '{resource_identifier}'")
        return result

    def collect_all_build_resources(self) -> ResourcesToBuildCollector:
        if False:
            i = 10
            return i + 15
        'Collect all buildable resources. Including Lambda functions and layers.\n\n        Returns\n        -------\n        ResourcesToBuildCollector\n            ResourcesToBuildCollector that contains all the buildable resources.\n        '
        result = ResourcesToBuildCollector()
        excludes: Tuple[str, ...] = self._exclude if self._exclude is not None else ()
        result.add_functions([f for f in self.function_provider.get_all() if f.name not in excludes and f.function_build_info.is_buildable()])
        result.add_layers([l for l in self.layer_provider.get_all() if l.name not in excludes and BuildContext.is_layer_buildable(l)])
        return result

    @property
    def is_building_specific_resource(self) -> bool:
        if False:
            return 10
        '\n        Whether customer requested to build a specific resource alone in isolation,\n        by specifying function_identifier to the build command.\n        Ex: sam build MyServerlessFunction\n        :return: True if user requested to build specific resource, False otherwise\n        '
        return bool(self._resource_identifier)

    def _collect_single_function_and_dependent_layers(self, resource_identifier: str, resource_collector: ResourcesToBuildCollector) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Populate resource_collector with function with provided identifier and all layers that function need to be\n        build in resource_collector\n        Parameters\n        ----------\n        resource_collector: Collector that will be populated with resources.\n        '
        function = self.function_provider.get(resource_identifier)
        if not function:
            return
        resource_collector.add_function(function)
        resource_collector.add_layers([l for l in function.layers if BuildContext.is_layer_buildable(l)])

    def _collect_single_buildable_layer(self, resource_identifier: str, resource_collector: ResourcesToBuildCollector) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Populate resource_collector with layer with provided identifier.\n\n        Parameters\n        ----------\n        resource_collector\n\n        Returns\n        -------\n\n        '
        layer = self.layer_provider.get(resource_identifier)
        if not layer:
            return
        if layer and layer.build_method is None:
            LOG.error('Layer %s is missing BuildMethod Metadata.', self._function_provider)
            raise MissingBuildMethodException(f'Build method missing in layer {resource_identifier}.')
        resource_collector.add_layer(layer)

    @staticmethod
    def is_layer_buildable(layer: LayerVersion):
        if False:
            for i in range(10):
                print('nop')
        if not layer.build_method:
            LOG.debug('Skip building layer without a build method: %s', layer.full_path)
            return False
        if isinstance(layer.codeuri, str) and layer.codeuri.endswith('.zip'):
            LOG.debug('Skip building zip layer: %s', layer.full_path)
            return False
        if layer.skip_build:
            LOG.debug('Skip building pre-built layer: %s', layer.full_path)
            return False
        return True
    _EXCLUDE_WARNING_MESSAGE = 'Resource expected to be built, but marked as excluded.\nBuilding anyways...'

    def _check_exclude_warning(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Prints warning message if a single resource to build is also being excluded\n        '
        excludes: Tuple[str, ...] = self._exclude if self._exclude is not None else ()
        if self._resource_identifier in excludes:
            LOG.warning(self._EXCLUDE_WARNING_MESSAGE)

    def _check_rust_cargo_experimental_flag(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Prints warning message and confirms if user wants to use beta feature\n        '
        WARNING_MESSAGE = 'Build method "rust-cargolambda" is a beta feature.\nPlease confirm if you would like to proceed\nYou can also enable this beta feature with "sam build --beta-features".'
        resources_to_build = self.get_resources_to_build()
        is_building_rust = False
        for function in resources_to_build.functions:
            if function.metadata and function.metadata.get('BuildMethod', '') == 'rust-cargolambda':
                is_building_rust = True
                break
        if is_building_rust:
            prompt_experimental(ExperimentalFlag.RustCargoLambda, WARNING_MESSAGE)

    @property
    def build_in_source(self) -> Optional[bool]:
        if False:
            while True:
                i = 10
        return self._build_in_source