import atexit
import contextlib
import os
import shutil
import subprocess
import sys
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Union
import dateutil.parser
import orjson
from dagster import AssetCheckResult, AssetCheckSeverity, AssetMaterialization, AssetObservation, AssetsDefinition, ConfigurableResource, OpExecutionContext, Output, get_dagster_logger
from dagster._annotations import public
from dagster._config.pythonic_config.pydantic_compat_layer import compat_model_validator
from dagster._core.errors import DagsterInvalidPropertyError
from dbt.contracts.results import NodeStatus, TestStatus
from dbt.node_types import NodeType
from dbt.version import __version__ as dbt_version
from packaging import version
from pydantic import Field, validator
from typing_extensions import Literal
from ..asset_utils import get_manifest_and_translator_from_dbt_assets, output_name_fn
from ..dagster_dbt_translator import DagsterDbtTranslator, validate_opt_translator, validate_translator
from ..dbt_manifest import DbtManifestParam, validate_manifest
from ..errors import DagsterDbtCliRuntimeError
from ..utils import ASSET_RESOURCE_TYPES, get_dbt_resource_props_by_dbt_unique_id_from_manifest
logger = get_dagster_logger()
DBT_EXECUTABLE = 'dbt'
DBT_PROJECT_YML_NAME = 'dbt_project.yml'
DBT_PROFILES_YML_NAME = 'profiles.yml'
PARTIAL_PARSE_FILE_NAME = 'partial_parse.msgpack'

def _get_dbt_target_path() -> Path:
    if False:
        print('Hello World!')
    return Path(os.getenv('DBT_TARGET_PATH', 'target'))

@dataclass
class DbtCliEventMessage:
    """The representation of a dbt CLI event.

    Args:
        raw_event (Dict[str, Any]): The raw event dictionary.
            See https://docs.getdbt.com/reference/events-logging#structured-logging for more
            information.
    """
    raw_event: Dict[str, Any]

    @classmethod
    def from_log(cls, log: str) -> 'DbtCliEventMessage':
        if False:
            while True:
                i = 10
        'Parse an event according to https://docs.getdbt.com/reference/events-logging#structured-logging.\n\n        We assume that the log format is json.\n        '
        raw_event: Dict[str, Any] = orjson.loads(log)
        return cls(raw_event=raw_event)

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.raw_event['info']['msg']

    @public
    def to_default_asset_events(self, manifest: DbtManifestParam, dagster_dbt_translator: DagsterDbtTranslator=DagsterDbtTranslator(), context: Optional[OpExecutionContext]=None) -> Iterator[Union[Output, AssetMaterialization, AssetObservation, AssetCheckResult]]:
        if False:
            for i in range(10):
                print('nop')
        'Convert a dbt CLI event to a set of corresponding Dagster events.\n\n        Args:\n            manifest (Union[Mapping[str, Any], str, Path]): The dbt manifest blob.\n            dagster_dbt_translator (DagsterDbtTranslator): Optionally, a custom translator for\n                linking dbt nodes to Dagster assets.\n\n        Returns:\n            Iterator[Union[Output, AssetMaterialization, AssetObservation, AssetCheckResult]]:\n                A set of corresponding Dagster events.\n\n                In a Dagster asset definition, the following are yielded:\n                - Output for refables (e.g. models, seeds, snapshots.)\n                - AssetCheckResult for dbt test results that are enabled as asset checks.\n                - AssetObservation for dbt test results that are not enabled as asset checks.\n\n                In a Dagster op definition, the following are yielded:\n                - AssetMaterialization for dbt test results that are not enabled as asset checks.\n                - AssetObservation for dbt test results.\n\n        '
        if self.raw_event['info']['level'] == 'debug':
            return
        event_node_info: Dict[str, Any] = self.raw_event['data'].get('node_info')
        if not event_node_info:
            return
        dagster_dbt_translator = validate_translator(dagster_dbt_translator)
        manifest = validate_manifest(manifest)
        if not manifest:
            logger.info('No dbt manifest was provided. Dagster events for dbt tests will not be created.')
        has_asset_def: bool = bool(context and context.has_assets_def)
        invocation_id: str = self.raw_event['info']['invocation_id']
        unique_id: str = event_node_info['unique_id']
        node_resource_type: str = event_node_info['resource_type']
        node_status: str = event_node_info['node_status']
        is_node_successful = node_status == NodeStatus.Success
        is_node_finished = bool(event_node_info.get('node_finished_at'))
        if node_resource_type in NodeType.refable() and is_node_successful:
            started_at = dateutil.parser.isoparse(event_node_info['node_started_at'])
            finished_at = dateutil.parser.isoparse(event_node_info['node_finished_at'])
            duration_seconds = (finished_at - started_at).total_seconds()
            if has_asset_def:
                yield Output(value=None, output_name=output_name_fn(event_node_info), metadata={'unique_id': unique_id, 'invocation_id': invocation_id, 'Execution Duration': duration_seconds})
            else:
                dbt_resource_props = manifest['nodes'][unique_id]
                asset_key = dagster_dbt_translator.get_asset_key(dbt_resource_props)
                yield AssetMaterialization(asset_key=asset_key, metadata={'unique_id': unique_id, 'invocation_id': invocation_id, 'Execution Duration': duration_seconds})
        elif manifest and node_resource_type == NodeType.Test and is_node_finished:
            upstream_unique_ids: List[str] = manifest['parent_map'][unique_id]
            test_resource_props = manifest['nodes'][unique_id]
            metadata = {'unique_id': unique_id, 'invocation_id': invocation_id, 'status': node_status}
            is_asset_check = dagster_dbt_translator.settings.enable_asset_checks
            attached_node_unique_id = test_resource_props.get('attached_node')
            is_generic_test = bool(attached_node_unique_id)
            if has_asset_def and is_asset_check and is_generic_test:
                is_test_successful = node_status == TestStatus.Pass
                severity = AssetCheckSeverity(test_resource_props['config']['severity'].upper())
                attached_node_resource_props: Dict[str, Any] = manifest['nodes'].get(attached_node_unique_id) or manifest['sources'].get(attached_node_unique_id)
                attached_node_asset_key = dagster_dbt_translator.get_asset_key(attached_node_resource_props)
                yield AssetCheckResult(passed=is_test_successful, asset_key=attached_node_asset_key, check_name=event_node_info['node_name'], metadata=metadata, severity=severity)
            else:
                for upstream_unique_id in upstream_unique_ids:
                    upstream_resource_props: Dict[str, Any] = manifest['nodes'].get(upstream_unique_id) or manifest['sources'].get(upstream_unique_id)
                    upstream_asset_key = dagster_dbt_translator.get_asset_key(upstream_resource_props)
                    yield AssetObservation(asset_key=upstream_asset_key, metadata=metadata)

@dataclass
class DbtCliInvocation:
    """The representation of an invoked dbt command.

    Args:
        process (subprocess.Popen): The process running the dbt command.
        manifest (Mapping[str, Any]): The dbt manifest blob.
        project_dir (Path): The path to the dbt project.
        target_path (Path): The path to the dbt target folder.
        raise_on_error (bool): Whether to raise an exception if the dbt command fails.
    """
    process: subprocess.Popen
    manifest: Mapping[str, Any]
    dagster_dbt_translator: DagsterDbtTranslator
    project_dir: Path
    target_path: Path
    raise_on_error: bool
    context: Optional[OpExecutionContext] = field(default=None, repr=False)
    _error_messages: List[str] = field(init=False, default_factory=list)

    @classmethod
    def run(cls, args: List[str], env: Dict[str, str], manifest: Mapping[str, Any], dagster_dbt_translator: DagsterDbtTranslator, project_dir: Path, target_path: Path, raise_on_error: bool, context: Optional[OpExecutionContext]) -> 'DbtCliInvocation':
        if False:
            i = 10
            return i + 15
        current_target_path = _get_dbt_target_path()
        partial_parse_file_path = current_target_path.joinpath(PARTIAL_PARSE_FILE_NAME) if current_target_path.is_absolute() else project_dir.joinpath(current_target_path, PARTIAL_PARSE_FILE_NAME)
        partial_parse_destination_target_path = target_path.joinpath(PARTIAL_PARSE_FILE_NAME)
        if partial_parse_file_path.exists() and (not partial_parse_destination_target_path.exists()):
            logger.info(f'Copying `{partial_parse_file_path}` to `{partial_parse_destination_target_path}` to take advantage of partial parsing.')
            partial_parse_destination_target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(partial_parse_file_path, partial_parse_destination_target_path)
        logger.info(f"Running dbt command: `{' '.join(args)}`.")
        process = subprocess.Popen(args=args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=project_dir)

        def cleanup_dbt_subprocess(process: subprocess.Popen) -> None:
            if False:
                while True:
                    i = 10
            if process.returncode is None:
                logger.info('The main process is being terminated, but the dbt command has not yet completed. Terminating the execution of dbt command.')
                process.terminate()
                process.wait()
        atexit.register(cleanup_dbt_subprocess, process)
        return cls(process=process, manifest=manifest, dagster_dbt_translator=dagster_dbt_translator, project_dir=project_dir, target_path=target_path, raise_on_error=raise_on_error, context=context)

    @public
    def wait(self) -> 'DbtCliInvocation':
        if False:
            for i in range(10):
                print('nop')
        'Wait for the dbt CLI process to complete.\n\n        Returns:\n            DbtCliInvocation: The current representation of the dbt CLI invocation.\n\n        Examples:\n            .. code-block:: python\n\n                from dagster_dbt import DbtCliResource\n\n                dbt = DbtCliResource(project_dir="/path/to/dbt/project")\n\n                dbt_cli_invocation = dbt.cli(["run"]).wait()\n        '
        list(self.stream_raw_events())
        return self

    @public
    def is_successful(self) -> bool:
        if False:
            while True:
                i = 10
        'Return whether the dbt CLI process completed successfully.\n\n        Returns:\n            bool: True, if the dbt CLI process returns with a zero exit code, and False otherwise.\n\n        Examples:\n            .. code-block:: python\n\n                from dagster_dbt import DbtCliResource\n\n                dbt = DbtCliResource(project_dir="/path/to/dbt/project")\n\n                dbt_cli_invocation = dbt.cli(["run"], raise_on_error=False)\n\n                if dbt_cli_invocation.is_successful():\n                    ...\n        '
        return self.process.wait() == 0

    @public
    def stream(self) -> Iterator[Union[Output, AssetMaterialization, AssetObservation, AssetCheckResult]]:
        if False:
            print('Hello World!')
        'Stream the events from the dbt CLI process and convert them to Dagster events.\n\n        Returns:\n            Iterator[Union[Output, AssetMaterialization, AssetObservation, AssetCheckResult]]:\n                A set of corresponding Dagster events.\n\n                In a Dagster asset definition, the following are yielded:\n                - Output for refables (e.g. models, seeds, snapshots.)\n                - AssetCheckResult for dbt test results that are enabled as asset checks.\n                - AssetObservation for dbt test results that are not enabled as asset checks.\n\n                In a Dagster op definition, the following are yielded:\n                - AssetMaterialization for dbt test results that are not enabled as asset checks.\n                - AssetObservation for dbt test results.\n\n        Examples:\n            .. code-block:: python\n\n                from pathlib import Path\n                from dagster_dbt import DbtCliResource, dbt_assets\n\n                @dbt_assets(manifest=Path("target", "manifest.json"))\n                def my_dbt_assets(context, dbt: DbtCliResource):\n                    yield from dbt.cli(["run"], context=context).stream()\n        '
        for event in self.stream_raw_events():
            yield from event.to_default_asset_events(manifest=self.manifest, dagster_dbt_translator=self.dagster_dbt_translator, context=self.context)

    @public
    def stream_raw_events(self) -> Iterator[DbtCliEventMessage]:
        if False:
            print('Hello World!')
        'Stream the events from the dbt CLI process.\n\n        Returns:\n            Iterator[DbtCliEventMessage]: An iterator of events from the dbt CLI process.\n        '
        with self.process.stdout or contextlib.nullcontext():
            for raw_line in self.process.stdout or []:
                log: str = raw_line.decode().strip()
                try:
                    event = DbtCliEventMessage.from_log(log=log)
                    is_error_message = event.raw_event['info']['level'] == 'error'
                    if is_error_message:
                        self._error_messages.append(str(event))
                    sys.stdout.write(str(event) + '\n')
                    sys.stdout.flush()
                    yield event
                except:
                    sys.stdout.write(log + '\n')
                    sys.stdout.flush()
        self._raise_on_error()

    @public
    def get_artifact(self, artifact: Union[Literal['manifest.json'], Literal['catalog.json'], Literal['run_results.json'], Literal['sources.json']]) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        'Retrieve a dbt artifact from the target path.\n\n        See https://docs.getdbt.com/reference/artifacts/dbt-artifacts for more information.\n\n        Args:\n            artifact (Union[Literal["manifest.json"], Literal["catalog.json"], Literal["run_results.json"], Literal["sources.json"]]): The name of the artifact to retrieve.\n\n        Returns:\n            Dict[str, Any]: The artifact as a dictionary.\n\n        Examples:\n            .. code-block:: python\n\n                from dagster_dbt import DbtCliResource\n\n                dbt = DbtCliResource(project_dir="/path/to/dbt/project")\n\n                dbt_cli_invocation = dbt.cli(["run"]).wait()\n\n                # Retrieve the run_results.json artifact.\n                run_results = dbt_cli_invocation.get_artifact("run_results.json")\n        '
        artifact_path = self.target_path.joinpath(artifact)
        return orjson.loads(artifact_path.read_bytes())

    def _format_error_messages(self) -> str:
        if False:
            i = 10
            return i + 15
        'Format the error messages from the dbt CLI process.'
        if not self._error_messages:
            return ''
        error_description = '\n'.join(self._error_messages)
        return f'\n\nErrors parsed from dbt logs:\n{error_description}'

    def _raise_on_error(self) -> None:
        if False:
            return 10
        'Ensure that the dbt CLI process has completed. If the process has not successfully\n        completed, then optionally raise an error.\n        '
        if not self.is_successful() and self.raise_on_error:
            log_path = self.target_path.joinpath('dbt.log')
            extra_description = ''
            if log_path.exists():
                extra_description = f', or view the dbt debug log: {log_path}'
            raise DagsterDbtCliRuntimeError(description=f'The dbt CLI process failed with exit code {self.process.returncode}. Check the stdout in the Dagster compute logs for the full information about the error{extra_description}.{self._format_error_messages()}')

class DbtCliResource(ConfigurableResource):
    """A resource used to execute dbt CLI commands.

    Attributes:
        project_dir (str): The path to the dbt project directory. This directory should contain a
            `dbt_project.yml`. See https://docs.getdbt.com/reference/dbt_project.yml for more
            information.
        global_config_flags (List[str]): A list of global flags configuration to pass to the dbt CLI
            invocation. See https://docs.getdbt.com/reference/global-configs for a full list of
            configuration.
        profiles_dir (Optional[str]): The path to the directory containing your dbt `profiles.yml`.
            By default, the current working directory is used, which is the dbt project directory.
            See https://docs.getdbt.com/docs/core/connect-data-platform/connection-profiles for more
            information.
        profile (Optional[str]): The profile from your dbt `profiles.yml` to use for execution. See
            https://docs.getdbt.com/docs/core/connect-data-platform/connection-profiles for more
            information.
        target (Optional[str]): The target from your dbt `profiles.yml` to use for execution. See
            https://docs.getdbt.com/docs/core/connect-data-platform/connection-profiles for more
            information.
        dbt_executable (str]): The path to the dbt executable. By default, this is `dbt`.

    Examples:
        Creating a dbt resource with only a reference to ``project_dir``:

        .. code-block:: python

            from dagster_dbt import DbtCliResource

            dbt = DbtCliResource(project_dir="/path/to/dbt/project")

        Creating a dbt resource with a custom ``profiles_dir``:

        .. code-block:: python

            from dagster_dbt import DbtCliResource

            dbt = DbtCliResource(
                project_dir="/path/to/dbt/project",
                profiles_dir="/path/to/dbt/project/profiles",
            )

        Creating a dbt resource with a custom ``profile`` and ``target``:

        .. code-block:: python

            from dagster_dbt import DbtCliResource

            dbt = DbtCliResource(
                project_dir="/path/to/dbt/project",
                profiles_dir="/path/to/dbt/project/profiles",
                profile="jaffle_shop",
                target="dev",
            )

        Creating a dbt resource with global configs, e.g. disabling colored logs with ``--no-use-color``:

        .. code-block:: python

            from dagster_dbt import DbtCliResource

            dbt = DbtCliResource(
                project_dir="/path/to/dbt/project",
                global_config_flags=["--no-use-color"],
            )

        Creating a dbt resource with custom dbt executable path:

        .. code-block:: python

            from dagster_dbt import DbtCliResource

            dbt = DbtCliResource(
                project_dir="/path/to/dbt/project",
                dbt_executable="/path/to/dbt/executable",
            )
    """
    project_dir: str = Field(description='The path to your dbt project directory. This directory should contain a `dbt_project.yml`. See https://docs.getdbt.com/reference/dbt_project.yml for more information.')
    global_config_flags: List[str] = Field(default=[], description='A list of global flags configuration to pass to the dbt CLI invocation. See https://docs.getdbt.com/reference/global-configs for a full list of configuration.')
    profiles_dir: Optional[str] = Field(default=None, description='The path to the directory containing your dbt `profiles.yml`. By default, the current working directory is used, which is the dbt project directory. See https://docs.getdbt.com/docs/core/connect-data-platform/connection-profiles for  more information.')
    profile: Optional[str] = Field(default=None, description='The profile from your dbt `profiles.yml` to use for execution. See https://docs.getdbt.com/docs/core/connect-data-platform/connection-profiles for more information.')
    target: Optional[str] = Field(default=None, description='The target from your dbt `profiles.yml` to use for execution. See https://docs.getdbt.com/docs/core/connect-data-platform/connection-profiles for more information.')
    dbt_executable: str = Field(default=DBT_EXECUTABLE, description='The path to the dbt executable.')

    @classmethod
    def _validate_absolute_path_exists(cls, path: Union[str, Path]) -> Path:
        if False:
            return 10
        absolute_path = Path(path).absolute()
        try:
            resolved_path = absolute_path.resolve(strict=True)
        except FileNotFoundError:
            raise ValueError(f"The absolute path of '{path}' ('{absolute_path}') does not exist")
        return resolved_path

    @classmethod
    def _validate_path_contains_file(cls, path: Path, file_name: str, error_message: str):
        if False:
            while True:
                i = 10
        if not path.joinpath(file_name).exists():
            raise ValueError(error_message)

    @validator('project_dir', 'profiles_dir', 'dbt_executable', pre=True)
    def convert_path_to_str(cls, v: Any) -> Any:
        if False:
            while True:
                i = 10
        'Validate that the path is converted to a string.'
        if isinstance(v, Path):
            resolved_path = cls._validate_absolute_path_exists(v)
            absolute_path = Path(v).absolute()
            try:
                resolved_path = absolute_path.resolve(strict=True)
            except FileNotFoundError:
                raise ValueError(f"The absolute path of '{v}' ('{absolute_path}') does not exist")
            return os.fspath(resolved_path)
        return v

    @validator('project_dir')
    def validate_project_dir(cls, project_dir: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        resolved_project_dir = cls._validate_absolute_path_exists(project_dir)
        cls._validate_path_contains_file(path=resolved_project_dir, file_name=DBT_PROJECT_YML_NAME, error_message=f'{resolved_project_dir} does not contain a {DBT_PROJECT_YML_NAME} file. Please specify a valid path to a dbt project.')
        return os.fspath(resolved_project_dir)

    @validator('profiles_dir')
    def validate_profiles_dir(cls, profiles_dir: Optional[str]) -> Optional[str]:
        if False:
            print('Hello World!')
        if profiles_dir is None:
            return None
        resolved_profiles_dir = cls._validate_absolute_path_exists(profiles_dir)
        cls._validate_path_contains_file(path=resolved_profiles_dir, file_name=DBT_PROFILES_YML_NAME, error_message=f'{resolved_profiles_dir} does not contain a {DBT_PROFILES_YML_NAME} file. Please specify a valid path to a dbt profile directory.')
        return os.fspath(resolved_profiles_dir)

    @validator('dbt_executable')
    def validate_dbt_executable(cls, dbt_executable: str) -> str:
        if False:
            i = 10
            return i + 15
        resolved_dbt_executable = shutil.which(dbt_executable)
        if not resolved_dbt_executable:
            raise ValueError(f"The dbt executable '{dbt_executable}' does not exist. Please specify a valid path to a dbt executable.")
        return dbt_executable

    @compat_model_validator(mode='before')
    def validate_dbt_version(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        'Validate that the dbt version is supported.'
        if version.parse(dbt_version) < version.parse('1.4.0'):
            raise ValueError(f'To use `dagster_dbt.DbtCliResource`, you must use `dbt-core>=1.4.0`. Currently, you are using `dbt-core=={dbt_version}`. Please install a compatible dbt-core version.')
        return values

    def _get_unique_target_path(self, *, context: Optional[OpExecutionContext]) -> Path:
        if False:
            i = 10
            return i + 15
        'Get a unique target path for the dbt CLI invocation.\n\n        Args:\n            context (Optional[OpExecutionContext]): The execution context.\n\n        Returns:\n            str: A unique target path for the dbt CLI invocation.\n        '
        unique_id = str(uuid.uuid4())[:7]
        path = unique_id
        if context:
            path = f'{context.op.name}-{context.run_id[:7]}-{unique_id}'
        current_target_path = _get_dbt_target_path()
        return current_target_path.joinpath(path)

    @public
    def cli(self, args: List[str], *, raise_on_error: bool=True, manifest: Optional[DbtManifestParam]=None, dagster_dbt_translator: Optional[DagsterDbtTranslator]=None, context: Optional[OpExecutionContext]=None, target_path: Optional[Path]=None) -> DbtCliInvocation:
        if False:
            i = 10
            return i + 15
        'Create a subprocess to execute a dbt CLI command.\n\n        Args:\n            args (List[str]): The dbt CLI command to execute.\n            raise_on_error (bool): Whether to raise an exception if the dbt CLI command fails.\n            manifest (Optional[Union[Mapping[str, Any], str, Path]]): The dbt manifest blob. If an\n                execution context from within `@dbt_assets` is provided to the context argument,\n                then the manifest provided to `@dbt_assets` will be used.\n            dagster_dbt_translator (Optional[DagsterDbtTranslator]): The translator to link dbt\n                nodes to Dagster assets. If an execution context from within `@dbt_assets` is\n                provided to the context argument, then the dagster_dbt_translator provided to\n                `@dbt_assets` will be used.\n            context (Optional[OpExecutionContext]): The execution context from within `@dbt_assets`.\n            target_path (Optional[Path]): An explicit path to a target folder to use to store and\n                retrieve dbt artifacts when running a dbt CLI command. If not provided, a unique\n                target path will be generated.\n\n        Returns:\n            DbtCliInvocation: A invocation instance that can be used to retrieve the output of the\n                dbt CLI command.\n\n        Examples:\n            Streaming Dagster events for dbt asset materializations and observations:\n\n            .. code-block:: python\n\n                from pathlib import Path\n\n                from dagster import AssetExecutionContext\n                from dagster_dbt import DbtCliResource, dbt_assets\n\n\n                @dbt_assets(manifest=Path("target", "manifest.json"))\n                def my_dbt_assets(context: AssetExecutionContext, dbt: DbtCliResource):\n                    yield from dbt.cli(["run"], context=context).stream()\n\n            Retrieving a dbt artifact after streaming the Dagster events:\n\n            .. code-block:: python\n\n                from pathlib import Path\n\n                from dagster import AssetExecutionContext\n                from dagster_dbt import DbtCliResource, dbt_assets\n\n\n                @dbt_assets(manifest=Path("target", "manifest.json"))\n                def my_dbt_assets(context: AssetExecutionContext, dbt: DbtCliResource):\n                    dbt_run_invocation = dbt.cli(["run"], context=context)\n\n                    yield from dbt_run_invocation.stream()\n\n                    # Retrieve the `run_results.json` dbt artifact as a dictionary:\n                    run_results_json = dbt_run_invocation.get_artifact("run_results.json")\n\n                    # Retrieve the `run_results.json` dbt artifact as a file path:\n                    run_results_path = dbt_run_invocation.target_path.joinpath("run_results.json")\n\n            Customizing the asset materialization metadata when streaming the Dagster events:\n\n            .. code-block:: python\n\n                from pathlib import Path\n\n                from dagster import AssetExecutionContext\n                from dagster_dbt import DbtCliResource, dbt_assets\n\n\n                @dbt_assets(manifest=Path("target", "manifest.json"))\n                def my_dbt_assets(context: AssetExecutionContext, dbt: DbtCliResource):\n                    dbt_cli_invocation = dbt.cli(["run"], context=context)\n\n                    for dagster_event in dbt_cli_invocation.stream():\n                        if isinstance(dagster_event, Output):\n                            context.add_output_metadata(\n                                metadata={\n                                    "my_custom_metadata": "my_custom_metadata_value",\n                                },\n                                output_name=dagster_event.output_name,\n                            )\n\n                        yield dagster_event\n\n            Suppressing exceptions from a dbt CLI command when a non-zero exit code is returned:\n\n            .. code-block:: python\n\n                from pathlib import Path\n\n                from dagster import AssetExecutionContext\n                from dagster_dbt import DbtCliResource, dbt_assets\n\n\n                @dbt_assets(manifest=Path("target", "manifest.json"))\n                def my_dbt_assets(context: AssetExecutionContext, dbt: DbtCliResource):\n                    dbt_run_invocation = dbt.cli(["run"], context=context, raise_on_error=False)\n\n                    if dbt_run_invocation.is_successful():\n                        yield from dbt_run_invocation.stream()\n                    else:\n                        ...\n\n            Invoking a dbt CLI command in a custom asset or op:\n\n            .. code-block:: python\n\n                import json\n\n                from dagster import asset, op\n                from dagster_dbt import DbtCliResource\n\n\n                @asset\n                def my_dbt_asset(dbt: DbtCliResource):\n                    dbt_macro_args = {"key": "value"}\n                    dbt.cli(["run-operation", "my-macro", json.dumps(dbt_macro_args)]).wait()\n\n\n                @op\n                def my_dbt_op(dbt: DbtCliResource):\n                    dbt_macro_args = {"key": "value"}\n                    dbt.cli(["run-operation", "my-macro", json.dumps(dbt_macro_args)]).wait()\n        '
        dagster_dbt_translator = validate_opt_translator(dagster_dbt_translator)
        target_path = target_path or self._get_unique_target_path(context=context)
        env = {**os.environ.copy(), 'PYTHONUNBUFFERED': '1', 'DBT_SEND_ANONYMOUS_USAGE_STATS': 'false', 'DBT_LOG_FORMAT': 'json', 'DBT_TARGET_PATH': os.fspath(target_path), 'DBT_LOG_PATH': os.fspath(target_path), **({'DBT_PROFILES_DIR': self.profiles_dir} if self.profiles_dir else {})}
        assets_def: Optional[AssetsDefinition] = None
        with suppress(DagsterInvalidPropertyError):
            assets_def = context.assets_def if context else None
        selection_args: List[str] = []
        dagster_dbt_translator = dagster_dbt_translator or DagsterDbtTranslator()
        if context and assets_def is not None:
            (manifest, dagster_dbt_translator) = get_manifest_and_translator_from_dbt_assets([assets_def])
            if version.parse(dbt_version) >= version.parse('1.5.0') and dagster_dbt_translator.settings.enable_asset_checks:
                env['DBT_INDIRECT_SELECTION'] = 'empty'
            selection_args = get_subset_selection_for_context(context=context, manifest=manifest, select=context.op.tags.get('dagster-dbt/select'), exclude=context.op.tags.get('dagster-dbt/exclude'))
        else:
            manifest = validate_manifest(manifest) if manifest else {}
        profile_args: List[str] = []
        if self.profile:
            profile_args = ['--profile', self.profile]
        if self.target:
            profile_args += ['--target', self.target]
        args = [self.dbt_executable, *self.global_config_flags, *args, *profile_args, *selection_args]
        project_dir = Path(self.project_dir)
        if not target_path.is_absolute():
            target_path = project_dir.joinpath(target_path)
        return DbtCliInvocation.run(args=args, env=env, manifest=manifest, dagster_dbt_translator=dagster_dbt_translator, project_dir=project_dir, target_path=target_path, raise_on_error=raise_on_error, context=context)

def get_subset_selection_for_context(context: OpExecutionContext, manifest: Mapping[str, Any], select: Optional[str], exclude: Optional[str]) -> List[str]:
    if False:
        return 10
    'Generate a dbt selection string to materialize the selected resources in a subsetted execution context.\n\n    See https://docs.getdbt.com/reference/node-selection/syntax#how-does-selection-work.\n\n    Args:\n        context (OpExecutionContext): The execution context for the current execution step.\n        select (Optional[str]): A dbt selection string to select resources to materialize.\n        exclude (Optional[str]): A dbt selection string to exclude resources from materializing.\n\n    Returns:\n        List[str]: dbt CLI arguments to materialize the selected resources in a\n            subsetted execution context.\n\n            If the current execution context is not performing a subsetted execution,\n            return CLI arguments composed of the inputed selection and exclusion arguments.\n    '
    default_dbt_selection = []
    if select:
        default_dbt_selection += ['--select', select]
    if exclude:
        default_dbt_selection += ['--exclude', exclude]
    dbt_resource_props_by_output_name = get_dbt_resource_props_by_output_name(manifest)
    dbt_resource_props_by_test_name = get_dbt_resource_props_by_test_name(manifest)
    if not context.is_subset:
        logger.info(f'A dbt subsetted execution is not being performed. Using the default dbt selection arguments `{default_dbt_selection}`.')
        return default_dbt_selection
    selected_dbt_resources = []
    for output_name in context.selected_output_names:
        dbt_resource_props = dbt_resource_props_by_output_name[output_name]
        fqn_selector = f"fqn:{'.'.join(dbt_resource_props['fqn'])}"
        selected_dbt_resources.append(fqn_selector)
    for (_, check_name) in context.selected_asset_check_keys:
        test_resource_props = dbt_resource_props_by_test_name[check_name]
        fqn_selector = f"fqn:{'.'.join(test_resource_props['fqn'])}"
        selected_dbt_resources.append(fqn_selector)
    union_selected_dbt_resources = ['--select'] + [' '.join(selected_dbt_resources)]
    logger.info(f'A dbt subsetted execution is being performed. Overriding default dbt selection arguments `{default_dbt_selection}` with arguments: `{union_selected_dbt_resources}`')
    return union_selected_dbt_resources

def get_dbt_resource_props_by_output_name(manifest: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    if False:
        i = 10
        return i + 15
    node_info_by_dbt_unique_id = get_dbt_resource_props_by_dbt_unique_id_from_manifest(manifest)
    return {output_name_fn(node): node for node in node_info_by_dbt_unique_id.values() if node['resource_type'] in ASSET_RESOURCE_TYPES}

def get_dbt_resource_props_by_test_name(manifest: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    if False:
        for i in range(10):
            print('nop')
    return {dbt_resource_props['name']: dbt_resource_props for (unique_id, dbt_resource_props) in manifest['nodes'].items() if unique_id.startswith('test')}