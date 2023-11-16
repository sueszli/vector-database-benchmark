from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, List, Mapping, Optional
from ._config import ConfigMap, ConfigValue
from ._output import OutputMap
from ._project_settings import ProjectSettings
from ._stack_settings import StackSettings
from ._tag import TagMap
PulumiFn = Callable[[], None]

class StackSummary:
    """A summary of the status of a given stack."""
    name: str
    current: bool
    update_in_progress: Optional[bool]
    last_update: Optional[datetime]
    resource_count: Optional[int]
    url: Optional[str]

    def __init__(self, name: str, current: bool, update_in_progress: Optional[bool]=None, last_update: Optional[datetime]=None, resource_count: Optional[int]=None, url: Optional[str]=None) -> None:
        if False:
            return 10
        self.name = name
        self.current = current
        self.update_in_progress = update_in_progress
        self.last_update = last_update
        self.resource_count = resource_count
        self.url = url

class WhoAmIResult:
    """The currently logged-in Pulumi identity."""
    user: str
    url: Optional[str]
    organizations: Optional[List[str]]

    def __init__(self, user: str, url: Optional[str]=None, organizations: Optional[List[str]]=None) -> None:
        if False:
            while True:
                i = 10
        self.user = user
        self.url = url
        self.organizations = organizations

class PluginInfo:
    name: str
    kind: str
    size: int
    last_used_time: datetime
    install_time: Optional[datetime]
    version: Optional[str]

    def __init__(self, name: str, kind: str, size: int, last_used_time: datetime, install_time: Optional[datetime]=None, version: Optional[str]=None) -> None:
        if False:
            return 10
        self.name = name
        self.kind = kind
        self.size = size
        self.install_time = install_time
        self.last_used = last_used_time
        self.version = version

class Deployment:
    version: Optional[int]
    deployment: Optional[Mapping[str, Any]]

    def __init__(self, version: Optional[int]=None, deployment: Optional[Mapping[str, Any]]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.version = version
        self.deployment = deployment

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'Deployment(version={self.version!r}, deployment={self.deployment!r})'

class Workspace(ABC):
    """
    Workspace is the execution context containing a single Pulumi project, a program, and multiple stacks.
    Workspaces are used to manage the execution environment, providing various utilities such as plugin
    installation, environment configuration ($PULUMI_HOME), and creation, deletion, and listing of Stacks.
    """
    work_dir: str
    '\n    The working directory to run Pulumi CLI commands\n    '
    pulumi_home: Optional[str]
    '\n    The directory override for CLI metadata if set.\n    This customizes the location of $PULUMI_HOME where metadata is stored and plugins are installed.\n    '
    secrets_provider: Optional[str]
    '\n    The secrets provider to use for encryption and decryption of stack secrets.\n    See: https://www.pulumi.com/docs/intro/concepts/secrets/#available-encryption-providers\n    '
    program: Optional[PulumiFn]
    '\n    The inline program `PulumiFn` to be used for Preview/Update operations if any.\n    If none is specified, the stack will refer to ProjectSettings for this information.\n    '
    env_vars: Mapping[str, str] = {}
    '\n    Environment values scoped to the current workspace. These will be supplied to every Pulumi command.\n    '
    pulumi_version: str
    '\n    The version of the underlying Pulumi CLI/Engine.\n    '

    @abstractmethod
    def project_settings(self) -> ProjectSettings:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the settings object for the current project if any.\n\n        :returns: ProjectSettings\n        '

    @abstractmethod
    def save_project_settings(self, settings: ProjectSettings) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overwrites the settings object in the current project.\n        There can only be a single project per workspace. Fails is new project name does not match old.\n\n        :param settings: The project settings to save.\n        '

    @abstractmethod
    def stack_settings(self, stack_name: str) -> StackSettings:
        if False:
            i = 10
            return i + 15
        '\n        Returns the settings object for the stack matching the specified stack name if any.\n\n        :param stack_name: The name of the stack.\n        :return: StackSettings\n        '

    @abstractmethod
    def save_stack_settings(self, stack_name: str, settings: StackSettings) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Overwrites the settings object for the stack matching the specified stack name.\n\n        :param stack_name: The name of the stack.\n        :param settings: The stack settings to save.\n        '

    @abstractmethod
    def serialize_args_for_op(self, stack_name: str) -> List[str]:
        if False:
            while True:
                i = 10
        '\n        A hook to provide additional args to CLI commands before they are executed.\n        Provided with stack name, returns a list of args to append to an invoked command ["--config=...", ]\n        LocalWorkspace does not utilize this extensibility point.\n\n        :param stack_name: The name of the stack.\n        '

    @abstractmethod
    def post_command_callback(self, stack_name: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        A hook executed after every command. Called with the stack name.\n        An extensibility point to perform workspace cleanup (CLI operations may create/modify a Pulumi.stack.yaml)\n        LocalWorkspace does not utilize this extensibility point.\n\n        :param stack_name: The name of the stack.\n        '

    @abstractmethod
    def get_config(self, stack_name: str, key: str, *, path: bool=False) -> ConfigValue:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the value associated with the specified stack name and key,\n        scoped to the Workspace.\n\n        :param stack_name: The name of the stack.\n        :param key: The key for the config item to get.\n        :param path: The key contains a path to a property in a map or list to get.\n        :returns: ConfigValue\n        '

    @abstractmethod
    def get_all_config(self, stack_name: str) -> ConfigMap:
        if False:
            return 10
        '\n        Returns the config map for the specified stack name, scoped to the current Workspace.\n\n        :param stack_name: The name of the stack.\n        :returns: ConfigMap\n        '

    @abstractmethod
    def set_config(self, stack_name: str, key: str, value: ConfigValue, *, path: bool=False) -> None:
        if False:
            print('Hello World!')
        '\n        Sets the specified key-value pair on the provided stack name.\n\n        :param stack_name: The name of the stack.\n        :param key: The config key to add.\n        :param value: The config value to add.\n        :param path: The key contains a path to a property in a map or list to set.\n        '

    @abstractmethod
    def set_all_config(self, stack_name: str, config: ConfigMap, *, path: bool=False) -> None:
        if False:
            return 10
        '\n        Sets all values in the provided config map for the specified stack name.\n\n        :param stack_name: The name of the stack.\n        :param config: A mapping of key to ConfigValue to set to config.\n        :param path: The keys contain a path to a property in a map or list to set.\n        '

    @abstractmethod
    def remove_config(self, stack_name: str, key: str, *, path: bool=False) -> None:
        if False:
            return 10
        '\n        Removes the specified key-value pair on the provided stack name.\n\n        :param stack_name: The name of the stack.\n        :param key: The key to remove from config.\n        :param path: The key contains a path to a property in a map or list to remove.\n        '

    @abstractmethod
    def remove_all_config(self, stack_name: str, keys: List[str], *, path: bool=False) -> None:
        if False:
            return 10
        '\n        Removes all values in the provided key list for the specified stack name.\n\n        :param stack_name: The name of the stack.\n        :param keys: The keys to remove from config.\n        :param path: The keys contain a path to a property in a map or list to remove.\n        '

    @abstractmethod
    def refresh_config(self, stack_name: str) -> None:
        if False:
            print('Hello World!')
        '\n        Gets and sets the config map used with the last update for Stack matching stack name.\n\n        :param stack_name: The name of the stack.\n        '

    @abstractmethod
    def get_tag(self, stack_name: str, key: str) -> str:
        if False:
            print('Hello World!')
        '\n        Returns the value associated with the specified stack name and key,\n        scoped to the Workspace.\n\n        :param stack_name: The name of the stack.\n        :param key: The key to use for the tag lookup.\n        :returns: str\n        '

    @abstractmethod
    def set_tag(self, stack_name: str, key: str, value: str) -> None:
        if False:
            return 10
        '\n        Sets the specified key-value pair on the provided stack name.\n\n        :param stack_name: The name of the stack.\n        :param key: The tag key to set.\n        :param value: The tag value to set.\n        '

    @abstractmethod
    def remove_tag(self, stack_name: str, key: str) -> None:
        if False:
            return 10
        '\n        Removes the specified key-value pair on the provided stack name.\n\n        :param stack_name: The name of the stack.\n        :param key: The tag key to remove.\n        '

    @abstractmethod
    def list_tags(self, stack_name: str) -> TagMap:
        if False:
            print('Hello World!')
        '\n        Returns the tag map for the specified tag name, scoped to the Workspace.\n\n        :param stack_name: The name of the stack.\n        :returns: TagMap\n        '

    @abstractmethod
    def who_am_i(self) -> WhoAmIResult:
        if False:
            i = 10
            return i + 15
        '\n        Returns the currently authenticated user.\n\n        :returns: WhoAmIResult\n        '

    @abstractmethod
    def stack(self) -> Optional[StackSummary]:
        if False:
            print('Hello World!')
        '\n        Returns a summary of the currently selected stack, if any.\n\n        :returns: Optional[StackSummary]\n        '

    @abstractmethod
    def create_stack(self, stack_name: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates and sets a new stack with the stack name, failing if one already exists.\n\n        :param str stack_name: The name of the stack to create\n        :returns: None\n        :raises CommandError Raised if a stack with the same name exists.\n        '

    @abstractmethod
    def select_stack(self, stack_name: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Selects and sets an existing stack matching the stack stack_name, failing if none exists.\n\n        :param stack_name: The name of the stack to select\n        :returns: None\n        :raises CommandError Raised if no matching stack exists.\n        '

    @abstractmethod
    def remove_stack(self, stack_name: str) -> None:
        if False:
            while True:
                i = 10
        '\n        Deletes the stack and all associated configuration and history.\n\n        :param stack_name: The name of the stack to remove\n        '

    @abstractmethod
    def list_stacks(self) -> List[StackSummary]:
        if False:
            print('Hello World!')
        '\n        Returns all Stacks created under the current Project.\n        This queries underlying backend and may return stacks not present in the Workspace\n        (as Pulumi.<stack>.yaml files).\n\n        :returns: List[StackSummary]\n        '

    @abstractmethod
    def install_plugin(self, name: str, version: str, kind: str='resource') -> None:
        if False:
            i = 10
            return i + 15
        '\n        Installs a plugin in the Workspace, for example to use cloud providers like AWS or GCP.\n\n        :param name: The name of the plugin to install.\n        :param version: The version to install.\n        :param kind: The kind of plugin.\n        '

    @abstractmethod
    def install_plugin_from_server(self, name: str, version: str, server: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Installs a plugin in the Workspace from a remote server, for example a third party plugin.\n\n        :param name: The name of the plugin to install.\n        :param version: The version to install.\n        :param server: The server to install from.\n        '

    @abstractmethod
    def remove_plugin(self, name: Optional[str]=None, version_range: Optional[str]=None, kind: str='resource') -> None:
        if False:
            while True:
                i = 10
        '\n        Removes a plugin from the Workspace matching the specified name and version.\n\n        :param name: The name of the plugin to remove.\n        :param version_range: The version range to remove.\n        :param kind: The kind of plugin.\n        '

    @abstractmethod
    def list_plugins(self) -> List[PluginInfo]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a list of all plugins installed in the Workspace.\n\n        :returns: List[PluginInfo]\n        '

    @abstractmethod
    def export_stack(self, stack_name: str) -> Deployment:
        if False:
            return 10
        "\n        ExportStack exports the deployment state of the stack matching the given name.\n        This can be combined with ImportStack to edit a stack's state (such as recovery from failed deployments).\n\n        :param stack_name: The name of the stack to export.\n        :returns: Deployment\n        "

    @abstractmethod
    def import_stack(self, stack_name: str, state: Deployment) -> None:
        if False:
            return 10
        "\n        ImportStack imports the specified deployment state into a pre-existing stack.\n        This can be combined with ExportStack to edit a stack's state (such as recovery from failed deployments).\n\n        :param stack_name: The name of the stack to import.\n        :param state: The deployment state to import.\n        "

    @abstractmethod
    def stack_outputs(self, stack_name: str) -> OutputMap:
        if False:
            print('Hello World!')
        '\n        Gets the current set of Stack outputs from the last Stack.up().\n\n        :param stack_name: The name of the stack.\n        :returns: OutputMap\n        '