"""
CLI configuration decorator to use TOML configuration files for click commands.
"""
import functools
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import click
from click.core import ParameterSource
from samcli.cli.context import get_cmd_names
from samcli.commands.exceptions import ConfigException
from samcli.lib.config.samconfig import DEFAULT_CONFIG_FILE_NAME, DEFAULT_ENV, SamConfig
__all__ = ('ConfigProvider', 'configuration_option', 'get_ctx_defaults')
LOG = logging.getLogger(__name__)

class ConfigProvider:
    """
    A parser for sam configuration files
    """

    def __init__(self, section=None, cmd_names=None):
        if False:
            print('Hello World!')
        '\n        The constructor for ConfigProvider class\n\n        Parameters\n        ----------\n        section\n            The section defined in the configuration file nested within `cmd`\n        cmd_names\n            The cmd_name defined in the configuration file\n        '
        self.section = section
        self.cmd_names = cmd_names

    def __call__(self, config_path: Path, config_env: str, cmd_names: List[str]) -> dict:
        if False:
            i = 10
            return i + 15
        '\n        Get resolved config based on the `file_path` for the configuration file,\n        `config_env` targeted inside the config file and corresponding `cmd_name`\n        as denoted by `click`.\n\n        Parameters\n        ----------\n        config_path: Path\n            The path of configuration file.\n        config_env: str\n            The name of the sectional config_env within configuration file.\n        cmd_names: List[str]\n            The sam command name as defined by click.\n\n        Returns\n        -------\n        dict\n            A dictionary containing the configuration parameters under specified config_env.\n        '
        resolved_config: dict = {}
        config_file_path = Path(os.path.abspath(config_path)) if config_path else Path(os.getcwd(), SamConfig.get_default_file(os.getcwd()))
        config_file_name = config_file_path.name
        config_file_dir = config_file_path.parents[0]
        samconfig = SamConfig(config_file_dir, config_file_name)
        if os.environ.get('SAM_DEBUG', '').lower() == 'true':
            LOG.setLevel(logging.DEBUG)
        LOG.debug('Config file location: %s', samconfig.path())
        if not samconfig.exists():
            LOG.debug("Config file '%s' does not exist", samconfig.path())
            return resolved_config
        if not self.cmd_names:
            self.cmd_names = cmd_names
        try:
            LOG.debug("Loading configuration values from [%s.%s.%s] (env.command_name.section) in config file at '%s'...", config_env, self.cmd_names, self.section, samconfig.path())
            resolved_config = dict(samconfig.get_all(self.cmd_names, self.section, env=config_env).items())
            handle_parse_options(resolved_config)
            LOG.debug('Configuration values successfully loaded.')
            LOG.debug('Configuration values are: %s', resolved_config)
        except KeyError as ex:
            LOG.debug("Error reading configuration from [%s.%s.%s] (env.command_name.section) in configuration file at '%s' with : %s", config_env, self.cmd_names, self.section, samconfig.path(), str(ex))
        except Exception as ex:
            LOG.debug('Error reading configuration file: %s %s', samconfig.path(), str(ex))
            raise ConfigException(f'Error reading configuration: {ex}') from ex
        return resolved_config

def handle_parse_options(resolved_config: dict) -> None:
    if False:
        while True:
            i = 10
    '\n    Click does some handling of options to convert them to the intended types.\n    When injecting the options to click through a samconfig, we should do a similar\n    parsing of the options to ensure we pass the intended type.\n\n    E.g. if multiple is defined in the click option but only a single value is passed,\n    handle it the same way click does by converting it to a list first.\n\n    Mutates the resolved_config object\n\n    Parameters\n    ----------\n    resolved_config: dict\n        Configuration options extracted from the configuration file\n    '
    options_map = get_options_map()
    for (config_name, config_value) in resolved_config.items():
        if config_name in options_map:
            try:
                allow_multiple = options_map[config_name].multiple
                if allow_multiple and (not isinstance(config_value, list)):
                    resolved_config[config_name] = [config_value]
                    LOG.debug(f"Adjusting value of {config_name} to be a list since this option is defined with 'multiple=True'")
            except (AttributeError, KeyError):
                LOG.debug(f'Unable to parse option: {config_name}. Leaving option as inputted')

def get_options_map() -> dict:
    if False:
        print('Hello World!')
    "\n    Attempt to get all of the options that exist for a command.\n    Return a mapping from each option name to that options' properties.\n\n    Returns\n    -------\n    dict\n        Dict of command options if successful, None otherwise\n    "
    try:
        command_options = click.get_current_context().command.params
        return {command_option.name: command_option for command_option in command_options}
    except AttributeError:
        LOG.debug('Unable to get parameters from click context.')
        return {}

def configuration_callback(cmd_name: str, option_name: str, saved_callback: Optional[Callable], provider: Callable, ctx: click.Context, param: click.Parameter, value):
    if False:
        return 10
    '\n    Callback for reading the config file.\n\n    Also takes care of calling user specified custom callback afterwards.\n\n    Parameters\n    ----------\n    cmd_name: str\n        The `sam` command name derived from click.\n    option_name: str\n        The name of the option. This is used for error messages.\n    saved_callback: Optional[Callable]\n        User-specified callback to be called later.\n    provider: Callable\n        A callable that parses the configuration file and returns a dictionary\n        of the configuration parameters. Will be called as\n        `provider(file_path, config_env, cmd_name)`.\n    ctx: click.Context\n        Click context\n    param: click.Parameter\n        Click parameter\n    value\n        Specified value for config_env\n\n    Returns\n    -------\n    The specified callback or the specified value for config_env.\n    '
    ctx.default_map = ctx.default_map or {}
    cmd_name = cmd_name or str(ctx.info_name)
    param.default = None
    config_env_name = ctx.params.get('config_env') or DEFAULT_ENV
    config_dir = getattr(ctx, 'samconfig_dir', None) or os.getcwd()
    config_file = SamConfig.get_default_file(config_dir=config_dir) if getattr(ctx.get_parameter_source('config_file'), 'name', '') == ParameterSource.DEFAULT.name else ctx.params.get('config_file') or SamConfig.get_default_file(config_dir=config_dir)
    config_file_path = config_file if os.path.isabs(config_file) else os.path.join(config_dir, config_file)
    if config_file and config_file != DEFAULT_CONFIG_FILE_NAME and (not (Path(config_file_path).absolute().is_file() or Path(config_file_path).absolute().is_fifo())):
        error_msg = f'Config file {config_file} does not exist or could not be read!'
        LOG.debug(error_msg)
        raise ConfigException(error_msg)
    config = get_ctx_defaults(cmd_name, provider, ctx, config_env_name=config_env_name, config_file=config_file_path)
    ctx.default_map.update(config)
    return saved_callback(ctx, param, config_env_name) if saved_callback else config_env_name

def get_ctx_defaults(cmd_name: str, provider: Callable, ctx: click.Context, config_env_name: str, config_file: Optional[str]=None) -> Any:
    if False:
        print('Hello World!')
    '\n    Get the set of the parameters that are needed to be set into the click command.\n\n    This function also figures out the command name by looking up current click context\'s parent\n    and constructing the parsed command name that is used in default configuration file.\n    If a given cmd_name is start-api, the parsed name is "local_start_api".\n    provider is called with `config_file`, `config_env_name` and `parsed_cmd_name`.\n\n    Parameters\n    ----------\n    cmd_name: str\n        The `sam` command name.\n    provider: Callable\n        The provider to be called for reading configuration file.\n    ctx: click.Context\n        Click context\n    config_env_name: str\n        The config-env within configuration file, sam configuration file will be relative to the\n        supplied original template if its path is not specified.\n    config_file: Optional[str]\n        The configuration file name.\n\n    Returns\n    -------\n    Any\n        A dictionary of defaults for parameters.\n    '
    return provider(config_file, config_env_name, get_cmd_names(cmd_name, ctx))

def save_command_line_args_to_config(ctx: click.Context, cmd_names: List[str], config_env_name: str, config_file: SamConfig):
    if False:
        while True:
            i = 10
    'Save the provided command line arguments to the provided config file.\n\n    Parameters\n    ----------\n    ctx: click.Context\n        Click context of the current session.\n    cmd_names: List[str]\n        List of representing the entire command. Ex: ["local", "generate-event", "s3", "put"]\n    config_env_name: str\n        Name of the config environment the command is being executed under. It will also serve as the environment that\n        the parameters are saved under.\n    config_file: SamConfig\n        Object representing the SamConfig file being read for this execution. It will also be the file to which the\n        parameters will be written.\n    '
    if not ctx.params.get('save_params', False):
        return
    params_to_exclude = ['save_params', 'config_file', 'config_env']
    saved_params = {}
    for (param_name, param_source) in ctx._parameter_source.items():
        if param_name in params_to_exclude:
            continue
        if param_source != ParameterSource.COMMANDLINE:
            continue
        param_value = ctx.params.get(param_name, None)
        if param_value is None:
            LOG.debug(f'Parameter {param_name} was not saved, as its value is None.')
            continue
        config_file.put(cmd_names, 'parameters', param_name, param_value, config_env_name)
        saved_params.update({param_name: param_value})
    config_file.flush()
    LOG.info(f"Saved parameters to config file '{config_file.filepath.name}' under environment '{config_env_name}': {saved_params}")

def save_params(func):
    if False:
        print('Hello World!')
    'Decorator for saving provided parameters to a config file, if the flag is set.'

    def wrapper(*args, **kwargs):
        if False:
            while True:
                i = 10
        ctx = click.get_current_context()
        cmd_names = get_cmd_names(ctx.info_name, ctx)
        save_command_line_args_to_config(ctx=ctx, cmd_names=cmd_names, config_env_name=ctx.params.get('config_env', None), config_file=SamConfig(getattr(ctx, 'samconfig_dir', os.getcwd()), ctx.params.get('config_file', None)))
        return func(*args, **kwargs)
    return wrapper

def save_params_option(func):
    if False:
        print('Hello World!')
    'Composite decorator to add --save-params flag to a command.\n\n    When used, this command should be placed as the LAST of the click option/argument decorators to preserve the flow\n    of execution. The decorator itself will add the --save-params option, and, if provided, save the provided commands\n    from the terminal to the config file.\n    '
    return click.option('--save-params', is_flag=True, help='Save the parameters provided via the command line to the configuration file.')(save_params(func))

def configuration_option(*param_decls, **attrs):
    if False:
        i = 10
        return i + 15
    '\n    Adds configuration file support to a click application.\n\n    This will create a hidden click option whose callback function loads configuration parameters from default\n    configuration environment [default] in default configuration file [samconfig.toml] in the template file\n    directory.\n\n    Note\n    ----\n    This decorator should be added to the top of parameter chain, right below click.command, before\n    any options are declared.\n\n    Example\n    -------\n    >>> @click.command("hello")\n        @configuration_option(provider=ConfigProvider(section="parameters"))\n        @click.option(\'--name\', type=click.String)\n        def hello(name):\n            print("Hello " + name)\n\n    Parameters\n    ----------\n    preconfig_decorator_list: list\n        A list of click option decorator which need to place before this function. For\n        example, if we want to add option "--config-file" and "--config-env" to allow customized configuration file\n        and configuration environment, we will use configuration_option as below:\n        @configuration_option(\n            preconfig_decorator_list=[decorator_customize_config_file, decorator_customize_config_env],\n            provider=ConfigProvider(section=CONFIG_SECTION),\n        )\n        By default, we enable these two options.\n    provider: Callable\n        A callable that parses the configuration file and returns a dictionary\n        of the configuration parameters. Will be called as\n        `provider(file_path, config_env, cmd_name)`\n    '

    def decorator_configuration_setup(f):
        if False:
            print('Hello World!')
        configuration_setup_params = ()
        configuration_setup_attrs = {}
        configuration_setup_attrs['help'] = 'This is a hidden click option whose callback function loads configuration parameters.'
        configuration_setup_attrs['is_eager'] = True
        configuration_setup_attrs['expose_value'] = False
        configuration_setup_attrs['hidden'] = True
        configuration_setup_attrs['type'] = click.STRING
        provider = attrs.pop('provider')
        saved_callback = attrs.pop('callback', None)
        partial_callback = functools.partial(configuration_callback, None, None, saved_callback, provider)
        configuration_setup_attrs['callback'] = partial_callback
        return click.option(*configuration_setup_params, **configuration_setup_attrs)(f)

    def composed_decorator(decorators):
        if False:
            i = 10
            return i + 15

        def decorator(f):
            if False:
                return 10
            for deco in decorators:
                f = deco(f)
            return f
        return decorator
    decorator_list = [decorator_configuration_setup]
    pre_config_decorators = attrs.pop('preconfig_decorator_list', [decorator_customize_config_file, decorator_customize_config_env])
    for decorator in pre_config_decorators:
        decorator_list.append(decorator)
    return composed_decorator(decorator_list)

def decorator_customize_config_file(f: Callable) -> Callable:
    if False:
        return 10
    "\n    CLI option to customize configuration file name. By default it is 'samconfig.toml' in project directory.\n    Ex: --config-file samconfig.toml\n\n    Parameters\n    ----------\n    f: Callable\n        Callback function passed by Click\n\n    Returns\n    -------\n    Callable\n        A Callback function\n    "
    config_file_attrs: Dict[str, Any] = {}
    config_file_param_decls = ('--config-file',)
    config_file_attrs['help'] = 'Configuration file containing default parameter values.'
    config_file_attrs['default'] = DEFAULT_CONFIG_FILE_NAME
    config_file_attrs['show_default'] = True
    config_file_attrs['is_eager'] = True
    config_file_attrs['required'] = False
    config_file_attrs['type'] = click.STRING
    return click.option(*config_file_param_decls, **config_file_attrs)(f)

def decorator_customize_config_env(f: Callable) -> Callable:
    if False:
        print('Hello World!')
    "\n    CLI option to customize configuration environment name. By default it is 'default'.\n    Ex: --config-env default\n\n    Parameters\n    ----------\n    f: Callable\n        Callback function passed by Click\n\n    Returns\n    -------\n    Callable\n        A Callback function\n    "
    config_env_attrs: Dict[str, Any] = {}
    config_env_param_decls = ('--config-env',)
    config_env_attrs['help'] = 'Environment name specifying default parameter values in the configuration file.'
    config_env_attrs['default'] = DEFAULT_ENV
    config_env_attrs['show_default'] = True
    config_env_attrs['is_eager'] = True
    config_env_attrs['required'] = False
    config_env_attrs['type'] = click.STRING
    return click.option(*config_env_param_decls, **config_env_attrs)(f)