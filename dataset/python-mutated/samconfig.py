"""
Class representing the samconfig.toml
"""
import logging
import os
from pathlib import Path
from typing import Any, Iterable
from samcli.lib.config.exceptions import FileParseException, SamConfigFileReadException, SamConfigVersionException
from samcli.lib.config.file_manager import FILE_MANAGER_MAPPER
from samcli.lib.config.version import SAM_CONFIG_VERSION, VERSION_KEY
from samcli.lib.telemetry.event import EventTracker
LOG = logging.getLogger(__name__)
DEFAULT_CONFIG_FILE_EXTENSION = '.toml'
DEFAULT_CONFIG_FILE = 'samconfig'
DEFAULT_CONFIG_FILE_NAME = DEFAULT_CONFIG_FILE + DEFAULT_CONFIG_FILE_EXTENSION
DEFAULT_ENV = 'default'
DEFAULT_GLOBAL_CMDNAME = 'global'

class SamConfig:
    """
    Class to represent `samconfig` config options.
    """

    def __init__(self, config_dir, filename=None):
        if False:
            while True:
                i = 10
        '\n        Initialize the class\n\n        Parameters\n        ----------\n        config_dir : string\n            Directory where the configuration file needs to be stored\n        filename : string\n            Optional. Name of the configuration file. It is recommended to stick with default so in the future we\n            could automatically support auto-resolving multiple config files within same directory.\n        '
        self.document = {}
        self.filepath = Path(config_dir, filename or self.get_default_file(config_dir=config_dir))
        file_extension = self.filepath.suffix
        self.file_manager = FILE_MANAGER_MAPPER.get(file_extension, None)
        if not self.file_manager:
            LOG.warning(f"The config file extension '{file_extension}' is not supported. Supported formats are: [{'|'.join(FILE_MANAGER_MAPPER.keys())}]")
            raise SamConfigFileReadException(f'The config file {self.filepath} uses an unsupported extension, and cannot be read.')
        self._read()
        EventTracker.track_event('SamConfigFileExtension', file_extension)

    def get_stage_configuration_names(self):
        if False:
            for i in range(10):
                print('nop')
        if self.document:
            return [stage for (stage, value) in self.document.items() if isinstance(value, dict)]
        return []

    def get_all(self, cmd_names, section, env=DEFAULT_ENV):
        if False:
            return 10
        '\n        Gets a value from the configuration file for the given environment, command and section\n\n        Parameters\n        ----------\n        cmd_names : list(str)\n            List of representing the entire command. Ex: ["local", "generate-event", "s3", "put"]\n        section : str\n            Specific section within the command to look into. e.g. `parameters`\n        env : str\n            Optional, Name of the environment\n\n        Returns\n        -------\n        dict\n            Dictionary of configuration options in the file. None, if the config doesn\'t exist.\n\n        Raises\n        ------\n        KeyError\n            If the config file does *not* have the specific section\n        '
        env = env or DEFAULT_ENV
        self.document = self._read()
        config_content = self.document.get(env, {})
        params = config_content.get(self.to_key(cmd_names), {}).get(section, {})
        if DEFAULT_GLOBAL_CMDNAME in config_content:
            global_params = config_content.get(DEFAULT_GLOBAL_CMDNAME, {}).get(section, {})
            global_params.update(params.copy())
            params = global_params.copy()
        return params

    def put(self, cmd_names, section, key, value, env=DEFAULT_ENV):
        if False:
            for i in range(10):
                print('nop')
        '\n        Writes the `key=value` under the given section. You have to call the `flush()` method after `put()` in\n        order to write the values back to the config file. Otherwise they will be just saved in-memory, available\n        for future access, but never saved back to the file.\n\n        Parameters\n        ----------\n        cmd_names : list(str)\n            List of representing the entire command. Ex: ["local", "generate-event", "s3", "put"]\n        section : str\n            Specific section within the command to look into. e.g. `parameters`\n        key : str\n            Key to write the data under\n        value : Any\n            Value to write. Could be any of the supported types.\n        env : str\n            Optional, Name of the environment\n        '
        cmd_name_key = self.to_key(cmd_names)
        env_content = self.document.get(env, {})
        cmd_content = env_content.get(cmd_name_key, {})
        param_content = cmd_content.get(section, {})
        if param_content:
            param_content.update({key: value})
        elif cmd_content:
            cmd_content.update({section: {key: value}})
        elif env_content:
            env_content.update({cmd_name_key: {section: {key: value}}})
        else:
            self.document.update({env: {cmd_name_key: {section: {key: value}}}})
        self._deduplicate_global_parameters(cmd_name_key, section, key, env)

    def put_comment(self, comment):
        if False:
            print('Hello World!')
        '\n        Write a comment section back to the file.\n\n        Parameters\n        ------\n        comment: str\n            A comment to write to the samconfg file\n        '
        self.document = self.file_manager.put_comment(self.document, comment)

    def flush(self):
        if False:
            i = 10
            return i + 15
        '\n        Write the data back to file\n        '
        self._write()

    def sanity_check(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sanity check the contents of samconfig\n        '
        try:
            self._read()
        except SamConfigFileReadException:
            return False
        else:
            return True

    def exists(self):
        if False:
            return 10
        return self.filepath.exists()

    def _ensure_exists(self):
        if False:
            i = 10
            return i + 15
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.filepath.touch()

    def path(self):
        if False:
            while True:
                i = 10
        return str(self.filepath)

    @staticmethod
    def config_dir(template_file_path=None):
        if False:
            while True:
                i = 10
        '\n        SAM Config file is always relative to the SAM Template. If it the template is not\n        given, then it is relative to cwd()\n        '
        if template_file_path:
            return os.path.dirname(template_file_path)
        return os.getcwd()

    def _read(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.document:
            try:
                self.document = self.file_manager.read(self.filepath)
            except FileParseException as e:
                raise SamConfigFileReadException(e) from e
        if self.document:
            self._version_sanity_check(self._version())
        return self.document

    def _write(self):
        if False:
            print('Hello World!')
        if not self.document:
            return
        self._ensure_exists()
        current_version = self._version() if self._version() else SAM_CONFIG_VERSION
        self.document.update({VERSION_KEY: current_version})
        self.file_manager.write(self.document, self.filepath)

    def _version(self):
        if False:
            for i in range(10):
                print('nop')
        return self.document.get(VERSION_KEY, None)

    def _deduplicate_global_parameters(self, cmd_name_key, section, key, env=DEFAULT_ENV):
        if False:
            return 10
        '\n        In case the global parameters contains the same key-value pair with command parameters,\n        we only keep the entry in global parameters\n\n        Parameters\n        ----------\n        cmd_name_key : str\n            key of command name\n\n        section : str\n            Specific section within the command to look into. e.g. `parameters`\n\n        key : str\n            Key to write the data under\n\n        env : str\n            Optional, Name of the environment\n        '
        global_params = self.document.get(env, {}).get(DEFAULT_GLOBAL_CMDNAME, {}).get(section, {})
        command_params = self.document.get(env, {}).get(cmd_name_key, {}).get(section, {})
        if cmd_name_key != DEFAULT_GLOBAL_CMDNAME and global_params and command_params and global_params.get(key) and (global_params.get(key) == command_params.get(key)):
            value = command_params.get(key)
            save_global_message = f'\n\tParameter "{key}={value}" in [{env}.{cmd_name_key}.{section}] is defined as a global parameter [{env}.{DEFAULT_GLOBAL_CMDNAME}.{section}].\n\tThis parameter will be only saved under [{env}.{DEFAULT_GLOBAL_CMDNAME}.{section}] in {self.filepath}.'
            LOG.info(save_global_message)
            del self.document[env][cmd_name_key][section][key]

    @staticmethod
    def get_default_file(config_dir: str) -> str:
        if False:
            while True:
                i = 10
        'Return a defaultly-named config file, if it exists, otherwise the current default.\n\n        Parameters\n        ----------\n        config_dir: str\n            The name of the directory where the config file is/will be stored.\n\n        Returns\n        -------\n        str\n            The name of the config file found, if it exists. In the case that it does not exist, the default config\n            file name is returned instead.\n        '
        config_files_found = 0
        config_file = DEFAULT_CONFIG_FILE_NAME
        for extension in reversed(list(FILE_MANAGER_MAPPER.keys())):
            filename = DEFAULT_CONFIG_FILE + extension
            if Path(config_dir, filename).exists():
                config_files_found += 1
                config_file = filename
        if config_files_found == 0:
            LOG.debug('No config file found in this directory.')
        elif config_files_found > 1:
            LOG.info(f"More than one samconfig file found; using {config_file}. To use another config file, please specify it using the '--config-file' flag.")
        return config_file

    @staticmethod
    def _version_sanity_check(version: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(version, float):
            raise SamConfigVersionException(f"'{VERSION_KEY}' key is not present or is in unrecognized format. ")

    @staticmethod
    def to_key(cmd_names: Iterable[str]) -> str:
        if False:
            return 10
        return '_'.join([cmd.replace('-', '_').replace(' ', '_') for cmd in cmd_names])