"""This module provides ``kedro.config`` with the functionality to load one
or more configuration files of yaml or json type from specified paths through OmegaConf.
"""
from __future__ import annotations
import io
import logging
import mimetypes
from pathlib import Path
from typing import Any, Callable, Iterable
import fsspec
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError, UnsupportedInterpolationType
from omegaconf.resolvers import oc
from yaml.parser import ParserError
from yaml.scanner import ScannerError
from kedro.config.abstract_config import AbstractConfigLoader, MissingConfigException
_config_logger = logging.getLogger(__name__)
_NO_VALUE = object()

class OmegaConfigLoader(AbstractConfigLoader):
    """Recursively scan directories (config paths) contained in ``conf_source`` for
    configuration files with a ``yaml``, ``yml`` or ``json`` extension, load and merge
    them through ``OmegaConf`` (https://omegaconf.readthedocs.io/)
    and return them in the form of a config dictionary.

    The first processed config path is the ``base`` directory inside
    ``conf_source``. The optional ``env`` argument can be used to specify a
    subdirectory of ``conf_source`` to process as a config path after ``base``.

    When the same top-level key appears in any two config files located in
    the same (sub)directory, a ``ValueError`` is raised.

    When the same key appears in any two config files located in different
    (sub)directories, the last processed config path takes precedence
    and overrides this key and any sub-keys.

    You can access the different configurations as follows:
    ::

        >>> import logging.config
        >>> from kedro.config import OmegaConfigLoader
        >>> from kedro.framework.project import settings
        >>>
        >>> conf_path = str(project_path / settings.CONF_SOURCE)
        >>> conf_loader = OmegaConfigLoader(conf_source=conf_path, env="local")
        >>>
        >>> conf_logging = conf_loader["logging"]
        >>> logging.config.dictConfig(conf_logging)  # set logging conf
        >>>
        >>> conf_catalog = conf_loader["catalog"]
        >>> conf_params = conf_loader["parameters"]

    ``OmegaConf`` supports variable interpolation in configuration
    https://omegaconf.readthedocs.io/en/2.2_branch/usage.html#merging-configurations. It is
    recommended to use this instead of yaml anchors with the ``OmegaConfigLoader``.

    This version of the ``OmegaConfigLoader`` does not support any of the built-in ``OmegaConf``
    resolvers. Support for resolvers might be added in future versions.

    To use this class, change the setting for the `CONFIG_LOADER_CLASS` constant
    in `settings.py`.

    Example:
    ::

        >>> # in settings.py
        >>> from kedro.config import OmegaConfigLoader
        >>>
        >>> CONFIG_LOADER_CLASS = OmegaConfigLoader

    """

    def __init__(self, conf_source: str, env: str=None, runtime_params: dict[str, Any]=None, *, config_patterns: dict[str, list[str]]=None, base_env: str='base', default_run_env: str='local', custom_resolvers: dict[str, Callable]=None):
        if False:
            for i in range(10):
                print('nop')
        'Instantiates a ``OmegaConfigLoader``.\n\n        Args:\n            conf_source: Path to use as root directory for loading configuration.\n            env: Environment that will take precedence over base.\n            runtime_params: Extra parameters passed to a Kedro run.\n            config_patterns: Regex patterns that specify the naming convention for configuration\n                files so they can be loaded. Can be customised by supplying config_patterns as\n                in `CONFIG_LOADER_ARGS` in `settings.py`.\n            base_env: Name of the base environment. Defaults to `"base"`.\n                This is used in the `conf_paths` property method to construct\n                the configuration paths.\n            default_run_env: Name of the default run environment. Defaults to `"local"`.\n                Can be overridden by supplying the `env` argument.\n            custom_resolvers: A dictionary of custom resolvers to be registered. For more information,\n             see here: https://omegaconf.readthedocs.io/en/2.3_branch/custom_resolvers.html#custom-resolvers\n        '
        self.base_env = base_env
        self.default_run_env = default_run_env
        self.config_patterns = {'catalog': ['catalog*', 'catalog*/**', '**/catalog*'], 'parameters': ['parameters*', 'parameters*/**', '**/parameters*'], 'credentials': ['credentials*', 'credentials*/**', '**/credentials*'], 'logging': ['logging*', 'logging*/**', '**/logging*'], 'globals': ['globals.yml']}
        self.config_patterns.update(config_patterns or {})
        OmegaConf.clear_resolver('oc.env')
        if custom_resolvers:
            self._register_new_resolvers(custom_resolvers)
        self._register_globals_resolver()
        (file_mimetype, _) = mimetypes.guess_type(conf_source)
        if file_mimetype == 'application/x-tar':
            self._protocol = 'tar'
        elif file_mimetype in ('application/zip', 'application/x-zip-compressed', 'application/zip-compressed'):
            self._protocol = 'zip'
        else:
            self._protocol = 'file'
        self._fs = fsspec.filesystem(protocol=self._protocol, fo=conf_source)
        super().__init__(conf_source=conf_source, env=env, runtime_params=runtime_params)
        try:
            self._globals = self['globals']
        except MissingConfigException:
            self._globals = {}

    def __setitem__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        if key == 'globals':
            self._globals = value
        super().__setitem__(key, value)

    def __getitem__(self, key) -> dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        "Get configuration files by key, load and merge them, and\n        return them in the form of a config dictionary.\n\n        Args:\n            key: Key of the configuration type to fetch.\n\n        Raises:\n            KeyError: If key provided isn't present in the config_patterns of this\n               ``OmegaConfigLoader`` instance.\n            MissingConfigException: If no configuration files exist matching the patterns\n                mapped to the provided key.\n\n        Returns:\n            Dict[str, Any]:  A Python dictionary with the combined\n               configuration from all configuration files.\n        "
        self._register_runtime_params_resolver()
        if key in self:
            return super().__getitem__(key)
        if key not in self.config_patterns:
            raise KeyError(f"No config patterns were found for '{key}' in your config loader")
        patterns = [*self.config_patterns[key]]
        if key == 'globals':
            OmegaConf.clear_resolver('runtime_params')
        read_environment_variables = key == 'credentials'
        processed_files: set[Path] = set()
        if self._protocol == 'file':
            base_path = str(Path(self.conf_source) / self.base_env)
        else:
            base_path = str(Path(self._fs.ls('', detail=False)[-1]) / self.base_env)
        try:
            base_config = self.load_and_merge_dir_config(base_path, patterns, key, processed_files, read_environment_variables)
        except UnsupportedInterpolationType as exc:
            if 'runtime_params' in str(exc):
                raise UnsupportedInterpolationType('The `runtime_params:` resolver is not supported for globals.')
            else:
                raise exc
        config = base_config
        run_env = self.env or self.default_run_env
        if self._protocol == 'file':
            env_path = str(Path(self.conf_source) / run_env)
        else:
            env_path = str(Path(self._fs.ls('', detail=False)[-1]) / run_env)
        try:
            env_config = self.load_and_merge_dir_config(env_path, patterns, key, processed_files, read_environment_variables)
        except UnsupportedInterpolationType as exc:
            if 'runtime_params' in str(exc):
                raise UnsupportedInterpolationType('The `runtime_params:` resolver is not supported for globals.')
            else:
                raise exc
        common_keys = config.keys() & env_config.keys()
        if common_keys:
            sorted_keys = ', '.join(sorted(common_keys))
            msg = "Config from path '%s' will override the following existing top-level config keys: %s"
            _config_logger.debug(msg, env_path, sorted_keys)
        config.update(env_config)
        if not processed_files and key != 'globals':
            raise MissingConfigException(f'No files of YAML or JSON format found in {base_path} or {env_path} matching the glob pattern(s): {[*self.config_patterns[key]]}')
        return config

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'OmegaConfigLoader(conf_source={self.conf_source}, env={self.env}, config_patterns={self.config_patterns})'

    def load_and_merge_dir_config(self, conf_path: str, patterns: Iterable[str], key: str, processed_files: set, read_environment_variables: bool | None=False) -> dict[str, Any]:
        if False:
            i = 10
            return i + 15
        "Recursively load and merge all configuration files in a directory using OmegaConf,\n        which satisfy a given list of glob patterns from a specific path.\n\n        Args:\n            conf_path: Path to configuration directory.\n            patterns: List of glob patterns to match the filenames against.\n            key: Key of the configuration type to fetch.\n            processed_files: Set of files read for a given configuration type.\n            read_environment_variables: Whether to resolve environment variables.\n\n        Raises:\n            MissingConfigException: If configuration path doesn't exist or isn't valid.\n            ValueError: If two or more configuration files contain the same key(s).\n            ParserError: If config file contains invalid YAML or JSON syntax.\n\n        Returns:\n            Resulting configuration dictionary.\n\n        "
        if not self._fs.isdir(Path(conf_path).as_posix()):
            raise MissingConfigException(f'Given configuration path either does not exist or is not a valid directory: {conf_path}')
        paths = []
        for pattern in patterns:
            for each in self._fs.glob(Path(f'{str(conf_path)}/{pattern}').as_posix()):
                if not self._is_hidden(each):
                    paths.append(Path(each))
        deduplicated_paths = set(paths)
        config_files_filtered = [path for path in deduplicated_paths if self._is_valid_config_path(path)]
        config_per_file = {}
        for config_filepath in config_files_filtered:
            try:
                with self._fs.open(str(config_filepath.as_posix())) as open_config:
                    tmp_fo = io.StringIO(open_config.read().decode('utf8'))
                    config = OmegaConf.load(tmp_fo)
                    processed_files.add(config_filepath)
                if read_environment_variables:
                    self._resolve_environment_variables(config)
                config_per_file[config_filepath] = config
            except (ParserError, ScannerError) as exc:
                line = exc.problem_mark.line
                cursor = exc.problem_mark.column
                raise ParserError(f'Invalid YAML or JSON file {Path(conf_path, config_filepath.name).as_posix()}, unable to read line {line}, position {cursor}.') from exc
        seen_file_to_keys = {file: set(config.keys()) for (file, config) in config_per_file.items()}
        aggregate_config = config_per_file.values()
        self._check_duplicates(seen_file_to_keys)
        if not aggregate_config:
            return {}
        if key == 'parameters':
            return OmegaConf.to_container(OmegaConf.merge(*aggregate_config, self.runtime_params), resolve=True)
        return {k: v for (k, v) in OmegaConf.to_container(OmegaConf.merge(*aggregate_config), resolve=True).items() if not k.startswith('_')}

    def _is_valid_config_path(self, path):
        if False:
            print('Hello World!')
        'Check if given path is a file path and file type is yaml or json.'
        posix_path = path.as_posix()
        return self._fs.isfile(str(posix_path)) and path.suffix in ['.yml', '.yaml', '.json']

    def _register_globals_resolver(self):
        if False:
            for i in range(10):
                print('nop')
        'Register the globals resolver'
        OmegaConf.register_new_resolver('globals', self._get_globals_value, replace=True)

    def _register_runtime_params_resolver(self):
        if False:
            return 10
        OmegaConf.register_new_resolver('runtime_params', self._get_runtime_value, replace=True)

    def _get_globals_value(self, variable, default_value=_NO_VALUE):
        if False:
            while True:
                i = 10
        'Return the globals values to the resolver'
        if variable.startswith('_'):
            raise InterpolationResolutionError("Keys starting with '_' are not supported for globals.")
        globals_oc = OmegaConf.create(self._globals)
        interpolated_value = OmegaConf.select(globals_oc, variable, default=default_value)
        if interpolated_value != _NO_VALUE:
            return interpolated_value
        else:
            raise InterpolationResolutionError(f"Globals key '{variable}' not found and no default value provided.")

    def _get_runtime_value(self, variable, default_value=_NO_VALUE):
        if False:
            i = 10
            return i + 15
        'Return the runtime params values to the resolver'
        runtime_oc = OmegaConf.create(self.runtime_params)
        interpolated_value = OmegaConf.select(runtime_oc, variable, default=default_value)
        if interpolated_value != _NO_VALUE:
            return interpolated_value
        else:
            raise InterpolationResolutionError(f"Runtime parameter '{variable}' not found and no default value provided.")

    @staticmethod
    def _register_new_resolvers(resolvers: dict[str, Callable]):
        if False:
            for i in range(10):
                print('nop')
        'Register custom resolvers'
        for (name, resolver) in resolvers.items():
            if not OmegaConf.has_resolver(name):
                msg = f'Registering new custom resolver: {name}'
                _config_logger.debug(msg)
                OmegaConf.register_new_resolver(name=name, resolver=resolver)

    @staticmethod
    def _check_duplicates(seen_files_to_keys: dict[Path, set[Any]]):
        if False:
            for i in range(10):
                print('nop')
        duplicates = []
        filepaths = list(seen_files_to_keys.keys())
        for (i, filepath1) in enumerate(filepaths, 1):
            config1 = seen_files_to_keys[filepath1]
            for filepath2 in filepaths[i:]:
                config2 = seen_files_to_keys[filepath2]
                combined_keys = config1 & config2
                overlapping_keys = {key for key in combined_keys if not key.startswith('_')}
                if overlapping_keys:
                    sorted_keys = ', '.join(sorted(overlapping_keys))
                    if len(sorted_keys) > 100:
                        sorted_keys = sorted_keys[:100] + '...'
                    duplicates.append(f'Duplicate keys found in {filepath1} and {filepath2}: {sorted_keys}')
        if duplicates:
            dup_str = '\n'.join(duplicates)
            raise ValueError(f'{dup_str}')

    @staticmethod
    def _resolve_environment_variables(config: dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        'Use the ``oc.env`` resolver to read environment variables and replace\n        them in-place, clearing the resolver after the operation is complete if\n        it was not registered beforehand.\n\n        Arguments:\n            config {Dict[str, Any]} -- The configuration dictionary to resolve.\n        '
        if not OmegaConf.has_resolver('oc.env'):
            OmegaConf.register_new_resolver('oc.env', oc.env)
            OmegaConf.resolve(config)
            OmegaConf.clear_resolver('oc.env')
        else:
            OmegaConf.resolve(config)

    def _is_hidden(self, path: str):
        if False:
            return 10
        'Check if path contains any hidden directory or is a hidden file'
        path = Path(path)
        conf_path = Path(self.conf_source).resolve().as_posix()
        if self._protocol == 'file':
            path = path.resolve()
        path = path.as_posix()
        if path.startswith(conf_path):
            path = path.replace(conf_path, '')
        parts = path.split(self._fs.sep)
        HIDDEN = '.'
        return any((part.startswith(HIDDEN) for part in parts))