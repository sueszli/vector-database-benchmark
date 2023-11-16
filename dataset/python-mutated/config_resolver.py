import json
import os
import re
import time
from collections import OrderedDict
from enum import auto
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from urllib.parse import urlencode
from urllib.parse import urlparse
from urllib.parse import urlsplit
import requests
import ruamel.yaml
from rich import progress
from ruamel.yaml import YAMLError
import semgrep.semgrep_interfaces.semgrep_output_v1 as out
from semgrep import __VERSION__
from semgrep.app import auth
from semgrep.console import console
from semgrep.constants import CLI_RULE_ID
from semgrep.constants import Colors
from semgrep.constants import DEFAULT_SEMGREP_APP_CONFIG_URL
from semgrep.constants import DEFAULT_SEMGREP_CONFIG_NAME
from semgrep.constants import ID_KEY
from semgrep.constants import MISSED_KEY
from semgrep.constants import PLEASE_FILE_ISSUE_TEXT
from semgrep.constants import RULES_KEY
from semgrep.error import INVALID_API_KEY_EXIT_CODE
from semgrep.error import InvalidRuleSchemaError
from semgrep.error import SemgrepError
from semgrep.error import UNPARSEABLE_YAML_EXIT_CODE
from semgrep.rule import Rule
from semgrep.rule import rule_without_metadata
from semgrep.rule_lang import EmptySpan
from semgrep.rule_lang import EmptyYamlException
from semgrep.rule_lang import parse_config_preserve_spans
from semgrep.rule_lang import Span
from semgrep.rule_lang import validate_yaml
from semgrep.rule_lang import YamlMap
from semgrep.rule_lang import YamlTree
from semgrep.state import get_state
from semgrep.util import is_config_suffix
from semgrep.util import is_rules
from semgrep.util import is_url
from semgrep.util import with_color
from semgrep.verbose_logging import getLogger
logger = getLogger(__name__)
AUTO_CONFIG_KEY = 'auto'
AUTO_CONFIG_LOCATION = 'c/auto'
MISSING_RULE_ID = 'no-rule-id'
DEFAULT_CONFIG = {'rules': [{'id': 'eqeq-is-bad', 'pattern': '$X == $X', 'message': '$X == $X is a useless equality check', 'languages': ['python'], 'severity': out.Error().to_json()}]}
CLOUD_PLATFORM_CONFIG_ID = 'semgrep-app-rules'
REGISTRY_CONFIG_ID = 'remote-registry'
NON_REGISTRY_REMOTE_CONFIG_ID = 'remote-url'

class ConfigFile(NamedTuple):
    config_id: Optional[str]
    contents: str
    config_path: str

class ConfigType(Enum):
    REGISTRY = auto()
    SEMGREP_CLOUD_PLATFORM = auto()
    LOCAL = auto()

class ConfigLoader:
    _origin = ConfigType.LOCAL
    _config_path = ''
    _project_url = None

    def __init__(self, config_str: str, project_url: Optional[str]=None, config_str_for_jsonnet: Optional[str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Mutates Metrics state!\n        Takes a user's inputted config_str and transforms it into the appropriate\n        path, checking whether the config string is a registry url or not. If it\n        is, also set the appropriate Metrics flag\n        "
        state = get_state()
        self._project_url = project_url
        self._origin = ConfigType.REGISTRY
        self._supports_fallback_config = False
        if config_str == 'r2c':
            state.metrics.add_feature('config', 'r2c')
            self._config_path = 'https://semgrep.dev/c/p/r2c'
        elif is_url(config_str):
            state.metrics.add_feature('config', 'url')
            self._config_path = config_str
        elif is_product_names(config_str):
            self._origin = ConfigType.SEMGREP_CLOUD_PLATFORM
            add_metrics_for_products(config_str)
            self._config_path = config_str
            self._supports_fallback_config = True
        elif is_registry_id(config_str):
            state.metrics.add_feature('config', f'registry:prefix-{config_str[0]}')
            self._config_path = registry_id_to_url(config_str)
        elif config_str == AUTO_CONFIG_KEY:
            state.metrics.add_feature('config', 'auto')
            self._config_path = f'{state.env.semgrep_url}/{AUTO_CONFIG_LOCATION}'
        else:
            state.metrics.add_feature('config', 'local')
            self._origin = ConfigType.LOCAL
            config_str = config_str_for_jsonnet if config_str_for_jsonnet else config_str
            self._config_path = str(Path(config_str).expanduser())
        if self.is_registry_url():
            state.metrics.is_using_registry = True
            state.metrics.add_registry_url(self._config_path)

    def load_config(self) -> List[ConfigFile]:
        if False:
            i = 10
            return i + 15
        "\n        Loads a config based on self's state.\n        A config path produces a list of ConfigFiles because\n        it may be a path to a folders of configs, each of\n        which produces a file\n        "
        if self._origin == ConfigType.REGISTRY:
            return [self._download_config()]
        elif self._origin == ConfigType.SEMGREP_CLOUD_PLATFORM:
            return [self._fetch_semgrep_cloud_platform_scan_config()]
        else:
            return self._load_config_from_local_path()

    def _nice_semgrep_url(self, url: str) -> str:
        if False:
            print('Hello World!')
        "\n        Alters semgrep.dev urls to let user\n        click through to the nice display page instead\n        of raw YAML.\n        Replaces '/c/' in semgrep urls with '/'.\n        "
        parsed = urlparse(url)
        if 'semgrep.dev' in parsed.netloc and parsed.path.startswith('/c'):
            return url.replace('/c/', '/')
        return url

    def _download_config(self) -> ConfigFile:
        if False:
            i = 10
            return i + 15
        '\n        Download a configuration from semgrep.dev\n        '
        try:
            return self._download_config_from_url(self._config_path)
        except Exception:
            if self._supports_fallback_config:
                try:
                    fallback_url = re.sub('^[^?]*', f'{get_state().env.fail_open_url}/config', self._config_path)
                    return self._download_config_from_url(fallback_url)
                except Exception:
                    pass
            raise

    def _download_config_from_url(self, url: str) -> ConfigFile:
        if False:
            i = 10
            return i + 15
        app_session = get_state().app_session
        logger.debug('Downloading config from %s', url)
        error = f'Failed to download configuration from {url}'
        try:
            resp = app_session.get(url, headers={'Accept': 'application/json'})
            if resp.status_code == requests.codes.ok:
                try:
                    rule_config = resp.json()['rule_config']
                    config_str = rule_config if isinstance(rule_config, str) else json.dumps(rule_config)
                    return ConfigFile(None, config_str, url)
                except Exception as ex:
                    logger.debug('Failed to decode JSON: %s', repr(ex))
                    return ConfigFile(None, resp.content.decode('utf-8', errors='replace'), url)
                finally:
                    logger.debug(f'Downloaded config from %s', url)
            error += f' HTTP {resp.status_code}.'
        except requests.exceptions.RetryError as ex:
            error += f' Failed after multiple attempts ({ex.args[0].reason})'
        logger.debug(error)
        raise SemgrepError(error)

    def _load_config_from_local_path(self) -> List[ConfigFile]:
        if False:
            i = 10
            return i + 15
        '\n        Return config file(s) as dictionary object\n        '
        loc = Path(self._config_path)
        logger.debug(f'Loading local config from {loc}')
        if loc.exists():
            if loc.is_file():
                config = [read_config_at_path(loc)]
            elif loc.is_dir():
                config = read_config_folder(loc)
            else:
                raise SemgrepError(f'config location `{loc}` is not a file or folder!')
        else:
            env = get_state().env
            addendum = ''
            if env.in_docker:
                addendum = ' (since you are running in docker, you cannot specify arbitrary paths on the host; they must be mounted into the container)'
            raise SemgrepError(f'WARNING: unable to find a config; path `{loc}` does not exist{addendum}')
        logger.debug(f'Done loading local config from {loc}')
        return config

    def is_registry_url(self) -> bool:
        if False:
            return 10
        return self._origin == ConfigType.REGISTRY

    def _project_metadata_for_standalone_scan(self, require_repo_name: bool) -> out.ProjectMetadata:
        if False:
            print('Hello World!')
        repo_name = os.environ.get('SEMGREP_REPO_NAME')
        if repo_name is None:
            if require_repo_name:
                raise SemgrepError(f'Need to set env var SEMGREP_REPO_NAME to use `--config {self._config_path}`')
            else:
                repo_name = 'unknown'
        return out.ProjectMetadata(semgrep_version=out.Version(__VERSION__), scan_environment='semgrep-scan', repository=repo_name, repo_url=None, branch=None, commit=None, commit_title=None, commit_author_email=None, commit_author_name=None, commit_author_username=None, commit_author_image_url=None, ci_job_url=None, on='unknown', pull_request_author_username=None, pull_request_author_image_url=None, pull_request_id=None, pull_request_title=None, is_full_scan=True)

    def _fetch_semgrep_cloud_platform_scan_config(self) -> ConfigFile:
        if False:
            return 10
        '\n        Download a configuration from semgrep.dev using new /api/cli/scans endpoint\n        '
        state = get_state()
        products = [out.Product.from_json(PRODUCT_NAMES[p]) for p in self._config_path.split(',')]
        require_repo_name = any((p.value in [out.SAST(), out.Secrets()] for p in products))
        request = out.ScanRequest(meta=out.RawJson({}), scan_metadata=out.ScanMetadata(cli_version=out.Version(__VERSION__), unique_id=out.Uuid(str(state.request_id)), requested_products=products, dry_run=True), project_metadata=self._project_metadata_for_standalone_scan(require_repo_name))
        try:
            return self._download_semgrep_cloud_platform_scan_config(request)
        except Exception:
            if self._supports_fallback_config:
                try:
                    return self._download_semgrep_cloud_platform_fallback_scan_config()
                except Exception:
                    pass
            raise

    def _download_semgrep_cloud_platform_scan_config(self, request: out.ScanRequest) -> ConfigFile:
        if False:
            print('Hello World!')
        state = get_state()
        url = f'{state.env.semgrep_url}/api/cli/scans'
        logger.debug('Downloading config from %s', url)
        error = f'Failed to download configuration from {url}'
        try:
            response = state.app_session.post(f'{state.env.semgrep_url}/api/cli/scans', json=request.to_json())
            if response.status_code == requests.codes.unauthorized:
                raise SemgrepError('Invalid API Key. Run `semgrep logout` and `semgrep login` again.', code=INVALID_API_KEY_EXIT_CODE)
            try:
                response.raise_for_status()
            except requests.RequestException:
                raise Exception(f'API server at {state.env.semgrep_url} returned this error: {response.text}')
            scan_response = out.ScanResponse.from_json(response.json())
            return ConfigFile(None, scan_response.config.rules.to_json_string(), url)
        except requests.exceptions.RetryError as ex:
            error += f' Failed after multiple attempts ({ex.args[0].reason})'
            logger.debug(error)
            raise SemgrepError(error)

    def _download_semgrep_cloud_platform_fallback_scan_config(self) -> ConfigFile:
        if False:
            while True:
                i = 10
        '\n        This function decides what fallback url to call if the semgrep cloud platform\n        scan config endpoint fails\n\n        ! This will manually rebuild the url until we have a better solution\n        '
        fallback_url = None
        if is_code(self._config_path):
            fallback_url = url_for_code()
        elif is_supply_chain(self._config_path):
            fallback_url = url_for_supply_chain()
        elif is_secrets(self._config_path):
            fallback_url = url_for_secrets()
        elif is_policy_id(self._config_path):
            fallback_url = url_for_policy()
        else:
            raise
        fallback_url = re.sub('^[^?]*', f'{get_state().env.fail_open_url}/config', fallback_url)
        return self._download_config_from_url(fallback_url)

def read_config_at_path(loc: Path, base_path: Optional[Path]=None) -> ConfigFile:
    if False:
        print('Hello World!')
    '\n    Assumes file at loc exists\n    '
    config_id = str(loc)
    if base_path:
        config_id = str(loc).replace(str(base_path), '')
    return ConfigFile(config_id, loc.read_text(), str(loc))

def read_config_folder(loc: Path, relative: bool=False) -> List[ConfigFile]:
    if False:
        return 10
    configs = []
    for l in loc.rglob('*'):
        correct_suffix = is_config_suffix(l)
        if not _is_hidden_config(l.relative_to(loc)) and correct_suffix:
            if l.is_file():
                configs.append(read_config_at_path(l, loc if relative else None))
    return configs

def parse_config_files(loaded_config_infos: List[ConfigFile]) -> Dict[str, YamlTree]:
    if False:
        return 10
    '\n    Parse a list of config files into rules\n    This assumes that config_id is set for local rules\n    but is None for registry rules\n    '
    config = {}
    for (config_id, contents, config_path) in progress.track(loaded_config_infos, description=f'  parsing {len(loaded_config_infos)} rules', transient=True, disable=len(loaded_config_infos) < 500, console=console):
        try:
            if not config_id:
                try:
                    remote_rule_netloc = urlsplit(config_path).netloc
                except ValueError:
                    remote_rule_netloc = 'invalid-url'
                config_id = REGISTRY_CONFIG_ID if is_url(config_path) and (remote_rule_netloc.endswith('.semgrep.dev') or remote_rule_netloc == 'semgrep.dev') else NON_REGISTRY_REMOTE_CONFIG_ID
                filename = f'{config_path[:20]}...'
            else:
                filename = config_path
            config.update(parse_config_string(config_id, contents, filename))
        except InvalidRuleSchemaError as e:
            if config_id == REGISTRY_CONFIG_ID or config_id == NON_REGISTRY_REMOTE_CONFIG_ID:
                notice = f'\nRules downloaded from {config_path} failed to parse.\nThis is likely because rules have been added that use functionality introduced in later versions of semgrep.\nPlease upgrade to latest version of semgrep (see https://semgrep.dev/docs/upgrading/) and try again.\n'
                notice_color = with_color(Colors.red, notice, bold=True)
                logger.error(notice_color)
                raise e
            else:
                raise e
    return config

def resolve_config(config_str: str, project_url: Optional[str]=None) -> Dict[str, YamlTree]:
    if False:
        for i in range(10):
            print('nop')
    'resolves if config arg is a registry entry, a url, or a file, folder, or loads from defaults if None'
    start_t = time.time()
    config_loader = ConfigLoader(config_str, project_url)
    config = parse_config_files(config_loader.load_config())
    if config:
        logger.debug(f'loaded {len(config)} configs in {time.time() - start_t}')
    return config

class Config:

    def __init__(self, valid_configs: Mapping[str, Sequence[Rule]], *, with_code_rules: bool=False, with_supply_chain: bool=False, with_secrets: bool=False, missed_rule_count: int=0) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Handles parsing and validating of config files\n        and exposes ability to get all rules in parsed config files\n        '
        self.valid = valid_configs
        self.with_code_rules = with_code_rules
        self.with_supply_chain = with_supply_chain
        self.with_secrets = with_secrets
        self.missed_rule_count = missed_rule_count

    @classmethod
    def from_pattern_lang(cls, pattern: str, lang: str, replacement: Optional[str]=None) -> Tuple['Config', Sequence[SemgrepError]]:
        if False:
            print('Hello World!')
        config_dict = manual_config(pattern, lang, replacement)
        (valid, errors, _) = cls._validate(config_dict)
        return (cls(valid), errors)

    @classmethod
    def from_rules_yaml(cls, config: str) -> Tuple['Config', Sequence[SemgrepError]]:
        if False:
            while True:
                i = 10
        config_dict: Dict[str, YamlTree] = {}
        errors: List[SemgrepError] = []
        try:
            resolved_config_key = CLOUD_PLATFORM_CONFIG_ID
            config_dict.update(parse_config_string(resolved_config_key, config, filename=None))
        except SemgrepError as e:
            errors.append(e)
        (valid, parse_errors, _) = cls._validate(config_dict)
        errors.extend(parse_errors)
        return (cls(valid), errors)

    @classmethod
    def from_config_list(cls, configs: Sequence[str], project_url: Optional[str]) -> Tuple['Config', Sequence[SemgrepError]]:
        if False:
            return 10
        '\n        Takes in list of files/directories and returns Config object as well as\n        list of errors parsing said config files\n\n        If empty list is passed, tries to read config file at default locations\n        '
        config_dict: Dict[str, YamlTree] = {}
        errors: List[SemgrepError] = []
        with_supply_chain = False
        with_code_rules = False
        with_secrets = False
        for (i, config) in enumerate(configs):
            try:
                resolved_config = resolve_config(config, project_url)
                if not resolved_config:
                    logger.verbose(f'Could not resolve config for {config}. Skipping.')
                    continue
                with_code_rules = with_code_rules or not is_supply_chain(config)
                with_supply_chain = with_supply_chain or is_supply_chain(config)
                with_secrets = with_secrets or is_secrets(config)
                for (resolved_config_key, resolved_config_yaml_tree) in resolved_config.items():
                    patched_resolved_config: Dict[str, YamlTree] = {}
                    patched_resolved_config[f'{resolved_config_key}_{i}'] = resolved_config_yaml_tree
                    config_dict.update(patched_resolved_config)
            except SemgrepError as e:
                errors.append(e)
        (valid, parse_errors, missed_rule_count) = cls._validate(config_dict)
        errors.extend(parse_errors)
        return (cls(valid, with_code_rules=with_code_rules, with_supply_chain=with_supply_chain, with_secrets=with_secrets, missed_rule_count=missed_rule_count), errors)

    def get_rules(self, no_rewrite_rule_ids: bool) -> List[Rule]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return list of rules\n\n        If no_rewrite_rule_ids is True will not add\n        path to config file to start of rule_ids\n        '
        configs = self.valid
        if not no_rewrite_rule_ids:
            self._rename_rule_ids(configs)
        return list(OrderedDict(((rule_without_metadata(rule), rule) for rules in configs.values() for rule in rules)).values())

    @staticmethod
    def _safe_relative_to(a: Path, b: Path) -> Path:
        if False:
            i = 10
            return i + 15
        try:
            return a.relative_to(b)
        except ValueError:
            return a

    @staticmethod
    def _sanitize_rule_id_fragment(s: str) -> str:
        if False:
            while True:
                i = 10
        "Make a valid fragment for a rule ID.\n\n        This removes characters that aren't allowed in Semgrep rule IDs.\n        The transformation is irreversible. The result may be an empty\n        string.\n\n        Rule ID format: [a-zA-Z0-9._-]*\n        "
        return re.sub('[^a-zA-Z0-9._-]', '', s)

    @staticmethod
    def _convert_config_id_to_prefix(config_id: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        at_path = Path(config_id)
        try:
            at_path = Config._safe_relative_to(at_path, Path.cwd())
        except FileNotFoundError:
            pass
        prefix = '.'.join(at_path.parts[:-1]).lstrip('./').lstrip('.')
        if len(prefix):
            prefix += '.'
        prefix = Config._sanitize_rule_id_fragment(prefix)
        return prefix

    @staticmethod
    def _rename_rule_ids(valid_configs: Mapping[str, Sequence[Rule]]) -> None:
        if False:
            i = 10
            return i + 15
        for (config_id, rules) in valid_configs.items():
            for rule in rules:
                rule.rename_id(f'{Config._convert_config_id_to_prefix(config_id)}{rule.id or MISSING_RULE_ID}')

    @staticmethod
    def _validate(config_dict: Mapping[str, YamlTree]) -> Tuple[Mapping[str, Sequence[Rule]], Sequence[SemgrepError], int]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Take configs and separate into valid and list of errors parsing the invalid ones\n        '
        errors: List[SemgrepError] = []
        valid: Dict[str, Any] = {}
        missed_rule_count = 0
        for (config_id, config_yaml_tree) in config_dict.items():
            config: YamlMap = config_yaml_tree.value
            if not isinstance(config, YamlMap):
                errors.append(SemgrepError(f'{config_id} was not a mapping'))
                continue
            missed_rule_container = config.get(MISSED_KEY)
            missed_rule_count += int(missed_rule_container.value) if missed_rule_container else 0
            rules = config.get(RULES_KEY)
            if rules is None:
                errors.append(InvalidRuleSchemaError(short_msg='missing keys', long_msg=f'{config_id} is missing `{RULES_KEY}` as top-level key', spans=[config_yaml_tree.span.truncate(lines=5)]))
                continue
            valid_rules = []
            for rule_dict in rules.value:
                try:
                    rule = validate_single_rule(config_id, rule_dict)
                except InvalidRuleSchemaError as ex:
                    errors.append(ex)
                else:
                    if isinstance(rule.product.value, out.Secrets) and (not config_id.startswith(REGISTRY_CONFIG_ID)) and (not config_id.startswith(CLOUD_PLATFORM_CONFIG_ID)):
                        rule.metadata.get('semgrep.dev', {}).get('rule', {})['origin'] = 'local'
                    valid_rules.append(rule)
            if valid_rules:
                valid[config_id] = valid_rules
        return (valid, errors, missed_rule_count)

def validate_single_rule(config_id: str, rule_yaml: YamlTree[YamlMap]) -> Rule:
    if False:
        return 10
    '\n    Validate that a rule dictionary contains all necessary keys\n    and can be correctly parsed.\n    '
    rule: YamlMap = rule_yaml.value
    return Rule.from_yamltree(rule_yaml)

def manual_config(pattern: str, lang: str, replacement: Optional[str]) -> Dict[str, YamlTree]:
    if False:
        for i in range(10):
            print('nop')
    'Create a fake rule when we only have a pattern and language\n\n    This is used when someone calls `semgrep scan -e print -l py`\n    '
    pattern_span = Span.from_string(pattern, filename='CLI Input')
    pattern_tree = YamlTree[str](value=pattern, span=pattern_span)
    error_span = Span.from_string(f'Semgrep bug generating manual config {PLEASE_FILE_ISSUE_TEXT}', filename=None)
    rules_key = {ID_KEY: CLI_RULE_ID, 'pattern': pattern_tree, 'message': pattern, 'languages': [lang], 'severity': out.Error().to_json()}
    if replacement:
        rules_key['fix'] = replacement
    return {'manual': YamlTree.wrap({RULES_KEY: [rules_key]}, span=error_span)}

def adjust_for_docker() -> None:
    if False:
        i = 10
        return i + 15
    'change into this folder so that all paths are relative to it'
    env = get_state().env
    if env.in_docker and (not env.in_gh_action):
        try:
            next(env.src_directory.iterdir())
        except (NotADirectoryError, StopIteration):
            raise SemgrepError(f"""Detected Docker environment without a code volume, please include '-v "${{PWD}}:{env.src_directory}"'""")
        else:
            os.chdir(env.src_directory)

def indent(msg: str) -> str:
    if False:
        return 10
    return '\n'.join(['\t' + line for line in msg.splitlines()])

def import_callback(base: str, path: str) -> Tuple[str, bytes]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Instructions to jsonnet for how to resolve\n    import expressions (`local $NAME = $PATH`).\n    The base is the directory of the file and the\n    path is $PATH in the local expression. We will\n    later pass this function to jsonnet, which will\n    use it when resolving imports. By implementing\n    this callback, we support yaml files (jsonnet\n    can otherwise only build against json files)\n    and config specifiers like `p/python`. We also\n    support a library path\n    '
    library_path = os.environ.get('R2C_INTERNAL_JSONNET_LIB')
    if library_path and (not os.path.isabs(library_path)):
        library_path = os.path.join(os.curdir, library_path)
    if library_path and os.path.exists(os.path.join(library_path, path)):
        final_path = os.path.join(library_path, path)
    else:
        final_path = os.path.join(base, path)
    logger.debug(f'import_callback for {path}, base = {base}, final = {final_path}')
    if final_path and (final_path.split('.')[-1] == 'yml' or final_path.split('.')[-1] == 'yaml'):
        logger.debug(f'loading yaml file {final_path}, converting to JSON on the fly')
        yaml = ruamel.yaml.YAML(typ='safe')
        with open(final_path) as fpi:
            data = yaml.load(fpi)
        contents = json.dumps(data)
        filename = final_path
        return (filename, contents.encode())
    logger.debug(f'defaulting to the config resolver for {path}')
    config_infos = ConfigLoader(path, None, final_path).load_config()
    if len(config_infos) == 0:
        raise SemgrepError(f'No valid configs imported')
    elif len(config_infos) > 1:
        raise SemgrepError(f'Currently configs cannot be imported from a directory')
    else:
        (_config_id, contents, config_path) = config_infos[0]
        return (config_path, contents.encode())

def parse_config_string(config_id: str, contents: str, filename: Optional[str]) -> Dict[str, YamlTree]:
    if False:
        return 10
    if not contents:
        raise SemgrepError(f'Empty configuration file {filename}', code=UNPARSEABLE_YAML_EXIT_CODE)
    if filename and filename.split('.')[-1] == 'jsonnet':
        logger.error('Support for Jsonnet rules is experimental and currently meant for internal use only. The syntax may change or be removed at any point.')
        try:
            import _jsonnet
        except ImportError:
            logger.error('Running jsonnet rules requires the python jsonnet library. Please run `pip install jsonnet` and try again.')
        contents = _jsonnet.evaluate_snippet(filename, contents, import_callback=import_callback)
    try:
        data = YamlTree.wrap(json.loads(contents), EmptySpan)
        validate_yaml(data)
        return {config_id: data}
    except json.decoder.JSONDecodeError:
        pass
    try:
        data = parse_config_preserve_spans(contents, filename)
    except EmptyYamlException:
        raise SemgrepError(f'Empty configuration file {filename}', code=UNPARSEABLE_YAML_EXIT_CODE)
    except YAMLError as se:
        raise SemgrepError(f'Invalid YAML file {config_id}:\n{indent(str(se))}', code=UNPARSEABLE_YAML_EXIT_CODE)
    return {config_id: data}

def _is_hidden_config(loc: Path) -> bool:
    if False:
        while True:
            i = 10
    '\n    Want to keep rules/.semgrep.yml but not path/.github/foo.yml\n    Also want to keep src/.semgrep/bad_pattern.yml but not ./.pre-commit-config.yaml\n    '
    return any((part != os.curdir and part != os.pardir and part.startswith('.') and (DEFAULT_SEMGREP_CONFIG_NAME not in part) for part in loc.parts))

def is_registry_id(config_str: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    '\n    Starts with r/, p/, s/ for registry, pack, and snippet respectively\n    '
    return config_str[:2] in {'r/', 'p/', 's/'}

def registry_id_to_url(registry_id: str) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Convert from registry_id to semgrep.dev url\n    '
    env = get_state().env
    return f'{env.semgrep_url}/{registry_id}'

def url_for_policy() -> str:
    if False:
        while True:
            i = 10
    '\n    Return url to download a policy for a given repo_name\n\n    For now uses envvar to know what repo_name is\n    '
    deployment_id = auth.get_deployment_id()
    if deployment_id is None:
        raise SemgrepError('Invalid API Key. Run `semgrep logout` and `semgrep login` again.')
    repo_name = os.environ.get('SEMGREP_REPO_NAME')
    if repo_name is None:
        raise SemgrepError('Need to set env var SEMGREP_REPO_NAME to use `--config policy`')
    env = get_state().env
    params = {'sca': False, 'is_secrets_scan': False, 'dry_run': True, 'full_scan': True, 'repo_name': repo_name, 'semgrep_version': __VERSION__}
    params_str = urlencode(params)
    return f'{env.semgrep_url}/{DEFAULT_SEMGREP_APP_CONFIG_URL}?{params_str}'
PRODUCT_NAMES = {'code': 'sast', 'policy': 'sast', 'secrets': 'secrets', 'supply-chain': 'sca'}

def is_product_names(config_str: str) -> bool:
    if False:
        return 10
    allowed = set(PRODUCT_NAMES.keys())
    names = set(config_str.split(','))
    return names <= allowed

def add_metrics_for_products(config_str: str) -> None:
    if False:
        return 10
    state = get_state()
    for product_name in config_str.split(','):
        if is_policy_id(product_name):
            state.metrics.add_feature('config', 'policy')
        else:
            state.metrics.add_feature('config', PRODUCT_NAMES[product_name])

def is_policy_id(config_str: str) -> bool:
    if False:
        print('Hello World!')
    return config_str == 'policy'

def legacy_url_for_scan(extra_params: Optional[dict]=None) -> str:
    if False:
        print('Hello World!')
    '\n    Generates a legacy scan url (api/agent/deployments/scans/config) to\n    fetch a scan configuration.\n    '
    env = get_state().env
    params = {'dry_run': True, 'full_scan': True, 'semgrep_version': __VERSION__}
    if extra_params:
        params.update(extra_params)
    if 'SEMGREP_REPO_NAME' in os.environ:
        params['repo_name'] = os.environ.get('SEMGREP_REPO_NAME')
    params_str = urlencode(params)
    return f'{env.semgrep_url}/{DEFAULT_SEMGREP_APP_CONFIG_URL}?{params_str}'

def url_for_code() -> str:
    if False:
        for i in range(10):
            print('nop')
    return legacy_url_for_scan()

def url_for_supply_chain() -> str:
    if False:
        for i in range(10):
            print('nop')
    return legacy_url_for_scan({'sca': True})

def url_for_secrets() -> str:
    if False:
        for i in range(10):
            print('nop')
    return legacy_url_for_scan({'is_secrets_scan': True})

def is_code(config_str: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return config_str == 'code'

def is_supply_chain(config_str: str) -> bool:
    if False:
        i = 10
        return i + 15
    return config_str == 'supply-chain'

def is_secrets(config_str: str) -> bool:
    if False:
        return 10
    return config_str == 'secrets'

def is_pack_id(config_str: str) -> bool:
    if False:
        while True:
            i = 10
    return config_str[:2] == 'p/'

def get_config(pattern: Optional[str], lang: Optional[str], config_strs: Sequence[str], *, project_url: Optional[str], replacement: Optional[str]=None) -> Tuple[Config, Sequence[SemgrepError]]:
    if False:
        return 10
    if pattern:
        if not lang:
            raise SemgrepError('language must be specified when a pattern is passed')
        (config, errors) = Config.from_pattern_lang(pattern, lang, replacement)
    elif len(config_strs) == 1 and is_rules(config_strs[0]):
        (config, errors) = Config.from_rules_yaml(config_strs[0])
    elif replacement:
        raise SemgrepError('command-line replacement flag can only be used with command-line pattern; when using a config file add the fix: key instead')
    else:
        (config, errors) = Config.from_config_list(config_strs, project_url)
    return (config, errors)