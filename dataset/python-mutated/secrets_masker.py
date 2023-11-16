"""Mask sensitive information from logs."""
from __future__ import annotations
import collections.abc
import logging
import sys
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, Iterable, Iterator, List, Pattern, TextIO, Tuple, TypeVar, Union
import re2
from airflow import settings
from airflow.compat.functools import cache
if TYPE_CHECKING:
    from kubernetes.client import V1EnvVar
    from airflow.typing_compat import TypeGuard
Redactable = TypeVar('Redactable', str, 'V1EnvVar', Dict[Any, Any], Tuple[Any, ...], List[Any])
Redacted = Union[Redactable, str]
log = logging.getLogger(__name__)
DEFAULT_SENSITIVE_FIELDS = frozenset({'access_token', 'api_key', 'apikey', 'authorization', 'passphrase', 'passwd', 'password', 'private_key', 'secret', 'token', 'keyfile_dict', 'service_account'})
'Names of fields (Connection extra, Variable key name etc.) that are deemed sensitive'
SECRETS_TO_SKIP_MASKING_FOR_TESTS = {'airflow'}

@cache
def get_sensitive_variables_fields():
    if False:
        for i in range(10):
            print('nop')
    'Get comma-separated sensitive Variable Fields from airflow.cfg.'
    from airflow.configuration import conf
    sensitive_fields = DEFAULT_SENSITIVE_FIELDS.copy()
    sensitive_variable_fields = conf.get('core', 'sensitive_var_conn_names')
    if sensitive_variable_fields:
        sensitive_fields |= frozenset({field.strip() for field in sensitive_variable_fields.split(',')})
    return sensitive_fields

def should_hide_value_for_key(name):
    if False:
        print('Hello World!')
    '\n    Return if the value for this given name should be hidden.\n\n    Name might be a Variable name, or key in conn.extra_dejson, for example.\n    '
    from airflow import settings
    if isinstance(name, str) and settings.HIDE_SENSITIVE_VAR_CONN_FIELDS:
        name = name.strip().lower()
        return any((s in name for s in get_sensitive_variables_fields()))
    return False

def mask_secret(secret: str | dict | Iterable, name: str | None=None) -> None:
    if False:
        while True:
            i = 10
    '\n    Mask a secret from appearing in the task logs.\n\n    If ``name`` is provided, then it will only be masked if the name matches\n    one of the configured "sensitive" names.\n\n    If ``secret`` is a dict or a iterable (excluding str) then it will be\n    recursively walked and keys with sensitive names will be hidden.\n    '
    if not secret:
        return
    _secrets_masker().add_mask(secret, name)

def redact(value: Redactable, name: str | None=None, max_depth: int | None=None) -> Redacted:
    if False:
        for i in range(10):
            print('nop')
    'Redact any secrets found in ``value``.'
    return _secrets_masker().redact(value, name, max_depth)

@cache
def _secrets_masker() -> SecretsMasker:
    if False:
        i = 10
        return i + 15
    for flt in logging.getLogger('airflow.task').filters:
        if isinstance(flt, SecretsMasker):
            return flt
    raise RuntimeError('Logging Configuration Error! No SecretsMasker found! If you have custom logging, please make sure you configure it taking airflow configuration as a base as explained at https://airflow.apache.org/docs/apache-airflow/stable/logging-monitoring/logging-tasks.html#advanced-configuration')

@cache
def _get_v1_env_var_type() -> type:
    if False:
        for i in range(10):
            print('nop')
    try:
        from kubernetes.client import V1EnvVar
    except ImportError:
        return type('V1EnvVar', (), {})
    return V1EnvVar

def _is_v1_env_var(v: Any) -> TypeGuard[V1EnvVar]:
    if False:
        print('Hello World!')
    return isinstance(v, _get_v1_env_var_type())

class SecretsMasker(logging.Filter):
    """Redact secrets from logs."""
    replacer: Pattern | None = None
    patterns: set[str]
    ALREADY_FILTERED_FLAG = '__SecretsMasker_filtered'
    MAX_RECURSION_DEPTH = 5

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.patterns = set()

    @cached_property
    def _record_attrs_to_ignore(self) -> Iterable[str]:
        if False:
            i = 10
            return i + 15
        record = logging.getLogRecordFactory()('x', logging.INFO, __file__, 1, '', (), exc_info=None, func='funcname')
        return frozenset(record.__dict__).difference({'msg', 'args'})

    def _redact_exception_with_context(self, exception):
        if False:
            while True:
                i = 10
        try:
            exception.args = (self.redact(v) for v in exception.args)
        except AttributeError:
            pass
        if exception.__context__:
            self._redact_exception_with_context(exception.__context__)
        if exception.__cause__ and exception.__cause__ is not exception.__context__:
            self._redact_exception_with_context(exception.__cause__)

    def filter(self, record) -> bool:
        if False:
            return 10
        if settings.MASK_SECRETS_IN_LOGS is not True:
            return True
        if self.ALREADY_FILTERED_FLAG in record.__dict__:
            return True
        if self.replacer:
            for (k, v) in record.__dict__.items():
                if k not in self._record_attrs_to_ignore:
                    record.__dict__[k] = self.redact(v)
            if record.exc_info and record.exc_info[1] is not None:
                exc = record.exc_info[1]
                self._redact_exception_with_context(exc)
        record.__dict__[self.ALREADY_FILTERED_FLAG] = True
        return True

    def _redact_all(self, item: Redactable, depth: int, max_depth: int=MAX_RECURSION_DEPTH) -> Redacted:
        if False:
            return 10
        if depth > max_depth or isinstance(item, str):
            return '***'
        if isinstance(item, dict):
            return {dict_key: self._redact_all(subval, depth + 1, max_depth) for (dict_key, subval) in item.items()}
        elif isinstance(item, (tuple, set)):
            return tuple((self._redact_all(subval, depth + 1, max_depth) for subval in item))
        elif isinstance(item, list):
            return list((self._redact_all(subval, depth + 1, max_depth) for subval in item))
        else:
            return item

    def _redact(self, item: Redactable, name: str | None, depth: int, max_depth: int) -> Redacted:
        if False:
            print('Hello World!')
        if depth > max_depth:
            return item
        try:
            if name and should_hide_value_for_key(name):
                return self._redact_all(item, depth, max_depth)
            if isinstance(item, dict):
                to_return = {dict_key: self._redact(subval, name=dict_key, depth=depth + 1, max_depth=max_depth) for (dict_key, subval) in item.items()}
                return to_return
            elif isinstance(item, Enum):
                return self._redact(item=item.value, name=name, depth=depth, max_depth=max_depth)
            elif _is_v1_env_var(item):
                tmp: dict = item.to_dict()
                if should_hide_value_for_key(tmp.get('name', '')) and 'value' in tmp:
                    tmp['value'] = '***'
                else:
                    return self._redact(item=tmp, name=name, depth=depth, max_depth=max_depth)
                return tmp
            elif isinstance(item, str):
                if self.replacer:
                    return self.replacer.sub('***', item)
                return item
            elif isinstance(item, (tuple, set)):
                return tuple((self._redact(subval, name=None, depth=depth + 1, max_depth=max_depth) for subval in item))
            elif isinstance(item, list):
                return [self._redact(subval, name=None, depth=depth + 1, max_depth=max_depth) for subval in item]
            else:
                return item
        except Exception as exc:
            log.warning('Unable to redact %r, please report this via <https://github.com/apache/airflow/issues>. Error was: %s: %s', item, type(exc).__name__, exc)
            return item

    def redact(self, item: Redactable, name: str | None=None, max_depth: int | None=None) -> Redacted:
        if False:
            for i in range(10):
                print('nop')
        'Redact an any secrets found in ``item``, if it is a string.\n\n        If ``name`` is given, and it\'s a "sensitive" name (see\n        :func:`should_hide_value_for_key`) then all string values in the item\n        is redacted.\n        '
        return self._redact(item, name, depth=0, max_depth=max_depth or self.MAX_RECURSION_DEPTH)

    @cached_property
    def _mask_adapter(self) -> None | Callable:
        if False:
            i = 10
            return i + 15
        'Pulls the secret mask adapter from config.\n\n        This lives in a function here to be cached and only hit the config once.\n        '
        from airflow.configuration import conf
        return conf.getimport('logging', 'secret_mask_adapter', fallback=None)

    @cached_property
    def _test_mode(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Pulls the unit test mode flag from config.\n\n        This lives in a function here to be cached and only hit the config once.\n        '
        from airflow.configuration import conf
        return conf.getboolean('core', 'unit_test_mode')

    def _adaptations(self, secret: str) -> Generator[str, None, None]:
        if False:
            for i in range(10):
                print('nop')
        'Yield the secret along with any adaptations to the secret that should be masked.'
        yield secret
        if self._mask_adapter:
            secret_or_secrets = self._mask_adapter(secret)
            if not isinstance(secret_or_secrets, str):
                yield from secret_or_secrets
            else:
                yield secret_or_secrets

    def add_mask(self, secret: str | dict | Iterable, name: str | None=None):
        if False:
            print('Hello World!')
        'Add a new secret to be masked to this filter instance.'
        if isinstance(secret, dict):
            for (k, v) in secret.items():
                self.add_mask(v, k)
        elif isinstance(secret, str):
            if not secret or (self._test_mode and secret in SECRETS_TO_SKIP_MASKING_FOR_TESTS):
                return
            new_mask = False
            for s in self._adaptations(secret):
                if s:
                    pattern = re2.escape(s)
                    if pattern not in self.patterns and (not name or should_hide_value_for_key(name)):
                        self.patterns.add(pattern)
                        new_mask = True
            if new_mask:
                self.replacer = re2.compile('|'.join(self.patterns))
        elif isinstance(secret, collections.abc.Iterable):
            for v in secret:
                self.add_mask(v, name)

class RedactedIO(TextIO):
    """IO class that redacts values going into stdout.

    Expected usage::

        with contextlib.redirect_stdout(RedactedIO()):
            ...  # Writes to stdout will be redacted.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.target = sys.stdout

    def __enter__(self) -> TextIO:
        if False:
            print('Hello World!')
        return self.target.__enter__()

    def __exit__(self, t, v, b) -> None:
        if False:
            return 10
        return self.target.__exit__(t, v, b)

    def __iter__(self) -> Iterator[str]:
        if False:
            return 10
        return iter(self.target)

    def __next__(self) -> str:
        if False:
            i = 10
            return i + 15
        return next(self.target)

    def close(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        return self.target.close()

    def fileno(self) -> int:
        if False:
            print('Hello World!')
        return self.target.fileno()

    def flush(self) -> None:
        if False:
            print('Hello World!')
        return self.target.flush()

    def isatty(self) -> bool:
        if False:
            print('Hello World!')
        return self.target.isatty()

    def read(self, n: int=-1) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.target.read(n)

    def readable(self) -> bool:
        if False:
            while True:
                i = 10
        return self.target.readable()

    def readline(self, n: int=-1) -> str:
        if False:
            return 10
        return self.target.readline(n)

    def readlines(self, n: int=-1) -> list[str]:
        if False:
            i = 10
            return i + 15
        return self.target.readlines(n)

    def seek(self, offset: int, whence: int=0) -> int:
        if False:
            print('Hello World!')
        return self.target.seek(offset, whence)

    def seekable(self) -> bool:
        if False:
            return 10
        return self.target.seekable()

    def tell(self) -> int:
        if False:
            i = 10
            return i + 15
        return self.target.tell()

    def truncate(self, s: int | None=None) -> int:
        if False:
            while True:
                i = 10
        return self.target.truncate(s)

    def writable(self) -> bool:
        if False:
            return 10
        return self.target.writable()

    def write(self, s: str) -> int:
        if False:
            while True:
                i = 10
        s = redact(s)
        return self.target.write(s)

    def writelines(self, lines) -> None:
        if False:
            while True:
                i = 10
        self.target.writelines(lines)