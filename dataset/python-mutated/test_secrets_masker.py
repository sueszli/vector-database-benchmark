from __future__ import annotations
import contextlib
import inspect
import logging
import logging.config
import os
import sys
import textwrap
from enum import Enum
from io import StringIO
from unittest.mock import patch
import pytest
from airflow import settings
from airflow.utils.log.secrets_masker import RedactedIO, SecretsMasker, mask_secret, redact, should_hide_value_for_key
from airflow.utils.state import DagRunState, JobState, State, TaskInstanceState
from tests.test_utils.config import conf_vars
settings.MASK_SECRETS_IN_LOGS = True
p = 'password'

class MyEnum(str, Enum):
    testname = 'testvalue'

@pytest.fixture
def logger(caplog):
    if False:
        print('Hello World!')
    logging.config.dictConfig({'version': 1, 'handlers': {__name__: {'class': 'logging.StreamHandler', 'stream': 'ext://sys.stdout'}}, 'loggers': {__name__: {'handlers': [__name__], 'level': logging.INFO, 'propagate': False}}, 'disable_existing_loggers': False})
    formatter = ShortExcFormatter('%(levelname)s %(message)s')
    logger = logging.getLogger(__name__)
    caplog.handler.setFormatter(formatter)
    logger.handlers = [caplog.handler]
    filt = SecretsMasker()
    logger.addFilter(filt)
    filt.add_mask('password')
    return logger

class TestSecretsMasker:

    def test_message(self, logger, caplog):
        if False:
            for i in range(10):
                print('nop')
        logger.info('XpasswordY')
        assert caplog.text == 'INFO X***Y\n'

    def test_args(self, logger, caplog):
        if False:
            i = 10
            return i + 15
        logger.info('Cannot connect to %s', 'user:password')
        assert caplog.text == 'INFO Cannot connect to user:***\n'

    def test_extra(self, logger, caplog):
        if False:
            for i in range(10):
                print('nop')
        logger.handlers[0].formatter = ShortExcFormatter('%(levelname)s %(message)s %(conn)s')
        logger.info('Cannot connect', extra={'conn': 'user:password'})
        assert caplog.text == 'INFO Cannot connect user:***\n'

    def test_exception(self, logger, caplog):
        if False:
            while True:
                i = 10
        try:
            conn = 'user:password'
            raise RuntimeError('Cannot connect to ' + conn)
        except RuntimeError:
            logger.exception('Err')
        line = lineno() - 4
        assert caplog.text == textwrap.dedent(f'            ERROR Err\n            Traceback (most recent call last):\n              File ".../test_secrets_masker.py", line {line}, in test_exception\n                raise RuntimeError("Cannot connect to " + conn)\n            RuntimeError: Cannot connect to user:***\n            ')

    def test_exception_not_raised(self, logger, caplog):
        if False:
            i = 10
            return i + 15
        '\n        Test that when ``logger.exception`` is called when there is no current exception we still log.\n\n        (This is a "bug" in user code, but we shouldn\'t die because of it!)\n        '
        logger.exception('Err')
        assert caplog.text == textwrap.dedent('            ERROR Err\n            NoneType: None\n            ')

    @pytest.mark.xfail(reason='Cannot filter secrets in traceback source')
    def test_exc_tb(self, logger, caplog):
        if False:
            return 10
        '\n        Show it is not possible to filter secrets in the source.\n\n        It is not possible to (regularly/reliably) filter out secrets that\n        appear directly in the source code. This is because the formatting of\n        exc_info is not done in the filter, it is done after the filter is\n        called, and fixing this "properly" is hard/impossible.\n\n        (It would likely need to construct a custom traceback that changed the\n        source. I have no idead if that is even possible)\n\n        This test illustrates that, but ix marked xfail in case someone wants to\n        fix this later.\n        '
        try:
            raise RuntimeError('Cannot connect to user:password')
        except RuntimeError:
            logger.exception('Err')
        line = lineno() - 4
        assert caplog.text == textwrap.dedent(f'            ERROR Err\n            Traceback (most recent call last):\n              File ".../test_secrets_masker.py", line {line}, in test_exc_tb\n                raise RuntimeError("Cannot connect to user:***)\n            RuntimeError: Cannot connect to user:***\n            ')

    def test_masking_in_implicit_context_exceptions(self, logger, caplog):
        if False:
            while True:
                i = 10
        '\n        Show that redacting password works in context exceptions.\n        '
        try:
            try:
                try:
                    raise RuntimeError(f'Cannot connect to user:{p}')
                except RuntimeError as ex1:
                    raise RuntimeError(f'Exception: {ex1}')
            except RuntimeError as ex2:
                raise RuntimeError(f'Exception: {ex2}')
        except RuntimeError:
            logger.exception('Err')
        assert 'user:password' not in caplog.text
        assert caplog.text.count('user:***') >= 2

    def test_masking_in_explicit_context_exceptions(self, logger, caplog):
        if False:
            for i in range(10):
                print('nop')
        '\n        Show that redacting password works in context exceptions.\n        '
        exception = None
        try:
            raise RuntimeError(f'Cannot connect to user:{p}')
        except RuntimeError as ex:
            exception = ex
        try:
            raise RuntimeError(f'Exception: {exception}') from exception
        except RuntimeError:
            logger.exception('Err')
        line = lineno() - 8
        assert caplog.text == textwrap.dedent(f'            ERROR Err\n            Traceback (most recent call last):\n              File ".../test_secrets_masker.py", line {line}, in test_masking_in_explicit_context_exceptions\n                raise RuntimeError(f"Cannot connect to user:{{p}}")\n            RuntimeError: Cannot connect to user:***\n\n            The above exception was the direct cause of the following exception:\n\n            Traceback (most recent call last):\n              File ".../test_secrets_masker.py", line {line + 4}, in test_masking_in_explicit_context_exceptions\n                raise RuntimeError(f"Exception: {{exception}}") from exception\n            RuntimeError: Exception: Cannot connect to user:***\n            ')

    @pytest.mark.parametrize(('name', 'value', 'expected_mask'), [(None, 'secret', {'secret'}), ('apikey', 'secret', {'secret'}), (None, {'apikey': 'secret', 'other': {'val': 'innocent', 'password': 'foo'}}, {'secret', 'foo'}), (None, ['secret', 'other'], {'secret', 'other'}), ('api_key', {'other': 'innoent'}, set()), (None, {'password': ''}, set()), (None, '', set())])
    def test_mask_secret(self, name, value, expected_mask):
        if False:
            print('Hello World!')
        filt = SecretsMasker()
        filt.add_mask(value, name)
        assert filt.patterns == expected_mask

    @pytest.mark.parametrize(('patterns', 'name', 'value', 'expected'), [({'secret'}, None, 'secret', '***'), ({'secret', 'foo'}, None, {'apikey': 'secret', 'other': {'val': 'innocent', 'password': 'foo'}}, {'apikey': '***', 'other': {'val': 'innocent', 'password': '***'}}), ({'secret', 'other'}, None, ['secret', 'other'], ['***', '***']), ({'secret', 'other'}, None, {'data': {'secret': 'secret'}}, {'data': {'secret': '***'}}), ({'secret', 'other'}, None, {1: {'secret': 'secret'}}, {1: {'secret': '***'}}), ({'secret'}, 'api_key', {'other': 'innoent', 'nested': ['x', 'y']}, {'other': '***', 'nested': ['***', '***']}), (set(), 'env', {'api_key': 'masked based on key name', 'other': 'foo'}, {'api_key': '***', 'other': 'foo'})])
    def test_redact(self, patterns, name, value, expected):
        if False:
            for i in range(10):
                print('nop')
        filt = SecretsMasker()
        for val in patterns:
            filt.add_mask(val)
        assert filt.redact(value, name) == expected

    def test_redact_filehandles(self, caplog):
        if False:
            i = 10
            return i + 15
        filt = SecretsMasker()
        with open('/dev/null', 'w') as handle:
            assert filt.redact(handle, None) == handle
        assert caplog.messages == []

    @pytest.mark.parametrize(('val', 'expected', 'max_depth'), [(['abc'], ['***'], None), (['abc'], ['***'], 1), ([[[['abc']]]], [[[['***']]]], None), ([[[[['abc']]]]], [[[[['***']]]]], None), ([[[[[['abc']]]]]], [[[[[['abc']]]]]], None), ([['abc']], [['abc']], 1)])
    def test_redact_max_depth(self, val, expected, max_depth):
        if False:
            return 10
        secrets_masker = SecretsMasker()
        secrets_masker.add_mask('abc')
        with patch('airflow.utils.log.secrets_masker._secrets_masker', return_value=secrets_masker):
            got = redact(val, max_depth=max_depth)
            assert got == expected

    @pytest.mark.parametrize('state, expected', [(DagRunState.SUCCESS, 'success'), (TaskInstanceState.FAILED, 'failed'), (JobState.RUNNING, 'running'), ([DagRunState.SUCCESS, DagRunState.RUNNING], ['success', 'running']), ([TaskInstanceState.FAILED, TaskInstanceState.SUCCESS], ['failed', 'success']), (State.failed_states, frozenset([TaskInstanceState.FAILED, TaskInstanceState.UPSTREAM_FAILED])), (MyEnum.testname, 'testvalue')])
    def test_redact_state_enum(self, logger, caplog, state, expected):
        if False:
            i = 10
            return i + 15
        logger.info('State: %s', state)
        assert caplog.text == f'INFO State: {expected}\n'
        assert 'TypeError' not in caplog.text

class TestShouldHideValueForKey:

    @pytest.mark.parametrize(('key', 'expected_result'), [('', False), (None, False), ('key', False), ('google_api_key', True), ('GOOGLE_API_KEY', True), ('GOOGLE_APIKEY', True), (1, False)])
    def test_hiding_defaults(self, key, expected_result):
        if False:
            return 10
        assert expected_result == should_hide_value_for_key(key)

    @pytest.mark.parametrize(('sensitive_variable_fields', 'key', 'expected_result'), [('key', 'TRELLO_KEY', True), ('key', 'TRELLO_API_KEY', True), ('key', 'GITHUB_APIKEY', True), ('key, token', 'TRELLO_TOKEN', True), ('mysecretword, mysensitivekey', 'GITHUB_mysecretword', True), (None, 'TRELLO_API', False), ('token', 'TRELLO_KEY', False), ('token, mysecretword', 'TRELLO_KEY', False)])
    def test_hiding_config(self, sensitive_variable_fields, key, expected_result):
        if False:
            return 10
        from airflow.utils.log.secrets_masker import get_sensitive_variables_fields
        with conf_vars({('core', 'sensitive_var_conn_names'): str(sensitive_variable_fields)}):
            get_sensitive_variables_fields.cache_clear()
            assert expected_result == should_hide_value_for_key(key)
        get_sensitive_variables_fields.cache_clear()

class ShortExcFormatter(logging.Formatter):
    """Don't include full path in exc_info messages"""

    def formatException(self, exc_info):
        if False:
            return 10
        formatted = super().formatException(exc_info)
        return formatted.replace(__file__, '.../' + os.path.basename(__file__))

def lineno():
    if False:
        i = 10
        return i + 15
    'Returns the current line number in our program.'
    return inspect.currentframe().f_back.f_lineno

class TestRedactedIO:

    @pytest.fixture(scope='class', autouse=True)
    def reset_secrets_masker(self):
        if False:
            i = 10
            return i + 15
        self.secrets_masker = SecretsMasker()
        with patch('airflow.utils.log.secrets_masker._secrets_masker', return_value=self.secrets_masker):
            mask_secret(p)
            yield

    def test_redacts_from_print(self, capsys):
        if False:
            for i in range(10):
                print('nop')
        print(p)
        stdout = capsys.readouterr().out
        assert stdout == f'{p}\n'
        assert '***' not in stdout
        with contextlib.redirect_stdout(RedactedIO()):
            print(p)
        stdout = capsys.readouterr().out
        assert stdout == '***\n'

    def test_write(self, capsys):
        if False:
            i = 10
            return i + 15
        RedactedIO().write(p)
        stdout = capsys.readouterr().out
        assert stdout == '***'

    def test_input_builtin(self, monkeypatch):
        if False:
            while True:
                i = 10
        '\n        Test that when redirect is inplace the `input()` builtin works.\n\n        This is used by debuggers!\n        '
        monkeypatch.setattr(sys, 'stdin', StringIO('a\n'))
        with contextlib.redirect_stdout(RedactedIO()):
            assert input() == 'a'

class TestMaskSecretAdapter:

    @pytest.fixture(scope='function', autouse=True)
    def reset_secrets_masker_and_skip_escape(self):
        if False:
            i = 10
            return i + 15
        self.secrets_masker = SecretsMasker()
        with patch('airflow.utils.log.secrets_masker._secrets_masker', return_value=self.secrets_masker):
            with patch('airflow.utils.log.secrets_masker.re2.escape', lambda x: x):
                yield

    def test_calling_mask_secret_adds_adaptations_for_returned_str(self):
        if False:
            return 10
        with conf_vars({('logging', 'secret_mask_adapter'): 'urllib.parse.quote'}):
            mask_secret('secret<>&', None)
        assert self.secrets_masker.patterns == {'secret%3C%3E%26', 'secret<>&'}

    def test_calling_mask_secret_adds_adaptations_for_returned_iterable(self):
        if False:
            print('Hello World!')
        with conf_vars({('logging', 'secret_mask_adapter'): 'urllib.parse.urlparse'}):
            mask_secret('https://airflow.apache.org/docs/apache-airflow/stable', 'password')
        assert self.secrets_masker.patterns == {'https', 'airflow.apache.org', '/docs/apache-airflow/stable', 'https://airflow.apache.org/docs/apache-airflow/stable'}

    def test_calling_mask_secret_not_set(self):
        if False:
            return 10
        with conf_vars({('logging', 'secret_mask_adapter'): None}):
            mask_secret('a secret')
        assert self.secrets_masker.patterns == {'a secret'}