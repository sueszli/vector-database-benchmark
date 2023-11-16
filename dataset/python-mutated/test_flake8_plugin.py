from __future__ import annotations
import ast
import pytest
from tools.flake8_plugin import SentryCheck

def _run(src: str, filename: str='getsentry/t.py') -> list[str]:
    if False:
        for i in range(10):
            print('nop')
    tree = ast.parse(src)
    return sorted(('t.py:{}:{}: {}'.format(*error) for error in SentryCheck(tree=tree, filename=filename).run()))

def test_S001():
    if False:
        i = 10
        return i + 15
    S001_py = 'class A:\n    def called_once():\n        pass\n\n\nA().called_once()\n'
    errors = _run(S001_py)
    assert errors == ['t.py:6:0: S001 Avoid using the called_once mock call as it is confusing and prone to causing invalid test behavior.']

def test_S002():
    if False:
        print('Hello World!')
    S002_py = 'print("print statements are not allowed")\n'
    errors = _run(S002_py)
    assert errors == ['t.py:1:0: S002 print functions or statements are not allowed.']

def test_S003():
    if False:
        i = 10
        return i + 15
    S003_py = 'import json\nimport simplejson\nfrom json import loads, load\nfrom simplejson import JSONDecoder, JSONDecodeError, _default_encoder\nimport sentry.utils.json as good_json\nfrom sentry.utils.json import JSONDecoder, JSONDecodeError\nfrom .json import Validator\n\n\ndef bad_code():\n    a = json.loads("\'\'")\n    b = simplejson.loads("\'\'")\n    c = loads("\'\'")\n    d = load()\n'
    errors = _run(S003_py)
    assert errors == ['t.py:1:0: S003 Use ``from sentry.utils import json`` instead.', 't.py:2:0: S003 Use ``from sentry.utils import json`` instead.', 't.py:3:0: S003 Use ``from sentry.utils import json`` instead.', 't.py:4:0: S003 Use ``from sentry.utils import json`` instead.']

def test_S004():
    if False:
        i = 10
        return i + 15
    S004_py = 'import unittest\nfrom something import func\n\n\nclass Test(unittest.TestCase):\n    def test(self):\n        with self.assertRaises(ValueError):\n            func()\n'
    errors = _run(S004_py)
    assert errors == ['t.py:7:13: S004 Use `pytest.raises` instead for better debuggability.']

def test_S005():
    if False:
        print('Hello World!')
    S005_py = 'from sentry.models import User\n'
    errors = _run(S005_py)
    assert errors == ['t.py:1:0: S005 Do not import models from sentry.models but the actual module']

def test_S006():
    if False:
        for i in range(10):
            print('nop')
    src = 'from django.utils.encoding import force_bytes\nfrom django.utils.encoding import force_str\n'
    assert _run(src, filename='src/sentry/whatever.py') == []
    errors = _run(src, filename='tests/test_foo.py')
    assert errors == ['t.py:1:0: S006 Do not use force_bytes / force_str -- test the types directly', 't.py:2:0: S006 Do not use force_bytes / force_str -- test the types directly']

def test_S007():
    if False:
        while True:
            i = 10
    src = 'from sentry.testutils.outbox import outbox_runner\n'
    assert _run(src, filename='tests/test_foo.py') == []
    assert _run(src, filename='src/sentry/testutils/silo.py') == []
    errors = _run(src, filename='src/sentry/api/endpoints/organization_details.py')
    assert errors == ['t.py:1:0: S007 Do not import sentry.testutils into production code.']
    src = 'import sentry.testutils.outbox as outbox_utils\n'
    assert _run(src, filename='tests/test_foo.py') == []
    errors = _run(src, filename='src/sentry/api/endpoints/organization_details.py')
    assert errors == ['t.py:1:0: S007 Do not import sentry.testutils into production code.']

@pytest.mark.parametrize('src', ('from pytz import utc', 'from pytz import UTC', 'pytz.utc', 'pytz.UTC'))
def test_S008(src):
    if False:
        while True:
            i = 10
    expected = ['t.py:1:0: S008 Use stdlib datetime.timezone.utc instead of pytz.utc / pytz.UTC']
    assert _run(src) == expected