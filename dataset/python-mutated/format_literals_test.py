from __future__ import annotations
import pytest
from pyupgrade._main import _fix_tokens

@pytest.mark.parametrize('s', ('"{0}"format(1)', pytest.param("'{}'.format(1)", id='already upgraded'), "'{'.format(1)", "'}'.format(1)", "x = ('{0} {1}',)\n", "'{0} {0}'.format(1)", "'{0:<{1}}'.format(1, 4)", "'{' '0}'.format(1)", '("{0}" # {1}\n"{2}").format(1, 2, 3)', 'f"{0}".format(a)', '"{}\\N{SNOWMAN}".format("")'))
def test_format_literals_noop(s):
    if False:
        for i in range(10):
            print('nop')
    assert _fix_tokens(s) == s

@pytest.mark.parametrize(('s', 'expected'), (("'{0}'.format(1)", "'{}'.format(1)"), ("'{0:x}'.format(30)", "'{:x}'.format(30)"), ("x = '{0}'.format(1)", "x = '{}'.format(1)"), ("'''{0}\n{1}\n'''.format(1, 2)", "'''{}\n{}\n'''.format(1, 2)"), ("'{0}' '{1}'.format(1, 2)", "'{}' '{}'.format(1, 2)"), ("print(\n    'foo{0}'\n    'bar{1}'.format(1, 2)\n)", "print(\n    'foo{}'\n    'bar{}'.format(1, 2)\n)"), ("print(\n    'foo{0}'  # ohai\n    'bar{1}'.format(1, 2)\n)", "print(\n    'foo{}'  # ohai\n    'bar{}'.format(1, 2)\n)"), ('x = "foo {0}" \\\n    "bar {1}".format(1, 2)', 'x = "foo {}" \\\n    "bar {}".format(1, 2)'), ('("{0}").format(1)', '("{}").format(1)'), pytest.param('"\\N{snowman} {0}".format(1)', '"\\N{snowman} {}".format(1)', id='named escape sequence')))
def test_format_literals(s, expected):
    if False:
        return 10
    assert _fix_tokens(s) == expected