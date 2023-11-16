from __future__ import annotations
import pytest
from pyupgrade._data import Settings
from pyupgrade._main import _fix_plugins

@pytest.mark.parametrize('s', ('(', "'{'.format(a)", "'}'.format(a)", '"{}" . format(x)', '"{}".format(\n    a,\n)', '"{} {}".format(*a)', '"{foo} {bar}".format(**b)"', '"{0} {0}".format(arg)', '"{x} {x}".format(arg)', '"{x.y} {x.z}".format(arg)', 'b"{} {}".format(a, b)', '"{:{}}".format(x, y)', '"{a[b]}".format(a=a)', '"{a.a[b]}".format(a=a)', '"{}{}".format(a)', '"{a}{b}".format(a=a)', '"{}".format(a[\'\\\\\'])', '"{}".format(a["b"])', "'{}'.format(a['b'])", "async def c(): return '{}'.format(await 3)", "async def c(): return '{}'.format(1 + await 3)"))
def test_fix_fstrings_noop(s):
    if False:
        i = 10
        return i + 15
    assert _fix_plugins(s, settings=Settings(min_version=(3, 6))) == s

@pytest.mark.parametrize(('s', 'expected'), (('"{} {}".format(a, b)', 'f"{a} {b}"'), ('"{1} {0}".format(a, b)', 'f"{b} {a}"'), ('"{x.y}".format(x=z)', 'f"{z.y}"'), ('"{.x} {.y}".format(a, b)', 'f"{a.x} {b.y}"'), ('"{} {}".format(a.b, c.d)', 'f"{a.b} {c.d}"'), ('"{}".format(a())', 'f"{a()}"'), ('"{}".format(a.b())', 'f"{a.b()}"'), ('"{}".format(a.b().c())', 'f"{a.b().c()}"'), ('"hello {}!".format(name)', 'f"hello {name}!"'), ('"{}{{}}{}".format(escaped, y)', 'f"{escaped}{{}}{y}"'), ('"{}{b}{}".format(a, c, b=b)', 'f"{a}{b}{c}"'), ('"{}".format(0x0)', 'f"{0x0}"'), pytest.param('"\\N{snowman} {}".format(a)', 'f"\\N{snowman} {a}"', id='named escape sequences'), pytest.param('u"foo{}".format(1)', 'f"foo{1}"', id='u-prefixed format')))
def test_fix_fstrings(s, expected):
    if False:
        return 10
    assert _fix_plugins(s, settings=Settings(min_version=(3, 6))) == expected

def test_fix_fstrings_await_py37():
    if False:
        print('Hello World!')
    s = "async def c(): return '{}'.format(await 1+foo())"
    expected = "async def c(): return f'{await 1+foo()}'"
    assert _fix_plugins(s, settings=Settings(min_version=(3, 7))) == expected