from __future__ import annotations
import pytest
from pyupgrade._data import Settings
from pyupgrade._main import _fix_plugins
from pyupgrade._plugins.percent_format import _parse_percent_format
from pyupgrade._plugins.percent_format import _percent_to_format
from pyupgrade._plugins.percent_format import _simplify_conversion_flag

@pytest.mark.parametrize(('s', 'expected'), (('""', (('""', None),)), ('"%%"', (('"', (None, None, None, None, '%')), ('"', None))), ('"%s"', (('"', (None, None, None, None, 's')), ('"', None))), ('"%s two! %s"', (('"', (None, None, None, None, 's')), (' two! ', (None, None, None, None, 's')), ('"', None))), ('"%(hi)s"', (('"', ('hi', None, None, None, 's')), ('"', None))), ('"%()s"', (('"', ('', None, None, None, 's')), ('"', None))), ('"%#o"', (('"', (None, '#', None, None, 'o')), ('"', None))), ('"% #0-+d"', (('"', (None, ' #0-+', None, None, 'd')), ('"', None))), ('"%5d"', (('"', (None, None, '5', None, 'd')), ('"', None))), ('"%*d"', (('"', (None, None, '*', None, 'd')), ('"', None))), ('"%.f"', (('"', (None, None, None, '.', 'f')), ('"', None))), ('"%.5f"', (('"', (None, None, None, '.5', 'f')), ('"', None))), ('"%.*f"', (('"', (None, None, None, '.*', 'f')), ('"', None))), ('"%ld"', (('"', (None, None, None, None, 'd')), ('"', None))), ('"%(complete)#4.4f"', (('"', ('complete', '#', '4', '.4', 'f')), ('"', None)))))
def test_parse_percent_format(s, expected):
    if False:
        i = 10
        return i + 15
    assert _parse_percent_format(s) == expected

@pytest.mark.parametrize(('s', 'expected'), (('%s', '{}'), ('%%%s', '%{}'), ('%(foo)s', '{foo}'), ('%2f', '{:2f}'), ('%r', '{!r}'), ('%a', '{!a}')))
def test_percent_to_format(s, expected):
    if False:
        return 10
    assert _percent_to_format(s) == expected

@pytest.mark.parametrize(('s', 'expected'), (('', ''), (' ', ' '), ('   ', ' '), ('#0- +', '#<+'), ('-', '<')))
def test_simplify_conversion_flag(s, expected):
    if False:
        while True:
            i = 10
    assert _simplify_conversion_flag(s) == expected

@pytest.mark.parametrize('s', ('"%s" % unknown_type', 'b"%s" % (b"bytestring",)', '"%*s" % (5, "hi")', '"%.*s" % (5, "hi")', '"%d" % (flt,)', '"%i" % (flt,)', '"%u" % (flt,)', '"%c" % (some_string,)', '"%#o" % (123,)', '"%()s" % {"": "empty"}', '"%4%" % ()', '"%.2r" % (1.25)', '"%.2a" % (1.25)', pytest.param('"%8s" % (None,)', id='unsafe width-string conversion'), 'i % 3', '"%s" % {"k": "v"}', '"%()s" % {"": "bar"}', '"%(1)s" % {"1": "bar"}', '"%(a)s" % {"a": 1, "a": 2}', '"%(ab)s" % {"a" "b": 1}', '"%(a)s" % {"a"  :  1}', '"%(1)s" % {1: 2, "1": 2}', '"%(and)s" % {"and": 2}', '"%" % {}', '"%(hi)" % {}', '"%2" % {}'))
def test_percent_format_noop(s):
    if False:
        for i in range(10):
            print('nop')
    assert _fix_plugins(s, settings=Settings()) == s

@pytest.mark.parametrize(('s', 'expected'), (('"trivial" % ()', '"trivial".format()'), ('"%s" % ("simple",)', '"{}".format("simple")'), ('"%s" % ("%s" % ("nested",),)', '"{}".format("{}".format("nested"))'), ('"%s%% percent" % (15,)', '"{}% percent".format(15)'), ('"%3f" % (15,)', '"{:3f}".format(15)'), ('"%-5f" % (5,)', '"{:<5f}".format(5)'), ('"%9f" % (5,)', '"{:9f}".format(5)'), ('"brace {} %s" % (1,)', '"brace {{}} {}".format(1)'), ('"%s" % (\n    "trailing comma",\n)\n', '"{}".format(\n    "trailing comma",\n)\n'), ('"%(k)s" % {"k": "v"}', '"{k}".format(k="v")'), ('"%(to_list)s" % {"to_list": []}', '"{to_list}".format(to_list=[])'), ('"%s \\N{snowman}" % (a,)', '"{} \\N{snowman}".format(a)'), ('"%(foo)s \\N{snowman}" % {"foo": 1}', '"{foo} \\N{snowman}".format(foo=1)')))
def test_percent_format(s, expected):
    if False:
        print('Hello World!')
    ret = _fix_plugins(s, settings=Settings())
    assert ret == expected

@pytest.mark.xfail
@pytest.mark.parametrize(('s', 'expected'), (('paren_continue = (\n    "foo %s "\n    "bar %s" % (x, y)\n)\n', 'paren_continue = (\n    "foo {} "\n    "bar {}".format(x, y)\n)\n'), ('paren_string = (\n    "foo %s "\n    "bar %s"\n) % (x, y)\n', 'paren_string = (\n    "foo {} "\n    "bar {}"\n).format(x, y)\n'), ('paren_continue = (\n    "foo %(foo)s "\n    "bar %(bar)s" % {"foo": x, "bar": y}\n)\n', 'paren_continue = (\n    "foo {foo} "\n    "bar {bar}".format(foo=x, bar=y)\n)\n'), ('paren_string = (\n    "foo %(foo)s "\n    "bar %(bar)s"\n) % {"foo": x, "bar": y}\n', 'paren_string = (\n    "foo {foo} "\n    "bar {bar}"\n).format(foo=x, bar=y)\n')))
def test_percent_format_todo(s, expected):
    if False:
        i = 10
        return i + 15
    ret = _fix_plugins(s, settings=Settings())
    assert ret == expected