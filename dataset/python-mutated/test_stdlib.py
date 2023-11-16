"""
Tests of various stdlib related things that could not be tested
with "Black Box Tests".
"""
from textwrap import dedent
import pytest

@pytest.mark.parametrize(['letter', 'expected'], [('n', ['name']), ('s', ['smart'])])
def test_namedtuple_str(letter, expected, Script):
    if False:
        print('Hello World!')
    source = dedent("        import collections\n        Person = collections.namedtuple('Person', 'name smart')\n        dave = Person('Dave', False)\n        dave.%s") % letter
    result = Script(source).complete()
    completions = set((r.name for r in result))
    assert completions == set(expected)

def test_namedtuple_list(Script):
    if False:
        return 10
    source = dedent("        import collections\n        Cat = collections.namedtuple('Person', ['legs', u'length', 'large'])\n        garfield = Cat(4, '85cm', True)\n        garfield.l")
    result = Script(source).complete()
    completions = set((r.name for r in result))
    assert completions == {'legs', 'length', 'large'}

def test_namedtuple_content(Script):
    if False:
        for i in range(10):
            print('nop')
    source = dedent("        import collections\n        Foo = collections.namedtuple('Foo', ['bar', 'baz'])\n        named = Foo(baz=4, bar=3.0)\n        unnamed = Foo(4, '')\n        ")

    def d(source):
        if False:
            return 10
        (x,) = Script(source).infer()
        return x.name
    assert d(source + 'unnamed.bar') == 'int'
    assert d(source + 'unnamed.baz') == 'str'
    assert d(source + 'named.bar') == 'float'
    assert d(source + 'named.baz') == 'int'

def test_nested_namedtuples(Script):
    if False:
        while True:
            i = 10
    '\n    From issue #730.\n    '
    s = Script(dedent("\n        import collections\n        Dataset = collections.namedtuple('Dataset', ['data'])\n        Datasets = collections.namedtuple('Datasets', ['train'])\n        train_x = Datasets(train=Dataset('data_value'))\n        train_x.train."))
    assert 'data' in [c.name for c in s.complete()]

def test_namedtuple_infer(Script):
    if False:
        return 10
    source = dedent("\n        from collections import namedtuple\n\n        Foo = namedtuple('Foo', 'id timestamp gps_timestamp attributes')\n        Foo")
    from jedi.api import Script
    (d1,) = Script(source).infer()
    assert d1.get_line_code() == 'class Foo(tuple):\n'
    assert d1.module_path is None
    assert d1.docstring() == 'Foo(id, timestamp, gps_timestamp, attributes)'

def test_re_sub(Script, environment):
    if False:
        print('Hello World!')
    '\n    This whole test was taken out of completion/stdlib.py, because of the\n    version differences.\n    '

    def run(code):
        if False:
            for i in range(10):
                print('nop')
        defs = Script(code).infer()
        return {d.name for d in defs}
    names = run("import re; re.sub('a', 'a', 'f')")
    assert names == {'str'}
    names = run("import re; re.sub('a', 'a')")
    assert names == {'str', 'bytes'}