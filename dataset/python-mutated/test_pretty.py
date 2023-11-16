"""This file originates in the IPython project and is made use of under the
following licensing terms:

The IPython licensing terms
IPython is licensed under the terms of the Modified BSD License (also known as
New or Revised or 3-Clause BSD), as follows:

Copyright (c) 2008-2014, IPython Development Team
Copyright (c) 2001-2007, Fernando Perez <fernando.perez@colorado.edu>
Copyright (c) 2001, Janko Hauser <jhauser@zscout.de>
Copyright (c) 2001, Nathaniel Gray <n8gray@caltech.edu>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

Neither the name of the IPython Development Team nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import re
import warnings
from collections import Counter, OrderedDict, defaultdict, deque
from enum import Enum, Flag
import pytest
from hypothesis.internal.compat import PYPY
from hypothesis.strategies._internal.numbers import SIGNALING_NAN
from hypothesis.vendor import pretty

class MyList:

    def __init__(self, content):
        if False:
            print('Hello World!')
        self.content = content

    def _repr_pretty_(self, p, cycle):
        if False:
            return 10
        if cycle:
            p.text('MyList(...)')
        else:
            with p.group(3, 'MyList(', ')'):
                for (i, child) in enumerate(self.content):
                    if i:
                        p.text(',')
                        p.breakable()
                    else:
                        p.breakable('')
                    p.pretty(child)

class MyDict(dict):

    def _repr_pretty_(self, p, cycle):
        if False:
            print('Hello World!')
        p.text('MyDict(...)')

class MyObj:

    def somemethod(self):
        if False:
            i = 10
            return i + 15
        pass

class Dummy1:

    def _repr_pretty_(self, p, cycle):
        if False:
            print('Hello World!')
        p.text('Dummy1(...)')

class Dummy2(Dummy1):
    _repr_pretty_ = None

class NoModule:
    pass
NoModule.__module__ = None

class Breaking:

    def _repr_pretty_(self, p, cycle):
        if False:
            print('Hello World!')
        with p.group(4, 'TG: ', ':'):
            p.text('Breaking(')
            p.break_()
            p.text(')')

class BreakingRepr:

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'Breaking(\n)'

class BreakingReprParent:

    def _repr_pretty_(self, p, cycle):
        if False:
            print('Hello World!')
        with p.group(4, 'TG: ', ':'):
            p.pretty(BreakingRepr())

class BadRepr:

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 1 / 0

def test_list():
    if False:
        print('Hello World!')
    assert pretty.pretty([]) == '[]'
    assert pretty.pretty([1]) == '[1]'

def test_dict():
    if False:
        while True:
            i = 10
    assert pretty.pretty({}) == '{}'
    assert pretty.pretty({1: 1}) == '{1: 1}'
    assert pretty.pretty({1: 1, 0: 0}) == '{1: 1, 0: 0}'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', BytesWarning)
        x = {'': 0, b'': 0}
    assert pretty.pretty(x) == "{'': 0, b'': 0}"

def test_tuple():
    if False:
        for i in range(10):
            print('nop')
    assert pretty.pretty(()) == '()'
    assert pretty.pretty((1,)) == '(1,)'
    assert pretty.pretty((1, 2)) == '(1, 2)'

class ReprDict(dict):

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'hi'

def test_dict_with_custom_repr():
    if False:
        i = 10
        return i + 15
    assert pretty.pretty(ReprDict()) == 'hi'

class ReprList(list):

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'bye'

class ReprSet(set):

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'cat'

def test_set_with_custom_repr():
    if False:
        print('Hello World!')
    assert pretty.pretty(ReprSet()) == 'cat'

def test_list_with_custom_repr():
    if False:
        i = 10
        return i + 15
    assert pretty.pretty(ReprList()) == 'bye'

def test_indentation():
    if False:
        while True:
            i = 10
    'Test correct indentation in groups.'
    count = 40
    gotoutput = pretty.pretty(MyList(range(count)))
    expectedoutput = 'MyList(\n' + ',\n'.join((f'   {i}' for i in range(count))) + ')'
    assert gotoutput == expectedoutput

def test_dispatch():
    if False:
        return 10
    'Test correct dispatching: The _repr_pretty_ method for MyDict must be\n    found before the registered printer for dict.'
    gotoutput = pretty.pretty(MyDict())
    expectedoutput = 'MyDict(...)'
    assert gotoutput == expectedoutput

def test_callability_checking():
    if False:
        return 10
    'Test that the _repr_pretty_ method is tested for callability and skipped\n    if not.'
    gotoutput = pretty.pretty(Dummy2())
    expectedoutput = 'Dummy1(...)'
    assert gotoutput == expectedoutput

def test_sets():
    if False:
        for i in range(10):
            print('nop')
    'Test that set and frozenset use Python 3 formatting.'
    objects = [set(), frozenset(), {1}, frozenset([1]), {1, 2}, frozenset([1, 2]), {-1, -2, -3}]
    expected = ['set()', 'frozenset()', '{1}', 'frozenset({1})', '{1, 2}', 'frozenset({1, 2})', '{-3, -2, -1}']
    for (obj, expected_output) in zip(objects, expected):
        got_output = pretty.pretty(obj)
        assert got_output == expected_output

def test_unsortable_set():
    if False:
        return 10
    xs = {1, 2, 3, 'foo', 'bar', 'baz', object()}
    p = pretty.pretty(xs)
    for x in xs:
        assert pretty.pretty(x) in p

def test_unsortable_dict():
    if False:
        for i in range(10):
            print('nop')
    xs = {k: 1 for k in [1, 2, 3, 'foo', 'bar', 'baz', object()]}
    p = pretty.pretty(xs)
    for x in xs:
        assert pretty.pretty(x) in p

def test_pprint_nomod():
    if False:
        i = 10
        return i + 15
    'Test that pprint works for classes with no __module__.'
    output = pretty.pretty(NoModule)
    assert output == 'NoModule'

def test_pprint_break():
    if False:
        while True:
            i = 10
    'Test that p.break_ produces expected output.'
    output = pretty.pretty(Breaking())
    expected = 'TG: Breaking(\n    ):'
    assert output == expected

def test_pprint_break_repr():
    if False:
        for i in range(10):
            print('nop')
    'Test that p.break_ is used in repr.'
    output = pretty.pretty(BreakingReprParent())
    expected = 'TG: Breaking(\n    ):'
    assert output == expected

def test_bad_repr():
    if False:
        while True:
            i = 10
    "Don't catch bad repr errors."
    with pytest.raises(ZeroDivisionError):
        pretty.pretty(BadRepr())

class BadException(Exception):

    def __str__(self):
        if False:
            return 10
        return -1

class ReallyBadRepr:
    __module__ = 1

    @property
    def __class__(self):
        if False:
            i = 10
            return i + 15
        raise ValueError('I am horrible')

    def __repr__(self):
        if False:
            return 10
        raise BadException

def test_really_bad_repr():
    if False:
        i = 10
        return i + 15
    with pytest.raises(BadException):
        pretty.pretty(ReallyBadRepr())

class SA:
    pass

class SB(SA):
    pass
try:
    super(SA).__self__

    def test_super_repr():
        if False:
            print('Hello World!')
        output = pretty.pretty(super(SA))
        assert 'SA' in output
        sb = SB()
        output = pretty.pretty(super(SA, sb))
        assert 'SA' in output
except AttributeError:

    def test_super_repr():
        if False:
            i = 10
            return i + 15
        pretty.pretty(super(SA))
        sb = SB()
        pretty.pretty(super(SA, sb))

def test_long_list():
    if False:
        for i in range(10):
            print('nop')
    lis = list(range(10000))
    p = pretty.pretty(lis)
    last2 = p.rsplit('\n', 2)[-2:]
    assert last2 == [' 999,', ' ...]']

def test_long_set():
    if False:
        while True:
            i = 10
    s = set(range(10000))
    p = pretty.pretty(s)
    last2 = p.rsplit('\n', 2)[-2:]
    assert last2 == [' 999,', ' ...}']

def test_long_tuple():
    if False:
        i = 10
        return i + 15
    tup = tuple(range(10000))
    p = pretty.pretty(tup)
    last2 = p.rsplit('\n', 2)[-2:]
    assert last2 == [' 999,', ' ...)']

def test_long_dict():
    if False:
        return 10
    d = {n: n for n in range(10000)}
    p = pretty.pretty(d)
    last2 = p.rsplit('\n', 2)[-2:]
    assert last2 == [' 999: 999,', ' ...}']

def test_unbound_method():
    if False:
        while True:
            i = 10
    assert pretty.pretty(MyObj.somemethod) == 'somemethod'

class MetaClass(type):

    def __new__(metacls, name):
        if False:
            i = 10
            return i + 15
        return type.__new__(metacls, name, (object,), {'name': name})

    def __repr__(cls):
        if False:
            i = 10
            return i + 15
        return f'[CUSTOM REPR FOR CLASS {cls.name}]'
ClassWithMeta = MetaClass('ClassWithMeta')

def test_metaclass_repr():
    if False:
        return 10
    output = pretty.pretty(ClassWithMeta)
    assert output == '[CUSTOM REPR FOR CLASS ClassWithMeta]'

def test_unicode_repr():
    if False:
        for i in range(10):
            print('nop')
    u = 'üniçodé'

    class C:

        def __repr__(self):
            if False:
                print('Hello World!')
            return u
    c = C()
    p = pretty.pretty(c)
    assert p == u
    p = pretty.pretty([c])
    assert p == f'[{u}]'

def test_basic_class():
    if False:
        print('Hello World!')

    def type_pprint_wrapper(obj, p, cycle):
        if False:
            return 10
        if obj is MyObj:
            type_pprint_wrapper.called = True
        return pretty._type_pprint(obj, p, cycle)
    type_pprint_wrapper.called = False
    printer = pretty.RepresentationPrinter()
    printer.type_pprinters[type] = type_pprint_wrapper
    printer.pretty(MyObj)
    output = printer.getvalue()
    assert output == f'{__name__}.MyObj'
    assert type_pprint_wrapper.called

def test_collections_defaultdict():
    if False:
        while True:
            i = 10
    a = defaultdict()
    a.default_factory = a
    b = defaultdict(list)
    b['key'] = b
    cases = [(defaultdict(list), 'defaultdict(list, {})'), (defaultdict(list, {'key': '-' * 50}), "defaultdict(list,\n            {'key': '--------------------------------------------------'})"), (a, 'defaultdict(defaultdict(...), {})'), (b, "defaultdict(list, {'key': defaultdict(...)})")]
    for (obj, expected) in cases:
        assert pretty.pretty(obj) == expected

@pytest.mark.skipif(PYPY, reason='slightly different on PyPy3')
def test_collections_ordereddict():
    if False:
        while True:
            i = 10
    a = OrderedDict()
    a['key'] = a
    cases = [(OrderedDict(), 'OrderedDict()'), (OrderedDict(((i, i) for i in range(1000, 1010))), 'OrderedDict([(1000, 1000),\n             (1001, 1001),\n             (1002, 1002),\n             (1003, 1003),\n             (1004, 1004),\n             (1005, 1005),\n             (1006, 1006),\n             (1007, 1007),\n             (1008, 1008),\n             (1009, 1009)])'), (a, "OrderedDict([('key', OrderedDict(...))])")]
    for (obj, expected) in cases:
        assert pretty.pretty(obj) == expected

def test_collections_deque():
    if False:
        return 10
    a = deque()
    a.append(a)
    cases = [(deque(), 'deque([])'), (deque([1, 2, 3]), 'deque([1, 2, 3])'), (deque((i for i in range(1000, 1020))), 'deque([1000,\n       1001,\n       1002,\n       1003,\n       1004,\n       1005,\n       1006,\n       1007,\n       1008,\n       1009,\n       1010,\n       1011,\n       1012,\n       1013,\n       1014,\n       1015,\n       1016,\n       1017,\n       1018,\n       1019])'), (a, 'deque([deque(...)])')]
    for (obj, expected) in cases:
        assert pretty.pretty(obj) == expected

def test_collections_counter():
    if False:
        while True:
            i = 10

    class MyCounter(Counter):
        pass
    cases = [(Counter(), 'Counter()'), (Counter(a=1), "Counter({'a': 1})"), (MyCounter(a=1), "MyCounter({'a': 1})")]
    for (obj, expected) in cases:
        assert pretty.pretty(obj) == expected

def test_cyclic_list():
    if False:
        while True:
            i = 10
    x = []
    x.append(x)
    assert pretty.pretty(x) == '[[...]]'

def test_cyclic_dequeue():
    if False:
        return 10
    x = deque()
    x.append(x)
    assert pretty.pretty(x) == 'deque([deque(...)])'

class HashItAnyway:

    def __init__(self, value):
        if False:
            return 10
        self.value = value

    def __hash__(self):
        if False:
            print('Hello World!')
        return 0

    def __eq__(self, other):
        if False:
            return 10
        return isinstance(other, HashItAnyway) and self.value == other.value

    def __ne__(self, other):
        if False:
            while True:
                i = 10
        return not self.__eq__(other)

    def _repr_pretty_(self, pretty, cycle):
        if False:
            print('Hello World!')
        pretty.pretty(self.value)

def test_cyclic_counter():
    if False:
        print('Hello World!')
    c = Counter()
    k = HashItAnyway(c)
    c[k] = 1
    assert pretty.pretty(c) == 'Counter({Counter(...): 1})'

def test_cyclic_dict():
    if False:
        for i in range(10):
            print('nop')
    x = {}
    k = HashItAnyway(x)
    x[k] = x
    assert pretty.pretty(x) == '{{...}: {...}}'

def test_cyclic_set():
    if False:
        i = 10
        return i + 15
    x = set()
    x.add(HashItAnyway(x))
    assert pretty.pretty(x) == '{{...}}'

class BigList(list):

    def _repr_pretty_(self, printer, cycle):
        if False:
            for i in range(10):
                print('nop')
        if cycle:
            return '[...]'
        else:
            with printer.group(open='[', close=']'):
                with printer.indent(5):
                    for v in self:
                        printer.pretty(v)
                        printer.breakable(',')

def test_print_with_indent():
    if False:
        print('Hello World!')
    pretty.pretty(BigList([1, 2, 3]))

class MyException(Exception):
    pass

def test_exception():
    if False:
        return 10
    assert pretty.pretty(ValueError('hi')) == "ValueError('hi')"
    assert pretty.pretty(ValueError('hi', 'there')) == "ValueError('hi', 'there')"
    assert 'test_pretty.' in pretty.pretty(MyException())

def test_re_evals():
    if False:
        i = 10
        return i + 15
    for r in [re.compile('hi'), re.compile('b\\nc', re.MULTILINE), re.compile(b'hi', 0), re.compile('foo', re.MULTILINE | re.UNICODE)]:
        r2 = eval(pretty.pretty(r), globals())
        assert r.pattern == r2.pattern
        assert r.flags == r2.flags

def test_print_builtin_function():
    if False:
        print('Hello World!')
    assert pretty.pretty(abs) == 'abs'

def test_pretty_function():
    if False:
        print('Hello World!')
    assert pretty.pretty(test_pretty_function) == 'test_pretty_function'

def test_breakable_at_group_boundary():
    if False:
        return 10
    assert '\n' in pretty.pretty([[], '0' * 80])

@pytest.mark.parametrize('obj, rep', [(float('nan'), 'nan'), (-float('nan'), '-nan'), (SIGNALING_NAN, 'nan  # Saw 1 signaling NaN'), (-SIGNALING_NAN, '-nan  # Saw 1 signaling NaN'), ((SIGNALING_NAN, SIGNALING_NAN), '(nan, nan)  # Saw 2 signaling NaNs')])
def test_nan_reprs(obj, rep):
    if False:
        for i in range(10):
            print('nop')
    assert pretty.pretty(obj) == rep

def _repr_call(*args, **kwargs):
    if False:
        return 10
    p = pretty.RepresentationPrinter()
    p.repr_call(*args, **kwargs)
    return p.getvalue()

@pytest.mark.parametrize('func_name', ['f', 'lambda: ...', 'lambda *args: ...'])
def test_repr_call(func_name):
    if False:
        for i in range(10):
            print('nop')
    fn = f'({func_name})' if func_name.startswith(('lambda:', 'lambda ')) else func_name
    aas = 'a' * 100
    assert _repr_call(func_name, (1, 2), {}) == f'{fn}(1, 2)'
    assert _repr_call(func_name, (aas,), {}) == f'{fn}(\n    {aas!r},\n)'
    assert _repr_call(func_name, (), {'a': 1, 'b': 2}) == f'{fn}(a=1, b=2)'
    assert _repr_call(func_name, (), {'x': aas}) == f'{fn}(\n    x={aas!r},\n)'

class AnEnum(Enum):
    SOME_MEMBER = 1

class Options(Flag):
    A = 1
    B = 2
    C = 4

class EvilReprOptions(Flag):
    A = 1
    B = 2

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return "can't parse this nonsense"

class LyingReprOptions(Flag):
    A = 1
    B = 2

    def __repr__(self):
        if False:
            while True:
                i = 10
        return 'LyingReprOptions.A|B|C'

@pytest.mark.parametrize('rep', ['AnEnum.SOME_MEMBER', 'Options.A', 'Options.A | Options.B', 'Options.A | Options.B | Options.C', 'Options(0)', 'EvilReprOptions.A', 'LyingReprOptions.A', 'EvilReprOptions.A | EvilReprOptions.B', 'LyingReprOptions.A | LyingReprOptions.B'])
def test_pretty_prints_enums_as_code(rep):
    if False:
        while True:
            i = 10
    assert pretty.pretty(eval(rep)) == rep

class Obj:

    def _repr_pretty_(self, p, cycle):
        if False:
            while True:
                i = 10
        'Exercise the IPython callback interface.'
        assert not cycle
        with p.indent(2):
            p.text('abc,')
            p.breakable(' ')
            p.break_()
        p.begin_group(8, '<')
        p.end_group(8, '>')

def test_supports_ipython_callback():
    if False:
        return 10
    assert pretty.pretty(Obj()) == 'abc, \n  <>'