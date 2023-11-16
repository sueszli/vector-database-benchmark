"""Tests for input manipulation machinery."""
import pytest
from IPython.core.prefilter import AutocallChecker

def test_prefilter():
    if False:
        print('Hello World!')
    'Test user input conversions'
    pairs = [('2+2', '2+2')]
    for (raw, correct) in pairs:
        assert ip.prefilter(raw) == correct

def test_prefilter_shadowed():
    if False:
        return 10

    def dummy_magic(line):
        if False:
            return 10
        pass
    prev_automagic_state = ip.automagic
    ip.automagic = True
    ip.autocall = 0
    try:
        for name in ['if', 'zip', 'get_ipython']:
            ip.register_magic_function(dummy_magic, magic_name=name)
            res = ip.prefilter(name + ' foo')
            assert res == name + ' foo'
            del ip.magics_manager.magics['line'][name]
        for name in ['fi', 'piz', 'nohtypi_teg']:
            ip.register_magic_function(dummy_magic, magic_name=name)
            res = ip.prefilter(name + ' foo')
            assert res != name + ' foo'
            del ip.magics_manager.magics['line'][name]
    finally:
        ip.automagic = prev_automagic_state

def test_autocall_binops():
    if False:
        return 10
    'See https://github.com/ipython/ipython/issues/81'
    ip.magic('autocall 2')
    f = lambda x: x
    ip.user_ns['f'] = f
    try:
        assert ip.prefilter('f 1') == 'f(1)'
        for t in ['f +1', 'f -1']:
            assert ip.prefilter(t) == t
        pm = ip.prefilter_manager
        ac = AutocallChecker(shell=pm.shell, prefilter_manager=pm, config=pm.config)
        try:
            ac.priority = 1
            ac.exclude_regexp = '^[,&^\\|\\*/]|^is |^not |^in |^and |^or '
            pm.sort_checkers()
            assert ip.prefilter('f -1') == 'f(-1)'
            assert ip.prefilter('f +1') == 'f(+1)'
        finally:
            pm.unregister_checker(ac)
    finally:
        ip.magic('autocall 0')
        del ip.user_ns['f']

def test_issue_114():
    if False:
        print('Hello World!')
    "Check that multiline string literals don't expand as magic\n    see http://github.com/ipython/ipython/issues/114"
    template = '"""\n%s\n"""'
    msp = ip.prefilter_manager.multi_line_specials
    ip.prefilter_manager.multi_line_specials = False
    try:
        for mgk in ip.magics_manager.lsmagic()['line']:
            raw = template % mgk
            assert ip.prefilter(raw) == raw
    finally:
        ip.prefilter_manager.multi_line_specials = msp

def test_prefilter_attribute_errors():
    if False:
        while True:
            i = 10
    'Capture exceptions thrown by user objects on attribute access.\n\n    See http://github.com/ipython/ipython/issues/988.'

    class X(object):

        def __getattr__(self, k):
            if False:
                while True:
                    i = 10
            raise ValueError('broken object')

        def __call__(self, x):
            if False:
                return 10
            return x
    ip.user_ns['x'] = X()
    ip.magic('autocall 2')
    try:
        ip.prefilter('x 1')
    finally:
        del ip.user_ns['x']
        ip.magic('autocall 0')

def test_autocall_should_support_unicode():
    if False:
        print('Hello World!')
    ip.magic('autocall 2')
    ip.user_ns['π'] = lambda x: x
    try:
        assert ip.prefilter('π 3') == 'π(3)'
    finally:
        ip.magic('autocall 0')
        del ip.user_ns['π']