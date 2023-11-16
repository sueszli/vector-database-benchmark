"""Tests for the TerminalInteractiveShell and related pieces."""
import sys
import unittest
import os
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from IPython.testing import tools as tt
from IPython.terminal.ptutils import _elide, _adjust_completion_text_based_on_context
from IPython.terminal.shortcuts.auto_suggest import NavigableAutoSuggestFromHistory

class TestAutoSuggest(unittest.TestCase):

    def test_changing_provider(self):
        if False:
            while True:
                i = 10
        ip = get_ipython()
        ip.autosuggestions_provider = None
        self.assertEqual(ip.auto_suggest, None)
        ip.autosuggestions_provider = 'AutoSuggestFromHistory'
        self.assertIsInstance(ip.auto_suggest, AutoSuggestFromHistory)
        ip.autosuggestions_provider = 'NavigableAutoSuggestFromHistory'
        self.assertIsInstance(ip.auto_suggest, NavigableAutoSuggestFromHistory)

class TestElide(unittest.TestCase):

    def test_elide(self):
        if False:
            i = 10
            return i + 15
        _elide('concatenate((a1, a2, ...), axis', '')
        _elide('concatenate((a1, a2, ..), . axis', '')
        self.assertEqual(_elide('aaaa.bbbb.ccccc.dddddd.eeeee.fffff.gggggg.hhhhhh', ''), 'aaaa.b…g.hhhhhh')
        test_string = os.sep.join(['', 10 * 'a', 10 * 'b', 10 * 'c', ''])
        expect_string = os.sep + 'a' + '…' + 'b' + os.sep + 10 * 'c'
        self.assertEqual(_elide(test_string, ''), expect_string)

    def test_elide_typed_normal(self):
        if False:
            return 10
        self.assertEqual(_elide('the quick brown fox jumped over the lazy dog', 'the quick brown fox', min_elide=10), 'the…fox jumped over the lazy dog')

    def test_elide_typed_short_match(self):
        if False:
            i = 10
            return i + 15
        '\n        if the match is too short we don\'t elide.\n        avoid the "the...the"\n        '
        self.assertEqual(_elide('the quick brown fox jumped over the lazy dog', 'the', min_elide=10), 'the quick brown fox jumped over the lazy dog')

    def test_elide_typed_no_match(self):
        if False:
            print('Hello World!')
        '\n        if the match is too short we don\'t elide.\n        avoid the "the...the"\n        '
        self.assertEqual(_elide('the quick brown fox jumped over the lazy dog', 'the quick red fox', min_elide=10), 'the quick brown fox jumped over the lazy dog')

class TestContextAwareCompletion(unittest.TestCase):

    def test_adjust_completion_text_based_on_context(self):
        if False:
            return 10
        self.assertEqual(_adjust_completion_text_based_on_context('arg1=', 'func1(a=)', 7), 'arg1')
        self.assertEqual(_adjust_completion_text_based_on_context('arg1=', 'func1(a)', 7), 'arg1=')
        self.assertEqual(_adjust_completion_text_based_on_context('arg1=', 'func1(a', 7), 'arg1=')
        self.assertEqual(_adjust_completion_text_based_on_context('%magic', 'func1(a=)', 7), '%magic')
        self.assertEqual(_adjust_completion_text_based_on_context('func2', 'func1(a=)', 7), 'func2')

class mock_input_helper(object):
    """Machinery for tests of the main interact loop.

    Used by the mock_input decorator.
    """

    def __init__(self, testgen):
        if False:
            return 10
        self.testgen = testgen
        self.exception = None
        self.ip = get_ipython()

    def __enter__(self):
        if False:
            while True:
                i = 10
        self.orig_prompt_for_code = self.ip.prompt_for_code
        self.ip.prompt_for_code = self.fake_input
        return self

    def __exit__(self, etype, value, tb):
        if False:
            print('Hello World!')
        self.ip.prompt_for_code = self.orig_prompt_for_code

    def fake_input(self):
        if False:
            return 10
        try:
            return next(self.testgen)
        except StopIteration:
            self.ip.keep_running = False
            return u''
        except:
            self.exception = sys.exc_info()
            self.ip.keep_running = False
            return u''

def mock_input(testfunc):
    if False:
        return 10
    'Decorator for tests of the main interact loop.\n\n    Write the test as a generator, yield-ing the input strings, which IPython\n    will see as if they were typed in at the prompt.\n    '

    def test_method(self):
        if False:
            print('Hello World!')
        testgen = testfunc(self)
        with mock_input_helper(testgen) as mih:
            mih.ip.interact()
        if mih.exception is not None:
            (etype, value, tb) = mih.exception
            import traceback
            traceback.print_tb(tb, file=sys.stdout)
            del tb
            raise value
    return test_method

class InteractiveShellTestCase(unittest.TestCase):

    def rl_hist_entries(self, rl, n):
        if False:
            while True:
                i = 10
        'Get last n readline history entries as a list'
        return [rl.get_history_item(rl.get_current_history_length() - x) for x in range(n - 1, -1, -1)]

    @mock_input
    def test_inputtransformer_syntaxerror(self):
        if False:
            i = 10
            return i + 15
        ip = get_ipython()
        ip.input_transformers_post.append(syntax_error_transformer)
        try:
            with tt.AssertPrints('4', suppress=False):
                yield u'print(2*2)'
            with tt.AssertPrints('SyntaxError: input contains', suppress=False):
                yield u'print(2345) # syntaxerror'
            with tt.AssertPrints('16', suppress=False):
                yield u'print(4*4)'
        finally:
            ip.input_transformers_post.remove(syntax_error_transformer)

    def test_repl_not_plain_text(self):
        if False:
            for i in range(10):
                print('nop')
        ip = get_ipython()
        formatter = ip.display_formatter
        assert formatter.active_types == ['text/plain']
        assert formatter.ipython_display_formatter.enabled

        class Test(object):

            def __repr__(self):
                if False:
                    while True:
                        i = 10
                return '<Test %i>' % id(self)

            def _repr_html_(self):
                if False:
                    print('Hello World!')
                return '<html>'
        obj = Test()
        (data, _) = formatter.format(obj)
        self.assertEqual(data, {'text/plain': repr(obj)})

        class Test2(Test):

            def _ipython_display_(self):
                if False:
                    return 10
                from IPython.display import display, HTML
                display(HTML('<custom>'))
        called = False

        def handler(data, metadata):
            if False:
                print('Hello World!')
            print('Handler called')
            nonlocal called
            called = True
        ip.display_formatter.active_types.append('text/html')
        ip.display_formatter.formatters['text/html'].enabled = True
        ip.mime_renderers['text/html'] = handler
        try:
            obj = Test()
            display(obj)
        finally:
            ip.display_formatter.formatters['text/html'].enabled = False
            del ip.mime_renderers['text/html']
        assert called == True

def syntax_error_transformer(lines):
    if False:
        i = 10
        return i + 15
    "Transformer that throws SyntaxError if 'syntaxerror' is in the code."
    for line in lines:
        pos = line.find('syntaxerror')
        if pos >= 0:
            e = SyntaxError('input contains "syntaxerror"')
            e.text = line
            e.offset = pos + 1
            raise e
    return lines

class TerminalMagicsTestCase(unittest.TestCase):

    def test_paste_magics_blankline(self):
        if False:
            while True:
                i = 10
        "Test that code with a blank line doesn't get split (gh-3246)."
        ip = get_ipython()
        s = 'def pasted_func(a):\n    b = a+1\n\n    return b'
        tm = ip.magics_manager.registry['TerminalMagics']
        tm.store_or_execute(s, name=None)
        self.assertEqual(ip.user_ns['pasted_func'](54), 55)