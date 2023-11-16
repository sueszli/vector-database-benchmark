from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
import mock
from tests.compat import unittest
from prompt_toolkit.document import Document
from haxor_news.completer import Completer
from haxor_news.settings import freelancer_post_id, who_is_hiring_post_id
from haxor_news.utils import TextUtils

class CompleterTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.completer = Completer(fuzzy_match=False, text_utils=TextUtils())
        self.completer_event = self.create_completer_event()

    def create_completer_event(self):
        if False:
            for i in range(10):
                print('nop')
        return mock.Mock()

    def _get_completions(self, command):
        if False:
            for i in range(10):
                print('nop')
        position = len(command)
        result = set(self.completer.get_completions(Document(text=command, cursor_position=position), self.completer_event))
        return result

    def verify_completions(self, commands, expected):
        if False:
            i = 10
            return i + 15
        result = set()
        for command in commands:
            result.update(self._get_completions(command))
        result_texts = []
        for item in result:
            result_texts.append(item.text)
        assert result_texts
        if len(expected) == 1:
            assert expected[0] in result_texts
        else:
            for item in expected:
                assert item in result_texts

    def test_blank(self):
        if False:
            return 10
        text = ''
        expected = set([])
        result = self._get_completions(text)
        assert result == expected

    def test_no_completions(self):
        if False:
            return 10
        text = 'foo'
        expected = set([])
        result = self._get_completions(text)
        assert result == expected

    def test_command(self):
        if False:
            while True:
                i = 10
        text = ['h']
        expected = ['hn']
        self.verify_completions(text, expected)

    def test_subcommand(self):
        if False:
            return 10
        text = ['hn as']
        expected = ['ask']
        self.verify_completions(text, expected)

    def test_arg_freelance(self):
        if False:
            return 10
        text = ['hn freelance ']
        expected = ['"(?i)(Python|Django)"']
        self.verify_completions(text, expected)

    def test_arg_hiring(self):
        if False:
            return 10
        text = ['hn hiring ']
        expected = ['"(?i)(Python|Django)"']
        self.verify_completions(text, expected)

    def test_arg_limit(self):
        if False:
            for i in range(10):
                print('nop')
        text = ['hn top ']
        expected = ['10']
        self.verify_completions(text, expected)

    def test_arg_user(self):
        if False:
            for i in range(10):
                print('nop')
        text = ['hn user ']
        expected = ['"user"']
        self.verify_completions(text, expected)

    def test_arg_view(self):
        if False:
            i = 10
            return i + 15
        text = ['hn view ']
        expected = ['1']
        self.verify_completions(text, expected)

    def test_option_freelance(self):
        if False:
            print('Hello World!')
        text = ['hn freelance "" ']
        expected = ['--id_post ' + str(freelancer_post_id), '-i ' + str(freelancer_post_id)]
        self.verify_completions(text, expected)

    def test_option_hiring(self):
        if False:
            i = 10
            return i + 15
        text = ['hn hiring "" ']
        expected = ['--id_post ' + str(who_is_hiring_post_id), '-i ' + str(who_is_hiring_post_id)]
        self.verify_completions(text, expected)

    def test_option_user(self):
        if False:
            print('Hello World!')
        text = ['hn user "" ']
        expected = ['--limit 10', '-l 10']
        self.verify_completions(text, expected)

    def test_option_view(self):
        if False:
            return 10
        text = ['hn view 0 ']
        expected = ['--comments_regex_query ""', '-cq ""', '--comments', '-c', '--comments_recent', '-cr', '--comments_unseen', '-cu', '--comments_hide_non_matching', '-ch', '--clear_cache', '-cc', '--browser', '-b']
        self.verify_completions(text, expected)

    def test_completing_option(self):
        if False:
            while True:
                i = 10
        text = ['hn view 0 -']
        expected = ['--comments_regex_query ""', '-cq ""', '--comments', '-c', '--comments_recent', '-cr', '--comments_unseen', '-cu', '--comments_hide_non_matching', '-ch', '--clear_cache', '-cc', '--browser', '-b']
        self.verify_completions(text, expected)

    def test_multiple_options(self):
        if False:
            while True:
                i = 10
        text = ['hn view 0 -c --brow']
        expected = ['--browser']
        self.verify_completions(text, expected)

    def test_fuzzy(self):
        if False:
            return 10
        text = ['hn vw']
        expected = ['view']
        self.completer.fuzzy_match = True
        self.verify_completions(text, expected)