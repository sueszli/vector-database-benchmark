from abc import ABC
from robot.errors import DataError
from robot.model import SuiteVisitor, TagPattern
from robot.utils import html_escape, Matcher, plural_or_not

class KeywordRemover(SuiteVisitor, ABC):
    message = 'Content removed using the --remove-keywords option.'

    def __init__(self):
        if False:
            print('Hello World!')
        self.removal_message = RemovalMessage(self.message)

    @classmethod
    def from_config(cls, conf):
        if False:
            while True:
                i = 10
        upper = conf.upper()
        if upper.startswith('NAME:'):
            return ByNameKeywordRemover(pattern=conf[5:])
        if upper.startswith('TAG:'):
            return ByTagKeywordRemover(pattern=conf[4:])
        try:
            return {'ALL': AllKeywordsRemover, 'PASSED': PassedKeywordRemover, 'FOR': ForLoopItemsRemover, 'WHILE': WhileLoopItemsRemover, 'WUKS': WaitUntilKeywordSucceedsRemover}[upper]()
        except KeyError:
            raise DataError(f"Expected 'ALL', 'PASSED', 'NAME:<pattern>', 'TAG:<pattern>', 'FOR' or 'WUKS', got '{conf}'.")

    def _clear_content(self, item):
        if False:
            print('Hello World!')
        if item.body:
            item.body.clear()
            self.removal_message.set_to(item)

    def _failed_or_warning_or_error(self, item):
        if False:
            while True:
                i = 10
        return not item.passed or self._warning_or_error(item)

    def _warning_or_error(self, item):
        if False:
            return 10
        finder = WarningAndErrorFinder()
        item.visit(finder)
        return finder.found

class AllKeywordsRemover(KeywordRemover):

    def start_body_item(self, item):
        if False:
            return 10
        self._clear_content(item)

    def start_if(self, item):
        if False:
            for i in range(10):
                print('nop')
        pass

    def start_if_branch(self, item):
        if False:
            for i in range(10):
                print('nop')
        self._clear_content(item)

    def start_try(self, item):
        if False:
            print('Hello World!')
        pass

    def start_try_branch(self, item):
        if False:
            print('Hello World!')
        self._clear_content(item)

class PassedKeywordRemover(KeywordRemover):

    def start_suite(self, suite):
        if False:
            i = 10
            return i + 15
        if not suite.statistics.failed:
            for keyword in (suite.setup, suite.teardown):
                if not self._warning_or_error(keyword):
                    self._clear_content(keyword)

    def visit_test(self, test):
        if False:
            return 10
        if not self._failed_or_warning_or_error(test):
            for item in test.body:
                self._clear_content(item)

    def visit_keyword(self, keyword):
        if False:
            print('Hello World!')
        pass

class ByNameKeywordRemover(KeywordRemover):

    def __init__(self, pattern):
        if False:
            while True:
                i = 10
        super().__init__()
        self._matcher = Matcher(pattern, ignore='_')

    def start_keyword(self, kw):
        if False:
            return 10
        if self._matcher.match(kw.full_name) and (not self._warning_or_error(kw)):
            self._clear_content(kw)

class ByTagKeywordRemover(KeywordRemover):

    def __init__(self, pattern):
        if False:
            print('Hello World!')
        super().__init__()
        self._pattern = TagPattern.from_string(pattern)

    def start_keyword(self, kw):
        if False:
            return 10
        if self._pattern.match(kw.tags) and (not self._warning_or_error(kw)):
            self._clear_content(kw)

class LoopItemsRemover(KeywordRemover, ABC):
    message = '{count} passing item{s} removed using the --remove-keywords option.'

    def _remove_from_loop(self, loop):
        if False:
            while True:
                i = 10
        before = len(loop.body)
        self._remove_keywords(loop.body)
        self.removal_message.set_to_if_removed(loop, before)

    def _remove_keywords(self, body):
        if False:
            return 10
        iterations = body.filter(messages=False)
        for it in iterations[:-1]:
            if not self._failed_or_warning_or_error(it):
                body.remove(it)

class ForLoopItemsRemover(LoopItemsRemover):

    def start_for(self, for_):
        if False:
            return 10
        self._remove_from_loop(for_)

class WhileLoopItemsRemover(LoopItemsRemover):

    def start_while(self, while_):
        if False:
            return 10
        self._remove_from_loop(while_)

class WaitUntilKeywordSucceedsRemover(KeywordRemover):
    message = '{count} failing item{s} removed using the --remove-keywords option.'

    def start_keyword(self, kw):
        if False:
            return 10
        if kw.owner == 'BuiltIn' and kw.name == 'Wait Until Keyword Succeeds':
            before = len(kw.body)
            self._remove_keywords(kw.body)
            self.removal_message.set_to_if_removed(kw, before)

    def _remove_keywords(self, body):
        if False:
            while True:
                i = 10
        keywords = body.filter(messages=False)
        if keywords:
            include_from_end = 2 if keywords[-1].passed else 1
            for kw in keywords[:-include_from_end]:
                if not self._warning_or_error(kw):
                    body.remove(kw)

class WarningAndErrorFinder(SuiteVisitor):

    def __init__(self):
        if False:
            print('Hello World!')
        self.found = False

    def start_suite(self, suite):
        if False:
            return 10
        return not self.found

    def start_test(self, test):
        if False:
            print('Hello World!')
        return not self.found

    def start_keyword(self, keyword):
        if False:
            for i in range(10):
                print('nop')
        return not self.found

    def visit_message(self, msg):
        if False:
            print('Hello World!')
        if msg.level in ('WARN', 'ERROR'):
            self.found = True

class RemovalMessage:

    def __init__(self, message):
        if False:
            for i in range(10):
                print('nop')
        self.message = message

    def set_to_if_removed(self, item, len_before):
        if False:
            for i in range(10):
                print('nop')
        removed = len_before - len(item.body)
        if removed:
            message = self.message.format(count=removed, s=plural_or_not(removed))
            self.set_to(item, message)

    def set_to(self, item, message=None):
        if False:
            i = 10
            return i + 15
        if not item.message:
            start = ''
        elif item.message.startswith('*HTML*'):
            start = item.message[6:].strip() + '<hr>'
        else:
            start = html_escape(item.message) + '<hr>'
        message = message or self.message
        item.message = f'*HTML* {start}<span class="robot-note">{message}</span>'