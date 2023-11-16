import re
from robot.output import LEVELS
from robot.result import Error, Keyword, Message, Return
from .jsbuildingcontext import JsBuildingContext
from .jsexecutionresult import JsExecutionResult
STATUSES = {'FAIL': 0, 'PASS': 1, 'SKIP': 2, 'NOT RUN': 3}
KEYWORD_TYPES = {'KEYWORD': 0, 'SETUP': 1, 'TEARDOWN': 2, 'FOR': 3, 'ITERATION': 4, 'IF': 5, 'ELSE IF': 6, 'ELSE': 7, 'RETURN': 8, 'VAR': 9, 'TRY': 10, 'EXCEPT': 11, 'FINALLY': 12, 'WHILE': 13, 'CONTINUE': 14, 'BREAK': 15, 'ERROR': 16}

class JsModelBuilder:

    def __init__(self, log_path=None, split_log=False, expand_keywords=None, prune_input_to_save_memory=False):
        if False:
            while True:
                i = 10
        self._context = JsBuildingContext(log_path, split_log, expand_keywords, prune_input_to_save_memory)

    def build_from(self, result_from_xml):
        if False:
            print('Hello World!')
        return JsExecutionResult(statistics=StatisticsBuilder().build(result_from_xml.statistics), suite=SuiteBuilder(self._context).build(result_from_xml.suite), errors=ErrorsBuilder(self._context).build(result_from_xml.errors), strings=self._context.strings, basemillis=self._context.basemillis, split_results=self._context.split_results, min_level=self._context.min_level, expand_keywords=self._context.expand_keywords)

class Builder:
    robot_note = re.compile('<span class="robot-note">(.*)</span>')

    def __init__(self, context: JsBuildingContext):
        if False:
            for i in range(10):
                print('nop')
        self._context = context
        self._string = self._context.string
        self._html = self._context.html
        self._timestamp = self._context.timestamp

    def _get_status(self, item, note_only=False):
        if False:
            return 10
        model = (STATUSES[item.status], self._timestamp(item.start_time), round(item.elapsed_time.total_seconds() * 1000))
        msg = item.message
        if not msg:
            return model
        if note_only:
            if msg.startswith('*HTML*'):
                match = self.robot_note.search(msg)
                if match:
                    index = self._string(match.group(1))
                    return model + (index,)
            return model
        if msg.startswith('*HTML*'):
            index = self._string(msg[6:].lstrip(), escape=False)
        else:
            index = self._string(msg)
        return model + (index,)

    def _build_body(self, body, split=False):
        if False:
            while True:
                i = 10
        splitting = self._context.start_splitting_if_needed(split)
        model = tuple([self._build_body_item(item) for item in body])
        return model if not splitting else self._context.end_splitting(model)

    def _build_body_item(self, item):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

class SuiteBuilder(Builder):

    def __init__(self, context):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(context)
        self._build_suite = self.build
        self._build_test = TestBuilder(context).build
        self._build_body_item = BodyItemBuilder(context).build

    def build(self, suite):
        if False:
            while True:
                i = 10
        with self._context.prune_input(suite.tests, suite.suites):
            stats = self._get_statistics(suite)
            fixture = []
            if suite.has_setup:
                fixture.append(suite.setup)
            if suite.has_teardown:
                fixture.append(suite.teardown)
            return (self._string(suite.name, attr=True), self._string(suite.source), self._context.relative_source(suite.source), self._html(suite.doc), tuple(self._yield_metadata(suite)), self._get_status(suite), tuple((self._build_suite(s) for s in suite.suites)), tuple((self._build_test(t) for t in suite.tests)), tuple((self._build_body_item(kw, split=True) for kw in fixture)), stats)

    def _yield_metadata(self, suite):
        if False:
            print('Hello World!')
        for (name, value) in suite.metadata.items():
            yield self._string(name)
            yield self._html(value)

    def _get_statistics(self, suite):
        if False:
            i = 10
            return i + 15
        stats = suite.statistics
        return (stats.total, stats.passed, stats.failed, stats.skipped)

class TestBuilder(Builder):

    def __init__(self, context):
        if False:
            return 10
        super().__init__(context)
        self._build_body_item = BodyItemBuilder(context).build

    def build(self, test):
        if False:
            for i in range(10):
                print('nop')
        body = self._get_body_items(test)
        with self._context.prune_input(test.body):
            return (self._string(test.name, attr=True), self._string(test.timeout), self._html(test.doc), tuple((self._string(t) for t in test.tags)), self._get_status(test), self._build_body(body, split=True))

    def _get_body_items(self, test):
        if False:
            for i in range(10):
                print('nop')
        body = test.body.flatten()
        if test.has_setup:
            body.insert(0, test.setup)
        if test.has_teardown:
            body.append(test.teardown)
        return body

class BodyItemBuilder(Builder):

    def __init__(self, context):
        if False:
            return 10
        super().__init__(context)
        self._build_body_item = self.build
        self._build_message = MessageBuilder(context).build

    def build(self, item, split=False):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(item, Message):
            return self._build_message(item)
        with self._context.prune_input(item.body):
            if isinstance(item, Keyword):
                return self._build_keyword(item, split)
            if isinstance(item, (Return, Error)):
                return self._build(item, args='    '.join(item.values), split=split)
            return self._build(item, item._log_name, split=split)

    def _build_keyword(self, kw: Keyword, split):
        if False:
            for i in range(10):
                print('nop')
        self._context.check_expansion(kw)
        body = kw.body.flatten()
        if kw.has_setup:
            body.insert(0, kw.setup)
        if kw.has_teardown:
            body.append(kw.teardown)
        return self._build(kw, kw.name, kw.owner, kw.timeout, kw.doc, '    '.join(kw.args), '    '.join(kw.assign), ', '.join(kw.tags), body, split=split)

    def _build(self, item, name='', owner='', timeout='', doc='', args='', assign='', tags='', body=None, split=False):
        if False:
            while True:
                i = 10
        if body is None:
            body = item.body.flatten()
        return (KEYWORD_TYPES[item.type], self._string(name, attr=True), self._string(owner, attr=True), self._string(timeout), self._html(doc), self._string(args), self._string(assign), self._string(tags), self._get_status(item, note_only=True), self._build_body(body, split))

class MessageBuilder(Builder):

    def build(self, msg):
        if False:
            for i in range(10):
                print('nop')
        if msg.level in ('WARN', 'ERROR'):
            self._context.create_link_target(msg)
        self._context.message_level(msg.level)
        return self._build(msg)

    def _build(self, msg):
        if False:
            for i in range(10):
                print('nop')
        return (self._timestamp(msg.timestamp), LEVELS[msg.level], self._string(msg.html_message, escape=False))

class StatisticsBuilder:

    def build(self, statistics):
        if False:
            while True:
                i = 10
        return (self._build_stats(statistics.total), self._build_stats(statistics.tags), self._build_stats(statistics.suite, exclude_empty=False))

    def _build_stats(self, stats, exclude_empty=True):
        if False:
            print('Hello World!')
        return tuple((stat.get_attributes(include_label=True, include_elapsed=True, exclude_empty=exclude_empty, html_escape=True) for stat in stats))

class ErrorsBuilder(Builder):

    def __init__(self, context):
        if False:
            while True:
                i = 10
        super().__init__(context)
        self._build_message = ErrorMessageBuilder(context).build

    def build(self, errors):
        if False:
            while True:
                i = 10
        with self._context.prune_input(errors.messages):
            return tuple((self._build_message(msg) for msg in errors))

class ErrorMessageBuilder(MessageBuilder):

    def build(self, msg):
        if False:
            print('Hello World!')
        model = self._build(msg)
        link = self._context.link(msg)
        return model if link is None else model + (link,)