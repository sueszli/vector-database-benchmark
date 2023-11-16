from robot.errors import DataError
from robot.model import SuiteVisitor
from robot.utils import ET, ETSource, get_error_message, html_escape
from .executionresult import Result, CombinedResult
from .flattenkeywordmatcher import FlattenByNameMatcher, FlattenByTypeMatcher, FlattenByTagMatcher
from .merger import Merger
from .xmlelementhandlers import XmlElementHandler

def ExecutionResult(*sources, **options):
    if False:
        for i in range(10):
            print('nop')
    'Factory method to constructs :class:`~.executionresult.Result` objects.\n\n    :param sources: XML source(s) containing execution results.\n        Can be specified as paths, opened file objects, or strings/bytes\n        containing XML directly. Support for bytes is new in RF 3.2.\n    :param options: Configuration options.\n        Using ``merge=True`` causes multiple results to be combined so that\n        tests in the latter results replace the ones in the original.\n        Setting ``rpa`` either to ``True`` (RPA mode) or ``False`` (test\n        automation) sets execution mode explicitly. By default it is got\n        from processed output files and conflicting modes cause an error.\n        Other options are passed directly to the\n        :class:`ExecutionResultBuilder` object used internally.\n    :returns: :class:`~.executionresult.Result` instance.\n\n    Should be imported by external code via the :mod:`robot.api` package.\n    See the :mod:`robot.result` package for a usage example.\n    '
    if not sources:
        raise DataError('One or more data source needed.')
    if options.pop('merge', False):
        return _merge_results(sources[0], sources[1:], options)
    if len(sources) > 1:
        return _combine_results(sources, options)
    return _single_result(sources[0], options)

def _merge_results(original, merged, options):
    if False:
        print('Hello World!')
    result = ExecutionResult(original, **options)
    merger = Merger(result, rpa=result.rpa)
    for path in merged:
        merged = ExecutionResult(path, **options)
        merger.merge(merged)
    return result

def _combine_results(sources, options):
    if False:
        for i in range(10):
            print('nop')
    return CombinedResult((ExecutionResult(src, **options) for src in sources))

def _single_result(source, options):
    if False:
        i = 10
        return i + 15
    ets = ETSource(source)
    result = Result(source, rpa=options.pop('rpa', None))
    try:
        return ExecutionResultBuilder(ets, **options).build(result)
    except IOError as err:
        error = err.strerror
    except:
        error = get_error_message()
    raise DataError(f"Reading XML source '{ets}' failed: {error}")

class ExecutionResultBuilder:
    """Builds :class:`~.executionresult.Result` objects based on output files.

    Instead of using this builder directly, it is recommended to use the
    :func:`ExecutionResult` factory method.
    """

    def __init__(self, source, include_keywords=True, flattened_keywords=None):
        if False:
            print('Hello World!')
        '\n        :param source: Path to the XML output file to build\n            :class:`~.executionresult.Result` objects from.\n        :param include_keywords: Controls whether to include keywords and control\n            structures like FOR and IF in the result or not. They are not needed\n            when generating only a report.\n        :param flattened_keywords: List of patterns controlling what keywords\n            and control structures to flatten. See the documentation of\n            the ``--flattenkeywords`` option for more details.\n        '
        self._source = source if isinstance(source, ETSource) else ETSource(source)
        self._include_keywords = include_keywords
        self._flattened_keywords = flattened_keywords

    def build(self, result):
        if False:
            for i in range(10):
                print('nop')
        handler = XmlElementHandler(result)
        with self._source as source:
            self._parse(source, handler.start, handler.end)
        result.handle_suite_teardown_failures()
        if not self._include_keywords:
            result.suite.visit(RemoveKeywords())
        return result

    def _parse(self, source, start, end):
        if False:
            print('Hello World!')
        context = ET.iterparse(source, events=('start', 'end'))
        if not self._include_keywords:
            context = self._omit_keywords(context)
        elif self._flattened_keywords:
            context = self._flatten_keywords(context, self._flattened_keywords)
        for (event, elem) in context:
            if event == 'start':
                start(elem)
            else:
                end(elem)
                elem.clear()

    def _omit_keywords(self, context):
        if False:
            while True:
                i = 10
        omitted_kws = 0
        for (event, elem) in context:
            omit = elem.tag in ('kw', 'for', 'if') and elem.get('type') != 'TEARDOWN'
            start = event == 'start'
            if omit and start:
                omitted_kws += 1
            if not omitted_kws:
                yield (event, elem)
            elif not start:
                elem.clear()
            if omit and (not start):
                omitted_kws -= 1

    def _flatten_keywords(self, context, flattened):
        if False:
            print('Hello World!')
        (name_match, by_name) = self._get_matcher(FlattenByNameMatcher, flattened)
        (type_match, by_type) = self._get_matcher(FlattenByTypeMatcher, flattened)
        (tags_match, by_tags) = self._get_matcher(FlattenByTagMatcher, flattened)
        started = -1
        tags = []
        containers = {'kw', 'for', 'while', 'iter', 'if', 'try'}
        inside = 0
        for (event, elem) in context:
            tag = elem.tag
            if event == 'start':
                if tag in containers:
                    inside += 1
                    if started >= 0:
                        started += 1
                    elif by_name and name_match(elem.get('name', ''), elem.get('owner') or elem.get('library')):
                        started = 0
                    elif by_type and type_match(tag):
                        started = 0
                    tags = []
            elif tag in containers:
                inside -= 1
            elif by_tags and inside and (started < 0) and (tag == 'tag'):
                tags.append(elem.text or '')
                if tags_match(tags):
                    started = 0
            elif started == 0 and tag == 'status':
                elem.text = self._create_flattened_message(elem.text)
            if started <= 0 or tag == 'msg':
                yield (event, elem)
            else:
                elem.clear()
            if started >= 0 and event == 'end' and (tag in containers):
                started -= 1

    def _create_flattened_message(self, original):
        if False:
            i = 10
            return i + 15
        if not original:
            start = ''
        elif original.startswith('*HTML*'):
            start = original[6:].strip() + '<hr>'
        else:
            start = html_escape(original) + '<hr>'
        return f'*HTML* {start}<span class="robot-note">Content flattened.</span>'

    def _get_matcher(self, matcher_class, flattened):
        if False:
            for i in range(10):
                print('nop')
        matcher = matcher_class(flattened)
        return (matcher.match, bool(matcher))

class RemoveKeywords(SuiteVisitor):

    def start_suite(self, suite):
        if False:
            for i in range(10):
                print('nop')
        suite.setup = None
        suite.teardown = None

    def visit_test(self, test):
        if False:
            while True:
                i = 10
        test.body = []