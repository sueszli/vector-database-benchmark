from datetime import datetime
from robot.errors import DataError

class XmlElementHandler:

    def __init__(self, execution_result, root_handler=None):
        if False:
            for i in range(10):
                print('nop')
        self._stack = [(root_handler or RootHandler(), execution_result)]

    def start(self, elem):
        if False:
            print('Hello World!')
        (handler, result) = self._stack[-1]
        handler = handler.get_child_handler(elem.tag)
        if result is not None:
            result = handler.start(elem, result)
        self._stack.append((handler, result))

    def end(self, elem):
        if False:
            for i in range(10):
                print('nop')
        (handler, result) = self._stack.pop()
        if result is not None:
            handler.end(elem, result)

class ElementHandler:
    element_handlers = {}
    tag = None
    children = frozenset()

    @classmethod
    def register(cls, handler):
        if False:
            while True:
                i = 10
        cls.element_handlers[handler.tag] = handler()
        return handler

    def get_child_handler(self, tag):
        if False:
            while True:
                i = 10
        if tag not in self.children:
            if not self.tag:
                raise DataError(f"Incompatible root element '{tag}'.")
            raise DataError(f"Incompatible child element '{tag}' for '{self.tag}'.")
        return self.element_handlers[tag]

    def start(self, elem, result):
        if False:
            for i in range(10):
                print('nop')
        return result

    def end(self, elem, result):
        if False:
            i = 10
            return i + 15
        pass

    def _legacy_timestamp(self, elem, attr_name):
        if False:
            for i in range(10):
                print('nop')
        ts = elem.get(attr_name)
        if ts == 'N/A' or not ts:
            return None
        ts = ts.ljust(24, '0')
        return datetime(int(ts[:4]), int(ts[4:6]), int(ts[6:8]), int(ts[9:11]), int(ts[12:14]), int(ts[15:17]), int(ts[18:24]))

class RootHandler(ElementHandler):
    children = frozenset(('robot',))

@ElementHandler.register
class RobotHandler(ElementHandler):
    tag = 'robot'
    children = frozenset(('suite', 'statistics', 'errors'))

    def start(self, elem, result):
        if False:
            for i in range(10):
                print('nop')
        generator = elem.get('generator', 'unknown').split()[0].upper()
        result.generated_by_robot = generator == 'ROBOT'
        if result.rpa is None:
            result.rpa = elem.get('rpa', 'false') == 'true'
        return result

@ElementHandler.register
class SuiteHandler(ElementHandler):
    tag = 'suite'
    children = frozenset(('doc', 'metadata', 'meta', 'status', 'kw', 'test', 'suite'))

    def start(self, elem, result):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(result, 'suite'):
            return result.suite.config(name=elem.get('name', ''), source=elem.get('source'), rpa=result.rpa)
        return result.suites.create(name=elem.get('name', ''), source=elem.get('source'), rpa=result.rpa)

    def get_child_handler(self, tag):
        if False:
            return 10
        if tag == 'status':
            return StatusHandler(set_status=False)
        return ElementHandler.get_child_handler(self, tag)

@ElementHandler.register
class TestHandler(ElementHandler):
    tag = 'test'
    children = frozenset(('doc', 'tags', 'tag', 'timeout', 'status', 'kw', 'if', 'for', 'try', 'while', 'variable', 'return', 'break', 'continue', 'error', 'msg'))

    def start(self, elem, result):
        if False:
            while True:
                i = 10
        lineno = elem.get('line')
        if lineno:
            lineno = int(lineno)
        return result.tests.create(name=elem.get('name', ''), lineno=lineno)

@ElementHandler.register
class KeywordHandler(ElementHandler):
    tag = 'kw'
    children = frozenset(('doc', 'arguments', 'arg', 'assign', 'var', 'tags', 'tag', 'timeout', 'status', 'msg', 'kw', 'if', 'for', 'try', 'while', 'variable', 'return', 'break', 'continue', 'error'))

    def start(self, elem, result):
        if False:
            while True:
                i = 10
        elem_type = elem.get('type')
        if not elem_type:
            creator = self._create_keyword
        else:
            creator = getattr(self, '_create_' + elem_type.lower())
        return creator(elem, result)

    def _create_keyword(self, elem, result):
        if False:
            while True:
                i = 10
        try:
            body = result.body
        except AttributeError:
            body = self._get_body_for_suite_level_keyword(result)
        return body.create_keyword(**self._get_keyword_attrs(elem))

    def _get_keyword_attrs(self, elem):
        if False:
            print('Hello World!')
        return {'name': elem.get('name', ''), 'owner': elem.get('owner') or elem.get('library'), 'source_name': elem.get('source_name') or elem.get('sourcename')}

    def _get_body_for_suite_level_keyword(self, result):
        if False:
            for i in range(10):
                print('nop')
        kw_type = 'teardown' if result.tests or result.suites else 'setup'
        keyword = getattr(result, kw_type)
        if not keyword:
            keyword.config(name=f'Implicit {kw_type}', status=keyword.PASS)
        return keyword.body

    def _create_setup(self, elem, result):
        if False:
            while True:
                i = 10
        return result.setup.config(**self._get_keyword_attrs(elem))

    def _create_teardown(self, elem, result):
        if False:
            for i in range(10):
                print('nop')
        return result.teardown.config(**self._get_keyword_attrs(elem))

    def _create_for(self, elem, result):
        if False:
            print('Hello World!')
        return result.body.create_keyword(name=elem.get('name'), type='FOR')

    def _create_foritem(self, elem, result):
        if False:
            print('Hello World!')
        return result.body.create_keyword(name=elem.get('name'), type='ITERATION')
    _create_iteration = _create_foritem

@ElementHandler.register
class ForHandler(ElementHandler):
    tag = 'for'
    children = frozenset(('var', 'value', 'iter', 'status', 'doc', 'msg', 'kw'))

    def start(self, elem, result):
        if False:
            print('Hello World!')
        return result.body.create_for(flavor=elem.get('flavor'), start=elem.get('start'), mode=elem.get('mode'), fill=elem.get('fill'))

@ElementHandler.register
class WhileHandler(ElementHandler):
    tag = 'while'
    children = frozenset(('iter', 'status', 'doc', 'msg', 'kw'))

    def start(self, elem, result):
        if False:
            while True:
                i = 10
        return result.body.create_while(condition=elem.get('condition'), limit=elem.get('limit'), on_limit=elem.get('on_limit'), on_limit_message=elem.get('on_limit_message'))

@ElementHandler.register
class IterationHandler(ElementHandler):
    tag = 'iter'
    children = frozenset(('var', 'doc', 'status', 'kw', 'if', 'for', 'msg', 'try', 'while', 'variable', 'return', 'break', 'continue', 'error'))

    def start(self, elem, result):
        if False:
            while True:
                i = 10
        return result.body.create_iteration()

@ElementHandler.register
class IfHandler(ElementHandler):
    tag = 'if'
    children = frozenset(('branch', 'status', 'doc', 'msg', 'kw'))

    def start(self, elem, result):
        if False:
            i = 10
            return i + 15
        return result.body.create_if()

@ElementHandler.register
class BranchHandler(ElementHandler):
    tag = 'branch'
    children = frozenset(('status', 'kw', 'if', 'for', 'try', 'while', 'msg', 'doc', 'variable', 'return', 'pattern', 'break', 'continue', 'error'))

    def start(self, elem, result):
        if False:
            for i in range(10):
                print('nop')
        if 'variable' in elem.attrib:
            elem.attrib['assign'] = elem.attrib.pop('variable')
        return result.body.create_branch(**elem.attrib)

@ElementHandler.register
class TryHandler(ElementHandler):
    tag = 'try'
    children = frozenset(('branch', 'status', 'doc', 'msg', 'kw'))

    def start(self, elem, result):
        if False:
            i = 10
            return i + 15
        return result.body.create_try()

@ElementHandler.register
class PatternHandler(ElementHandler):
    tag = 'pattern'
    children = frozenset()

    def end(self, elem, result):
        if False:
            print('Hello World!')
        result.patterns += (elem.text or '',)

@ElementHandler.register
class VariableHandler(ElementHandler):
    tag = 'variable'
    children = frozenset(('var', 'status', 'msg', 'kw'))

    def start(self, elem, result):
        if False:
            print('Hello World!')
        return result.body.create_var(name=elem.get('name', ''), scope=elem.get('scope'), separator=elem.get('separator'))

@ElementHandler.register
class ReturnHandler(ElementHandler):
    tag = 'return'
    children = frozenset(('value', 'status', 'msg', 'kw'))

    def start(self, elem, result):
        if False:
            return 10
        return result.body.create_return()

@ElementHandler.register
class ContinueHandler(ElementHandler):
    tag = 'continue'
    children = frozenset(('status', 'msg', 'kw'))

    def start(self, elem, result):
        if False:
            for i in range(10):
                print('nop')
        return result.body.create_continue()

@ElementHandler.register
class BreakHandler(ElementHandler):
    tag = 'break'
    children = frozenset(('status', 'msg', 'kw'))

    def start(self, elem, result):
        if False:
            i = 10
            return i + 15
        return result.body.create_break()

@ElementHandler.register
class ErrorHandler(ElementHandler):
    tag = 'error'
    children = frozenset(('status', 'msg', 'value'))

    def start(self, elem, result):
        if False:
            for i in range(10):
                print('nop')
        return result.body.create_error()

@ElementHandler.register
class MessageHandler(ElementHandler):
    tag = 'msg'

    def end(self, elem, result):
        if False:
            for i in range(10):
                print('nop')
        self._create_message(elem, result.body.create_message)

    def _create_message(self, elem, creator):
        if False:
            return 10
        if 'time' in elem.attrib:
            timestamp = elem.attrib['time']
        else:
            timestamp = self._legacy_timestamp(elem, 'timestamp')
        creator(elem.text or '', elem.get('level', 'INFO'), elem.get('html') in ('true', 'yes'), timestamp)

class ErrorMessageHandler(MessageHandler):

    def end(self, elem, result):
        if False:
            while True:
                i = 10
        self._create_message(elem, result.messages.create)

@ElementHandler.register
class StatusHandler(ElementHandler):
    tag = 'status'

    def __init__(self, set_status=True):
        if False:
            i = 10
            return i + 15
        self.set_status = set_status

    def end(self, elem, result):
        if False:
            for i in range(10):
                print('nop')
        if self.set_status:
            result.status = elem.get('status', 'FAIL')
        if 'start' in elem.attrib:
            result.start_time = elem.attrib['start']
            result.elapsed_time = float(elem.attrib['elapsed'])
        else:
            result.start_time = self._legacy_timestamp(elem, 'starttime')
            result.end_time = self._legacy_timestamp(elem, 'endtime')
        if elem.text:
            result.message = elem.text

@ElementHandler.register
class DocHandler(ElementHandler):
    tag = 'doc'

    def end(self, elem, result):
        if False:
            for i in range(10):
                print('nop')
        try:
            result.doc = elem.text or ''
        except AttributeError:
            result.message = elem.text or ''

@ElementHandler.register
class MetadataHandler(ElementHandler):
    tag = 'metadata'
    children = frozenset(('item',))

@ElementHandler.register
class MetadataItemHandler(ElementHandler):
    tag = 'item'

    def end(self, elem, result):
        if False:
            return 10
        result.metadata[elem.get('name', '')] = elem.text or ''

@ElementHandler.register
class MetaHandler(ElementHandler):
    tag = 'meta'

    def end(self, elem, result):
        if False:
            i = 10
            return i + 15
        result.metadata[elem.get('name', '')] = elem.text or ''

@ElementHandler.register
class TagsHandler(ElementHandler):
    tag = 'tags'
    children = frozenset(('tag',))

@ElementHandler.register
class TagHandler(ElementHandler):
    tag = 'tag'

    def end(self, elem, result):
        if False:
            i = 10
            return i + 15
        result.tags.add(elem.text or '')

@ElementHandler.register
class TimeoutHandler(ElementHandler):
    tag = 'timeout'

    def end(self, elem, result):
        if False:
            i = 10
            return i + 15
        result.timeout = elem.get('value')

@ElementHandler.register
class AssignHandler(ElementHandler):
    tag = 'assign'
    children = frozenset(('var',))

@ElementHandler.register
class VarHandler(ElementHandler):
    tag = 'var'

    def end(self, elem, result):
        if False:
            i = 10
            return i + 15
        value = elem.text or ''
        if result.type in (result.KEYWORD, result.FOR):
            result.assign += (value,)
        elif result.type == result.ITERATION:
            result.assign[elem.get('name')] = value
        elif result.type == result.VAR:
            result.value += (value,)
        else:
            raise DataError(f"Invalid element '{elem}' for result '{result!r}'.")

@ElementHandler.register
class ArgumentsHandler(ElementHandler):
    tag = 'arguments'
    children = frozenset(('arg',))

@ElementHandler.register
class ArgumentHandler(ElementHandler):
    tag = 'arg'

    def end(self, elem, result):
        if False:
            print('Hello World!')
        result.args += (elem.text or '',)

@ElementHandler.register
class ValueHandler(ElementHandler):
    tag = 'value'

    def end(self, elem, result):
        if False:
            return 10
        result.values += (elem.text or '',)

@ElementHandler.register
class ErrorsHandler(ElementHandler):
    tag = 'errors'

    def start(self, elem, result):
        if False:
            return 10
        return result.errors

    def get_child_handler(self, tag):
        if False:
            for i in range(10):
                print('nop')
        return ErrorMessageHandler()

@ElementHandler.register
class StatisticsHandler(ElementHandler):
    tag = 'statistics'

    def get_child_handler(self, tag):
        if False:
            i = 10
            return i + 15
        return self