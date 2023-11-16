from datetime import datetime
from robot.utils import NullMarkupWriter, safe_str, XmlWriter
from robot.version import get_full_version
from robot.result.visitor import ResultVisitor
from .loggerapi import LoggerApi
from .loggerhelper import IsLogged

class XmlLoggerAdapter(LoggerApi):

    def __init__(self, path, log_level='TRACE', rpa=False, generator='Robot'):
        if False:
            print('Hello World!')
        self.logger = XmlLogger(path, log_level, rpa, generator)

    @property
    def flatten_level(self):
        if False:
            i = 10
            return i + 15
        return self.logger.flatten_level

    def close(self):
        if False:
            while True:
                i = 10
        self.logger.close()

    def set_log_level(self, level):
        if False:
            return 10
        return self.logger.set_log_level(level)

    def start_suite(self, data, result):
        if False:
            print('Hello World!')
        self.logger.start_suite(result)

    def end_suite(self, data, result):
        if False:
            for i in range(10):
                print('nop')
        self.logger.end_suite(result)

    def start_test(self, data, result):
        if False:
            while True:
                i = 10
        self.logger.start_test(result)

    def end_test(self, data, result):
        if False:
            for i in range(10):
                print('nop')
        self.logger.end_test(result)

    def start_keyword(self, data, result):
        if False:
            return 10
        self.logger.start_keyword(result)

    def end_keyword(self, data, result):
        if False:
            while True:
                i = 10
        self.logger.end_keyword(result)

    def start_for(self, data, result):
        if False:
            for i in range(10):
                print('nop')
        self.logger.start_for(result)

    def end_for(self, data, result):
        if False:
            while True:
                i = 10
        self.logger.end_for(result)

    def start_for_iteration(self, data, result):
        if False:
            while True:
                i = 10
        self.logger.start_for_iteration(result)

    def end_for_iteration(self, data, result):
        if False:
            i = 10
            return i + 15
        self.logger.end_for_iteration(result)

    def start_while(self, data, result):
        if False:
            print('Hello World!')
        self.logger.start_while(result)

    def end_while(self, data, result):
        if False:
            return 10
        self.logger.end_while(result)

    def start_while_iteration(self, data, result):
        if False:
            while True:
                i = 10
        self.logger.start_while_iteration(result)

    def end_while_iteration(self, data, result):
        if False:
            print('Hello World!')
        self.logger.end_while_iteration(result)

    def start_if(self, data, result):
        if False:
            i = 10
            return i + 15
        self.logger.start_if(result)

    def end_if(self, data, result):
        if False:
            return 10
        self.logger.end_if(result)

    def start_if_branch(self, data, result):
        if False:
            print('Hello World!')
        self.logger.start_if_branch(result)

    def end_if_branch(self, data, result):
        if False:
            i = 10
            return i + 15
        self.logger.end_if_branch(result)

    def start_try(self, data, result):
        if False:
            for i in range(10):
                print('nop')
        self.logger.start_try(result)

    def end_try(self, data, result):
        if False:
            while True:
                i = 10
        self.logger.end_try(result)

    def start_try_branch(self, data, result):
        if False:
            for i in range(10):
                print('nop')
        self.logger.start_try_branch(result)

    def end_try_branch(self, data, result):
        if False:
            i = 10
            return i + 15
        self.logger.end_try_branch(result)

    def start_var(self, data, result):
        if False:
            i = 10
            return i + 15
        self.logger.start_var(result)

    def end_var(self, data, result):
        if False:
            print('Hello World!')
        self.logger.end_var(result)

    def start_break(self, data, result):
        if False:
            print('Hello World!')
        self.logger.start_break(result)

    def end_break(self, data, result):
        if False:
            return 10
        self.logger.end_break(result)

    def start_continue(self, data, result):
        if False:
            while True:
                i = 10
        self.logger.start_continue(result)

    def end_continue(self, data, result):
        if False:
            return 10
        self.logger.end_continue(result)

    def start_return(self, data, result):
        if False:
            while True:
                i = 10
        self.logger.start_return(result)

    def end_return(self, data, result):
        if False:
            print('Hello World!')
        self.logger.end_return(result)

    def start_error(self, data, result):
        if False:
            i = 10
            return i + 15
        self.logger.start_error(result)

    def end_error(self, data, result):
        if False:
            i = 10
            return i + 15
        self.logger.end_error(result)

    def log_message(self, message):
        if False:
            i = 10
            return i + 15
        self.logger.log_message(message)

    def message(self, message):
        if False:
            return 10
        self.logger.message(message)

class XmlLogger(ResultVisitor):

    def __init__(self, path, log_level='TRACE', rpa=False, generator='Robot'):
        if False:
            i = 10
            return i + 15
        self._log_message_is_logged = IsLogged(log_level)
        self._error_message_is_logged = IsLogged('WARN')
        self._writer = self._xml_writer = self._get_writer(path, rpa, generator)
        self.flatten_level = 0
        self._errors = []

    def _get_writer(self, path, rpa, generator):
        if False:
            i = 10
            return i + 15
        if not path:
            return NullMarkupWriter()
        writer = XmlWriter(path, write_empty=False, usage='output')
        writer.start('robot', {'generator': get_full_version(generator), 'generated': datetime.now().isoformat(), 'rpa': 'true' if rpa else 'false', 'schemaversion': '5'})
        return writer

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        self.start_errors()
        for msg in self._errors:
            self._write_message(msg)
        self.end_errors()
        self._writer.end('robot')
        self._writer.close()

    def set_log_level(self, level):
        if False:
            for i in range(10):
                print('nop')
        return self._log_message_is_logged.set_level(level)

    def message(self, msg):
        if False:
            return 10
        if self._error_message_is_logged(msg.level):
            self._errors.append(msg)

    def log_message(self, msg):
        if False:
            return 10
        if self._log_message_is_logged(msg.level):
            self._write_message(msg)

    def _write_message(self, msg):
        if False:
            for i in range(10):
                print('nop')
        attrs = {'time': msg.timestamp.isoformat() if msg.timestamp else None, 'level': msg.level}
        if msg.html:
            attrs['html'] = 'true'
        self._xml_writer.element('msg', msg.message, attrs)

    def start_keyword(self, kw):
        if False:
            i = 10
            return i + 15
        attrs = {'name': kw.name, 'owner': kw.owner}
        if kw.type != 'KEYWORD':
            attrs['type'] = kw.type
        if kw.source_name:
            attrs['source_name'] = kw.source_name
        self._writer.start('kw', attrs)
        self._write_list('var', kw.assign)
        self._write_list('arg', [safe_str(a) for a in kw.args])
        self._write_list('tag', kw.tags)
        self._writer.element('doc', kw.doc)
        if kw.tags.robot('flatten'):
            self.flatten_level += 1
            self._writer = NullMarkupWriter()

    def end_keyword(self, kw):
        if False:
            while True:
                i = 10
        if kw.tags.robot('flatten'):
            self.flatten_level -= 1
            if self.flatten_level == 0:
                self._writer = self._xml_writer
        if kw.timeout:
            self._writer.element('timeout', attrs={'value': str(kw.timeout)})
        self._write_status(kw)
        self._writer.end('kw')

    def start_if(self, if_):
        if False:
            while True:
                i = 10
        self._writer.start('if')

    def end_if(self, if_):
        if False:
            return 10
        self._write_status(if_)
        self._writer.end('if')

    def start_if_branch(self, branch):
        if False:
            for i in range(10):
                print('nop')
        self._writer.start('branch', {'type': branch.type, 'condition': branch.condition})

    def end_if_branch(self, branch):
        if False:
            print('Hello World!')
        self._write_status(branch)
        self._writer.end('branch')

    def start_for(self, for_):
        if False:
            while True:
                i = 10
        self._writer.start('for', {'flavor': for_.flavor, 'start': for_.start, 'mode': for_.mode, 'fill': for_.fill})
        for name in for_.assign:
            self._writer.element('var', name)
        for value in for_.values:
            self._writer.element('value', value)

    def end_for(self, for_):
        if False:
            i = 10
            return i + 15
        self._write_status(for_)
        self._writer.end('for')

    def start_for_iteration(self, iteration):
        if False:
            return 10
        self._writer.start('iter')
        for (name, value) in iteration.assign.items():
            self._writer.element('var', value, {'name': name})

    def end_for_iteration(self, iteration):
        if False:
            for i in range(10):
                print('nop')
        self._write_status(iteration)
        self._writer.end('iter')

    def start_try(self, root):
        if False:
            for i in range(10):
                print('nop')
        self._writer.start('try')

    def end_try(self, root):
        if False:
            while True:
                i = 10
        self._write_status(root)
        self._writer.end('try')

    def start_try_branch(self, branch):
        if False:
            for i in range(10):
                print('nop')
        if branch.type == branch.EXCEPT:
            self._writer.start('branch', attrs={'type': 'EXCEPT', 'pattern_type': branch.pattern_type, 'assign': branch.assign})
            self._write_list('pattern', branch.patterns)
        else:
            self._writer.start('branch', attrs={'type': branch.type})

    def end_try_branch(self, branch):
        if False:
            print('Hello World!')
        self._write_status(branch)
        self._writer.end('branch')

    def start_while(self, while_):
        if False:
            return 10
        self._writer.start('while', attrs={'condition': while_.condition, 'limit': while_.limit, 'on_limit': while_.on_limit, 'on_limit_message': while_.on_limit_message})

    def end_while(self, while_):
        if False:
            return 10
        self._write_status(while_)
        self._writer.end('while')

    def start_while_iteration(self, iteration):
        if False:
            return 10
        self._writer.start('iter')

    def end_while_iteration(self, iteration):
        if False:
            return 10
        self._write_status(iteration)
        self._writer.end('iter')

    def start_var(self, var):
        if False:
            return 10
        attr = {'name': var.name}
        if var.scope is not None:
            attr['scope'] = var.scope
        if var.separator is not None:
            attr['separator'] = var.separator
        self._writer.start('variable', attr, write_empty=True)
        for val in var.value:
            self._writer.element('var', val)

    def end_var(self, var):
        if False:
            print('Hello World!')
        self._write_status(var)
        self._writer.end('variable')

    def start_return(self, return_):
        if False:
            return 10
        self._writer.start('return')
        for value in return_.values:
            self._writer.element('value', value)

    def end_return(self, return_):
        if False:
            return 10
        self._write_status(return_)
        self._writer.end('return')

    def start_continue(self, continue_):
        if False:
            for i in range(10):
                print('nop')
        self._writer.start('continue')

    def end_continue(self, continue_):
        if False:
            while True:
                i = 10
        self._write_status(continue_)
        self._writer.end('continue')

    def start_break(self, break_):
        if False:
            i = 10
            return i + 15
        self._writer.start('break')

    def end_break(self, break_):
        if False:
            i = 10
            return i + 15
        self._write_status(break_)
        self._writer.end('break')

    def start_error(self, error):
        if False:
            for i in range(10):
                print('nop')
        self._writer.start('error')
        for value in error.values:
            self._writer.element('value', value)

    def end_error(self, error):
        if False:
            print('Hello World!')
        self._write_status(error)
        self._writer.end('error')

    def start_test(self, test):
        if False:
            i = 10
            return i + 15
        self._writer.start('test', {'id': test.id, 'name': test.name, 'line': str(test.lineno or '')})

    def end_test(self, test):
        if False:
            return 10
        self._writer.element('doc', test.doc)
        self._write_list('tag', test.tags)
        if test.timeout:
            self._writer.element('timeout', attrs={'value': str(test.timeout)})
        self._write_status(test)
        self._writer.end('test')

    def start_suite(self, suite):
        if False:
            i = 10
            return i + 15
        attrs = {'id': suite.id, 'name': suite.name}
        if suite.source:
            attrs['source'] = str(suite.source)
        self._writer.start('suite', attrs)

    def end_suite(self, suite):
        if False:
            print('Hello World!')
        self._writer.element('doc', suite.doc)
        for (name, value) in suite.metadata.items():
            self._writer.element('meta', value, {'name': name})
        self._write_status(suite)
        self._writer.end('suite')

    def start_statistics(self, stats):
        if False:
            for i in range(10):
                print('nop')
        self._writer.start('statistics')

    def end_statistics(self, stats):
        if False:
            print('Hello World!')
        self._writer.end('statistics')

    def start_total_statistics(self, total_stats):
        if False:
            return 10
        self._writer.start('total')

    def end_total_statistics(self, total_stats):
        if False:
            while True:
                i = 10
        self._writer.end('total')

    def start_tag_statistics(self, tag_stats):
        if False:
            for i in range(10):
                print('nop')
        self._writer.start('tag')

    def end_tag_statistics(self, tag_stats):
        if False:
            i = 10
            return i + 15
        self._writer.end('tag')

    def start_suite_statistics(self, tag_stats):
        if False:
            print('Hello World!')
        self._writer.start('suite')

    def end_suite_statistics(self, tag_stats):
        if False:
            i = 10
            return i + 15
        self._writer.end('suite')

    def visit_stat(self, stat):
        if False:
            return 10
        self._writer.element('stat', stat.name, stat.get_attributes(values_as_strings=True))

    def start_errors(self, errors=None):
        if False:
            i = 10
            return i + 15
        self._writer.start('errors')

    def end_errors(self, errors=None):
        if False:
            print('Hello World!')
        self._writer.end('errors')

    def _write_list(self, tag, items):
        if False:
            i = 10
            return i + 15
        for item in items:
            self._writer.element(tag, item)

    def _write_status(self, item):
        if False:
            return 10
        attrs = {'status': item.status, 'start': item.start_time.isoformat() if item.start_time else None, 'elapsed': format(item.elapsed_time.total_seconds(), 'f')}
        self._writer.element('status', item.message, attrs)