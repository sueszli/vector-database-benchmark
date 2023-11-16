from __future__ import annotations
import collections
import contextlib
import itertools
import re
from .. import event
from ..engine import url
from ..engine.default import DefaultDialect
from ..schema import BaseDDLElement

class AssertRule:
    is_consumed = False
    errormessage = None
    consume_statement = True

    def process_statement(self, execute_observed):
        if False:
            for i in range(10):
                print('nop')
        pass

    def no_more_statements(self):
        if False:
            return 10
        assert False, 'All statements are complete, but pending assertion rules remain'

class SQLMatchRule(AssertRule):
    pass

class CursorSQL(SQLMatchRule):

    def __init__(self, statement, params=None, consume_statement=True):
        if False:
            i = 10
            return i + 15
        self.statement = statement
        self.params = params
        self.consume_statement = consume_statement

    def process_statement(self, execute_observed):
        if False:
            while True:
                i = 10
        stmt = execute_observed.statements[0]
        if self.statement != stmt.statement or (self.params is not None and self.params != stmt.parameters):
            self.consume_statement = True
            self.errormessage = 'Testing for exact SQL %s parameters %s received %s %s' % (self.statement, self.params, stmt.statement, stmt.parameters)
        else:
            execute_observed.statements.pop(0)
            self.is_consumed = True
            if not execute_observed.statements:
                self.consume_statement = True

class CompiledSQL(SQLMatchRule):

    def __init__(self, statement, params=None, dialect='default', enable_returning=True):
        if False:
            return 10
        self.statement = statement
        self.params = params
        self.dialect = dialect
        self.enable_returning = enable_returning

    def _compare_sql(self, execute_observed, received_statement):
        if False:
            while True:
                i = 10
        stmt = re.sub('[\\n\\t]', '', self.statement)
        return received_statement == stmt

    def _compile_dialect(self, execute_observed):
        if False:
            i = 10
            return i + 15
        if self.dialect == 'default':
            dialect = DefaultDialect()
            dialect.supports_default_metavalue = True
            if self.enable_returning:
                dialect.insert_returning = dialect.update_returning = dialect.delete_returning = True
                dialect.use_insertmanyvalues = True
                dialect.supports_multivalues_insert = True
                dialect.update_returning_multifrom = True
                dialect.delete_returning_multifrom = True
                assert dialect.insert_executemany_returning
            return dialect
        else:
            return url.URL.create(self.dialect).get_dialect()()

    def _received_statement(self, execute_observed):
        if False:
            for i in range(10):
                print('nop')
        'reconstruct the statement and params in terms\n        of a target dialect, which for CompiledSQL is just DefaultDialect.'
        context = execute_observed.context
        compare_dialect = self._compile_dialect(execute_observed)
        cache_key = None
        extracted_parameters = None
        if 'schema_translate_map' in context.execution_options:
            map_ = context.execution_options['schema_translate_map']
        else:
            map_ = None
        if isinstance(execute_observed.clauseelement, BaseDDLElement):
            compiled = execute_observed.clauseelement.compile(dialect=compare_dialect, schema_translate_map=map_)
        else:
            compiled = execute_observed.clauseelement.compile(cache_key=cache_key, dialect=compare_dialect, column_keys=context.compiled.column_keys, for_executemany=context.compiled.for_executemany, schema_translate_map=map_)
        _received_statement = re.sub('[\\n\\t]', '', str(compiled))
        parameters = execute_observed.parameters
        if not parameters:
            _received_parameters = [compiled.construct_params(extracted_parameters=extracted_parameters)]
        else:
            _received_parameters = [compiled.construct_params(m, extracted_parameters=extracted_parameters) for m in parameters]
        return (_received_statement, _received_parameters)

    def process_statement(self, execute_observed):
        if False:
            i = 10
            return i + 15
        context = execute_observed.context
        (_received_statement, _received_parameters) = self._received_statement(execute_observed)
        params = self._all_params(context)
        equivalent = self._compare_sql(execute_observed, _received_statement)
        if equivalent:
            if params is not None:
                all_params = list(params)
                all_received = list(_received_parameters)
                while all_params and all_received:
                    param = dict(all_params.pop(0))
                    for (idx, received) in enumerate(list(all_received)):
                        for param_key in param:
                            if param_key not in received or received[param_key] != param[param_key]:
                                break
                        else:
                            del all_received[idx]
                            break
                    else:
                        equivalent = False
                        break
                if all_params or all_received:
                    equivalent = False
        if equivalent:
            self.is_consumed = True
            self.errormessage = None
        else:
            self.errormessage = self._failure_message(execute_observed, params) % {'received_statement': _received_statement, 'received_parameters': _received_parameters}

    def _all_params(self, context):
        if False:
            for i in range(10):
                print('nop')
        if self.params:
            if callable(self.params):
                params = self.params(context)
            else:
                params = self.params
            if not isinstance(params, list):
                params = [params]
            return params
        else:
            return None

    def _failure_message(self, execute_observed, expected_params):
        if False:
            for i in range(10):
                print('nop')
        return 'Testing for compiled statement\n%r partial params %s, received\n%%(received_statement)r with params %%(received_parameters)r' % (self.statement.replace('%', '%%'), repr(expected_params).replace('%', '%%'))

class RegexSQL(CompiledSQL):

    def __init__(self, regex, params=None, dialect='default', enable_returning=False):
        if False:
            for i in range(10):
                print('nop')
        SQLMatchRule.__init__(self)
        self.regex = re.compile(regex)
        self.orig_regex = regex
        self.params = params
        self.dialect = dialect
        self.enable_returning = enable_returning

    def _failure_message(self, execute_observed, expected_params):
        if False:
            for i in range(10):
                print('nop')
        return 'Testing for compiled statement ~%r partial params %s, received %%(received_statement)r with params %%(received_parameters)r' % (self.orig_regex.replace('%', '%%'), repr(expected_params).replace('%', '%%'))

    def _compare_sql(self, execute_observed, received_statement):
        if False:
            return 10
        return bool(self.regex.match(received_statement))

class DialectSQL(CompiledSQL):

    def _compile_dialect(self, execute_observed):
        if False:
            print('Hello World!')
        return execute_observed.context.dialect

    def _compare_no_space(self, real_stmt, received_stmt):
        if False:
            print('Hello World!')
        stmt = re.sub('[\\n\\t]', '', real_stmt)
        return received_stmt == stmt

    def _received_statement(self, execute_observed):
        if False:
            i = 10
            return i + 15
        (received_stmt, received_params) = super()._received_statement(execute_observed)
        for real_stmt in execute_observed.statements:
            if self._compare_no_space(real_stmt.statement, received_stmt):
                break
        else:
            raise AssertionError("Can't locate compiled statement %r in list of statements actually invoked" % received_stmt)
        return (received_stmt, execute_observed.context.compiled_parameters)

    def _dialect_adjusted_statement(self, dialect):
        if False:
            return 10
        paramstyle = dialect.paramstyle
        stmt = re.sub('[\\n\\t]', '', self.statement)
        stmt = stmt.replace('::', '!!')
        if paramstyle == 'pyformat':
            stmt = re.sub(':([\\w_]+)', '%(\\1)s', stmt)
        else:
            repl = None
            if paramstyle == 'qmark':
                repl = '?'
            elif paramstyle == 'format':
                repl = '%s'
            elif paramstyle.startswith('numeric'):
                counter = itertools.count(1)
                num_identifier = '$' if paramstyle == 'numeric_dollar' else ':'

                def repl(m):
                    if False:
                        for i in range(10):
                            print('nop')
                    return f'{num_identifier}{next(counter)}'
            stmt = re.sub(':([\\w_]+)', repl, stmt)
        stmt = stmt.replace('!!', '::')
        return stmt

    def _compare_sql(self, execute_observed, received_statement):
        if False:
            for i in range(10):
                print('nop')
        stmt = self._dialect_adjusted_statement(execute_observed.context.dialect)
        return received_statement == stmt

    def _failure_message(self, execute_observed, expected_params):
        if False:
            i = 10
            return i + 15
        return 'Testing for compiled statement\n%r partial params %s, received\n%%(received_statement)r with params %%(received_parameters)r' % (self._dialect_adjusted_statement(execute_observed.context.dialect).replace('%', '%%'), repr(expected_params).replace('%', '%%'))

class CountStatements(AssertRule):

    def __init__(self, count):
        if False:
            while True:
                i = 10
        self.count = count
        self._statement_count = 0

    def process_statement(self, execute_observed):
        if False:
            i = 10
            return i + 15
        self._statement_count += 1

    def no_more_statements(self):
        if False:
            print('Hello World!')
        if self.count != self._statement_count:
            assert False, 'desired statement count %d does not match %d' % (self.count, self._statement_count)

class AllOf(AssertRule):

    def __init__(self, *rules):
        if False:
            print('Hello World!')
        self.rules = set(rules)

    def process_statement(self, execute_observed):
        if False:
            i = 10
            return i + 15
        for rule in list(self.rules):
            rule.errormessage = None
            rule.process_statement(execute_observed)
            if rule.is_consumed:
                self.rules.discard(rule)
                if not self.rules:
                    self.is_consumed = True
                break
            elif not rule.errormessage:
                self.errormessage = None
                break
        else:
            self.errormessage = list(self.rules)[0].errormessage

class EachOf(AssertRule):

    def __init__(self, *rules):
        if False:
            while True:
                i = 10
        self.rules = list(rules)

    def process_statement(self, execute_observed):
        if False:
            i = 10
            return i + 15
        if not self.rules:
            self.is_consumed = True
            self.consume_statement = False
        while self.rules:
            rule = self.rules[0]
            rule.process_statement(execute_observed)
            if rule.is_consumed:
                self.rules.pop(0)
            elif rule.errormessage:
                self.errormessage = rule.errormessage
            if rule.consume_statement:
                break
        if not self.rules:
            self.is_consumed = True

    def no_more_statements(self):
        if False:
            for i in range(10):
                print('nop')
        if self.rules and (not self.rules[0].is_consumed):
            self.rules[0].no_more_statements()
        elif self.rules:
            super().no_more_statements()

class Conditional(EachOf):

    def __init__(self, condition, rules, else_rules):
        if False:
            print('Hello World!')
        if condition:
            super().__init__(*rules)
        else:
            super().__init__(*else_rules)

class Or(AllOf):

    def process_statement(self, execute_observed):
        if False:
            i = 10
            return i + 15
        for rule in self.rules:
            rule.process_statement(execute_observed)
            if rule.is_consumed:
                self.is_consumed = True
                break
        else:
            self.errormessage = list(self.rules)[0].errormessage

class SQLExecuteObserved:

    def __init__(self, context, clauseelement, multiparams, params):
        if False:
            print('Hello World!')
        self.context = context
        self.clauseelement = clauseelement
        if multiparams:
            self.parameters = multiparams
        elif params:
            self.parameters = [params]
        else:
            self.parameters = []
        self.statements = []

    def __repr__(self):
        if False:
            return 10
        return str(self.statements)

class SQLCursorExecuteObserved(collections.namedtuple('SQLCursorExecuteObserved', ['statement', 'parameters', 'context', 'executemany'])):
    pass

class SQLAsserter:

    def __init__(self):
        if False:
            print('Hello World!')
        self.accumulated = []

    def _close(self):
        if False:
            for i in range(10):
                print('nop')
        self._final = self.accumulated
        del self.accumulated

    def assert_(self, *rules):
        if False:
            return 10
        rule = EachOf(*rules)
        observed = list(self._final)
        while observed:
            statement = observed.pop(0)
            rule.process_statement(statement)
            if rule.is_consumed:
                break
            elif rule.errormessage:
                assert False, rule.errormessage
        if observed:
            assert False, 'Additional SQL statements remain:\n%s' % observed
        elif not rule.is_consumed:
            rule.no_more_statements()

@contextlib.contextmanager
def assert_engine(engine):
    if False:
        return 10
    asserter = SQLAsserter()
    orig = []

    @event.listens_for(engine, 'before_execute')
    def connection_execute(conn, clauseelement, multiparams, params, execution_options):
        if False:
            print('Hello World!')
        orig[:] = (clauseelement, multiparams, params)

    @event.listens_for(engine, 'after_cursor_execute')
    def cursor_execute(conn, cursor, statement, parameters, context, executemany):
        if False:
            while True:
                i = 10
        if not context:
            return
        if asserter.accumulated and asserter.accumulated[-1].context is context:
            obs = asserter.accumulated[-1]
        else:
            obs = SQLExecuteObserved(context, orig[0], orig[1], orig[2])
            asserter.accumulated.append(obs)
        obs.statements.append(SQLCursorExecuteObserved(statement, parameters, context, executemany))
    try:
        yield asserter
    finally:
        event.remove(engine, 'after_cursor_execute', cursor_execute)
        event.remove(engine, 'before_execute', connection_execute)
        asserter._close()