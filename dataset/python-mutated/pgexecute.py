import logging
import traceback
from collections import namedtuple
import re
import pgspecial as special
import psycopg
import psycopg.sql
from psycopg.conninfo import make_conninfo
import sqlparse
from .packages.parseutils.meta import FunctionMetadata, ForeignKey
_logger = logging.getLogger(__name__)
ViewDef = namedtuple('ViewDef', 'nspname relname relkind viewdef reloptions checkoption')

def remove_beginning_comments(command):
    if False:
        return 10
    pattern = '^(/\\*.*?\\*/|--.*?)(?:\\n|$)'
    cleaned_command = command
    comments = []
    match = re.match(pattern, cleaned_command, re.DOTALL)
    while match:
        comments.append(match.group())
        cleaned_command = cleaned_command[len(match.group()):].lstrip()
        match = re.match(pattern, cleaned_command, re.DOTALL)
    return [cleaned_command, comments]

def register_typecasters(connection):
    if False:
        return 10
    "Casts date and timestamp values to string, resolves issues with out-of-range\n    dates (e.g. BC) which psycopg can't handle"
    for forced_text_type in ['date', 'time', 'timestamp', 'timestamptz', 'bytea', 'json', 'jsonb']:
        connection.adapters.register_loader(forced_text_type, psycopg.types.string.TextLoader)

class ProtocolSafeCursor(psycopg.Cursor):
    """This class wraps and suppresses Protocol Errors with pgbouncer database.
    See https://github.com/dbcli/pgcli/pull/1097.
    Pgbouncer database is a virtual database with its own set of commands."""

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.protocol_error = False
        self.protocol_message = ''
        super().__init__(*args, **kwargs)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        if self.protocol_error:
            raise StopIteration
        return super().__iter__()

    def fetchall(self):
        if False:
            while True:
                i = 10
        if self.protocol_error:
            return [(self.protocol_message,)]
        return super().fetchall()

    def fetchone(self):
        if False:
            for i in range(10):
                print('nop')
        if self.protocol_error:
            return (self.protocol_message,)
        return super().fetchone()

    def execute(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        try:
            super().execute(*args, **kwargs)
            self.protocol_error = False
            self.protocol_message = ''
        except psycopg.errors.ProtocolViolation as ex:
            self.protocol_error = True
            self.protocol_message = str(ex)
            _logger.debug('%s: %s' % (ex.__class__.__name__, ex))

class PGExecute:
    search_path_query = '\n        SELECT * FROM unnest(current_schemas(true))'
    schemata_query = '\n        SELECT  nspname\n        FROM    pg_catalog.pg_namespace\n        ORDER BY 1 '
    tables_query = '\n        SELECT  n.nspname schema_name,\n                c.relname table_name\n        FROM    pg_catalog.pg_class c\n                LEFT JOIN pg_catalog.pg_namespace n\n                    ON n.oid = c.relnamespace\n        WHERE   c.relkind = ANY(%s)\n        ORDER BY 1,2;'
    databases_query = '\n        SELECT d.datname\n        FROM pg_catalog.pg_database d\n        ORDER BY 1'
    full_databases_query = '\n        SELECT d.datname as "Name",\n            pg_catalog.pg_get_userbyid(d.datdba) as "Owner",\n            pg_catalog.pg_encoding_to_char(d.encoding) as "Encoding",\n            d.datcollate as "Collate",\n            d.datctype as "Ctype",\n            pg_catalog.array_to_string(d.datacl, E\'\n\') AS "Access privileges"\n        FROM pg_catalog.pg_database d\n        ORDER BY 1'
    socket_directory_query = "\n        SELECT setting\n        FROM pg_settings\n        WHERE name = 'unix_socket_directories'\n    "
    view_definition_query = "\n        WITH v AS (SELECT %s::pg_catalog.regclass::pg_catalog.oid AS v_oid)\n        SELECT nspname, relname, relkind,\n               pg_catalog.pg_get_viewdef(c.oid, true),\n               array_remove(array_remove(c.reloptions,'check_option=local'),\n                            'check_option=cascaded') AS reloptions,\n               CASE\n                 WHEN 'check_option=local' = ANY (c.reloptions) THEN 'LOCAL'::text\n                 WHEN 'check_option=cascaded' = ANY (c.reloptions) THEN 'CASCADED'::text\n                 ELSE NULL\n               END AS checkoption\n        FROM pg_catalog.pg_class c\n        LEFT JOIN pg_catalog.pg_namespace n ON (c.relnamespace = n.oid)\n        JOIN v ON (c.oid = v.v_oid)"
    function_definition_query = '\n        WITH f AS\n            (SELECT %s::pg_catalog.regproc::pg_catalog.oid AS f_oid)\n        SELECT pg_catalog.pg_get_functiondef(f.f_oid)\n        FROM f'

    def __init__(self, database=None, user=None, password=None, host=None, port=None, dsn=None, **kwargs):
        if False:
            print('Hello World!')
        self._conn_params = {}
        self._is_virtual_database = None
        self.conn = None
        self.dbname = None
        self.user = None
        self.password = None
        self.host = None
        self.port = None
        self.server_version = None
        self.extra_args = None
        self.connect(database, user, password, host, port, dsn, **kwargs)
        self.reset_expanded = None

    def is_virtual_database(self):
        if False:
            i = 10
            return i + 15
        if self._is_virtual_database is None:
            self._is_virtual_database = self.is_protocol_error()
        return self._is_virtual_database

    def copy(self):
        if False:
            return 10
        'Returns a clone of the current executor.'
        return self.__class__(**self._conn_params)

    def connect(self, database=None, user=None, password=None, host=None, port=None, dsn=None, **kwargs):
        if False:
            i = 10
            return i + 15
        conn_params = self._conn_params.copy()
        new_params = {'dbname': database, 'user': user, 'password': password, 'host': host, 'port': port, 'dsn': dsn}
        new_params.update(kwargs)
        if new_params['dsn']:
            new_params = {'dsn': new_params['dsn'], 'password': new_params['password']}
            if new_params['password']:
                new_params['dsn'] = make_conninfo(new_params['dsn'], password=new_params.pop('password'))
        conn_params.update({k: v for (k, v) in new_params.items() if v})
        if 'dsn' in conn_params:
            other_params = {k: v for (k, v) in conn_params.items() if k != 'dsn'}
            conn_info = make_conninfo(conn_params['dsn'], **other_params)
        else:
            conn_info = make_conninfo(**conn_params)
        conn = psycopg.connect(conn_info)
        conn.cursor_factory = ProtocolSafeCursor
        self._conn_params = conn_params
        if self.conn:
            self.conn.close()
        self.conn = conn
        self.conn.autocommit = True
        dsn_parameters = conn.info.get_parameters()
        if dsn_parameters:
            self.dbname = dsn_parameters.get('dbname')
            self.user = dsn_parameters.get('user')
            self.host = dsn_parameters.get('host')
            self.port = dsn_parameters.get('port')
        else:
            self.dbname = conn_params.get('database')
            self.user = conn_params.get('user')
            self.host = conn_params.get('host')
            self.port = conn_params.get('port')
        self.password = password
        self.extra_args = kwargs
        if not self.host:
            self.host = 'pgbouncer' if self.is_virtual_database() else self.get_socket_directory()
        self.pid = conn.info.backend_pid
        self.superuser = conn.info.parameter_status('is_superuser') in ('on', '1')
        self.server_version = conn.info.parameter_status('server_version') or ''
        if not self.is_virtual_database():
            register_typecasters(conn)

    @property
    def short_host(self):
        if False:
            for i in range(10):
                print('nop')
        if ',' in self.host:
            (host, _, _) = self.host.partition(',')
        else:
            host = self.host
        (short_host, _, _) = host.partition('.')
        return short_host

    def _select_one(self, cur, sql):
        if False:
            for i in range(10):
                print('nop')
        '\n        Helper method to run a select and retrieve a single field value\n        :param cur: cursor\n        :param sql: string\n        :return: string\n        '
        cur.execute(sql)
        return cur.fetchone()

    def failed_transaction(self):
        if False:
            while True:
                i = 10
        return self.conn.info.transaction_status == psycopg.pq.TransactionStatus.INERROR

    def valid_transaction(self):
        if False:
            print('Hello World!')
        status = self.conn.info.transaction_status
        return status == psycopg.pq.TransactionStatus.ACTIVE or status == psycopg.pq.TransactionStatus.INTRANS

    def run(self, statement, pgspecial=None, exception_formatter=None, on_error_resume=False, explain_mode=False):
        if False:
            return 10
        'Execute the sql in the database and return the results.\n\n        :param statement: A string containing one or more sql statements\n        :param pgspecial: PGSpecial object\n        :param exception_formatter: A callable that accepts an Exception and\n               returns a formatted (title, rows, headers, status) tuple that can\n               act as a query result. If an exception_formatter is not supplied,\n               psycopg2 exceptions are always raised.\n        :param on_error_resume: Bool. If true, queries following an exception\n               (assuming exception_formatter has been supplied) continue to\n               execute.\n\n        :return: Generator yielding tuples containing\n                 (title, rows, headers, status, query, success, is_special)\n        '
        statement = statement.strip()
        if not statement:
            yield (None, None, None, None, statement, False, False)
        removed_comments = []
        sqlarr = []
        cleaned_command = ''
        (cleaned_command, removed_comments) = remove_beginning_comments(statement)
        sqlarr = sqlparse.split(cleaned_command)
        if len(removed_comments) > 0:
            sqlarr = removed_comments + sqlarr
        for sql in sqlarr:
            sql = sql.rstrip(';')
            sql = sqlparse.format(sql, strip_comments=False).strip()
            if not sql:
                continue
            try:
                if explain_mode:
                    sql = self.explain_prefix() + sql
                elif pgspecial:
                    if sql.endswith('\\G'):
                        if not pgspecial.expanded_output:
                            pgspecial.expanded_output = True
                            self.reset_expanded = True
                        sql = sql[:-2].strip()
                    _logger.debug('Trying a pgspecial command. sql: %r', sql)
                    try:
                        cur = self.conn.cursor()
                    except psycopg.InterfaceError:
                        cur = None
                    try:
                        response = pgspecial.execute(cur, sql)
                        if cur and cur.protocol_error:
                            yield (None, None, None, cur.protocol_message, statement, False, False)
                            self.connect()
                            continue
                        for result in response:
                            if len(result) < 7:
                                yield (result + (sql, True, True))
                            else:
                                yield result
                        continue
                    except special.CommandNotFound:
                        pass
                yield (self.execute_normal_sql(sql) + (sql, True, False))
            except psycopg.DatabaseError as e:
                _logger.error('sql: %r, error: %r', sql, e)
                _logger.error('traceback: %r', traceback.format_exc())
                if self._must_raise(e) or not exception_formatter:
                    raise
                yield (None, None, None, exception_formatter(e), sql, False, False)
                if not on_error_resume:
                    break
            finally:
                if self.reset_expanded:
                    pgspecial.expanded_output = False
                    self.reset_expanded = None

    def _must_raise(self, e):
        if False:
            i = 10
            return i + 15
        "Return true if e is an error that should not be caught in ``run``.\n\n        An uncaught error will prompt the user to reconnect; as long as we\n        detect that the connection is still open, we catch the error, as\n        reconnecting won't solve that problem.\n\n        :param e: DatabaseError. An exception raised while executing a query.\n\n        :return: Bool. True if ``run`` must raise this exception.\n\n        "
        return self.conn.closed != 0

    def execute_normal_sql(self, split_sql):
        if False:
            for i in range(10):
                print('nop')
        'Returns tuple (title, rows, headers, status)'
        _logger.debug('Regular sql statement. sql: %r', split_sql)
        title = ''

        def handle_notices(n):
            if False:
                while True:
                    i = 10
            nonlocal title
            title = f'{n.message_primary}\n{n.message_detail}\n{title}'
        self.conn.add_notice_handler(handle_notices)
        if self.is_virtual_database() and 'show help' in split_sql.lower():
            res = self.conn.pgconn.exec_(split_sql.encode())
            return (title, None, None, res.command_status.decode())
        cur = self.conn.cursor()
        cur.execute(split_sql)
        if cur.description:
            headers = [x[0] for x in cur.description]
            return (title, cur, headers, cur.statusmessage)
        elif cur.protocol_error:
            _logger.debug('Protocol error, unsupported command.')
            return (title, None, None, cur.protocol_message)
        else:
            _logger.debug('No rows in result.')
            return (title, None, None, cur.statusmessage)

    def search_path(self):
        if False:
            i = 10
            return i + 15
        'Returns the current search path as a list of schema names'
        try:
            with self.conn.cursor() as cur:
                _logger.debug('Search path query. sql: %r', self.search_path_query)
                cur.execute(self.search_path_query)
                return [x[0] for x in cur.fetchall()]
        except psycopg.ProgrammingError:
            fallback = 'SELECT * FROM current_schemas(true)'
            with self.conn.cursor() as cur:
                _logger.debug('Search path query. sql: %r', fallback)
                cur.execute(fallback)
                return cur.fetchone()[0]

    def view_definition(self, spec):
        if False:
            i = 10
            return i + 15
        'Returns the SQL defining views described by `spec`'
        with self.conn.cursor() as cur:
            sql = self.view_definition_query
            _logger.debug('View Definition Query. sql: %r\nspec: %r', sql, spec)
            try:
                cur.execute(sql, (spec,))
            except psycopg.ProgrammingError:
                raise RuntimeError(f'View {spec} does not exist.')
            result = ViewDef(*cur.fetchone())
            if result.relkind == 'm':
                template = 'CREATE OR REPLACE MATERIALIZED VIEW {name} AS \n{stmt}'
            else:
                template = 'CREATE OR REPLACE VIEW {name} AS \n{stmt}'
            return psycopg.sql.SQL(template).format(name=psycopg.sql.Identifier(result.nspname, result.relname), stmt=psycopg.sql.SQL(result.viewdef)).as_string(self.conn)

    def function_definition(self, spec):
        if False:
            while True:
                i = 10
        'Returns the SQL defining functions described by `spec`'
        with self.conn.cursor() as cur:
            sql = self.function_definition_query
            _logger.debug('Function Definition Query. sql: %r\nspec: %r', sql, spec)
            try:
                cur.execute(sql, (spec,))
                result = cur.fetchone()
                return result[0]
            except psycopg.ProgrammingError:
                raise RuntimeError(f'Function {spec} does not exist.')

    def schemata(self):
        if False:
            while True:
                i = 10
        'Returns a list of schema names in the database'
        with self.conn.cursor() as cur:
            _logger.debug('Schemata Query. sql: %r', self.schemata_query)
            cur.execute(self.schemata_query)
            return [x[0] for x in cur.fetchall()]

    def _relations(self, kinds=('r', 'p', 'f', 'v', 'm')):
        if False:
            while True:
                i = 10
        "Get table or view name metadata\n\n        :param kinds: list of postgres relkind filters:\n                'r' - table\n                'p' - partitioned table\n                'f' - foreign table\n                'v' - view\n                'm' - materialized view\n        :return: (schema_name, rel_name) tuples\n        "
        with self.conn.cursor() as cur:
            cur.execute(self.tables_query, [kinds])
            yield from cur

    def tables(self):
        if False:
            i = 10
            return i + 15
        'Yields (schema_name, table_name) tuples'
        yield from self._relations(kinds=['r', 'p', 'f'])

    def views(self):
        if False:
            i = 10
            return i + 15
        'Yields (schema_name, view_name) tuples.\n\n        Includes both views and and materialized views\n        '
        yield from self._relations(kinds=['v', 'm'])

    def _columns(self, kinds=('r', 'p', 'f', 'v', 'm')):
        if False:
            for i in range(10):
                print('nop')
        "Get column metadata for tables and views\n\n        :param kinds: kinds: list of postgres relkind filters:\n                'r' - table\n                'p' - partitioned table\n                'f' - foreign table\n                'v' - view\n                'm' - materialized view\n        :return: list of (schema_name, relation_name, column_name, column_type) tuples\n        "
        if self.conn.info.server_version >= 80400:
            columns_query = '\n                SELECT  nsp.nspname schema_name,\n                        cls.relname table_name,\n                        att.attname column_name,\n                        att.atttypid::regtype::text type_name,\n                        att.atthasdef AS has_default,\n                        pg_catalog.pg_get_expr(def.adbin, def.adrelid, true) as default\n                FROM    pg_catalog.pg_attribute att\n                        INNER JOIN pg_catalog.pg_class cls\n                            ON att.attrelid = cls.oid\n                        INNER JOIN pg_catalog.pg_namespace nsp\n                            ON cls.relnamespace = nsp.oid\n                        LEFT OUTER JOIN pg_attrdef def\n                            ON def.adrelid = att.attrelid\n                            AND def.adnum = att.attnum\n                WHERE   cls.relkind = ANY(%s)\n                        AND NOT att.attisdropped\n                        AND att.attnum  > 0\n                ORDER BY 1, 2, att.attnum'
        else:
            columns_query = '\n                SELECT  nsp.nspname schema_name,\n                        cls.relname table_name,\n                        att.attname column_name,\n                        typ.typname type_name,\n                        NULL AS has_default,\n                        NULL AS default\n                FROM    pg_catalog.pg_attribute att\n                        INNER JOIN pg_catalog.pg_class cls\n                            ON att.attrelid = cls.oid\n                        INNER JOIN pg_catalog.pg_namespace nsp\n                            ON cls.relnamespace = nsp.oid\n                        INNER JOIN pg_catalog.pg_type typ\n                            ON typ.oid = att.atttypid\n                WHERE   cls.relkind = ANY(%s)\n                        AND NOT att.attisdropped\n                        AND att.attnum  > 0\n                ORDER BY 1, 2, att.attnum'
        with self.conn.cursor() as cur:
            cur.execute(columns_query, [kinds])
            yield from cur

    def table_columns(self):
        if False:
            while True:
                i = 10
        yield from self._columns(kinds=['r', 'p', 'f'])

    def view_columns(self):
        if False:
            while True:
                i = 10
        yield from self._columns(kinds=['v', 'm'])

    def databases(self):
        if False:
            print('Hello World!')
        with self.conn.cursor() as cur:
            _logger.debug('Databases Query. sql: %r', self.databases_query)
            cur.execute(self.databases_query)
            return [x[0] for x in cur.fetchall()]

    def full_databases(self):
        if False:
            for i in range(10):
                print('nop')
        with self.conn.cursor() as cur:
            _logger.debug('Databases Query. sql: %r', self.full_databases_query)
            cur.execute(self.full_databases_query)
            headers = [x[0] for x in cur.description]
            return (cur.fetchall(), headers, cur.statusmessage)

    def is_protocol_error(self):
        if False:
            for i in range(10):
                print('nop')
        query = 'SELECT 1'
        with self.conn.cursor() as cur:
            _logger.debug('Simple Query. sql: %r', query)
            cur.execute(query)
            return bool(cur.protocol_error)

    def get_socket_directory(self):
        if False:
            return 10
        with self.conn.cursor() as cur:
            _logger.debug('Socket directory Query. sql: %r', self.socket_directory_query)
            cur.execute(self.socket_directory_query)
            result = cur.fetchone()
            return result[0] if result else ''

    def foreignkeys(self):
        if False:
            while True:
                i = 10
        'Yields ForeignKey named tuples'
        if self.conn.info.server_version < 90000:
            return
        with self.conn.cursor() as cur:
            query = "\n                SELECT s_p.nspname AS parentschema,\n                       t_p.relname AS parenttable,\n                       unnest((\n                        select\n                            array_agg(attname ORDER BY i)\n                        from\n                            (select unnest(confkey) as attnum, generate_subscripts(confkey, 1) as i) x\n                            JOIN pg_catalog.pg_attribute c USING(attnum)\n                            WHERE c.attrelid = fk.confrelid\n                        )) AS parentcolumn,\n                       s_c.nspname AS childschema,\n                       t_c.relname AS childtable,\n                       unnest((\n                        select\n                            array_agg(attname ORDER BY i)\n                        from\n                            (select unnest(conkey) as attnum, generate_subscripts(conkey, 1) as i) x\n                            JOIN pg_catalog.pg_attribute c USING(attnum)\n                            WHERE c.attrelid = fk.conrelid\n                        )) AS childcolumn\n                FROM pg_catalog.pg_constraint fk\n                JOIN pg_catalog.pg_class      t_p ON t_p.oid = fk.confrelid\n                JOIN pg_catalog.pg_namespace  s_p ON s_p.oid = t_p.relnamespace\n                JOIN pg_catalog.pg_class      t_c ON t_c.oid = fk.conrelid\n                JOIN pg_catalog.pg_namespace  s_c ON s_c.oid = t_c.relnamespace\n                WHERE fk.contype = 'f';\n                "
            _logger.debug('Functions Query. sql: %r', query)
            cur.execute(query)
            for row in cur:
                yield ForeignKey(*row)

    def functions(self):
        if False:
            while True:
                i = 10
        'Yields FunctionMetadata named tuples'
        if self.conn.info.server_version >= 110000:
            query = "\n                SELECT n.nspname schema_name,\n                        p.proname func_name,\n                        p.proargnames,\n                        COALESCE(proallargtypes::regtype[], proargtypes::regtype[])::text[],\n                        p.proargmodes,\n                        prorettype::regtype::text return_type,\n                        p.prokind = 'a' is_aggregate,\n                        p.prokind = 'w' is_window,\n                        p.proretset is_set_returning,\n                        d.deptype = 'e' is_extension,\n                        pg_get_expr(proargdefaults, 0) AS arg_defaults\n                FROM pg_catalog.pg_proc p\n                        INNER JOIN pg_catalog.pg_namespace n\n                            ON n.oid = p.pronamespace\n                LEFT JOIN pg_depend d ON d.objid = p.oid and d.deptype = 'e'\n                WHERE p.prorettype::regtype != 'trigger'::regtype\n                ORDER BY 1, 2\n                "
        elif self.conn.info.server_version > 90000:
            query = "\n                SELECT n.nspname schema_name,\n                        p.proname func_name,\n                        p.proargnames,\n                        COALESCE(proallargtypes::regtype[], proargtypes::regtype[])::text[],\n                        p.proargmodes,\n                        prorettype::regtype::text return_type,\n                        p.proisagg is_aggregate,\n                        p.proiswindow is_window,\n                        p.proretset is_set_returning,\n                        d.deptype = 'e' is_extension,\n                        pg_get_expr(proargdefaults, 0) AS arg_defaults\n                FROM pg_catalog.pg_proc p\n                        INNER JOIN pg_catalog.pg_namespace n\n                            ON n.oid = p.pronamespace\n                LEFT JOIN pg_depend d ON d.objid = p.oid and d.deptype = 'e'\n                WHERE p.prorettype::regtype != 'trigger'::regtype\n                ORDER BY 1, 2\n                "
        elif self.conn.info.server_version >= 80400:
            query = "\n                SELECT n.nspname schema_name,\n                        p.proname func_name,\n                        p.proargnames,\n                        COALESCE(proallargtypes::regtype[], proargtypes::regtype[])::text[],\n                        p.proargmodes,\n                        prorettype::regtype::text,\n                        p.proisagg is_aggregate,\n                        false is_window,\n                        p.proretset is_set_returning,\n                        d.deptype = 'e' is_extension,\n                        NULL AS arg_defaults\n                FROM pg_catalog.pg_proc p\n                        INNER JOIN pg_catalog.pg_namespace n\n                            ON n.oid = p.pronamespace\n                LEFT JOIN pg_depend d ON d.objid = p.oid and d.deptype = 'e'\n                WHERE p.prorettype::regtype != 'trigger'::regtype\n                ORDER BY 1, 2\n                "
        else:
            query = "\n                SELECT n.nspname schema_name,\n                        p.proname func_name,\n                        p.proargnames,\n                        NULL arg_types,\n                        NULL arg_modes,\n                        '' ret_type,\n                        p.proisagg is_aggregate,\n                        false is_window,\n                        p.proretset is_set_returning,\n                        d.deptype = 'e' is_extension,\n                        NULL AS arg_defaults\n                FROM pg_catalog.pg_proc p\n                        INNER JOIN pg_catalog.pg_namespace n\n                            ON n.oid = p.pronamespace\n                LEFT JOIN pg_depend d ON d.objid = p.oid and d.deptype = 'e'\n                WHERE p.prorettype::regtype != 'trigger'::regtype\n                ORDER BY 1, 2\n                "
        with self.conn.cursor() as cur:
            _logger.debug('Functions Query. sql: %r', query)
            cur.execute(query)
            for row in cur:
                yield FunctionMetadata(*row)

    def datatypes(self):
        if False:
            i = 10
            return i + 15
        'Yields tuples of (schema_name, type_name)'
        with self.conn.cursor() as cur:
            if self.conn.info.server_version > 90000:
                query = "\n                    SELECT n.nspname schema_name,\n                           t.typname type_name\n                    FROM   pg_catalog.pg_type t\n                           INNER JOIN pg_catalog.pg_namespace n\n                              ON n.oid = t.typnamespace\n                    WHERE ( t.typrelid = 0  -- non-composite types\n                            OR (  -- composite type, but not a table\n                                  SELECT c.relkind = 'c'\n                                  FROM pg_catalog.pg_class c\n                                  WHERE c.oid = t.typrelid\n                                )\n                          )\n                          AND NOT EXISTS( -- ignore array types\n                                SELECT  1\n                                FROM    pg_catalog.pg_type el\n                                WHERE   el.oid = t.typelem AND el.typarray = t.oid\n                              )\n                          AND n.nspname <> 'pg_catalog'\n                          AND n.nspname <> 'information_schema'\n                    ORDER BY 1, 2;\n                    "
            else:
                query = "\n                    SELECT n.nspname schema_name,\n                      pg_catalog.format_type(t.oid, NULL) type_name\n                    FROM pg_catalog.pg_type t\n                         LEFT JOIN pg_catalog.pg_namespace n ON n.oid = t.typnamespace\n                    WHERE (t.typrelid = 0 OR (SELECT c.relkind = 'c' FROM pg_catalog.pg_class c WHERE c.oid = t.typrelid))\n                      AND t.typname !~ '^_'\n                          AND n.nspname <> 'pg_catalog'\n                          AND n.nspname <> 'information_schema'\n                      AND pg_catalog.pg_type_is_visible(t.oid)\n                    ORDER BY 1, 2;\n                "
            _logger.debug('Datatypes Query. sql: %r', query)
            cur.execute(query)
            yield from cur

    def casing(self):
        if False:
            while True:
                i = 10
        'Yields the most common casing for names used in db functions'
        with self.conn.cursor() as cur:
            query = "\n          WITH Words AS (\n                SELECT regexp_split_to_table(prosrc, '\\W+') AS Word, COUNT(1)\n                FROM pg_catalog.pg_proc P\n                JOIN pg_catalog.pg_namespace N ON N.oid = P.pronamespace\n                JOIN pg_catalog.pg_language L ON L.oid = P.prolang\n                WHERE L.lanname IN ('sql', 'plpgsql')\n                AND N.nspname NOT IN ('pg_catalog', 'information_schema')\n                GROUP BY Word\n            ),\n            OrderWords AS (\n                SELECT Word,\n                    ROW_NUMBER() OVER(PARTITION BY LOWER(Word) ORDER BY Count DESC)\n                FROM Words\n                WHERE Word ~* '.*[a-z].*'\n            ),\n            Names AS (\n                --Column names\n                SELECT attname AS Name\n                FROM pg_catalog.pg_attribute\n                UNION -- Table/view names\n                SELECT relname\n                FROM pg_catalog.pg_class\n                UNION -- Function names\n                SELECT proname\n                FROM pg_catalog.pg_proc\n                UNION -- Type names\n                SELECT typname\n                FROM pg_catalog.pg_type\n                UNION -- Schema names\n                SELECT nspname\n                FROM pg_catalog.pg_namespace\n                UNION -- Parameter names\n                SELECT unnest(proargnames)\n                FROM pg_proc\n            )\n            SELECT Word\n            FROM OrderWords\n            WHERE LOWER(Word) IN (SELECT Name FROM Names)\n            AND Row_Number = 1;\n            "
            _logger.debug('Casing Query. sql: %r', query)
            cur.execute(query)
            for row in cur:
                yield row[0]

    def explain_prefix(self):
        if False:
            for i in range(10):
                print('nop')
        return 'EXPLAIN (ANALYZE, COSTS, VERBOSE, BUFFERS, FORMAT JSON) '