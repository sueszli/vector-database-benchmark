from contextlib import contextmanager
import sqlalchemy as sa
from sqlalchemy.ext import compiler
from sqlalchemy.sql.expression import ClauseElement
from sqlalchemy.sql.expression import Executable

class InsertFromSelect(Executable, ClauseElement):
    _execution_options = Executable._execution_options.union({'autocommit': True})

    def __init__(self, table, select):
        if False:
            return 10
        self.table = table
        self.select = select

@compiler.compiles(InsertFromSelect)
def _visit_insert_from_select(element, compiler, **kw):
    if False:
        while True:
            i = 10
    return f'INSERT INTO {compiler.process(element.table, asfrom=True)} {compiler.process(element.select)}'

def sa_version():
    if False:
        i = 10
        return i + 15
    if hasattr(sa, '__version__'):

        def tryint(s):
            if False:
                while True:
                    i = 10
            try:
                return int(s)
            except (ValueError, TypeError):
                return -1
        return tuple(map(tryint, sa.__version__.split('.')))
    return (0, 0, 0)

def Table(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Wrap table creation to add any necessary dialect-specific options'
    kwargs['mysql_character_set'] = 'utf8'
    return sa.Table(*args, **kwargs)

@contextmanager
def withoutSqliteForeignKeys(engine, connection=None):
    if False:
        print('Hello World!')
    conn = connection
    if engine.dialect.name == 'sqlite':
        if conn is None:
            conn = engine.connect()
        assert not getattr(engine, 'fk_disabled', False)
        engine.fk_disabled = True
        conn.execute('pragma foreign_keys=OFF')
    try:
        yield
    finally:
        if engine.dialect.name == 'sqlite':
            engine.fk_disabled = False
            conn.execute('pragma foreign_keys=ON')
            if connection is None:
                conn.close()