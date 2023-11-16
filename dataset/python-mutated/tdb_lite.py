import sqlalchemy as sa
import cPickle as pickle

class tdb_lite(object):

    def __init__(self, gc):
        if False:
            while True:
                i = 10
        self.gc = gc

    def make_metadata(self, engine):
        if False:
            i = 10
            return i + 15
        metadata = sa.MetaData(engine)
        metadata.bind.echo = self.gc.sqlprinting
        return metadata

    def index_str(self, table, name, on, where=None):
        if False:
            return 10
        index_str = 'create index idx_%s_' % name
        index_str += table.name
        index_str += ' on ' + table.name + ' (%s)' % on
        if where:
            index_str += ' where %s' % where
        return index_str

    def create_table(self, table, index_commands=None):
        if False:
            i = 10
            return i + 15
        t = table
        if self.gc.db_create_tables:
            if not t.bind.has_table(t.name):
                t.create(checkfirst=False)
                if index_commands:
                    for i in index_commands:
                        t.bind.execute(i)

    def py2db(self, val, return_kind=False):
        if False:
            i = 10
            return i + 15
        if isinstance(val, bool):
            val = 't' if val else 'f'
            kind = 'bool'
        elif isinstance(val, (str, unicode)):
            kind = 'str'
        elif isinstance(val, (int, float, long)):
            kind = 'num'
        elif val is None:
            kind = 'none'
        else:
            kind = 'pickle'
            val = pickle.dumps(val)
        if return_kind:
            return (val, kind)
        else:
            return val

    def db2py(self, val, kind):
        if False:
            return 10
        if kind == 'bool':
            val = True if val is 't' else False
        elif kind == 'num':
            try:
                val = int(val)
            except ValueError:
                val = float(val)
        elif kind == 'none':
            val = None
        elif kind == 'pickle':
            val = pickle.loads(val)
        return val