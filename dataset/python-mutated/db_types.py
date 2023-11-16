from __future__ import annotations
import sqlalchemy as sa
from alembic import context
from lazy_object_proxy import Proxy

def _mssql_TIMESTAMP():
    if False:
        i = 10
        return i + 15
    from sqlalchemy.dialects import mssql

    class DATETIME2(mssql.DATETIME2):

        def __init__(self, *args, precision=6, **kwargs):
            if False:
                i = 10
                return i + 15
            super().__init__(*args, precision=precision, **kwargs)
    return DATETIME2

def _mysql_TIMESTAMP():
    if False:
        return 10
    from sqlalchemy.dialects import mysql

    class TIMESTAMP(mysql.TIMESTAMP):

        def __init__(self, *args, fsp=6, timezone=True, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__(*args, fsp=fsp, timezone=timezone, **kwargs)
    return TIMESTAMP

def _sa_TIMESTAMP():
    if False:
        for i in range(10):
            print('nop')

    class TIMESTAMP(sa.TIMESTAMP):

        def __init__(self, *args, timezone=True, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__(*args, timezone=timezone, **kwargs)
    return TIMESTAMP

def _sa_StringID():
    if False:
        for i in range(10):
            print('nop')
    from airflow.models.base import StringID
    return StringID

def __getattr__(name):
    if False:
        return 10
    if name in ['TIMESTAMP', 'StringID']:

        def lazy_load():
            if False:
                i = 10
                return i + 15
            dialect = context.get_bind().dialect.name
            module = globals()
            type_ = module.get(f'_{dialect}_{name}', None) or module.get(f'_sa_{name}')
            val = module[name] = type_()
            return val
        return Proxy(lazy_load)
    raise AttributeError(f'module {__name__} has no attribute {name}')