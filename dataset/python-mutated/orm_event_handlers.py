from __future__ import annotations
import logging
import os
import time
import traceback
import sqlalchemy.orm.mapper
from sqlalchemy import event, exc
from airflow.configuration import conf
log = logging.getLogger(__name__)

def setup_event_handlers(engine):
    if False:
        i = 10
        return i + 15
    'Setups event handlers.'
    from airflow.models import import_all_models
    event.listen(sqlalchemy.orm.mapper, 'before_configured', import_all_models, once=True)

    @event.listens_for(engine, 'connect')
    def connect(dbapi_connection, connection_record):
        if False:
            for i in range(10):
                print('nop')
        connection_record.info['pid'] = os.getpid()
    if engine.dialect.name == 'sqlite':

        @event.listens_for(engine, 'connect')
        def set_sqlite_pragma(dbapi_connection, connection_record):
            if False:
                return 10
            cursor = dbapi_connection.cursor()
            cursor.execute('PRAGMA foreign_keys=ON')
            cursor.close()
    if engine.dialect.name == 'mysql':

        @event.listens_for(engine, 'connect')
        def set_mysql_timezone(dbapi_connection, connection_record):
            if False:
                i = 10
                return i + 15
            cursor = dbapi_connection.cursor()
            cursor.execute("SET time_zone = '+00:00'")
            cursor.close()

    @event.listens_for(engine, 'checkout')
    def checkout(dbapi_connection, connection_record, connection_proxy):
        if False:
            print('Hello World!')
        pid = os.getpid()
        if connection_record.info['pid'] != pid:
            connection_record.connection = connection_proxy.connection = None
            raise exc.DisconnectionError(f"Connection record belongs to pid {connection_record.info['pid']}, attempting to check out in pid {pid}")
    if conf.getboolean('debug', 'sqlalchemy_stats', fallback=False):

        @event.listens_for(engine, 'before_cursor_execute')
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            if False:
                while True:
                    i = 10
            conn.info.setdefault('query_start_time', []).append(time.perf_counter())

        @event.listens_for(engine, 'after_cursor_execute')
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            if False:
                for i in range(10):
                    print('nop')
            total = time.perf_counter() - conn.info['query_start_time'].pop()
            file_name = [f"'{f.name}':{f.filename}:{f.lineno}" for f in traceback.extract_stack() if 'sqlalchemy' not in f.filename][-1]
            stack = [f for f in traceback.extract_stack() if 'sqlalchemy' not in f.filename]
            stack_info = '>'.join([f"{f.filename.rpartition('/')[-1]}:{f.name}" for f in stack][-3:])
            conn.info.setdefault('query_start_time', []).append(time.monotonic())
            log.info('@SQLALCHEMY %s |$ %s |$ %s |$  %s ', total, file_name, stack_info, statement.replace('\n', ' '))