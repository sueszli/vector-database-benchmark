from sqlalchemy import create_engine

def _get_version(conn):
    if False:
        i = 10
        return i + 15
    sql = "SELECT UNIQUE CLIENT_DRIVER FROM V$SESSION_CONNECT_INFO WHERE SID = SYS_CONTEXT('USERENV', 'SID')"
    return conn.exec_driver_sql(sql).scalar()

def run_thin_mode(url, queue, **kw):
    if False:
        for i in range(10):
            print('nop')
    e = create_engine(url, **kw)
    with e.connect() as conn:
        res = _get_version(conn)
        queue.put((res, e.dialect.is_thin_mode(conn)))
    e.dispose()

def run_thick_mode(url, queue, **kw):
    if False:
        i = 10
        return i + 15
    e = create_engine(url, **kw)
    with e.connect() as conn:
        res = _get_version(conn)
        queue.put((res, e.dialect.is_thin_mode(conn)))
    e.dispose()