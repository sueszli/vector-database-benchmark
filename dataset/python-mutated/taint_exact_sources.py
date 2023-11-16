from framework import db_access

def fn(params):
    if False:
        for i in range(10):
            print('nop')
    params['sql'] = 'select value from table where x = %s' % params['test']
    db_access.mysql_dict(params)

def fn(params):
    if False:
        return 10
    params['sql'] = 'select value from table where x = %s' % db_access.escape(params['test'])
    db_access.mysql_dict(params)

def fn(params):
    if False:
        i = 10
        return i + 15
    params['sql'] = 'select xyz from table'
    results = db_access.mysql_dict(params)
    params['sql'] = 'delete from table2 where field = %s' % results
    db_access.mysql_update(params)
    for res in results:
        params['sql'] = 'delete from table2 where field = %s' % res['xyz']
        db_access.mysql_update(params)

def fn(params):
    if False:
        print('Hello World!')
    params['name'] = 'test'
    params['sql'] = 'select * from params where name = %(name)s' % params
    db_access.mysql_update(params)

def fn(params):
    if False:
        return 10
    alt = params
    params['sql'] = 'select * from params where name = %(name)s' % alt
    db_access.mysql_update(params)

def fn(params):
    if False:
        i = 10
        return i + 15
    alt = params
    params['name'] = 'x'
    params['sql'] = 'select * from params where name = %(name)s' % alt
    db_access.mysql_update(params)

def fn(params):
    if False:
        while True:
            i = 10
    alt = params.copy()
    params['name'] = 'x'
    params['sql'] = 'select * from params where name = %(name)s' % alt
    db_access.mysql_update(params)