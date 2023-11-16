query1 = 'SELECT %s FROM table' % (var,)
query2 = 'SELECT var FROM ' + table
query3 = 'SELECT ' + val + ' FROM ' + table
query4 = 'SELECT {} FROM table;'.format(var)
query5 = f'SELECT * FROM table WHERE var = {var}'
query6 = 'DELETE FROM table WHERE var = %s' % (var,)
query7 = 'DELETE FROM table WHERE VAR = ' + var
query8 = 'DELETE FROM ' + table + 'WHERE var = ' + var
query9 = 'DELETE FROM table WHERE var = {}'.format(var)
query10 = f'DELETE FROM table WHERE var = {var}'
query11 = 'INSERT INTO table VALUES (%s)' % (var,)
query12 = 'INSERT INTO TABLE VALUES (' + var + ')'
query13 = 'INSERT INTO {} VALUES ({})'.format(table, var)
query14 = f'INSERT INTO {table} VALUES var = {var}'
query15 = 'UPDATE %s SET var = %s' % (table, var)
query16 = 'UPDATE ' + table + ' SET var = ' + var
query17 = 'UPDATE {} SET var = {}'.format(table, var)
query18 = f'UPDATE {table} SET var = {var}'
query19 = 'select %s from table' % (var,)
query20 = 'select var from ' + table
query21 = 'select ' + val + ' from ' + table
query22 = 'select {} from table;'.format(var)
query23 = f'select * from table where var = {var}'
query24 = 'delete from table where var = %s' % (var,)
query25 = 'delete from table where var = ' + var
query26 = 'delete from ' + table + 'where var = ' + var
query27 = 'delete from table where var = {}'.format(var)
query28 = f'delete from table where var = {var}'
query29 = 'insert into table values (%s)' % (var,)
query30 = 'insert into table values (' + var + ')'
query31 = 'insert into {} values ({})'.format(table, var)
query32 = f'insert into {table} values var = {var}'
query33 = 'update %s set var = %s' % (table, var)
query34 = 'update ' + table + ' set var = ' + var
query35 = 'update {} set var = {}'.format(table, var)
query36 = f'update {table} set var = {var}'

def query37():
    if False:
        print('Hello World!')
    return '\n    SELECT *\n    FROM table\n    WHERE var = %s\n    ' % var

def query38():
    if False:
        return 10
    return '\n    SELECT *\n    FROM TABLE\n    WHERE var =\n    ' + var

def query39():
    if False:
        i = 10
        return i + 15
    return '\n    SELECT *\n    FROM table\n    WHERE var = {}\n    '.format(var)

def query40():
    if False:
        for i in range(10):
            print('nop')
    return f'\n    SELECT *\n    FROM table\n    WHERE var = {var}\n    '

def query41():
    if False:
        i = 10
        return i + 15
    return f'SELECT * FROM table WHERE var = {var}'
query42 = cursor.execute('SELECT * FROM table WHERE var = %s' % var)
query43 = cursor.execute(f'SELECT * FROM table WHERE var = {var}')
query44 = cursor.execute('SELECT * FROM table WHERE var = {}'.format(var))
query45 = cursor.executemany('SELECT * FROM table WHERE var = %s' % var, [])
query = 'SELECT * FROM table WHERE id = 1'
query = 'DELETE FROM table WHERE id = 1'
query = 'INSERT INTO table VALUES (1)'
query = 'UPDATE table SET id = 1'
cursor.execute('SELECT * FROM table WHERE id = %s', var)
cursor.execute('SELECT * FROM table WHERE id = 1')
cursor.executemany('SELECT * FROM table WHERE id = %s', [var, var2])
query = 'INSERT table VALUES (%s)' % (var,)
query = 'REPLACE INTO table VALUES (%s)' % (var,)
query = 'REPLACE table VALUES (%s)' % (var,)
query = "Deselect something that is not SQL even though it has a ' from ' somewhere in %s." % 'there'