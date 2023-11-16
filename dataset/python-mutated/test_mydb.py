from fixtures.mydb import MyDB
import pytest

@pytest.fixture(scope='module')
def cur():
    if False:
        while True:
            i = 10
    print('setting up')
    db = MyDB()
    conn = db.connect('server')
    curs = conn.cursor()
    yield curs
    curs.close()
    conn.close()
    print('closing DB')

def test_johns_id(cur):
    if False:
        i = 10
        return i + 15
    id = cur.execute('select id from employee_db where name=John')
    assert id == 123

def test_toms_id(cur):
    if False:
        print('Hello World!')
    id = cur.execute('select id from employee_db where name=Tom')
    assert id == 789