import pytest

@pytest.mark.usefixtures('foo_table')
def test_issue105(db):
    if False:
        for i in range(10):
            print('nop')
    assert db.query('select count(*) as n from foo').scalar() == 0