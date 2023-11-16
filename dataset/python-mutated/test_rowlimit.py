import pytest
from unittest.mock import Mock
from pgcli.main import PGCli

@pytest.fixture(scope='module')
def default_pgcli_obj():
    if False:
        return 10
    return PGCli()

@pytest.fixture(scope='module')
def DEFAULT(default_pgcli_obj):
    if False:
        print('Hello World!')
    return default_pgcli_obj.row_limit

@pytest.fixture(scope='module')
def LIMIT(DEFAULT):
    if False:
        return 10
    return DEFAULT + 1000

@pytest.fixture(scope='module')
def over_default(DEFAULT):
    if False:
        i = 10
        return i + 15
    over_default_cursor = Mock()
    over_default_cursor.configure_mock(rowcount=DEFAULT + 10)
    return over_default_cursor

@pytest.fixture(scope='module')
def over_limit(LIMIT):
    if False:
        while True:
            i = 10
    over_limit_cursor = Mock()
    over_limit_cursor.configure_mock(rowcount=LIMIT + 10)
    return over_limit_cursor

@pytest.fixture(scope='module')
def low_count():
    if False:
        while True:
            i = 10
    low_count_cursor = Mock()
    low_count_cursor.configure_mock(rowcount=1)
    return low_count_cursor

def test_row_limit_with_LIMIT_clause(LIMIT, over_limit):
    if False:
        i = 10
        return i + 15
    cli = PGCli(row_limit=LIMIT)
    stmt = 'SELECT * FROM students LIMIT 1000'
    result = cli._should_limit_output(stmt, over_limit)
    assert result is False
    cli = PGCli(row_limit=0)
    result = cli._should_limit_output(stmt, over_limit)
    assert result is False

def test_row_limit_without_LIMIT_clause(LIMIT, over_limit):
    if False:
        print('Hello World!')
    cli = PGCli(row_limit=LIMIT)
    stmt = 'SELECT * FROM students'
    result = cli._should_limit_output(stmt, over_limit)
    assert result is True
    cli = PGCli(row_limit=0)
    result = cli._should_limit_output(stmt, over_limit)
    assert result is False

def test_row_limit_on_non_select(over_limit):
    if False:
        i = 10
        return i + 15
    cli = PGCli()
    stmt = "UPDATE students SET name='Boby'"
    result = cli._should_limit_output(stmt, over_limit)
    assert result is False
    cli = PGCli(row_limit=0)
    result = cli._should_limit_output(stmt, over_limit)
    assert result is False