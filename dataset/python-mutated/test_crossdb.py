from datasette.cli import cli
from click.testing import CliRunner
import urllib
import sqlite3
from .fixtures import app_client_two_attached_databases_crossdb_enabled

def test_crossdb_join(app_client_two_attached_databases_crossdb_enabled):
    if False:
        for i in range(10):
            print('nop')
    app_client = app_client_two_attached_databases_crossdb_enabled
    sql = "\n    select\n      'extra database' as db,\n      pk,\n      text1,\n      text2\n    from\n      [extra database].searchable\n    union all\n    select\n      'fixtures' as db,\n      pk,\n      text1,\n      text2\n    from\n      fixtures.searchable\n    "
    response = app_client.get('/_memory.json?' + urllib.parse.urlencode({'sql': sql, '_shape': 'array'}))
    assert response.status == 200
    assert response.json == [{'db': 'extra database', 'pk': 1, 'text1': 'barry cat', 'text2': 'terry dog'}, {'db': 'extra database', 'pk': 2, 'text1': 'terry dog', 'text2': 'sara weasel'}, {'db': 'fixtures', 'pk': 1, 'text1': 'barry cat', 'text2': 'terry dog'}, {'db': 'fixtures', 'pk': 2, 'text1': 'terry dog', 'text2': 'sara weasel'}]

def test_crossdb_warning_if_too_many_databases(tmp_path_factory):
    if False:
        for i in range(10):
            print('nop')
    db_dir = tmp_path_factory.mktemp('dbs')
    dbs = []
    for i in range(11):
        path = str(db_dir / 'db_{}.db'.format(i))
        conn = sqlite3.connect(path)
        conn.execute('vacuum')
        dbs.append(path)
    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(cli, ['serve', '--crossdb', '--get', '/'] + dbs, catch_exceptions=False)
    assert 'Warning: --crossdb only works with the first 10 attached databases' in result.stderr

def test_crossdb_attached_database_list_display(app_client_two_attached_databases_crossdb_enabled):
    if False:
        return 10
    app_client = app_client_two_attached_databases_crossdb_enabled
    response = app_client.get('/_memory')
    for fragment in ('databases are attached to this connection', '<li><strong>fixtures</strong> - ', '<li><strong>extra database</strong> - '):
        assert fragment in response.text