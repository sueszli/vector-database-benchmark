import pytest
from pytest_mock import MockFixture
from superset.exceptions import SupersetSecurityException
from superset.extensions import appbuilder
from superset.security.manager import SupersetSecurityManager

def test_security_manager(app_context: None) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that the security manager can be built.\n    '
    sm = SupersetSecurityManager(appbuilder)
    assert sm

def test_raise_for_access_query_default_schema(mocker: MockFixture, app_context: None) -> None:
    if False:
        while True:
            i = 10
    '\n    Test that the DB default schema is used in non-qualified table names.\n\n    For example, in Postgres, for the following query:\n\n        > SELECT * FROM foo;\n\n    We should check that the user has access to the `public` schema, regardless of the\n    schema set in the query.\n    '
    sm = SupersetSecurityManager(appbuilder)
    mocker.patch.object(sm, 'can_access_database', return_value=False)
    mocker.patch.object(sm, 'get_schema_perm', return_value='[PostgreSQL].[public]')
    SqlaTable = mocker.patch('superset.connectors.sqla.models.SqlaTable')
    SqlaTable.query_datasources_by_name.return_value = []
    database = mocker.MagicMock()
    database.get_default_schema_for_query.return_value = 'public'
    query = mocker.MagicMock()
    query.database = database
    query.sql = 'SELECT * FROM ab_user'
    mocker.patch.object(sm, 'can_access', return_value=True)
    assert sm.raise_for_access(database=None, datasource=None, query=query, query_context=None, table=None, viz=None) is None
    sm.can_access.assert_called_with('schema_access', '[PostgreSQL].[public]')
    mocker.patch.object(sm, 'can_access', return_value=False)
    with pytest.raises(SupersetSecurityException) as excinfo:
        sm.raise_for_access(database=None, datasource=None, query=query, query_context=None, table=None, viz=None)
    assert str(excinfo.value) == 'You need access to the following tables: `public.ab_user`,\n            `all_database_access` or `all_datasource_access` permission'