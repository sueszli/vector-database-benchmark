from typing import Optional
import pytest
from sqlalchemy.engine.url import make_url
from superset.exceptions import SupersetSecurityException
from superset.security.analytics_db_safety import check_sqlalchemy_uri
from tests.integration_tests.test_app import app

@pytest.mark.parametrize('sqlalchemy_uri, error, error_message', [('postgres://user:password@test.com', False, None), ('sqlite:///home/superset/bad.db', True, 'SQLiteDialect_pysqlite cannot be used as a data source for security reasons.'), ('sqlite+pysqlite:///home/superset/bad.db', True, 'SQLiteDialect_pysqlite cannot be used as a data source for security reasons.'), ('sqlite+aiosqlite:///home/superset/bad.db', True, 'SQLiteDialect_pysqlite cannot be used as a data source for security reasons.'), ('sqlite+pysqlcipher:///home/superset/bad.db', True, 'SQLiteDialect_pysqlite cannot be used as a data source for security reasons.'), ('sqlite+:///home/superset/bad.db', True, 'SQLiteDialect_pysqlite cannot be used as a data source for security reasons.'), ('sqlite+new+driver:///home/superset/bad.db', True, 'SQLiteDialect_pysqlite cannot be used as a data source for security reasons.'), ('sqlite+new+:///home/superset/bad.db', True, 'SQLiteDialect_pysqlite cannot be used as a data source for security reasons.'), ('shillelagh:///home/superset/bad.db', True, 'shillelagh cannot be used as a data source for security reasons.'), ('shillelagh+apsw:///home/superset/bad.db', True, 'shillelagh cannot be used as a data source for security reasons.'), ('shillelagh+:///home/superset/bad.db', False, None), ('shillelagh+something:///home/superset/bad.db', False, None)])
def test_check_sqlalchemy_uri(sqlalchemy_uri: str, error: bool, error_message: Optional[str]):
    if False:
        return 10
    with app.app_context():
        if error:
            with pytest.raises(SupersetSecurityException) as excinfo:
                check_sqlalchemy_uri(make_url(sqlalchemy_uri))
                assert str(excinfo.value) == error_message
        else:
            check_sqlalchemy_uri(make_url(sqlalchemy_uri))