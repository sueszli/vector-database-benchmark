import importlib
import os
from pathlib import Path
import pytest
from ...utils import needs_pydanticv1

@needs_pydanticv1
def test_testing_dbs(tmp_path_factory: pytest.TempPathFactory):
    if False:
        i = 10
        return i + 15
    tmp_path = tmp_path_factory.mktemp('data')
    cwd = os.getcwd()
    os.chdir(tmp_path)
    test_db = Path('./test.db')
    if test_db.is_file():
        test_db.unlink()
    from docs_src.sql_databases.sql_app.tests import test_sql_app
    importlib.reload(test_sql_app)
    test_sql_app.test_create_user()
    if test_db.is_file():
        test_db.unlink()
    os.chdir(cwd)