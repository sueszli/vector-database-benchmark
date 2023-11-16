import sqlite3
import pytest
import salt.modules.mac_assistive as assistive
from salt.exceptions import CommandExecutionError
from tests.support.mock import patch
BIGSUR_DB_SCHEMA = 'CREATE TABLE admin (key TEXT PRIMARY KEY NOT NULL, value INTEGER NOT NULL);\nCREATE TABLE policies ( id              INTEGER NOT NULL PRIMARY KEY,   bundle_id       TEXT    NOT NULL,       uuid            TEXT    NOT NULL,       display         TEXT    NOT NULL,       UNIQUE (bundle_id, uuid));\nCREATE TABLE active_policy (    client          TEXT    NOT NULL,       client_type     INTEGER NOT NULL,       policy_id       INTEGER NOT NULL,       PRIMARY KEY (client, client_type),      FOREIGN KEY (policy_id) REFERENCES policies(id) ON DELETE CASCADE ON UPDATE CASCADE);\nCREATE INDEX active_policy_id ON active_policy(policy_id);\nCREATE TABLE access_overrides ( service         TEXT    NOT NULL PRIMARY KEY);\nCREATE TABLE expired (    service        TEXT        NOT NULL,     client         TEXT        NOT NULL,     client_type    INTEGER     NOT NULL,     csreq          BLOB,     last_modified  INTEGER     NOT NULL ,     expired_at     INTEGER     NOT NULL DEFAULT (CAST(strftime(\'%s\',\'now\') AS INTEGER)),     PRIMARY KEY (service, client, client_type));\nCREATE TABLE IF NOT EXISTS "access" (    service        TEXT        NOT NULL,     client         TEXT        NOT NULL,     client_type    INTEGER     NOT NULL,     auth_value     INTEGER     NOT NULL,     auth_reason    INTEGER     NOT NULL,     auth_version   INTEGER     NOT NULL,     csreq          BLOB,     policy_id      INTEGER,     indirect_object_identifier_type    INTEGER,     indirect_object_identifier         TEXT NOT NULL DEFAULT \'UNUSED\',     indirect_object_code_identity      BLOB,     flags          INTEGER,     last_modified  INTEGER     NOT NULL DEFAULT (CAST(strftime(\'%s\',\'now\') AS INTEGER)),     PRIMARY KEY (service, client, client_type, indirect_object_identifier),    FOREIGN KEY (policy_id) REFERENCES policies(id) ON DELETE CASCADE ON UPDATE CASCADE);\n'
CATALINA_DB_SCHEMA = 'CREATE TABLE admin (key TEXT PRIMARY KEY NOT NULL, value INTEGER NOT NULL);\nCREATE TABLE policies ( id              INTEGER NOT NULL PRIMARY KEY,   bundle_id       TEXT    NOT NULL,       uuid            TEXT    NOT NULL,       display         TEXT    NOT NULL,       UNIQUE (bundle_id, uuid));\nCREATE TABLE active_policy (    client          TEXT    NOT NULL,       client_type     INTEGER NOT NULL,       policy_id       INTEGER NOT NULL,       PRIMARY KEY (client, client_type),      FOREIGN KEY (policy_id) REFERENCES policies(id) ON DELETE CASCADE ON UPDATE CASCADE);\nCREATE INDEX active_policy_id ON active_policy(policy_id);\nCREATE TABLE access_overrides ( service         TEXT    NOT NULL PRIMARY KEY);\nCREATE TABLE expired (    service        TEXT        NOT NULL,     client         TEXT        NOT NULL,     client_type    INTEGER     NOT NULL,     csreq          BLOB,     last_modified  INTEGER     NOT NULL ,     expired_at     INTEGER     NOT NULL DEFAULT (CAST(strftime(\'%s\',\'now\') AS INTEGER)),     PRIMARY KEY (service, client, client_type));\nCREATE TABLE IF NOT EXISTS "access" (    service        TEXT        NOT NULL,     client         TEXT        NOT NULL,     client_type    INTEGER     NOT NULL,     auth_value     INTEGER     NOT NULL,     auth_reason    INTEGER     NOT NULL,     auth_version   INTEGER     NOT NULL,     csreq          BLOB,     policy_id      INTEGER,     indirect_object_identifier_type    INTEGER,     indirect_object_identifier         TEXT NOT NULL DEFAULT \'UNUSED\',     indirect_object_code_identity      BLOB,     flags          INTEGER,     last_modified  INTEGER     NOT NULL DEFAULT (CAST(strftime(\'%s\',\'now\') AS INTEGER)),     PRIMARY KEY (service, client, client_type, indirect_object_identifier),    FOREIGN KEY (policy_id) REFERENCES policies(id) ON DELETE CASCADE ON UPDATE CASCADE);\n'

@pytest.fixture(params=('Catalina', 'BigSur'))
def macos_version(request):
    if False:
        while True:
            i = 10
    return request.param

@pytest.fixture(autouse=True)
def tcc_db_path(tmp_path, macos_version):
    if False:
        print('Hello World!')
    db = tmp_path / 'tcc.db'
    if macos_version == 'BigSur':
        schema = BIGSUR_DB_SCHEMA
    elif macos_version == 'Catalina':
        schema = CATALINA_DB_SCHEMA
    else:
        pytest.fail("Don't know how to handle {}".format(macos_version))
    conn = sqlite3.connect(str(db))
    with conn:
        for stmt in schema.splitlines():
            conn.execute(stmt)
    return str(db)

@pytest.fixture
def configure_loader_modules(tcc_db_path):
    if False:
        return 10
    return {assistive: {'TCC_DB_PATH': tcc_db_path}}

def test_install_assistive_bundle():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test installing a bundle ID as being allowed to run with assistive access\n    '
    assert assistive.install('foo')

def test_install_assistive_error():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test installing a bundle ID as being allowed to run with assistive access\n    '
    with patch.object(assistive.TccDB, 'install', side_effect=sqlite3.Error('Foo')):
        pytest.raises(CommandExecutionError, assistive.install, 'foo')

def test_installed_bundle():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test checking to see if a bundle id is installed as being able to use assistive access\n    '
    assistive.install('foo')
    assert assistive.installed('foo')

def test_installed_bundle_not():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test checking to see if a bundle id is installed as being able to use assistive access\n    '
    assert not assistive.installed('foo')

def test_enable_assistive():
    if False:
        while True:
            i = 10
    '\n    Test enabling a bundle ID as being allowed to run with assistive access\n    '
    assistive.install('foo', enable=False)
    assert assistive.enable_('foo', True)

def test_enable_error():
    if False:
        return 10
    '\n    Test enabled a bundle ID that throws a command error\n    '
    with patch.object(assistive.TccDB, 'enable', side_effect=sqlite3.Error('Foo')):
        pytest.raises(CommandExecutionError, assistive.enable_, 'foo')

def test_enable_false():
    if False:
        return 10
    "\n    Test return of enable function when app isn't found.\n    "
    assert not assistive.enable_('foo')

def test_enabled_assistive():
    if False:
        while True:
            i = 10
    '\n    Test enabling a bundle ID as being allowed to run with assistive access\n    '
    assistive.install('foo')
    assert assistive.enabled('foo')

def test_enabled_assistive_false():
    if False:
        i = 10
        return i + 15
    '\n    Test if a bundle ID is disabled for assistive access\n    '
    assistive.install('foo', enable=False)
    assert not assistive.enabled('foo')

def test_remove_assistive():
    if False:
        while True:
            i = 10
    '\n    Test removing an assitive bundle.\n    '
    assistive.install('foo')
    assert assistive.remove('foo')

def test_remove_assistive_error():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test removing an assitive bundle.\n    '
    with patch.object(assistive.TccDB, 'remove', side_effect=sqlite3.Error('Foo')):
        pytest.raises(CommandExecutionError, assistive.remove, 'foo')