import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from cme.cmedb import delete_workspace, CMEDBMenu
from cme.first_run import first_run_setup
from cme.loaders.protocolloader import ProtocolLoader
from cme.logger import CMEAdapter
from cme.paths import WS_PATH
from sqlalchemy.dialects.sqlite import Insert

@pytest.fixture(scope='session')
def db_engine():
    if False:
        print('Hello World!')
    db_path = os.path.join(WS_PATH, 'test/smb.db')
    db_engine = create_engine(f'sqlite:///{db_path}', isolation_level='AUTOCOMMIT', future=True)
    yield db_engine
    db_engine.dispose()

@pytest.fixture(scope='session')
def db_setup(db_engine):
    if False:
        print('Hello World!')
    proto = 'smb'
    logger = CMEAdapter()
    first_run_setup(logger)
    p_loader = ProtocolLoader()
    protocols = p_loader.get_protocols()
    CMEDBMenu.create_workspace('test', p_loader, protocols)
    protocol_db_path = p_loader.get_protocols()[proto]['dbpath']
    protocol_db_object = getattr(p_loader.load_protocol(protocol_db_path), 'database')
    database_obj = protocol_db_object(db_engine)
    database_obj.reflect_tables()
    yield database_obj
    database_obj.shutdown_db()
    delete_workspace('test')

@pytest.fixture(scope='function')
def db(db_setup):
    if False:
        i = 10
        return i + 15
    yield db_setup
    db_setup.clear_database()

@pytest.fixture(scope='session')
def sess(db_engine):
    if False:
        for i in range(10):
            print('nop')
    session_factory = sessionmaker(bind=db_engine, expire_on_commit=True)
    Session = scoped_session(session_factory)
    sess = Session()
    yield sess
    sess.close()

def test_add_host(db):
    if False:
        return 10
    db.add_host('127.0.0.1', 'localhost', 'TEST.DEV', 'Windows Testing 2023', False, True, True, True, False, False)
    inserted_host = db.get_hosts()
    assert len(inserted_host) == 1
    host = inserted_host[0]
    assert host.id == 1
    assert host.ip == '127.0.0.1'
    assert host.hostname == 'localhost'
    assert host.os == 'Windows Testing 2023'
    assert host.smbv1 is False
    assert host.signing is True
    assert host.spooler is True
    assert host.zerologon is True
    assert host.petitpotam is False
    assert host.dc is False

def test_update_host(db, sess):
    if False:
        print('Hello World!')
    host = {'ip': '127.0.0.1', 'hostname': 'localhost', 'domain': 'TEST.DEV', 'os': 'Windows Testing 2023', 'smbv1': True, 'signing': False, 'spooler': True, 'zerologon': False, 'petitpotam': False, 'dc': False}
    iq = Insert(db.HostsTable)
    sess.execute(iq, [host])
    db.add_host('127.0.0.1', 'localhost', 'TEST.DEV', 'Windows Testing 2023 Updated', False, True, False, False, False, False)
    inserted_host = db.get_hosts()
    assert len(inserted_host) == 1
    host = inserted_host[0]
    assert host.id == 1
    assert host.ip == '127.0.0.1'
    assert host.hostname == 'localhost'
    assert host.os == 'Windows Testing 2023 Updated'
    assert host.smbv1 is False
    assert host.signing is True
    assert host.spooler is False
    assert host.zerologon is False
    assert host.petitpotam is False
    assert host.dc is False

def test_add_credential():
    if False:
        return 10
    pass

def test_update_credential():
    if False:
        i = 10
        return i + 15
    pass

def test_remove_credential():
    if False:
        return 10
    pass

def test_add_admin_user():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_get_admin_relations():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_remove_admin_relation():
    if False:
        print('Hello World!')
    pass

def test_is_credential_valid():
    if False:
        print('Hello World!')
    pass

def test_get_credentials():
    if False:
        print('Hello World!')
    pass

def test_get_credential():
    if False:
        return 10
    pass

def test_is_credential_local():
    if False:
        while True:
            i = 10
    pass

def test_is_host_valid():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_get_hosts():
    if False:
        return 10
    pass

def test_is_group_valid():
    if False:
        while True:
            i = 10
    pass

def test_add_group():
    if False:
        return 10
    pass

def test_get_groups():
    if False:
        print('Hello World!')
    pass

def test_get_group_relations():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_remove_group_relations():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_is_user_valid():
    if False:
        while True:
            i = 10
    pass

def test_get_users():
    if False:
        return 10
    pass

def test_get_user():
    if False:
        while True:
            i = 10
    pass

def test_get_domain_controllers():
    if False:
        i = 10
        return i + 15
    pass

def test_is_share_valid():
    if False:
        print('Hello World!')
    pass

def test_add_share():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_get_shares():
    if False:
        print('Hello World!')
    pass

def test_get_shares_by_access():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_get_users_with_share_access():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_add_domain_backupkey():
    if False:
        print('Hello World!')
    pass

def test_get_domain_backupkey():
    if False:
        print('Hello World!')
    pass

def test_is_dpapi_secret_valid():
    if False:
        while True:
            i = 10
    pass

def test_add_dpapi_secrets():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_get_dpapi_secrets():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_add_loggedin_relation():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_get_loggedin_relations():
    if False:
        print('Hello World!')
    pass

def test_remove_loggedin_relations():
    if False:
        while True:
            i = 10
    pass