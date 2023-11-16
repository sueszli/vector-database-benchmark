import datetime
import os
import pytest
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from flask import current_app
from flask_principal import Identity
from flask_principal import identity_changed
from lemur import create_app
from lemur.auth.service import create_token
from lemur.common.utils import parse_private_key
from lemur.database import db as _db
from lemur.tests.vectors import INTERMEDIATE_KEY
from lemur.tests.vectors import ROOTCA_CERT_STR
from lemur.tests.vectors import ROOTCA_KEY
from lemur.tests.vectors import SAN_CERT_KEY
from sqlalchemy.sql import text
from .factories import ApiKeyFactory
from .factories import AsyncAuthorityFactory
from .factories import AuthorityFactory
from .factories import CACertificateFactory
from .factories import CertificateFactory
from .factories import CryptoAuthorityFactory
from .factories import DestinationFactory
from .factories import EndpointFactory
from .factories import InvalidCertificateFactory
from .factories import NotificationFactory
from .factories import PendingCertificateFactory
from .factories import RoleFactory
from .factories import RotationPolicyFactory
from .factories import SourceFactory
from .factories import UserFactory

def pytest_runtest_setup(item):
    if False:
        i = 10
        return i + 15
    if 'slow' in item.keywords and (not item.config.getoption('--runslow')):
        pytest.skip('need --runslow option to run')
    if 'incremental' in item.keywords:
        previousfailed = getattr(item.parent, '_previousfailed', None)
        if previousfailed is not None:
            pytest.xfail(f'previous test failed ({previousfailed.name})')

def pytest_runtest_makereport(item, call):
    if False:
        while True:
            i = 10
    if 'incremental' in item.keywords:
        if call.excinfo is not None:
            parent = item.parent
            parent._previousfailed = item

@pytest.fixture(scope='session')
def app(request):
    if False:
        return 10
    '\n    Creates a new Flask application for a test duration.\n    Uses application factory `create_app`.\n    '
    _app = create_app(config_path=os.path.dirname(os.path.realpath(__file__)) + '/conf.py')
    ctx = _app.app_context()
    ctx.push()
    yield _app
    ctx.pop()

@pytest.fixture(scope='session')
def db(app, request):
    if False:
        print('Hello World!')
    _db.drop_all()
    _db.engine.execute(text('CREATE EXTENSION IF NOT EXISTS pg_trgm'))
    _db.create_all()
    _db.app = app
    UserFactory()
    r = RoleFactory(name='admin')
    u = UserFactory(roles=[r])
    rp = RotationPolicyFactory(name='default')
    ApiKeyFactory(user=u)
    _db.session.commit()
    yield _db
    _db.drop_all()

@pytest.fixture(scope='function')
def session(db, request):
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates a new database session with (with working transaction)\n    for test duration.\n    '
    db.session.begin_nested()
    yield db.session
    db.session.rollback()

@pytest.fixture(scope='function')
def client(app, session, client):
    if False:
        i = 10
        return i + 15
    yield client

@pytest.fixture
def authority(session):
    if False:
        for i in range(10):
            print('nop')
    a = AuthorityFactory()
    session.commit()
    return a

@pytest.fixture
def crypto_authority(session):
    if False:
        for i in range(10):
            print('nop')
    a = CryptoAuthorityFactory()
    session.commit()
    return a

@pytest.fixture
def async_authority(session):
    if False:
        print('Hello World!')
    a = AsyncAuthorityFactory()
    session.commit()
    return a

@pytest.fixture
def destination(session):
    if False:
        i = 10
        return i + 15
    d = DestinationFactory()
    session.commit()
    return d

@pytest.fixture
def source(session):
    if False:
        for i in range(10):
            print('nop')
    s = SourceFactory()
    session.commit()
    return s

@pytest.fixture
def notification(session):
    if False:
        print('Hello World!')
    n = NotificationFactory()
    session.commit()
    return n

@pytest.fixture
def certificate(session):
    if False:
        print('Hello World!')
    u = UserFactory()
    a = AuthorityFactory()
    c = CertificateFactory(user=u, authority=a)
    session.commit()
    return c

@pytest.fixture
def endpoint(session):
    if False:
        print('Hello World!')
    s = SourceFactory()
    e = EndpointFactory(source=s)
    session.commit()
    return e

@pytest.fixture
def role(session):
    if False:
        while True:
            i = 10
    r = RoleFactory()
    session.commit()
    return r

@pytest.fixture
def user(session):
    if False:
        for i in range(10):
            print('nop')
    u = UserFactory()
    session.commit()
    user_token = create_token(u)
    token = {'Authorization': 'Basic ' + user_token}
    return {'user': u, 'token': token}

@pytest.fixture
def pending_certificate(session):
    if False:
        return 10
    u = UserFactory()
    a = AsyncAuthorityFactory()
    p = PendingCertificateFactory(user=u, authority=a)
    session.commit()
    return p

@pytest.fixture
def pending_certificate_from_full_chain_ca(session):
    if False:
        return 10
    u = UserFactory()
    a = AuthorityFactory()
    p = PendingCertificateFactory(user=u, authority=a)
    session.commit()
    return p

@pytest.fixture
def pending_certificate_from_partial_chain_ca(session):
    if False:
        return 10
    u = UserFactory()
    c = CACertificateFactory(body=ROOTCA_CERT_STR, private_key=ROOTCA_KEY, chain=None)
    a = AuthorityFactory(authority_certificate=c)
    p = PendingCertificateFactory(user=u, authority=a)
    session.commit()
    return p

@pytest.fixture
def invalid_certificate(session):
    if False:
        print('Hello World!')
    u = UserFactory()
    a = AsyncAuthorityFactory()
    i = InvalidCertificateFactory(user=u, authority=a)
    session.commit()
    return i

@pytest.fixture
def admin_user(session):
    if False:
        print('Hello World!')
    u = UserFactory()
    admin_role = RoleFactory(name='admin')
    u.roles.append(admin_role)
    session.commit()
    user_token = create_token(u)
    token = {'Authorization': 'Basic ' + user_token}
    return {'user': u, 'token': token}

@pytest.fixture
def async_issuer_plugin():
    if False:
        while True:
            i = 10
    from lemur.plugins.base import register
    from .plugins.issuer_plugin import TestAsyncIssuerPlugin
    register(TestAsyncIssuerPlugin)
    return TestAsyncIssuerPlugin

@pytest.fixture
def issuer_plugin():
    if False:
        while True:
            i = 10
    from lemur.plugins.base import register
    from .plugins.issuer_plugin import TestIssuerPlugin
    register(TestIssuerPlugin)
    return TestIssuerPlugin

@pytest.fixture
def notification_plugin():
    if False:
        while True:
            i = 10
    from lemur.plugins.base import register
    from .plugins.notification_plugin import TestNotificationPlugin
    register(TestNotificationPlugin)
    return TestNotificationPlugin

@pytest.fixture
def destination_plugin():
    if False:
        while True:
            i = 10
    from lemur.plugins.base import register
    from .plugins.destination_plugin import TestDestinationPlugin
    register(TestDestinationPlugin)
    return TestDestinationPlugin

@pytest.fixture
def source_plugin():
    if False:
        while True:
            i = 10
    from lemur.plugins.base import register
    from .plugins.source_plugin import TestSourcePlugin
    register(TestSourcePlugin)
    return TestSourcePlugin

@pytest.fixture(scope='function')
def logged_in_user(session, app):
    if False:
        for i in range(10):
            print('nop')
    with app.test_request_context():
        identity_changed.send(current_app._get_current_object(), identity=Identity(1))
        yield

@pytest.fixture(scope='function')
def logged_in_admin(session, app):
    if False:
        return 10
    with app.test_request_context():
        identity_changed.send(current_app._get_current_object(), identity=Identity(2))
        yield

@pytest.fixture
def private_key():
    if False:
        print('Hello World!')
    return parse_private_key(SAN_CERT_KEY)

@pytest.fixture
def issuer_private_key():
    if False:
        for i in range(10):
            print('nop')
    return parse_private_key(INTERMEDIATE_KEY)

@pytest.fixture
def cert_builder(private_key):
    if False:
        for i in range(10):
            print('nop')
    return x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, 'foo.com')])).issuer_name(x509.Name([x509.NameAttribute(x509.NameOID.COMMON_NAME, 'foo.com')])).serial_number(1).public_key(private_key.public_key()).not_valid_before(datetime.datetime(2017, 12, 22)).not_valid_after(datetime.datetime(2040, 1, 1))

@pytest.fixture
def selfsigned_cert(cert_builder, private_key):
    if False:
        return 10
    return cert_builder.sign(private_key, hashes.SHA256(), default_backend())

@pytest.fixture(scope='function')
def aws_credentials():
    if False:
        while True:
            i = 10
    os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
    os.environ['AWS_SECURITY_TOKEN'] = 'testing'
    os.environ['AWS_SESSION_TOKEN'] = 'testing'