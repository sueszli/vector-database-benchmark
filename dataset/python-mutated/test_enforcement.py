"""Tests the Enforcement API"""
from pathlib import Path
import pytest
from oso import Oso
from oso.exceptions import ForbiddenError, NotFoundError
from polar import exceptions
from .test_oso import Bar, Company, Foo, User, Widget
test_oso_file = Path(__file__).parent / 'test_oso.polar'

@pytest.fixture
def test_oso():
    if False:
        while True:
            i = 10
    oso = Oso()
    oso.register_class(User, name='test_oso::User')
    oso.register_class(Widget, name='test_oso::Widget')
    oso.register_class(Company, name='test_oso::Company')
    oso.register_class(Foo)
    oso.register_class(Bar)
    oso.load_file(test_oso_file)
    return oso

def test_authorize(test_oso):
    if False:
        print('Hello World!')
    actor = User(name='guest')
    resource = Widget(id='1')
    action = 'read'
    test_oso.authorize(actor, action, resource)
    test_oso.authorize({'username': 'guest'}, action, resource)
    test_oso.authorize('guest', action, resource)
    actor = User(name='president')
    action = 'create'
    resource = Company(id='1')
    test_oso.authorize(actor, action, resource)
    test_oso.authorize({'username': 'president'}, action, resource)

def test_fail_authorize(test_oso):
    if False:
        while True:
            i = 10
    actor = User(name='guest')
    resource = Widget(id='1')
    action = 'not_allowed'
    with pytest.raises(ForbiddenError):
        test_oso.authorize(actor, action, resource)
    with pytest.raises(ForbiddenError):
        test_oso.authorize({'username': 'guest'}, action, resource)
    resource = Company(id='1')
    with pytest.raises(NotFoundError):
        test_oso.authorize({'username': 'guest'}, action, resource)

def test_authorized_actions(test_oso):
    if False:
        while True:
            i = 10
    test_oso.clear_rules()
    with open(test_oso_file, 'rb') as f:
        policy = f.read().decode('utf-8')
        policy1 = policy + 'allow(_actor: test_oso::User{name: "Sally"}, action, _resource: test_oso::Widget{id: "1"}) if\n                       action in ["CREATE", "UPDATE"];'
        test_oso.load_str(policy1)
        user = User(name='Sally')
        resource = Widget(id='1')
        assert test_oso.authorized_actions(user, resource) == {'read', 'CREATE', 'UPDATE'}
        test_oso.clear_rules()
        policy2 = policy + 'allow(_actor: test_oso::User{name: "John"}, _action, _resource: test_oso::Widget{id: "1"});'
        test_oso.load_str(policy2)
        user = User(name='John')
        with pytest.raises(exceptions.OsoError):
            test_oso.authorized_actions(user, resource)
        assert test_oso.authorized_actions(user, resource, allow_wildcard=True) == {'*'}

def test_authorize_request(test_oso):
    if False:
        return 10

    class Request:

        def __init__(self, method, path) -> None:
            if False:
                for i in range(10):
                    print('nop')
            self.method = method
            self.path = path
    policy = '\n    allow_request("graham", request: Request) if\n        request.path.startswith("/repos");\n\n    allow_request(user: test_oso::User, request: Request) if\n        request.path.startswith("/account")\n        and user.verified;\n    '
    verified = User('verified')
    verified.verified = True
    test_oso.clear_rules()
    test_oso.register_class(Request)
    test_oso.load_str(policy)
    test_oso.authorize_request('graham', Request('GET', '/repos/1'))
    with pytest.raises(ForbiddenError):
        test_oso.authorize_request('sam', Request('GET', '/repos/1'))
    test_oso.authorize_request(verified, Request('GET', '/account'))
    with pytest.raises(ForbiddenError):
        test_oso.authorize_request('graham', Request('GET', '/account'))

def test_authorize_field(test_oso):
    if False:
        return 10
    admin = User(name='president')
    guest = User(name='guest')
    company = Company(id='1')
    resource = Widget(id=company.id)
    test_oso.authorize_field(admin, 'update', resource, 'name')
    with pytest.raises(ForbiddenError):
        test_oso.authorize_field(guest, 'update', resource, 'foo')
    test_oso.authorize_field(guest, 'read', resource, 'name')
    with pytest.raises(ForbiddenError):
        test_oso.authorize_field(guest, 'read', resource, 'private_field')

def test_authorized_fields(test_oso):
    if False:
        i = 10
        return i + 15
    admin = User(name='president')
    guest = User(name='guest')
    company = Company(id='1')
    resource = Widget(id=company.id)
    assert test_oso.authorized_fields(admin, 'update', resource) == {'name', 'purpose', 'private_field'}
    assert test_oso.authorized_fields(guest, 'update', resource) == set()
    assert test_oso.authorized_fields(admin, 'read', resource) == {'name', 'purpose', 'private_field'}
    assert test_oso.authorized_fields(guest, 'read', resource) == {'name', 'purpose'}

def test_custom_errors():
    if False:
        for i in range(10):
            print('nop')

    class TestNotFound(Exception):
        pass

    class TestForbidden(Exception):
        pass
    oso = Oso(not_found_error=lambda : TestNotFound, forbidden_error=TestForbidden)
    oso.load_str('allow("graham", "read", "bar");')
    with pytest.raises(TestForbidden):
        oso.authorize('graham', 'frob', 'bar')
    with pytest.raises(TestNotFound):
        oso.authorize('sam', 'frob', 'bar')

def test_custom_read_action():
    if False:
        i = 10
        return i + 15
    oso = Oso(read_action='fetch')
    oso.load_str('allow("graham", "fetch", "bar");')
    with pytest.raises(NotFoundError):
        oso.authorize('not graham', 'frob', 'bar')
    with pytest.raises(ForbiddenError):
        oso.authorize('graham', 'frob', 'bar')
if __name__ == '__main__':
    pytest.main([__file__])