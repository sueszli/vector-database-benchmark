"""Tests the Polar API as an external consumer"""
from pathlib import Path
import pytest
from oso import Oso
from polar import exceptions
actors = {'guest': '1', 'president': '1'}

class User:
    name: str = ''
    verified: bool = False

    def __init__(self, name=''):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.verified = False

    def companies(self):
        if False:
            return 10
        yield Company(id='0')
        yield Company(id=actors[self.name])

class Widget:
    id: str = ''
    actions = ('read', 'create')

    def __init__(self, id):
        if False:
            while True:
                i = 10
        self.id = id

    def company(self):
        if False:
            return 10
        return Company(id=self.id)

class Company:
    roles = ('guest', 'admin')

    def __init__(self, id, default_role=''):
        if False:
            while True:
                i = 10
        self.id = id
        self.default_role = default_role

    def role(self, actor: User):
        if False:
            while True:
                i = 10
        if actor.name == 'president':
            return 'admin'
        else:
            return 'guest'

    def __eq__(self, other):
        if False:
            return 10
        return self.id == other.id
test_oso_file = Path(__file__).parent / 'test_oso.polar'

@pytest.fixture
def test_oso():
    if False:
        print('Hello World!')
    oso = Oso()
    oso.register_class(User, name='test_oso::User')
    oso.register_class(Widget, name='test_oso::Widget')
    oso.register_class(Company, name='test_oso::Company')
    oso.register_class(Foo)
    oso.register_class(Bar)
    oso.load_file(test_oso_file)
    return oso

def test_sanity(test_oso):
    if False:
        while True:
            i = 10
    pass

class Foo:

    def __init__(self, foo):
        if False:
            print('Hello World!')
        self.foo = foo

class Bar(Foo):

    def __init__(self, bar):
        if False:
            while True:
                i = 10
        super()
        self.bar = bar

def test_is_allowed(test_oso):
    if False:
        i = 10
        return i + 15
    actor = User(name='guest')
    resource = Widget(id='1')
    action = 'read'
    assert test_oso.is_allowed(actor, action, resource)
    assert test_oso.is_allowed({'username': 'guest'}, action, resource)
    assert test_oso.is_allowed('guest', action, resource)
    actor = User(name='president')
    action = 'create'
    resource = Company(id='1')
    assert test_oso.is_allowed(actor, action, resource)
    assert test_oso.is_allowed({'username': 'president'}, action, resource)

def test_query_rule(test_oso):
    if False:
        return 10
    actor = User(name='guest')
    resource = Widget(id='1')
    action = 'read'
    assert list(test_oso.query_rule('allow', actor, action, resource))

def test_fail(test_oso):
    if False:
        while True:
            i = 10
    actor = User(name='guest')
    resource = Widget(id='1')
    action = 'not_allowed'
    assert not test_oso.is_allowed(actor, action, resource)
    assert not test_oso.is_allowed({'username': 'guest'}, action, resource)

def test_instance_from_external_call(test_oso):
    if False:
        print('Hello World!')
    user = User(name='guest')
    resource = Company(id='1')
    assert test_oso.is_allowed(user, 'frob', resource)
    assert test_oso.is_allowed({'username': 'guest'}, 'frob', resource)

def test_allow_model(test_oso):
    if False:
        while True:
            i = 10
    'Test user auditor can list companies but not widgets'
    user = User(name='auditor')
    assert not test_oso.is_allowed(user, 'list', Widget)
    assert test_oso.is_allowed(user, 'list', Company)

def test_get_allowed_actions(test_oso):
    if False:
        while True:
            i = 10
    test_oso.clear_rules()
    with open(test_oso_file, 'rb') as f:
        policy = f.read().decode('utf-8')
        policy1 = policy + 'allow(_actor: test_oso::User{name: "Sally"}, action, _resource: test_oso::Widget{id: "1"}) if\n        action in ["CREATE", "UPDATE"];'
        test_oso.load_str(policy1)
        user = User(name='Sally')
        resource = Widget(id='1')
        assert set(test_oso.get_allowed_actions(user, resource)) == {'read', 'CREATE', 'UPDATE'}
        test_oso.clear_rules()
        policy2 = policy + 'allow(_actor: test_oso::User{name: "John"}, _action, _resource: test_oso::Widget{id: "1"});'
        test_oso.load_str(policy2)
        user = User(name='John')
        with pytest.raises(exceptions.OsoError):
            test_oso.get_allowed_actions(user, resource)
        assert set(test_oso.get_allowed_actions(user, resource, allow_wildcard=True)) == {'*'}

def test_forall_with_dot_lookup_and_method_call():
    if False:
        for i in range(10):
            print('nop')
    'Thanks to user Alex Pearce for this test case!'
    import uuid
    from dataclasses import dataclass, field
    from typing import List
    from oso import ForbiddenError, NotFoundError, Oso

    @dataclass(frozen=True)
    class User:
        name: str
        scopes: List[str]
        id: str = field(default_factory=lambda : str(uuid.uuid4()))

        def has_scope(self, scope: str):
            if False:
                while True:
                    i = 10
            print(f'Checking scope {scope}')
            return scope in self.scopes

    @dataclass(frozen=True)
    class Request:
        scopes: List[str] = field(default_factory=list)

    def check_request(actor, request):
        if False:
            for i in range(10):
                print('nop')
        'Helper to convert an Oso exception to a True/False decision.'
        try:
            oso.authorize_request(actor, request)
        except (ForbiddenError, NotFoundError):
            return False
        return True

    def expect(value, expected):
        if False:
            i = 10
            return i + 15
        assert value == expected
    oso = Oso()
    oso.clear_rules()
    oso.register_class(User)
    oso.register_class(Request)
    oso.load_str('\n# allow(actor: Actor, action: String, resource: Resource) if\n#    has_permission(actor, action, resource);\nallow(_, _, _);\n\n# A Token is authorised if has all scopes required by the route being accessed\n# in the request\nallow_request(user: User, request: Request) if\n  request_scopes = request.scopes and\n  forall(scope in request_scopes, user.has_scope(scope));\n  # forall(scope in request.scopes, scope in user.scopes);\n    ')
    user = User(name='Dave', scopes=['xyz'])
    expect(check_request(user, Request()), True)
    expect(check_request(user, Request(scopes=['xyz'])), True)
    expect(check_request(user, Request(scopes=['xyzxyz'])), False)
if __name__ == '__main__':
    pytest.main([__file__])