"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    Test cases for salt.modules.keystone
"""
import pytest
import salt.modules.config as config
import salt.modules.keystone as keystone
from tests.support.mock import MagicMock, call, patch

class MockEC2:
    """
    Mock of EC2 class
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.access = ''
        self.secret = ''
        self.tenant_id = ''
        self.user_id = ''
        self.connection_args = ''
        self.profile = ''

    @staticmethod
    def create(userid, tenantid):
        if False:
            return 10
        '\n        Mock of create method\n        '
        cr_ec2 = MockEC2()
        cr_ec2.tenant_id = tenantid
        cr_ec2.user_id = userid
        return cr_ec2

    def delete(self, userid, accesskey):
        if False:
            while True:
                i = 10
        '\n        Mock of delete method\n        '
        self.access = accesskey
        self.user_id = userid
        return True

    @staticmethod
    def get(user_id, access, profile, **connection_args):
        if False:
            i = 10
            return i + 15
        '\n        Mock of get method\n        '
        cr_ec2 = MockEC2()
        cr_ec2.profile = profile
        cr_ec2.access = access
        cr_ec2.user_id = user_id
        cr_ec2.connection_args = connection_args
        return cr_ec2

    @staticmethod
    def list(user_id):
        if False:
            while True:
                i = 10
        '\n        Mock of list method\n        '
        cr_ec2 = MockEC2()
        cr_ec2.user_id = user_id
        return [cr_ec2]

class MockEndpoints:
    """
    Mock of Endpoints class
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.id = '007'
        self.region = 'RegionOne'
        self.adminurl = 'adminurl'
        self.internalurl = 'internalurl'
        self.publicurl = 'publicurl'
        self.service_id = '117'

    @staticmethod
    def list():
        if False:
            print('Hello World!')
        '\n        Mock of list method\n        '
        return [MockEndpoints()]

    @staticmethod
    def create(region, service_id, publicurl, adminurl, internalurl):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mock of create method\n        '
        return (region, service_id, publicurl, adminurl, internalurl)

    @staticmethod
    def delete(id):
        if False:
            i = 10
            return i + 15
        '\n        Mock of delete method\n        '
        return id

class MockServices:
    """
    Mock of Services class
    """
    flag = None

    def __init__(self):
        if False:
            print('Hello World!')
        self.id = '117'
        self.name = 'iptables'
        self.description = 'description'
        self.type = 'type'

    @staticmethod
    def create(name, service_type, description):
        if False:
            return 10
        '\n        Mock of create method\n        '
        service = MockServices()
        service.id = '005'
        service.name = name
        service.description = description
        service.type = service_type
        return service

    def get(self, service_id):
        if False:
            while True:
                i = 10
        '\n        Mock of get method\n        '
        service = MockServices()
        if self.flag == 1:
            service.id = 'asd'
            return [service]
        elif self.flag == 2:
            service.id = service_id
            return service
        return [service]

    def list(self):
        if False:
            while True:
                i = 10
        '\n        Mock of list method\n        '
        service = MockServices()
        if self.flag == 1:
            service.id = 'asd'
            return [service]
        return [service]

    @staticmethod
    def delete(service_id):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mock of delete method\n        '
        return service_id

class MockRoles:
    """
    Mock of Roles class
    """
    flag = None

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.id = '113'
        self.name = 'nova'
        self.user_id = '446'
        self.tenant_id = 'a1a1'

    @staticmethod
    def create(name):
        if False:
            return 10
        '\n        Mock of create method\n        '
        return name

    def get(self, role_id):
        if False:
            return 10
        '\n        Mock of get method\n        '
        role = MockRoles()
        if self.flag == 1:
            role.id = None
            return role
        role.id = role_id
        return role

    @staticmethod
    def list():
        if False:
            while True:
                i = 10
        '\n        Mock of list method\n        '
        return [MockRoles()]

    @staticmethod
    def delete(role):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mock of delete method\n        '
        return role

    @staticmethod
    def add_user_role(user_id, role_id, tenant_id):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mock of add_user_role method\n        '
        return (user_id, role_id, tenant_id)

    @staticmethod
    def remove_user_role(user_id, role_id, tenant_id):
        if False:
            return 10
        '\n        Mock of remove_user_role method\n        '
        return (user_id, role_id, tenant_id)

    @staticmethod
    def roles_for_user(user, tenant):
        if False:
            while True:
                i = 10
        '\n        Mock of roles_for_user method\n        '
        role = MockRoles()
        role.user_id = user
        role.tenant_id = tenant
        return [role]

class MockTenants:
    """
    Mock of Tenants class
    """
    flag = None

    def __init__(self):
        if False:
            print('Hello World!')
        self.id = '446'
        self.name = 'nova'
        self.description = 'description'
        self.enabled = 'True'

    @staticmethod
    def create(name, description, enabled):
        if False:
            while True:
                i = 10
        '\n        Mock of create method\n        '
        tenant = MockTenants()
        tenant.name = name
        tenant.description = description
        tenant.enabled = enabled
        return tenant

    def get(self, tenant_id):
        if False:
            return 10
        '\n        Mock of get method\n        '
        tenant = MockTenants()
        if self.flag == 1:
            tenant.id = None
            return tenant
        tenant.id = tenant_id
        return tenant

    @staticmethod
    def list():
        if False:
            i = 10
            return i + 15
        '\n        Mock of list method\n        '
        return [MockTenants()]

    @staticmethod
    def delete(tenant_id):
        if False:
            i = 10
            return i + 15
        '\n        Mock of delete method\n        '
        return tenant_id

class MockServiceCatalog:
    """
    Mock of ServiceCatalog class
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.id = '446'
        self.expires = 'No'
        self.user_id = 'admin'
        self.tenant_id = 'ae04'

    def get_token(self):
        if False:
            return 10
        '\n        Mock of get_token method\n        '
        return {'id': self.id, 'expires': self.expires, 'user_id': self.user_id, 'tenant_id': self.tenant_id}

class MockUsers:
    """
    Mock of Users class
    """
    flag = None

    def __init__(self):
        if False:
            return 10
        self.id = '446'
        self.name = 'nova'
        self.email = 'salt@saltstack.com'
        self.enabled = 'True'
        self.tenant_id = 'a1a1'
        self.password = 'salt'

    def create(self, name, password, email, tenant_id, enabled):
        if False:
            return 10
        '\n        Mock of create method\n        '
        user = MockUsers()
        user.name = name
        user.password = password
        user.email = email
        user.enabled = enabled
        self.tenant_id = tenant_id
        return user

    def get(self, user_id):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mock of get method\n        '
        user = MockUsers()
        if self.flag == 1:
            user.id = None
            return user
        user.id = user_id
        return user

    @staticmethod
    def list():
        if False:
            for i in range(10):
                print('nop')
        '\n        Mock of list method\n        '
        return [MockUsers()]

    @staticmethod
    def delete(user_id):
        if False:
            return 10
        '\n        Mock of delete method\n        '
        return user_id

    @staticmethod
    def update(user, name, email, enabled):
        if False:
            print('Hello World!')
        '\n        Mock of update method\n        '
        return (user, name, email, enabled)

    @staticmethod
    def update_password(user, password):
        if False:
            return 10
        '\n        Mock of update_password method\n        '
        return (user, password)

class Unauthorized(Exception):
    """
    The base exception class for all exceptions.
    """

    def __init__(self, message='Test'):
        if False:
            i = 10
            return i + 15
        super().__init__(message)
        self.msg = message

class AuthorizationFailure(Exception):
    """
    Additional exception class to Unauthorized.
    """

    def __init__(self, message='Test'):
        if False:
            return 10
        super().__init__(message)
        self.msg = message

class MockExceptions:
    """
    Mock of exceptions class
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.Unauthorized = Unauthorized
        self.AuthorizationFailure = AuthorizationFailure

class MockKeystoneClient:
    """
    Mock of keystoneclient module
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.exceptions = MockExceptions()

class MockClient:
    """
    Mock of Client class
    """
    flag = None

    def __init__(self, profile=None, **conn_args):
        if False:
            return 10
        self.ec2 = MockEC2()
        self.endpoints = MockEndpoints()
        self.services = MockServices()
        self.roles = MockRoles()
        self.tenants = MockTenants()
        self.service_catalog = MockServiceCatalog()
        self.users = MockUsers()

    def Client(self, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Mock of Client method\n        '
        if self.flag == 1:
            raise Unauthorized
        return True

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {keystone: {'auth': MockClient, 'client': MockClient(), 'keystoneclient': MockKeystoneClient(), '__salt__': {'config.get': config.get}, '__opts__': {}}, config: {'__opts__': {}}}

def test_ec2_credentials_create():
    if False:
        while True:
            i = 10
    '\n    Test if it create EC2-compatible credentials for user per tenant\n    '
    assert keystone.ec2_credentials_create() == {'Error': 'Could not resolve User ID'}
    assert keystone.ec2_credentials_create(user_id='salt') == {'Error': 'Could not resolve Tenant ID'}
    assert keystone.ec2_credentials_create(user_id='salt', tenant_id='72278') == {'access': '', 'tenant_id': '72278', 'secret': '', 'user_id': 'salt'}

def test_ec2_credentials_delete():
    if False:
        print('Hello World!')
    '\n    Test if it delete EC2-compatible credentials\n    '
    assert keystone.ec2_credentials_delete() == {'Error': 'Could not resolve User ID'}
    assert keystone.ec2_credentials_delete(user_id='salt', access_key='72278') == 'ec2 key "72278" deleted under user id "salt"'

def test_ec2_credentials_get():
    if False:
        print('Hello World!')
    '\n    Test if it return ec2_credentials for a user\n    (keystone ec2-credentials-get)\n    '
    assert keystone.ec2_credentials_get() == {'Error': 'Unable to resolve user id'}
    assert keystone.ec2_credentials_get(user_id='salt') == {'Error': 'Access key is required'}
    assert keystone.ec2_credentials_get(user_id='salt', access='72278', profile='openstack1') == {'salt': {'access': '72278', 'secret': '', 'tenant': '', 'user_id': 'salt'}}

def test_ec2_credentials_list():
    if False:
        print('Hello World!')
    '\n    Test if it return a list of ec2_credentials\n    for a specific user (keystone ec2-credentials-list)\n    '
    assert keystone.ec2_credentials_list() == {'Error': 'Unable to resolve user id'}
    assert keystone.ec2_credentials_list(user_id='salt', profile='openstack1') == {'salt': {'access': '', 'secret': '', 'tenant_id': '', 'user_id': 'salt'}}

def test_endpoint_get():
    if False:
        print('Hello World!')
    '\n    Test if it return a specific endpoint (keystone endpoint-get)\n    '
    assert keystone.endpoint_get('nova', 'RegionOne', profile='openstack') == {'Error': 'Could not find the specified service'}
    ret = {'Error': 'Could not find endpoint for the specified service'}
    MockServices.flag = 1
    assert keystone.endpoint_get('iptables', 'RegionOne', profile='openstack') == ret
    MockServices.flag = 0
    assert keystone.endpoint_get('iptables', 'RegionOne', profile='openstack') == {'adminurl': 'adminurl', 'id': '007', 'internalurl': 'internalurl', 'publicurl': 'publicurl', 'region': 'RegionOne', 'service_id': '117'}

def test_endpoint_list():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it return a list of available endpoints\n    (keystone endpoints-list)\n    '
    assert keystone.endpoint_list(profile='openstack1') == {'007': {'adminurl': 'adminurl', 'id': '007', 'internalurl': 'internalurl', 'publicurl': 'publicurl', 'region': 'RegionOne', 'service_id': '117'}}

def test_endpoint_create():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it create an endpoint for an Openstack service\n    '
    assert keystone.endpoint_create('nova') == {'Error': 'Could not find the specified service'}
    MockServices.flag = 2
    assert keystone.endpoint_create('iptables', 'http://public/url', 'http://internal/url', 'http://adminurl/url', 'RegionOne') == {'adminurl': 'adminurl', 'id': '007', 'internalurl': 'internalurl', 'publicurl': 'publicurl', 'region': 'RegionOne', 'service_id': '117'}

def test_endpoint_delete():
    if False:
        return 10
    '\n    Test if it delete an endpoint for an Openstack service\n    '
    ret = {'Error': 'Could not find any endpoints for the service'}
    assert keystone.endpoint_delete('nova', 'RegionOne') == ret
    with patch.object(keystone, 'endpoint_get', MagicMock(side_effect=[{'id': '117'}, None])):
        assert keystone.endpoint_delete('iptables', 'RegionOne')

def test_role_create():
    if False:
        print('Hello World!')
    '\n    Test if it create named role\n    '
    assert keystone.role_create('nova') == {'Error': 'Role "nova" already exists'}
    assert keystone.role_create('iptables') == {'Error': 'Unable to resolve role id'}

def test_role_delete():
    if False:
        return 10
    '\n    Test if it delete a role (keystone role-delete)\n    '
    assert keystone.role_delete() == {'Error': 'Unable to resolve role id'}
    assert keystone.role_delete('iptables') == 'Role ID iptables deleted'

def test_role_get():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it return a specific roles (keystone role-get)\n    '
    assert keystone.role_get() == {'Error': 'Unable to resolve role id'}
    assert keystone.role_get(name='nova') == {'nova': {'id': '113', 'name': 'nova'}}

def test_role_list():
    if False:
        i = 10
        return i + 15
    '\n    Test if it return a list of available roles (keystone role-list)\n    '
    assert keystone.role_list() == {'nova': {'id': '113', 'name': 'nova', 'tenant_id': 'a1a1', 'user_id': '446'}}

def test_service_create():
    if False:
        i = 10
        return i + 15
    '\n    Test if it add service to Keystone service catalog\n    '
    MockServices.flag = 2
    assert keystone.service_create('nova', 'compute', 'OpenStack Service') == {'iptables': {'description': 'description', 'id': '005', 'name': 'iptables', 'type': 'type'}}

def test_service_delete():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it delete a service from Keystone service catalog\n    '
    assert keystone.service_delete('iptables') == 'Keystone service ID "iptables" deleted'

def test_service_get():
    if False:
        print('Hello World!')
    '\n    Test if it return a list of available services (keystone services-list)\n    '
    MockServices.flag = 0
    assert keystone.service_get() == {'Error': 'Unable to resolve service id'}
    MockServices.flag = 2
    assert keystone.service_get(service_id='c965') == {'iptables': {'description': 'description', 'id': 'c965', 'name': 'iptables', 'type': 'type'}}

def test_service_list():
    if False:
        return 10
    '\n    Test if it return a list of available services (keystone services-list)\n    '
    MockServices.flag = 0
    assert keystone.service_list(profile='openstack1') == {'iptables': {'description': 'description', 'id': '117', 'name': 'iptables', 'type': 'type'}}

def test_tenant_create():
    if False:
        i = 10
        return i + 15
    '\n    Test if it create a keystone tenant\n    '
    assert keystone.tenant_create('nova') == {'nova': {'description': 'description', 'id': '446', 'name': 'nova', 'enabled': 'True'}}

def test_tenant_delete():
    if False:
        i = 10
        return i + 15
    '\n    Test if it delete a tenant (keystone tenant-delete)\n    '
    assert keystone.tenant_delete() == {'Error': 'Unable to resolve tenant id'}
    assert keystone.tenant_delete('nova') == 'Tenant ID nova deleted'

def test_tenant_get():
    if False:
        return 10
    '\n    Test if it return a specific tenants (keystone tenant-get)\n    '
    assert keystone.tenant_get() == {'Error': 'Unable to resolve tenant id'}
    assert keystone.tenant_get(tenant_id='446') == {'nova': {'description': 'description', 'id': '446', 'name': 'nova', 'enabled': 'True'}}

def test_tenant_list():
    if False:
        while True:
            i = 10
    '\n    Test if it return a list of available tenants (keystone tenants-list)\n    '
    assert keystone.tenant_list() == {'nova': {'description': 'description', 'id': '446', 'name': 'nova', 'enabled': 'True'}}

def test_tenant_update():
    if False:
        print('Hello World!')
    "\n    Test if it update a tenant's information (keystone tenant-update)\n    "
    assert keystone.tenant_update() == {'Error': 'Unable to resolve tenant id'}

def test_token_get():
    if False:
        print('Hello World!')
    '\n    Test if it return the configured tokens (keystone token-get)\n    '
    assert keystone.token_get() == {'expires': 'No', 'id': '446', 'tenant_id': 'ae04', 'user_id': 'admin'}

def test_user_list():
    if False:
        print('Hello World!')
    '\n    Test if it return a list of available users (keystone user-list)\n    '
    assert keystone.user_list() == {'nova': {'name': 'nova', 'tenant_id': 'a1a1', 'enabled': 'True', 'id': '446', 'password': 'salt', 'email': 'salt@saltstack.com'}}

def test_user_get():
    if False:
        print('Hello World!')
    '\n    Test if it return a specific users (keystone user-get)\n    '
    assert keystone.user_get() == {'Error': 'Unable to resolve user id'}
    assert keystone.user_get(user_id='446') == {'nova': {'name': 'nova', 'tenant_id': 'a1a1', 'enabled': 'True', 'id': '446', 'password': 'salt', 'email': 'salt@saltstack.com'}}

def test_user_create():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it create a user (keystone user-create)\n    '
    assert keystone.user_create(name='nova', password='salt', email='salt@saltstack.com', tenant_id='a1a1') == {'nova': {'name': 'nova', 'tenant_id': 'a1a1', 'enabled': 'True', 'id': '446', 'password': 'salt', 'email': 'salt@saltstack.com'}}

def test_user_delete():
    if False:
        i = 10
        return i + 15
    '\n    Test if it delete a user (keystone user-delete)\n    '
    assert keystone.user_delete() == {'Error': 'Unable to resolve user id'}
    assert keystone.user_delete('nova') == 'User ID nova deleted'

def test_user_update():
    if False:
        print('Hello World!')
    "\n    Test if it update a user's information (keystone user-update)\n    "
    assert keystone.user_update() == {'Error': 'Unable to resolve user id'}
    assert keystone.user_update('nova') == 'Info updated for user ID nova'

def test_user_verify_password():
    if False:
        i = 10
        return i + 15
    "\n    Test if it verify a user's password\n    "
    mock = MagicMock(return_value='http://127.0.0.1:35357/v2.0')
    with patch.dict(keystone.__salt__, {'config.option': mock}):
        assert keystone.user_verify_password() == {'Error': 'Unable to resolve user name'}
        assert keystone.user_verify_password(user_id='446', name='nova')
        MockClient.flag = 1
        assert not keystone.user_verify_password(user_id='446', name='nova')

def test_user_password_update():
    if False:
        i = 10
        return i + 15
    "\n    Test if it update a user's password (keystone user-password-update)\n    "
    assert keystone.user_password_update() == {'Error': 'Unable to resolve user id'}
    assert keystone.user_password_update('nova') == 'Password updated for user ID nova'

def test_user_role_add():
    if False:
        return 10
    '\n    Test if it add role for user in tenant (keystone user-role-add)\n    '
    assert keystone.user_role_add(user='nova', tenant='nova', role='nova') == '"nova" role added for user "nova" for "nova" tenant/project'
    MockRoles.flag = 1
    assert keystone.user_role_add(user='nova', tenant='nova', role='nova') == {'Error': 'Unable to resolve role id'}
    MockTenants.flag = 1
    assert keystone.user_role_add(user='nova', tenant='nova') == {'Error': 'Unable to resolve tenant/project id'}
    MockUsers.flag = 1
    assert keystone.user_role_add(user='nova') == {'Error': 'Unable to resolve user id'}

def test_user_role_remove():
    if False:
        i = 10
        return i + 15
    '\n    Test if it add role for user in tenant (keystone user-role-add)\n    '
    MockUsers.flag = 1
    assert keystone.user_role_remove(user='nova') == {'Error': 'Unable to resolve user id'}
    MockUsers.flag = 0
    MockTenants.flag = 1
    assert keystone.user_role_remove(user='nova', tenant='nova') == {'Error': 'Unable to resolve tenant/project id'}
    MockTenants.flag = 0
    MockRoles.flag = 1
    assert keystone.user_role_remove(user='nova', tenant='nova', role='nova') == {'Error': 'Unable to resolve role id'}
    ret = '"nova" role removed for user "nova" under "nova" tenant'
    MockRoles.flag = 0
    assert keystone.user_role_remove(user='nova', tenant='nova', role='nova') == ret

def test_user_role_list():
    if False:
        i = 10
        return i + 15
    '\n    Test if it return a list of available user_roles\n    (keystone user-roles-list)\n    '
    assert keystone.user_role_list(user='nova') == {'Error': 'Unable to resolve user or tenant/project id'}
    assert keystone.user_role_list(user_name='nova', tenant_name='nova') == {'nova': {'id': '113', 'name': 'nova', 'tenant_id': '446', 'user_id': '446'}}

def test_api_version_verify_ssl():
    if False:
        return 10
    '\n    test api_version when using verify_ssl\n    '
    test_verify = [True, False, None]
    conn_args = {'keystone.user': 'admin', 'connection_password': 'password', 'connection_tenant': 'admin', 'connection_tenant_id': 'id', 'connection_auth_url': 'https://127.0.0.1/v2.0/', 'connection_verify_ssl': True}
    http_ret = {'dict': {'version': {'id': 'id_test'}}}
    for verify in test_verify:
        mock_http = MagicMock(return_value=http_ret)
        patch_http = patch('salt.utils.http.query', mock_http)
        conn_args['connection_verify_ssl'] = verify
        if verify is None:
            conn_args.pop('connection_verify_ssl')
            verify = True
        with patch_http:
            ret = keystone.api_version(**conn_args)
        assert mock_http.call_args_list == [call('https://127.0.0.1/v2.0/', decode=True, decode_type='json', verify_ssl=verify)]