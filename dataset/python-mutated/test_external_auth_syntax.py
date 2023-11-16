import urllib.parse
import attr
import pytest
import salt.utils.json
import salt.utils.yaml
import tests.support.netapi as netapi
from salt.netapi.rest_tornado import saltnado
pytestmark = [pytest.mark.destructive_test, pytest.mark.skip_if_not_root]
ACCOUNT_USERNAME = 'saltdev-syntax'
ACCOUNT_GROUP_NAME = '{}-group'.format(ACCOUNT_USERNAME)

@attr.s(frozen=True, slots=True)
class ExternalAuthConfig:
    eauth = attr.ib()
    pam_key = attr.ib(repr=False)
    pam_config = attr.ib(repr=False)
    expected_perms = attr.ib(repr=False)
    fixture_id = attr.ib(repr=False)
    auto = attr.ib(init=False)
    pam = attr.ib(init=False)

    @auto.default
    def _set_auto(self):
        if False:
            return 10
        return {'*': ['grains.*'], ACCOUNT_USERNAME: ['@wheel'], '{}%'.format(ACCOUNT_GROUP_NAME): ['@runner']}

    @pam.default
    def _set_pam(self):
        if False:
            i = 10
            return i + 15
        return {self.pam_key: self.pam_config}

@pytest.fixture(scope='module')
def netapi_account():
    if False:
        i = 10
        return i + 15
    with pytest.helpers.create_account(username=ACCOUNT_USERNAME, password='saltdev', group_name=ACCOUNT_GROUP_NAME) as account:
        yield account

def external_auth_ids(value):
    if False:
        i = 10
        return i + 15
    return value.fixture_id

@pytest.fixture(params=(ExternalAuthConfig(eauth='pam', pam_key=ACCOUNT_USERNAME, pam_config=['test.*'], expected_perms=['test.*'], fixture_id='by-user-pam'), ExternalAuthConfig(eauth='auto', pam_key=ACCOUNT_USERNAME, pam_config=['test.*'], expected_perms=['@wheel', 'grains.*'], fixture_id='by-user-auto'), ExternalAuthConfig(eauth='pam', pam_key='{}%'.format(ACCOUNT_GROUP_NAME), pam_config=['grains.*'], expected_perms=['grains.*'], fixture_id='by-group-pam'), ExternalAuthConfig(eauth='auto', pam_key='{}%'.format(ACCOUNT_GROUP_NAME), pam_config=['@wheel', 'grains.*'], expected_perms=['@wheel', 'grains.*'], fixture_id='by-group-auto'), ExternalAuthConfig(eauth='pam', pam_key=ACCOUNT_USERNAME, pam_config=[{'G@id:master2': ['@jobs']}, {'G@id:master1': ['@jobs']}], expected_perms=[{'G@id:master1': ['@jobs']}, {'G@id:master2': ['@jobs']}], fixture_id='by-user-by-minion-pam'), ExternalAuthConfig(eauth='auto', pam_key=ACCOUNT_USERNAME, pam_config=[{'G@id:master2': ['@jobs']}, {'G@id:master1': ['@jobs']}], expected_perms=['@wheel', 'grains.*'], fixture_id='by-user-by-minion-auto'), ExternalAuthConfig(eauth='pam', pam_key=ACCOUNT_USERNAME, pam_config=['@wheel'], expected_perms=['@wheel'], fixture_id='by-user-by-@wheel-pam'), ExternalAuthConfig(eauth='auto', pam_key=ACCOUNT_USERNAME, pam_config=['@wheel'], expected_perms=['@wheel', 'grains.*'], fixture_id='by-user-by-@wheel-auto'), ExternalAuthConfig(eauth='pam', pam_key=ACCOUNT_USERNAME, pam_config=['@runner'], expected_perms=['@runner'], fixture_id='by-user-by-@runner-pam'), ExternalAuthConfig(eauth='auto', pam_key=ACCOUNT_USERNAME, pam_config=['@runner'], expected_perms=['@wheel', 'grains.*'], fixture_id='by-user-by-@runner-auto'), ExternalAuthConfig(eauth='pam', pam_key=ACCOUNT_USERNAME, pam_config=['@jobs'], expected_perms=['@jobs'], fixture_id='by-user-by-@jobs-pam'), ExternalAuthConfig(eauth='auto', pam_key=ACCOUNT_USERNAME, pam_config=['@jobs'], expected_perms=['@wheel', 'grains.*'], fixture_id='by-user-by-@jobs-auto'), ExternalAuthConfig(eauth='pam', pam_key='{}%'.format(ACCOUNT_GROUP_NAME), pam_config=['@wheel'], expected_perms=['@wheel'], fixture_id='by-group-by-@wheel-pam'), ExternalAuthConfig(eauth='auto', pam_key='{}%'.format(ACCOUNT_GROUP_NAME), pam_config=['@wheel'], expected_perms=['@wheel', 'grains.*'], fixture_id='by-group-by-@wheel-auto'), ExternalAuthConfig(eauth='pam', pam_key='{}%'.format(ACCOUNT_GROUP_NAME), pam_config=['@runner'], expected_perms=['@runner'], fixture_id='by-group-by-@runner-pam'), ExternalAuthConfig(eauth='auto', pam_key='{}%'.format(ACCOUNT_GROUP_NAME), pam_config=['@runner'], expected_perms=['@wheel', 'grains.*'], fixture_id='by-group-by-@runner-auto'), ExternalAuthConfig(eauth='pam', pam_key='{}%'.format(ACCOUNT_GROUP_NAME), pam_config=['@jobs'], expected_perms=['@jobs'], fixture_id='by-group-by-@jobs-pam'), ExternalAuthConfig(eauth='auto', pam_key='{}%'.format(ACCOUNT_GROUP_NAME), pam_config=['@jobs'], expected_perms=['@wheel', 'grains.*'], fixture_id='by-group-by-@jobs-auto'), ExternalAuthConfig(eauth='pam', pam_key=ACCOUNT_USERNAME, pam_config=[{'@runner': ['active']}], expected_perms=[{'@runner': ['active']}], fixture_id='by-user-by-@wheel/@runner/@jobs-module-pam'), ExternalAuthConfig(eauth='auto', pam_key=ACCOUNT_USERNAME, pam_config=[{'@runner': ['active']}], expected_perms=['@wheel', 'grains.*'], fixture_id='by-user-by-@wheel/@runner/@jobs-module-auto'), ExternalAuthConfig(eauth='pam', pam_key=ACCOUNT_USERNAME, pam_config=[{'*': [{'my_mod.*': {'args': ['a1.*', '.*', 'a3.*'], 'kwargs': {'kwa': 'kwa.*', 'kwb': 'kwb'}}}]}], expected_perms=[{'*': [{'my_mod.*': {'args': ['a1.*', '.*', 'a3.*'], 'kwargs': {'kwa': 'kwa.*', 'kwb': 'kwb'}}}]}], fixture_id='by-user-by-module-args-kwargs-pam'), ExternalAuthConfig(eauth='auto', pam_key=ACCOUNT_USERNAME, pam_config=[{'*': [{'my_mod.*': {'args': ['a1.*', '.*', 'a3.*'], 'kwargs': {'kwa': 'kwa.*', 'kwb': 'kwb'}}}]}], expected_perms=['@wheel', 'grains.*'], fixture_id='by-user-by-module-args-kwargs-auto')), ids=external_auth_ids)
def external_auth(request):
    if False:
        while True:
            i = 10
    return request.param

@pytest.fixture
def auth_creds(external_auth, netapi_account):
    if False:
        while True:
            i = 10
    return {'username': netapi_account.username, 'password': netapi_account.password, 'eauth': external_auth.eauth}

@pytest.fixture
def client_config(client_config, external_auth):
    if False:
        for i in range(10):
            print('nop')
    client_config['external_auth'] = {'auto': external_auth.auto, 'pam': external_auth.pam}
    return client_config

@pytest.fixture
def http_server(io_loop, app, netapi_port, content_type_map):
    if False:
        return 10
    with netapi.TestsTornadoHttpServer(io_loop=io_loop, app=app, port=netapi_port, client_headers={'Content-Type': content_type_map['form']}) as server:
        yield server

@pytest.fixture
def app_urls():
    if False:
        return 10
    return [('/login', saltnado.SaltAuthHandler)]

async def test_perms(http_client, auth_creds, external_auth):
    response = await http_client.fetch('/login', method='POST', body=urllib.parse.urlencode(auth_creds))
    assert response.code == 200
    response_obj = salt.utils.json.loads(response.body)['return'][0]
    perms = response_obj['perms']
    assert perms == external_auth.expected_perms