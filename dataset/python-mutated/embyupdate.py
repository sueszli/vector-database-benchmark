"""Updates the Emby Library whenever the beets library is changed.

    emby:
        host: localhost
        port: 8096
        username: user
        apikey: apikey
        password: password
"""
import hashlib
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit, urlunsplit
import requests
from beets import config
from beets.plugins import BeetsPlugin

def api_url(host, port, endpoint):
    if False:
        print('Hello World!')
    'Returns a joined url.\n\n    Takes host, port and endpoint and generates a valid emby API url.\n\n    :param host: Hostname of the emby server\n    :param port: Portnumber of the emby server\n    :param endpoint: API endpoint\n    :type host: str\n    :type port: int\n    :type endpoint: str\n    :returns: Full API url\n    :rtype: str\n    '
    hostname_list = [host]
    if host.startswith('http://') or host.startswith('https://'):
        hostname = ''.join(hostname_list)
    else:
        hostname_list.insert(0, 'http://')
        hostname = ''.join(hostname_list)
    joined = urljoin('{hostname}:{port}'.format(hostname=hostname, port=port), endpoint)
    (scheme, netloc, path, query_string, fragment) = urlsplit(joined)
    query_params = parse_qs(query_string)
    query_params['format'] = ['json']
    new_query_string = urlencode(query_params, doseq=True)
    return urlunsplit((scheme, netloc, path, new_query_string, fragment))

def password_data(username, password):
    if False:
        for i in range(10):
            print('nop')
    'Returns a dict with username and its encoded password.\n\n    :param username: Emby username\n    :param password: Emby password\n    :type username: str\n    :type password: str\n    :returns: Dictionary with username and encoded password\n    :rtype: dict\n    '
    return {'username': username, 'password': hashlib.sha1(password.encode('utf-8')).hexdigest(), 'passwordMd5': hashlib.md5(password.encode('utf-8')).hexdigest()}

def create_headers(user_id, token=None):
    if False:
        return 10
    'Return header dict that is needed to talk to the Emby API.\n\n    :param user_id: Emby user ID\n    :param token: Authentication token for Emby\n    :type user_id: str\n    :type token: str\n    :returns: Headers for requests\n    :rtype: dict\n    '
    headers = {}
    authorization = 'MediaBrowser UserId="{user_id}", Client="other", Device="beets", DeviceId="beets", Version="0.0.0"'.format(user_id=user_id)
    headers['x-emby-authorization'] = authorization
    if token:
        headers['x-mediabrowser-token'] = token
    return headers

def get_token(host, port, headers, auth_data):
    if False:
        while True:
            i = 10
    'Return token for a user.\n\n    :param host: Emby host\n    :param port: Emby port\n    :param headers: Headers for requests\n    :param auth_data: Username and encoded password for authentication\n    :type host: str\n    :type port: int\n    :type headers: dict\n    :type auth_data: dict\n    :returns: Access Token\n    :rtype: str\n    '
    url = api_url(host, port, '/Users/AuthenticateByName')
    r = requests.post(url, headers=headers, data=auth_data)
    return r.json().get('AccessToken')

def get_user(host, port, username):
    if False:
        i = 10
        return i + 15
    'Return user dict from server or None if there is no user.\n\n    :param host: Emby host\n    :param port: Emby port\n    :username: Username\n    :type host: str\n    :type port: int\n    :type username: str\n    :returns: Matched Users\n    :rtype: list\n    '
    url = api_url(host, port, '/Users/Public')
    r = requests.get(url)
    user = [i for i in r.json() if i['Name'] == username]
    return user

class EmbyUpdate(BeetsPlugin):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        config['emby'].add({'host': 'http://localhost', 'port': 8096, 'apikey': None, 'password': None})
        self.register_listener('database_change', self.listen_for_db_change)

    def listen_for_db_change(self, lib, model):
        if False:
            while True:
                i = 10
        'Listens for beets db change and register the update for the end.'
        self.register_listener('cli_exit', self.update)

    def update(self, lib):
        if False:
            print('Hello World!')
        'When the client exists try to send refresh request to Emby.'
        self._log.info('Updating Emby library...')
        host = config['emby']['host'].get()
        port = config['emby']['port'].get()
        username = config['emby']['username'].get()
        password = config['emby']['password'].get()
        userid = config['emby']['userid'].get()
        token = config['emby']['apikey'].get()
        if not any([password, token]):
            self._log.warning('Provide at least Emby password or apikey.')
            return
        if not userid:
            user = get_user(host, port, username)
            if not user:
                self._log.warning(f'User {username} could not be found.')
                return
            userid = user[0]['Id']
        if not token:
            auth_data = password_data(username, password)
            headers = create_headers(userid)
            token = get_token(host, port, headers, auth_data)
            if not token:
                self._log.warning('Could not get token for user {0}', username)
                return
        headers = create_headers(userid, token=token)
        url = api_url(host, port, '/Library/Refresh')
        r = requests.post(url, headers=headers)
        if r.status_code != 204:
            self._log.warning('Update could not be triggered')
        else:
            self._log.info('Update triggered.')