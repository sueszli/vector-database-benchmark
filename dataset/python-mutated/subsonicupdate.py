"""Updates Subsonic library on Beets import
Your Beets configuration file should contain
a "subsonic" section like the following:
    subsonic:
        url: https://mydomain.com:443/subsonic
        user: username
        pass: password
        auth: token
For older Subsonic versions, token authentication
is not supported, use password instead:
    subsonic:
        url: https://mydomain.com:443/subsonic
        user: username
        pass: password
        auth: pass
"""
import hashlib
import random
import string
from binascii import hexlify
import requests
from beets import config
from beets.plugins import BeetsPlugin
__author__ = 'https://github.com/maffo999'

class SubsonicUpdate(BeetsPlugin):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        config['subsonic'].add({'user': 'admin', 'pass': 'admin', 'url': 'http://localhost:4040', 'auth': 'token'})
        config['subsonic']['pass'].redact = True
        self.register_listener('database_change', self.db_change)
        self.register_listener('smartplaylist_update', self.spl_update)

    def db_change(self, lib, model):
        if False:
            print('Hello World!')
        self.register_listener('cli_exit', self.start_scan)

    def spl_update(self):
        if False:
            while True:
                i = 10
        self.register_listener('cli_exit', self.start_scan)

    @staticmethod
    def __create_token():
        if False:
            return 10
        'Create salt and token from given password.\n\n        :return: The generated salt and hashed token\n        '
        password = config['subsonic']['pass'].as_str()
        r = string.ascii_letters + string.digits
        salt = ''.join([random.choice(r) for _ in range(6)])
        salted_password = password + salt
        token = hashlib.md5(salted_password.encode('utf-8')).hexdigest()
        return (salt, token)

    @staticmethod
    def __format_url(endpoint):
        if False:
            i = 10
            return i + 15
        'Get the Subsonic URL to trigger the given endpoint.\n        Uses either the url config option or the deprecated host, port,\n        and context_path config options together.\n\n        :return: Endpoint for updating Subsonic\n        '
        url = config['subsonic']['url'].as_str()
        if url and url.endswith('/'):
            url = url[:-1]
        if not url:
            host = config['subsonic']['host'].as_str()
            port = config['subsonic']['port'].get(int)
            context_path = config['subsonic']['contextpath'].as_str()
            if context_path == '/':
                context_path = ''
            url = f'http://{host}:{port}{context_path}'
        return url + f'/rest/{endpoint}'

    def start_scan(self):
        if False:
            i = 10
            return i + 15
        user = config['subsonic']['user'].as_str()
        auth = config['subsonic']['auth'].as_str()
        url = self.__format_url('startScan')
        self._log.debug('URL is {0}', url)
        self._log.debug('auth type is {0}', config['subsonic']['auth'])
        if auth == 'token':
            (salt, token) = self.__create_token()
            payload = {'u': user, 't': token, 's': salt, 'v': '1.13.0', 'c': 'beets', 'f': 'json'}
        elif auth == 'password':
            password = config['subsonic']['pass'].as_str()
            encpass = hexlify(password.encode()).decode()
            payload = {'u': user, 'p': f'enc:{encpass}', 'v': '1.12.0', 'c': 'beets', 'f': 'json'}
        else:
            return
        try:
            response = requests.get(url, params=payload)
            json = response.json()
            if response.status_code == 200 and json['subsonic-response']['status'] == 'ok':
                count = json['subsonic-response']['scanStatus']['count']
                self._log.info(f'Updating Subsonic; scanning {count} tracks')
            elif response.status_code == 200 and json['subsonic-response']['status'] == 'failed':
                error_message = json['subsonic-response']['error']['message']
                self._log.error(f'Error: {error_message}')
            else:
                self._log.error('Error: {0}', json)
        except Exception as error:
            self._log.error(f'Error: {error}')