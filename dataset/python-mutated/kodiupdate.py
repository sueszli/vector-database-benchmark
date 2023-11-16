"""Updates a Kodi library whenever the beets library is changed.
This is based on the Plex Update plugin.

Put something like the following in your config.yaml to configure:
    kodi:
        host: localhost
        port: 8080
        user: user
        pwd: secret
"""
import requests
from beets import config
from beets.plugins import BeetsPlugin

def update_kodi(host, port, user, password):
    if False:
        while True:
            i = 10
    'Sends request to the Kodi api to start a library refresh.'
    url = f'http://{host}:{port}/jsonrpc'
    'Content-Type: application/json is mandatory\n    according to the kodi jsonrpc documentation'
    headers = {'Content-Type': 'application/json'}
    payload = {'jsonrpc': '2.0', 'method': 'AudioLibrary.Scan', 'id': 1}
    r = requests.post(url, auth=(user, password), json=payload, headers=headers)
    return r

class KodiUpdate(BeetsPlugin):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        config['kodi'].add([{'host': 'localhost', 'port': 8080, 'user': 'kodi', 'pwd': 'kodi'}])
        config['kodi']['pwd'].redact = True
        self.register_listener('database_change', self.listen_for_db_change)

    def listen_for_db_change(self, lib, model):
        if False:
            i = 10
            return i + 15
        'Listens for beets db change and register the update'
        self.register_listener('cli_exit', self.update)

    def update(self, lib):
        if False:
            print('Hello World!')
        'When the client exists try to send refresh request to Kodi server.'
        self._log.info('Requesting a Kodi library update...')
        kodi = config['kodi'].get()
        if not isinstance(kodi, list):
            kodi = [kodi]
        for instance in kodi:
            try:
                r = update_kodi(instance['host'], instance['port'], instance['user'], instance['pwd'])
                r.raise_for_status()
                json = r.json()
                if json.get('result') != 'OK':
                    self._log.warning('Kodi update failed: JSON response was {0!r}', json)
                    continue
                self._log.info('Kodi update triggered for {0}:{1}', instance['host'], instance['port'])
            except requests.exceptions.RequestException as e:
                self._log.warning('Kodi update failed: {0}', str(e))
                continue