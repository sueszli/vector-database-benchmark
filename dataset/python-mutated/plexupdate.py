"""Updates an Plex library whenever the beets library is changed.

Plex Home users enter the Plex Token to enable updating.
Put something like the following in your config.yaml to configure:
    plex:
        host: localhost
        port: 32400
        token: token
"""
from urllib.parse import urlencode, urljoin
from xml.etree import ElementTree
import requests
from beets import config
from beets.plugins import BeetsPlugin

def get_music_section(host, port, token, library_name, secure, ignore_cert_errors):
    if False:
        while True:
            i = 10
    'Getting the section key for the music library in Plex.'
    api_endpoint = append_token('library/sections', token)
    url = urljoin('{}://{}:{}'.format(get_protocol(secure), host, port), api_endpoint)
    r = requests.get(url, verify=not ignore_cert_errors)
    tree = ElementTree.fromstring(r.content)
    for child in tree.findall('Directory'):
        if child.get('title') == library_name:
            return child.get('key')

def update_plex(host, port, token, library_name, secure, ignore_cert_errors):
    if False:
        while True:
            i = 10
    'Ignore certificate errors if configured to.'
    if ignore_cert_errors:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    'Sends request to the Plex api to start a library refresh.\n    '
    section_key = get_music_section(host, port, token, library_name, secure, ignore_cert_errors)
    api_endpoint = f'library/sections/{section_key}/refresh'
    api_endpoint = append_token(api_endpoint, token)
    url = urljoin('{}://{}:{}'.format(get_protocol(secure), host, port), api_endpoint)
    r = requests.get(url, verify=not ignore_cert_errors)
    return r

def append_token(url, token):
    if False:
        return 10
    'Appends the Plex Home token to the api call if required.'
    if token:
        url += '?' + urlencode({'X-Plex-Token': token})
    return url

def get_protocol(secure):
    if False:
        i = 10
        return i + 15
    if secure:
        return 'https'
    else:
        return 'http'

class PlexUpdate(BeetsPlugin):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        config['plex'].add({'host': 'localhost', 'port': 32400, 'token': '', 'library_name': 'Music', 'secure': False, 'ignore_cert_errors': False})
        config['plex']['token'].redact = True
        self.register_listener('database_change', self.listen_for_db_change)

    def listen_for_db_change(self, lib, model):
        if False:
            return 10
        'Listens for beets db change and register the update for the end'
        self.register_listener('cli_exit', self.update)

    def update(self, lib):
        if False:
            while True:
                i = 10
        'When the client exists try to send refresh request to Plex server.'
        self._log.info('Updating Plex library...')
        try:
            update_plex(config['plex']['host'].get(), config['plex']['port'].get(), config['plex']['token'].get(), config['plex']['library_name'].get(), config['plex']['secure'].get(bool), config['plex']['ignore_cert_errors'].get(bool))
            self._log.info('... started.')
        except requests.exceptions.RequestException:
            self._log.warning('Update failed.')