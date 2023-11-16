from os import urandom
from json import loads
import requests
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyType
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
from base64 import urlsafe_b64encode
import hashlib
try:
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.ciphers import Cipher
    from cryptography.hazmat.primitives.ciphers import algorithms
    from cryptography.hazmat.primitives.ciphers import modes
    from cryptography.hazmat.backends import default_backend
    NOTIFY_SIMPLEPUSH_ENABLED = True
except ImportError:
    NOTIFY_SIMPLEPUSH_ENABLED = False

class NotifySimplePush(NotifyBase):
    """
    A wrapper for SimplePush Notifications
    """
    enabled = NOTIFY_SIMPLEPUSH_ENABLED
    requirements = {'packages_required': 'cryptography'}
    service_name = 'SimplePush'
    service_url = 'https://simplepush.io/'
    secure_protocol = 'spush'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_simplepush'
    notify_url = 'https://api.simplepush.io/send'
    body_maxlen = 10000
    title_maxlen = 1024
    templates = ('{schema}://{apikey}', '{schema}://{salt}:{password}@{apikey}')
    template_tokens = dict(NotifyBase.template_tokens, **{'apikey': {'name': _('API Key'), 'type': 'string', 'private': True, 'required': True}, 'password': {'name': _('Password'), 'type': 'string', 'private': True}, 'salt': {'name': _('Salt'), 'type': 'string', 'private': True, 'map_to': 'user'}})
    template_args = dict(NotifyBase.template_args, **{'event': {'name': _('Event'), 'type': 'string'}})

    def __init__(self, apikey, event=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Initialize SimplePush Object\n        '
        super().__init__(**kwargs)
        self.apikey = validate_regex(apikey)
        if not self.apikey:
            msg = 'An invalid SimplePush API Key ({}) was specified.'.format(apikey)
            self.logger.warning(msg)
            raise TypeError(msg)
        if event:
            self.event = validate_regex(event)
            if not self.event:
                msg = 'An invalid SimplePush Event Name ({}) was specified.'.format(event)
                self.logger.warning(msg)
                raise TypeError(msg)
        else:
            self.event = None
        self._iv = None
        self._iv_hex = None
        self._key = None

    def _encrypt(self, content):
        if False:
            return 10
        '\n        Encrypts message for use with SimplePush\n        '
        if self._iv is None:
            self._iv = urandom(algorithms.AES.block_size // 8)
            self._iv_hex = ''.join(['{:02x}'.format(ord(self._iv[idx:idx + 1])) for idx in range(len(self._iv))]).upper()
            self._key = bytes(bytearray.fromhex(hashlib.sha1('{}{}'.format(self.password, self.user).encode('utf-8')).hexdigest()[0:32]))
        padder = padding.PKCS7(algorithms.AES.block_size).padder()
        content = padder.update(content.encode()) + padder.finalize()
        encryptor = Cipher(algorithms.AES(self._key), modes.CBC(self._iv), default_backend()).encryptor()
        return urlsafe_b64encode(encryptor.update(content) + encryptor.finalize())

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            print('Hello World!')
        '\n        Perform SimplePush Notification\n        '
        headers = {'User-Agent': self.app_id, 'Content-type': 'application/x-www-form-urlencoded'}
        payload = {'key': self.apikey}
        if self.password and self.user:
            body = self._encrypt(body)
            title = self._encrypt(title)
            payload.update({'encrypted': 'true', 'iv': self._iv_hex})
        payload.update({'msg': body, 'title': title})
        if self.event:
            payload['event'] = self.event
        self.logger.debug('SimplePush POST URL: %s (cert_verify=%r)' % (self.notify_url, self.verify_certificate))
        self.logger.debug('SimplePush Payload: %s' % str(payload))
        status_str = None
        status = None
        self.throttle()
        try:
            r = requests.post(self.notify_url, data=payload, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            try:
                json_response = loads(r.content)
                status_str = json_response.get('message')
                status = json_response.get('status')
            except (TypeError, ValueError, AttributeError):
                pass
            if r.status_code != requests.codes.ok or status != 'OK':
                status_str = status_str if status_str else NotifyBase.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send SimplePush notification:{}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent SimplePush notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending SimplePush notification.')
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = self.url_parameters(*args, privacy=privacy, **kwargs)
        if self.event:
            params['event'] = self.event
        auth = ''
        if self.user and self.password:
            auth = '{salt}:{password}@'.format(salt=self.pprint(self.user, privacy, mode=PrivacyMode.Secret, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
        return '{schema}://{auth}{apikey}/?{params}'.format(schema=self.secure_protocol, auth=auth, apikey=self.pprint(self.apikey, privacy, safe=''), params=NotifySimplePush.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            print('Hello World!')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['apikey'] = NotifySimplePush.unquote(results['host'])
        if 'event' in results['qsd'] and len(results['qsd']['event']):
            results['event'] = NotifySimplePush.unquote(results['qsd']['event'])
        return results