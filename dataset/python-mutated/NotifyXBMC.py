import requests
from json import dumps
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyType
from ..common import NotifyImageSize
from ..utils import parse_bool
from ..AppriseLocale import gettext_lazy as _

class NotifyXBMC(NotifyBase):
    """
    A wrapper for XBMC/KODI Notifications
    """
    service_name = 'Kodi/XBMC'
    service_url = 'http://kodi.tv/'
    xbmc_protocol = 'xbmc'
    xbmc_secure_protocol = 'xbmcs'
    kodi_protocol = 'kodi'
    kodi_secure_protocol = 'kodis'
    protocol = (xbmc_protocol, kodi_protocol)
    secure_protocol = (xbmc_secure_protocol, kodi_secure_protocol)
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_kodi'
    request_rate_per_sec = 0
    body_max_line_count = 2
    xbmc_default_port = 8080
    image_size = NotifyImageSize.XY_128
    xbmc_remote_protocol = 2
    kodi_remote_protocol = 6
    templates = ('{schema}://{host}', '{schema}://{host}:{port}', '{schema}://{user}:{password}@{host}', '{schema}://{user}:{password}@{host}:{port}')
    template_tokens = dict(NotifyBase.template_tokens, **{'host': {'name': _('Hostname'), 'type': 'string', 'required': True}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}, 'user': {'name': _('Username'), 'type': 'string'}, 'password': {'name': _('Password'), 'type': 'string', 'private': True}})
    template_args = dict(NotifyBase.template_args, **{'duration': {'name': _('Duration'), 'type': 'int', 'min': 1, 'default': 12}, 'image': {'name': _('Include Image'), 'type': 'bool', 'default': True, 'map_to': 'include_image'}})

    def __init__(self, include_image=True, duration=None, **kwargs):
        if False:
            return 10
        '\n        Initialize XBMC/KODI Object\n        '
        super().__init__(**kwargs)
        self.duration = self.template_args['duration']['default'] if not (isinstance(duration, int) and self.template_args['duration']['min'] > 0) else duration
        self.schema = 'https' if self.secure else 'http'
        self.headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json'}
        self.protocol = kwargs.get('protocol', self.xbmc_remote_protocol)
        self.include_image = include_image

    def _payload_60(self, title, body, notify_type, **kwargs):
        if False:
            print('Hello World!')
        '\n        Builds payload for KODI API v6.0\n\n        Returns (headers, payload)\n        '
        payload = {'jsonrpc': '2.0', 'method': 'GUI.ShowNotification', 'params': {'title': title, 'message': body, 'displaytime': int(self.duration * 1000)}, 'id': 1}
        image_url = None if not self.include_image else self.image_url(notify_type)
        if image_url:
            payload['params']['image'] = image_url
            if notify_type is NotifyType.FAILURE:
                payload['type'] = 'error'
            elif notify_type is NotifyType.WARNING:
                payload['type'] = 'warning'
            else:
                payload['type'] = 'info'
        return (self.headers, dumps(payload))

    def _payload_20(self, title, body, notify_type, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Builds payload for XBMC API v2.0\n\n        Returns (headers, payload)\n        '
        payload = {'jsonrpc': '2.0', 'method': 'GUI.ShowNotification', 'params': {'title': title, 'message': body, 'displaytime': int(self.duration * 1000)}, 'id': 1}
        image_url = None if not self.include_image else self.image_url(notify_type)
        if image_url:
            payload['params']['image'] = image_url
        return (self.headers, dumps(payload))

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform XBMC/KODI Notification\n        '
        if self.protocol == self.xbmc_remote_protocol:
            (headers, payload) = self._payload_20(title, body, notify_type, **kwargs)
        else:
            (headers, payload) = self._payload_60(title, body, notify_type, **kwargs)
        auth = None
        if self.user:
            auth = (self.user, self.password)
        url = '%s://%s' % (self.schema, self.host)
        if self.port:
            url += ':%d' % self.port
        url += '/jsonrpc'
        self.logger.debug('XBMC/KODI POST URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
        self.logger.debug('XBMC/KODI Payload: %s' % str(payload))
        self.throttle()
        try:
            r = requests.post(url, data=payload, headers=headers, auth=auth, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyXBMC.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send XBMC/KODI notification: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent XBMC/KODI notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending XBMC/KODI notification.')
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'image': 'yes' if self.include_image else 'no', 'duration': str(self.duration)}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        auth = ''
        if self.user and self.password:
            auth = '{user}:{password}@'.format(user=NotifyXBMC.quote(self.user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
        elif self.user:
            auth = '{user}@'.format(user=NotifyXBMC.quote(self.user, safe=''))
        default_schema = self.xbmc_protocol if self.protocol <= self.xbmc_remote_protocol else self.kodi_protocol
        default_port = 443 if self.secure else self.xbmc_default_port
        if self.secure:
            default_schema += 's'
        return '{schema}://{auth}{hostname}{port}/?{params}'.format(schema=default_schema, auth=auth, hostname=self.host, port='' if not self.port or self.port == default_port else ':{}'.format(self.port), params=NotifyXBMC.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            while True:
                i = 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url)
        if not results:
            return results
        if results.get('schema', '').startswith('xbmc'):
            results['protocol'] = NotifyXBMC.xbmc_remote_protocol
            if not results['port']:
                results['port'] = NotifyXBMC.xbmc_default_port
        else:
            results['protocol'] = NotifyXBMC.kodi_remote_protocol
        results['include_image'] = parse_bool(results['qsd'].get('image', True))
        try:
            results['duration'] = abs(int(results['qsd'].get('duration')))
        except (TypeError, ValueError):
            pass
        return results