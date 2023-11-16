import re
import requests
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyType
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _

class NoticaMode:
    """
    Tracks if we're accessing the notica upstream server or a locally hosted
    one.
    """
    SELFHOSTED = 'selfhosted'
    OFFICIAL = 'official'
NOTICA_MODES = (NoticaMode.SELFHOSTED, NoticaMode.OFFICIAL)

class NotifyNotica(NotifyBase):
    """
    A wrapper for Notica Notifications
    """
    service_name = 'Notica'
    service_url = 'https://notica.us/'
    protocol = 'notica'
    secure_protocol = 'noticas'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_notica'
    notify_url = 'https://notica.us/?{token}'
    title_maxlen = 0
    templates = ('{schema}://{token}', '{schema}://{host}/{token}', '{schema}://{host}:{port}/{token}', '{schema}://{user}@{host}/{token}', '{schema}://{user}@{host}:{port}/{token}', '{schema}://{user}:{password}@{host}/{token}', '{schema}://{user}:{password}@{host}:{port}/{token}', '{schema}://{host}{path}/{token}', '{schema}://{host}:{port}/{path}/{token}', '{schema}://{user}@{host}/{path}/{token}', '{schema}://{user}@{host}:{port}{path}/{token}', '{schema}://{user}:{password}@{host}{path}/{token}', '{schema}://{user}:{password}@{host}:{port}/{path}/{token}')
    template_tokens = dict(NotifyBase.template_tokens, **{'token': {'name': _('Token'), 'type': 'string', 'private': True, 'required': True, 'regex': '^\\?*(?P<token>[^/]+)\\s*$'}, 'host': {'name': _('Hostname'), 'type': 'string'}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}, 'user': {'name': _('Username'), 'type': 'string'}, 'password': {'name': _('Password'), 'type': 'string', 'private': True}, 'path': {'name': _('Path'), 'type': 'string', 'map_to': 'fullpath', 'default': '/'}})
    template_kwargs = {'headers': {'name': _('HTTP Header'), 'prefix': '+'}}

    def __init__(self, token, headers=None, **kwargs):
        if False:
            return 10
        '\n        Initialize Notica Object\n        '
        super().__init__(**kwargs)
        self.token = validate_regex(token)
        if not self.token:
            msg = 'An invalid Notica Token ({}) was specified.'.format(token)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.mode = NoticaMode.SELFHOSTED if self.host else NoticaMode.OFFICIAL
        self.fullpath = kwargs.get('fullpath')
        if not isinstance(self.fullpath, str):
            self.fullpath = '/'
        self.headers = {}
        if headers:
            self.headers.update(headers)
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            print('Hello World!')
        '\n        Perform Notica Notification\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/x-www-form-urlencoded'}
        payload = 'd:{}'.format(body)
        auth = None
        if self.mode is NoticaMode.OFFICIAL:
            notify_url = self.notify_url.format(token=self.token)
        else:
            headers.update(self.headers)
            if self.user:
                auth = (self.user, self.password)
            schema = 'https' if self.secure else 'http'
            notify_url = '%s://%s' % (schema, self.host)
            if isinstance(self.port, int):
                notify_url += ':%d' % self.port
            notify_url += '{fullpath}?token={token}'.format(fullpath=self.fullpath, token=self.token)
        self.logger.debug('Notica POST URL: %s (cert_verify=%r)' % (notify_url, self.verify_certificate))
        self.logger.debug('Notica Payload: %s' % str(payload))
        self.throttle()
        try:
            r = requests.post(notify_url.format(token=self.token), data=payload, headers=headers, auth=auth, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyNotica.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send Notica notification:{}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent Notica notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending Notica notification.')
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = self.url_parameters(*args, privacy=privacy, **kwargs)
        if self.mode == NoticaMode.OFFICIAL:
            return '{schema}://{token}/?{params}'.format(schema=self.protocol, token=self.pprint(self.token, privacy, safe=''), params=NotifyNotica.urlencode(params))
        params.update({'+{}'.format(k): v for (k, v) in self.headers.items()})
        auth = ''
        if self.user and self.password:
            auth = '{user}:{password}@'.format(user=NotifyNotica.quote(self.user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
        elif self.user:
            auth = '{user}@'.format(user=NotifyNotica.quote(self.user, safe=''))
        default_port = 443 if self.secure else 80
        return '{schema}://{auth}{hostname}{port}{fullpath}{token}/?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, auth=auth, hostname=NotifyNotica.quote(self.host, safe=''), port='' if self.port is None or self.port == default_port else ':{}'.format(self.port), fullpath=NotifyNotica.quote(self.fullpath, safe='/'), token=self.pprint(self.token, privacy, safe=''), params=NotifyNotica.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            print('Hello World!')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        entries = NotifyNotica.split_path(results['fullpath'])
        if not entries:
            results['mode'] = NoticaMode.OFFICIAL
            results['token'] = NotifyNotica.unquote(results['host'])
            results['host'] = None
        else:
            results['mode'] = NoticaMode.SELFHOSTED
            results['token'] = entries.pop()
            results['fullpath'] = '/' if not entries else '/{}/'.format('/'.join(entries))
            results['headers'] = {NotifyNotica.unquote(x): NotifyNotica.unquote(y) for (x, y) in results['qsd+'].items()}
        return results

    @staticmethod
    def parse_native_url(url):
        if False:
            return 10
        '\n        Support https://notica.us/?abc123\n        '
        result = re.match('^https?://notica\\.us/?\\??(?P<token>[^&]+)([&\\s]*(?P<params>.+))?$', url, re.I)
        if result:
            return NotifyNotica.parse_url('{schema}://{token}/{params}'.format(schema=NotifyNotica.protocol, token=result.group('token'), params='' if not result.group('params') else '?{}'.format(result.group('params'))))
        return None