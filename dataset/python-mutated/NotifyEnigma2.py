import requests
from json import loads
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyType
from ..AppriseLocale import gettext_lazy as _

class Enigma2MessageType:
    INFO = 1
    WARNING = 2
    ERROR = 3
MESSAGE_MAPPING = {NotifyType.INFO: Enigma2MessageType.INFO, NotifyType.SUCCESS: Enigma2MessageType.INFO, NotifyType.WARNING: Enigma2MessageType.WARNING, NotifyType.FAILURE: Enigma2MessageType.ERROR}

class NotifyEnigma2(NotifyBase):
    """
    A wrapper for Enigma2 Notifications
    """
    service_name = 'Enigma2'
    service_url = 'https://dreambox.de/'
    protocol = 'enigma2'
    secure_protocol = 'enigma2s'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_enigma2'
    title_maxlen = 0
    body_maxlen = 1000
    request_rate_per_sec = 0.5
    templates = ('{schema}://{host}', '{schema}://{host}:{port}', '{schema}://{user}@{host}', '{schema}://{user}@{host}:{port}', '{schema}://{user}:{password}@{host}', '{schema}://{user}:{password}@{host}:{port}', '{schema}://{host}/{fullpath}', '{schema}://{host}:{port}/{fullpath}', '{schema}://{user}@{host}/{fullpath}', '{schema}://{user}@{host}:{port}/{fullpath}', '{schema}://{user}:{password}@{host}/{fullpath}', '{schema}://{user}:{password}@{host}:{port}/{fullpath}')
    template_tokens = dict(NotifyBase.template_tokens, **{'host': {'name': _('Hostname'), 'type': 'string', 'required': True}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}, 'user': {'name': _('Username'), 'type': 'string'}, 'password': {'name': _('Password'), 'type': 'string', 'private': True}, 'fullpath': {'name': _('Path'), 'type': 'string'}})
    template_args = dict(NotifyBase.template_args, **{'timeout': {'name': _('Server Timeout'), 'type': 'int', 'default': 13, 'min': -1}})
    template_kwargs = {'headers': {'name': _('HTTP Header'), 'prefix': '+'}}

    def __init__(self, timeout=None, headers=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialize Enigma2 Object\n\n        headers can be a dictionary of key/value pairs that you want to\n        additionally include as part of the server headers to post with\n        '
        super().__init__(**kwargs)
        try:
            self.timeout = int(timeout)
            if self.timeout < self.template_args['timeout']['min']:
                self.timeout = self.template_args['timeout']['min']
        except (ValueError, TypeError):
            self.timeout = self.template_args['timeout']['default']
        self.fullpath = kwargs.get('fullpath')
        if not isinstance(self.fullpath, str):
            self.fullpath = '/'
        self.headers = {}
        if headers:
            self.headers.update(headers)
        return

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'timeout': str(self.timeout)}
        params.update({'+{}'.format(k): v for (k, v) in self.headers.items()})
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        auth = ''
        if self.user and self.password:
            auth = '{user}:{password}@'.format(user=NotifyEnigma2.quote(self.user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
        elif self.user:
            auth = '{user}@'.format(user=NotifyEnigma2.quote(self.user, safe=''))
        default_port = 443 if self.secure else 80
        return '{schema}://{auth}{hostname}{port}{fullpath}?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, auth=auth, hostname=self.host, port='' if self.port is None or self.port == default_port else ':{}'.format(self.port), fullpath=NotifyEnigma2.quote(self.fullpath, safe='/'), params=NotifyEnigma2.urlencode(params))

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform Enigma2 Notification\n        '
        headers = {'User-Agent': self.app_id}
        params = {'text': body, 'type': MESSAGE_MAPPING.get(notify_type, Enigma2MessageType.INFO), 'timeout': self.timeout}
        headers.update(self.headers)
        auth = None
        if self.user:
            auth = (self.user, self.password)
        schema = 'https' if self.secure else 'http'
        url = '%s://%s' % (schema, self.host)
        if isinstance(self.port, int):
            url += ':%d' % self.port
        url += self.fullpath.rstrip('/') + '/api/message'
        self.logger.debug('Enigma2 POST URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
        self.logger.debug('Enigma2 Parameters: %s' % str(params))
        self.throttle()
        try:
            r = requests.get(url, params=params, headers=headers, auth=auth, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyEnigma2.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send Enigma2 notification: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            try:
                result = loads(r.content).get('result', False)
            except (AttributeError, TypeError, ValueError):
                result = False
            if not result:
                self.logger.warning('Failed to send Enigma2 notification: There was no server acknowledgement.')
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            self.logger.info('Sent Enigma2 notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending Enigma2 notification to %s.' % self.host)
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    @staticmethod
    def parse_url(url):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url)
        if not results:
            return results
        results['headers'] = {NotifyEnigma2.unquote(x): NotifyEnigma2.unquote(y) for (x, y) in results['qsd+'].items()}
        if 'timeout' in results['qsd'] and len(results['qsd']['timeout']):
            results['timeout'] = results['qsd']['timeout']
        return results