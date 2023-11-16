import requests
from json import dumps
from .NotifyBase import NotifyBase
from ..common import NotifyType, NotifyFormat
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _

class GotifyPriority:
    LOW = 0
    MODERATE = 3
    NORMAL = 5
    HIGH = 8
    EMERGENCY = 10
GOTIFY_PRIORITIES = {GotifyPriority.LOW: 'low', GotifyPriority.MODERATE: 'moderate', GotifyPriority.NORMAL: 'normal', GotifyPriority.HIGH: 'high', GotifyPriority.EMERGENCY: 'emergency'}
GOTIFY_PRIORITY_MAP = {'l': GotifyPriority.LOW, 'm': GotifyPriority.MODERATE, 'n': GotifyPriority.NORMAL, 'h': GotifyPriority.HIGH, 'e': GotifyPriority.EMERGENCY, '10': GotifyPriority.EMERGENCY, '0': GotifyPriority.LOW, '1': GotifyPriority.LOW, '2': GotifyPriority.LOW, '3': GotifyPriority.MODERATE, '4': GotifyPriority.MODERATE, '5': GotifyPriority.NORMAL, '6': GotifyPriority.NORMAL, '7': GotifyPriority.NORMAL, '8': GotifyPriority.HIGH, '9': GotifyPriority.HIGH}

class NotifyGotify(NotifyBase):
    """
    A wrapper for Gotify Notifications
    """
    service_name = 'Gotify'
    service_url = 'https://github.com/gotify/server'
    protocol = 'gotify'
    secure_protocol = 'gotifys'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_gotify'
    request_rate_per_sec = 0
    templates = ('{schema}://{host}/{token}', '{schema}://{host}:{port}/{token}', '{schema}://{host}{path}{token}', '{schema}://{host}:{port}{path}{token}')
    template_tokens = dict(NotifyBase.template_tokens, **{'token': {'name': _('Token'), 'type': 'string', 'private': True, 'required': True}, 'host': {'name': _('Hostname'), 'type': 'string', 'required': True}, 'path': {'name': _('Path'), 'type': 'string', 'map_to': 'fullpath', 'default': '/'}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}})
    template_args = dict(NotifyBase.template_args, **{'priority': {'name': _('Priority'), 'type': 'choice:int', 'values': GOTIFY_PRIORITIES, 'default': GotifyPriority.NORMAL}})

    def __init__(self, token, priority=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize Gotify Object\n\n        '
        super().__init__(**kwargs)
        self.token = validate_regex(token)
        if not self.token:
            msg = 'An invalid Gotify Token ({}) was specified.'.format(token)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.fullpath = kwargs.get('fullpath', '/')
        self.priority = int(NotifyGotify.template_args['priority']['default'] if priority is None else next((v for (k, v) in GOTIFY_PRIORITY_MAP.items() if str(priority).lower().startswith(k)), NotifyGotify.template_args['priority']['default']))
        if self.secure:
            self.schema = 'https'
        else:
            self.schema = 'http'
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform Gotify Notification\n        '
        url = '%s://%s' % (self.schema, self.host)
        if self.port:
            url += ':%d' % self.port
        url += '{fullpath}message'.format(fullpath=self.fullpath)
        payload = {'priority': self.priority, 'title': title, 'message': body}
        if self.notify_format == NotifyFormat.MARKDOWN:
            payload['extras'] = {'client::display': {'contentType': 'text/markdown'}}
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json', 'X-Gotify-Key': self.token}
        self.logger.debug('Gotify POST URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
        self.logger.debug('Gotify Payload: %s' % str(payload))
        self.throttle()
        try:
            r = requests.post(url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyGotify.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send Gotify notification: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent Gotify notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending Gotify notification to %s.' % self.host)
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'priority': GOTIFY_PRIORITIES[self.template_args['priority']['default']] if self.priority not in GOTIFY_PRIORITIES else GOTIFY_PRIORITIES[self.priority]}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        default_port = 443 if self.secure else 80
        return '{schema}://{hostname}{port}{fullpath}{token}/?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, hostname=self.host, port='' if self.port is None or self.port == default_port else ':{}'.format(self.port), fullpath=NotifyGotify.quote(self.fullpath, safe='/'), token=self.pprint(self.token, privacy, safe=''), params=NotifyGotify.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url)
        if not results:
            return results
        entries = NotifyBase.split_path(results['fullpath'])
        try:
            results['token'] = entries.pop()
        except IndexError:
            results['token'] = None
        results['fullpath'] = '/' if not entries else '/{}/'.format('/'.join(entries))
        if 'priority' in results['qsd'] and len(results['qsd']['priority']):
            results['priority'] = NotifyGotify.unquote(results['qsd']['priority'])
        return results