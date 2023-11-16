import re
import requests
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import parse_bool
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _

class NotificoFormat:
    Reset = '\x0f'
    Bold = '\x02'
    Italic = '\x1d'
    Underline = '\x1f'
    BGSwap = '\x16'

class NotificoColor:
    Reset = '\x03'
    White = '\x0300'
    Black = '\x0301'
    Blue = '\x0302'
    Green = '\x0303'
    Red = '\x0304'
    Brown = '\x0305'
    Purple = '\x0306'
    Orange = '\x0307'
    Yellow = ('\x0308',)
    LightGreen = '\x0309'
    Teal = '\x0310'
    LightCyan = '\x0311'
    LightBlue = '\x0312'
    Violet = '\x0313'
    Grey = '\x0314'
    LightGrey = '\x0315'

class NotifyNotifico(NotifyBase):
    """
    A wrapper for Notifico Notifications
    """
    service_name = 'Notifico'
    service_url = 'https://n.tkte.ch'
    protocol = 'notifico'
    secure_protocol = 'notifico'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_notifico'
    notify_url = 'https://n.tkte.ch/h/{proj}/{hook}'
    title_maxlen = 0
    body_maxlen = 512
    templates = ('{schema}://{project_id}/{msghook}',)
    template_tokens = dict(NotifyBase.template_tokens, **{'project_id': {'name': _('Project ID'), 'type': 'string', 'required': True, 'private': True, 'regex': ('^[0-9]+$', '')}, 'msghook': {'name': _('Message Hook'), 'type': 'string', 'required': True, 'private': True, 'regex': ('^[a-z0-9]+$', 'i')}})
    template_args = dict(NotifyBase.template_args, **{'color': {'name': _('IRC Colors'), 'type': 'bool', 'default': True}, 'prefix': {'name': _('Prefix'), 'type': 'bool', 'default': True}})

    def __init__(self, project_id, msghook, color=True, prefix=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize Notifico Object\n        '
        super().__init__(**kwargs)
        self.project_id = validate_regex(project_id, *self.template_tokens['project_id']['regex'])
        if not self.project_id:
            msg = 'An invalid Notifico Project ID ({}) was specified.'.format(project_id)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.msghook = validate_regex(msghook, *self.template_tokens['msghook']['regex'])
        if not self.msghook:
            msg = 'An invalid Notifico Message Token ({}) was specified.'.format(msghook)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.prefix = prefix
        self.color = color
        self.api_url = self.notify_url.format(proj=self.project_id, hook=self.msghook)
        return

    def url(self, privacy=False, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'color': 'yes' if self.color else 'no', 'prefix': 'yes' if self.prefix else 'no'}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        return '{schema}://{proj}/{hook}/?{params}'.format(schema=self.secure_protocol, proj=self.pprint(self.project_id, privacy, safe=''), hook=self.pprint(self.msghook, privacy, safe=''), params=NotifyNotifico.urlencode(params))

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        wrapper to _send since we can alert more then one channel\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8'}
        color = ''
        token = ''
        if notify_type == NotifyType.INFO:
            color = NotificoColor.Teal
            token = 'i'
        elif notify_type == NotifyType.SUCCESS:
            color = NotificoColor.LightGreen
            token = '✔'
        elif notify_type == NotifyType.WARNING:
            color = NotificoColor.Orange
            token = '!'
        elif notify_type == NotifyType.FAILURE:
            color = NotificoColor.Red
            token = '✗'
        if self.color:
            body = re.sub('\\\\x03(\\d{0,2})', '\\\\x03\\g<1>', body)
        else:
            body = re.sub('\\\\x03(\\d{1,2}(,[0-9]{1,2})?)?', '', body)
        payload = {'payload': body if not self.prefix else '{}[{}]{} {}{}{}: {}{}'.format(color if self.color else '', token, NotificoColor.Reset if self.color else '', NotificoFormat.Bold if self.color else '', self.app_id, NotificoFormat.Reset if self.color else '', body, NotificoFormat.Reset if self.color else '')}
        self.logger.debug('Notifico GET URL: %s (cert_verify=%r)' % (self.api_url, self.verify_certificate))
        self.logger.debug('Notifico Payload: %s' % str(payload))
        self.throttle()
        try:
            r = requests.get(self.api_url, params=payload, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyNotifico.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send Notifico notification: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent Notifico notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending Notifico notification.')
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    @staticmethod
    def parse_url(url):
        if False:
            while True:
                i = 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['project_id'] = NotifyNotifico.unquote(results['host'])
        try:
            results['msghook'] = NotifyNotifico.split_path(results['fullpath'])[0]
        except IndexError:
            results['msghook'] = None
        results['color'] = parse_bool(results['qsd'].get('color', True))
        results['prefix'] = parse_bool(results['qsd'].get('prefix', True))
        return results

    @staticmethod
    def parse_native_url(url):
        if False:
            while True:
                i = 10
        '\n        Support https://n.tkte.ch/h/PROJ_ID/MESSAGE_HOOK/\n        '
        result = re.match('^https?://n\\.tkte\\.ch/h/(?P<proj>[0-9]+)/(?P<hook>[A-Z0-9]+)/?(?P<params>\\?.+)?$', url, re.I)
        if result:
            return NotifyNotifico.parse_url('{schema}://{proj}/{hook}/{params}'.format(schema=NotifyNotifico.secure_protocol, proj=result.group('proj'), hook=result.group('hook'), params='' if not result.group('params') else result.group('params')))
        return None