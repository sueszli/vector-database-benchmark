import re
import requests
from ..common import NotifyType
from .NotifyBase import NotifyBase
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _

class NotifyServerChan(NotifyBase):
    """
    A wrapper for ServerChan Notifications
    """
    service_name = 'ServerChan'
    service_url = 'https://sct.ftqq.com/'
    secure_protocol = 'schan'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_serverchan'
    notify_url = 'https://sctapi.ftqq.com/{token}.send'
    templates = ('{schema}://{token}',)
    template_tokens = dict(NotifyBase.template_tokens, **{'token': {'name': _('Token'), 'type': 'string', 'private': True, 'required': True, 'regex': ('^[a-z0-9]+$', 'i')}})

    def __init__(self, token, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Initialize ServerChan Object\n        '
        super().__init__(**kwargs)
        self.token = validate_regex(token, *self.template_tokens['token']['regex'])
        if not self.token:
            msg = 'An invalid ServerChan API Token ({}) was specified.'.format(token)
            self.logger.warning(msg)
            raise TypeError(msg)

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            print('Hello World!')
        '\n        Perform ServerChan Notification\n        '
        payload = {'title': title, 'desp': body}
        notify_url = self.notify_url.format(token=self.token)
        self.logger.debug('ServerChan URL: {} (cert_verify={})'.format(notify_url, self.verify_certificate))
        self.logger.debug('ServerChan Payload: {}'.format(payload))
        self.throttle()
        try:
            r = requests.post(notify_url, data=payload)
            if r.status_code != requests.codes.ok:
                status_str = NotifyServerChan.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send ServerChan notification: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent ServerChan notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occured sending ServerChan notification.')
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def url(self, privacy=False):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        return '{schema}://{token}'.format(schema=self.secure_protocol, token=self.pprint(self.token, privacy, safe=''))

    @staticmethod
    def parse_url(url):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to substantiate this object.\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        pattern = 'schan://([a-zA-Z0-9]+)/' + ('?' if not url.endswith('/') else '')
        result = re.match(pattern, url)
        results['token'] = result.group(1) if result else ''
        return results