import requests
from json import dumps
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
UUID4_RE = '[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}'

class NotifyTechulusPush(NotifyBase):
    """
    A wrapper for Techulus Push Notifications
    """
    service_name = 'Techulus Push'
    service_url = 'https://push.techulus.com'
    secure_protocol = 'push'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_techulus'
    notify_url = 'https://push.techulus.com/api/v1/notify'
    body_maxlen = 1000
    templates = ('{schema}://{apikey}',)
    template_tokens = dict(NotifyBase.template_tokens, **{'apikey': {'name': _('API Key'), 'type': 'string', 'private': True, 'required': True, 'regex': ('^{}$'.format(UUID4_RE), 'i')}})

    def __init__(self, apikey, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize Techulus Push Object\n        '
        super().__init__(**kwargs)
        self.apikey = validate_regex(apikey, *self.template_tokens['apikey']['regex'])
        if not self.apikey:
            msg = 'An invalid Techulus Push API key ({}) was specified.'.format(apikey)
            self.logger.warning(msg)
            raise TypeError(msg)

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            return 10
        '\n        Perform Techulus Push Notification\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json', 'x-api-key': self.apikey}
        payload = {'title': title, 'body': body}
        self.logger.debug('Techulus Push POST URL: %s (cert_verify=%r)' % (self.notify_url, self.verify_certificate))
        self.logger.debug('Techulus Push Payload: %s' % str(payload))
        self.throttle()
        try:
            r = requests.post(self.notify_url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code not in (requests.codes.ok, requests.codes.no_content):
                status_str = NotifyTechulusPush.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send Techulus Push notification: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent Techulus Push notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending Techulus Push notification.')
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            return 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = self.url_parameters(*args, privacy=privacy, **kwargs)
        return '{schema}://{apikey}/?{params}'.format(schema=self.secure_protocol, apikey=self.pprint(self.apikey, privacy, safe=''), params=NotifyTechulusPush.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            while True:
                i = 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['apikey'] = NotifyTechulusPush.unquote(results['host'])
        return results