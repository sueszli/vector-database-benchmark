import requests
from json import dumps
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
KUMULOS_HTTP_ERROR_MAP = {401: 'Unauthorized - Invalid API and/or Server Key.', 422: 'Unprocessable Entity - The request was unparsable.', 400: 'Bad Request - Targeted users do not exist or have unsubscribed.'}

class NotifyKumulos(NotifyBase):
    """
    A wrapper for Kumulos Notifications
    """
    service_name = 'Kumulos'
    service_url = 'https://kumulos.com/'
    secure_protocol = 'kumulos'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_kumulos'
    notify_url = 'https://messages.kumulos.com/v2/notifications'
    title_maxlen = 64
    body_maxlen = 240
    templates = ('{schema}://{apikey}/{serverkey}',)
    template_tokens = dict(NotifyBase.template_tokens, **{'apikey': {'name': _('API Key'), 'type': 'string', 'private': True, 'required': True, 'regex': ('^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$', 'i')}, 'serverkey': {'name': _('Server Key'), 'type': 'string', 'private': True, 'required': True, 'regex': ('^[A-Z0-9+]{36}$', 'i')}})

    def __init__(self, apikey, serverkey, **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialize Kumulos Object\n        '
        super().__init__(**kwargs)
        self.apikey = validate_regex(apikey, *self.template_tokens['apikey']['regex'])
        if not self.apikey:
            msg = 'An invalid Kumulos API Key ({}) was specified.'.format(apikey)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.serverkey = validate_regex(serverkey, *self.template_tokens['serverkey']['regex'])
        if not self.serverkey:
            msg = 'An invalid Kumulos Server Key ({}) was specified.'.format(serverkey)
            self.logger.warning(msg)
            raise TypeError(msg)

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Perform Kumulos Notification\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json', 'Accept': 'application/json'}
        payload = {'target': {'broadcast': True}, 'content': {'title': title, 'message': body}}
        auth = (self.apikey, self.serverkey)
        self.logger.debug('Kumulos POST URL: %s (cert_verify=%r)' % (self.notify_url, self.verify_certificate))
        self.logger.debug('Kumulos Payload: %s' % str(payload))
        self.throttle()
        try:
            r = requests.post(self.notify_url, data=dumps(payload), headers=headers, auth=auth, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyKumulos.http_response_code_lookup(r.status_code, KUMULOS_HTTP_ERROR_MAP)
                self.logger.warning('Failed to send Kumulos notification: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent Kumulos notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending Kumulos notification.')
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            return 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = self.url_parameters(*args, privacy=privacy, **kwargs)
        return '{schema}://{apikey}/{serverkey}/?{params}'.format(schema=self.secure_protocol, apikey=self.pprint(self.apikey, privacy, safe=''), serverkey=self.pprint(self.serverkey, privacy, safe=''), params=NotifyKumulos.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            print('Hello World!')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['apikey'] = NotifyKumulos.unquote(results['host'])
        try:
            results['serverkey'] = NotifyKumulos.split_path(results['fullpath'])[0]
        except IndexError:
            results['serverkey'] = None
        return results