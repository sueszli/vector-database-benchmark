import re
import requests
from json import dumps
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..common import NotifyFormat
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
WEBEX_HTTP_ERROR_MAP = {401: 'Unauthorized - Invalid Token.', 415: 'Unsuported media specified', 429: 'To many consecutive requests were made.', 503: 'Service is overloaded, try again later'}

class NotifyWebexTeams(NotifyBase):
    """
    A wrapper for Webex Teams Notifications
    """
    service_name = 'Cisco Webex Teams'
    service_url = 'https://webex.teams.com/'
    secure_protocol = ('wxteams', 'webex')
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_wxteams'
    notify_url = 'https://api.ciscospark.com/v1/webhooks/incoming/'
    body_maxlen = 1000
    title_maxlen = 0
    notify_format = NotifyFormat.MARKDOWN
    templates = ('{schema}://{token}',)
    template_tokens = dict(NotifyBase.template_tokens, **{'token': {'name': _('Token'), 'type': 'string', 'private': True, 'required': True, 'regex': ('^[a-z0-9]{80,160}$', 'i')}})

    def __init__(self, token, **kwargs):
        if False:
            return 10
        '\n        Initialize Webex Teams Object\n        '
        super().__init__(**kwargs)
        self.token = validate_regex(token, *self.template_tokens['token']['regex'])
        if not self.token:
            msg = 'The Webex Teams token specified ({}) is invalid.'.format(token)
            self.logger.warning(msg)
            raise TypeError(msg)

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            return 10
        '\n        Perform Webex Teams Notification\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json'}
        url = '{}/{}'.format(self.notify_url, self.token)
        payload = {'markdown' if self.notify_format == NotifyFormat.MARKDOWN else 'text': body}
        self.logger.debug('Webex Teams POST URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
        self.logger.debug('Webex Teams Payload: %s' % str(payload))
        self.throttle()
        try:
            r = requests.post(url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code not in (requests.codes.ok, requests.codes.no_content):
                status_str = NotifyWebexTeams.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send Webex Teams notification: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent Webex Teams notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending Webex Teams notification.')
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = self.url_parameters(*args, privacy=privacy, **kwargs)
        return '{schema}://{token}/?{params}'.format(schema=self.secure_protocol[0], token=self.pprint(self.token, privacy, safe=''), params=NotifyWebexTeams.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            return 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['token'] = NotifyWebexTeams.unquote(results['host'])
        return results

    @staticmethod
    def parse_native_url(url):
        if False:
            print('Hello World!')
        '\n        Support https://api.ciscospark.com/v1/webhooks/incoming/WEBHOOK_TOKEN\n        '
        result = re.match('^https?://(api\\.ciscospark\\.com|webexapis\\.com)/v[1-9][0-9]*/webhooks/incoming/(?P<webhook_token>[A-Z0-9_-]+)/?(?P<params>\\?.+)?$', url, re.I)
        if result:
            return NotifyWebexTeams.parse_url('{schema}://{webhook_token}/{params}'.format(schema=NotifyWebexTeams.secure_protocol[0], webhook_token=result.group('webhook_token'), params='' if not result.group('params') else result.group('params')))
        return None