import requests
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..common import NotifyFormat
from ..utils import validate_regex
from ..utils import parse_bool
from ..AppriseLocale import gettext_lazy as _

class NotifyPushMe(NotifyBase):
    """
    A wrapper for PushMe Notifications
    """
    service_name = 'PushMe'
    service_url = 'https://push.i-i.me/'
    protocol = 'pushme'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_pushme'
    notify_url = 'https://push.i-i.me/'
    templates = ('{schema}://{token}',)
    template_tokens = dict(NotifyBase.template_tokens, **{'token': {'name': _('Token'), 'type': 'string', 'private': True, 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'token': {'alias_of': 'token'}, 'push_key': {'alias_of': 'token'}, 'status': {'name': _('Show Status'), 'type': 'bool', 'default': True}})

    def __init__(self, token, status=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialize PushMe Object\n        '
        super().__init__(**kwargs)
        self.token = validate_regex(token)
        if not self.token:
            msg = 'An invalid PushMe Token ({}) was specified.'.format(token)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.status = status
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Perform PushMe Notification\n        '
        headers = {'User-Agent': self.app_id}
        params = {'push_key': self.token, 'title': title if not self.status else '{} {}'.format(self.asset.ascii(notify_type), title), 'content': body, 'type': 'markdown' if self.notify_format == NotifyFormat.MARKDOWN else 'text'}
        self.logger.debug('PushMe POST URL: %s (cert_verify=%r)' % (self.notify_url, self.verify_certificate))
        self.logger.debug('PushMe Payload: %s' % str(params))
        self.throttle()
        try:
            r = requests.post(self.notify_url, params=params, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyPushMe.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send PushMe notification:{}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent PushMe notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending PushMe notification.')
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'status': 'yes' if self.status else 'no'}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        return '{schema}://{token}/?{params}'.format(schema=self.protocol, token=self.pprint(self.token, privacy, safe=''), params=NotifyPushMe.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            return 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['token'] = NotifyPushMe.unquote(results['host'])
        if 'token' in results['qsd'] and len(results['qsd']['token']):
            results['token'] = NotifyPushMe.unquote(results['qsd']['token'])
        elif 'push_key' in results['qsd'] and len(results['qsd']['push_key']):
            results['token'] = NotifyPushMe.unquote(results['qsd']['push_key'])
        results['status'] = parse_bool(results['qsd'].get('status', True))
        return results