import requests
from .NotifyBase import NotifyBase
from ..common import NotifyImageSize
from ..common import NotifyType
from ..utils import parse_bool
from ..AppriseLocale import gettext_lazy as _
from ..utils import validate_regex

class NotifyFaast(NotifyBase):
    """
    A wrapper for Faast Notifications
    """
    service_name = 'Faast'
    service_url = 'http://www.faast.io/'
    protocol = 'faast'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_faast'
    notify_url = 'https://www.appnotifications.com/account/notifications.json'
    image_size = NotifyImageSize.XY_72
    templates = ('{schema}://{authtoken}',)
    template_tokens = dict(NotifyBase.template_tokens, **{'authtoken': {'name': _('Authorization Token'), 'type': 'string', 'private': True, 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'image': {'name': _('Include Image'), 'type': 'bool', 'default': True, 'map_to': 'include_image'}})

    def __init__(self, authtoken, include_image=True, **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialize Faast Object\n        '
        super().__init__(**kwargs)
        self.authtoken = validate_regex(authtoken)
        if not self.authtoken:
            msg = 'An invalid Faast Authentication Token ({}) was specified.'.format(authtoken)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.include_image = include_image
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Perform Faast Notification\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'multipart/form-data'}
        payload = {'user_credentials': self.authtoken, 'title': title, 'message': body}
        image_url = None if not self.include_image else self.image_url(notify_type)
        if image_url:
            payload['icon_url'] = image_url
        self.logger.debug('Faast POST URL: %s (cert_verify=%r)' % (self.notify_url, self.verify_certificate))
        self.logger.debug('Faast Payload: %s' % str(payload))
        self.throttle()
        try:
            r = requests.post(self.notify_url, data=payload, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyFaast.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send Faast notification:{}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent Faast notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending Faast notification.')
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'image': 'yes' if self.include_image else 'no'}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        return '{schema}://{authtoken}/?{params}'.format(schema=self.protocol, authtoken=self.pprint(self.authtoken, privacy, safe=''), params=NotifyFaast.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            while True:
                i = 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['authtoken'] = NotifyFaast.unquote(results['host'])
        results['include_image'] = parse_bool(results['qsd'].get('image', True))
        return results