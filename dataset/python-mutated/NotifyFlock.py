import re
import requests
from json import dumps
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..common import NotifyFormat
from ..common import NotifyImageSize
from ..utils import parse_list
from ..utils import parse_bool
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
FLOCK_HTTP_ERROR_MAP = {401: 'Unauthorized - Invalid Token.'}
IS_CHANNEL_RE = re.compile('^(#|g:)(?P<id>[A-Z0-9_]+)$', re.I)
IS_USER_RE = re.compile('^(@|u:)?(?P<id>[A-Z0-9_]+)$', re.I)

class NotifyFlock(NotifyBase):
    """
    A wrapper for Flock Notifications
    """
    service_name = 'Flock'
    service_url = 'https://flock.com/'
    secure_protocol = 'flock'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_flock'
    notify_url = 'https://api.flock.com/hooks/sendMessage'
    notify_api = 'https://api.flock.co/v1/chat.sendMessage'
    image_size = NotifyImageSize.XY_72
    templates = ('{schema}://{token}', '{schema}://{botname}@{token}', '{schema}://{botname}@{token}/{targets}', '{schema}://{token}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'token': {'name': _('Access Key'), 'type': 'string', 'regex': ('^[a-z0-9-]+$', 'i'), 'private': True, 'required': True}, 'botname': {'name': _('Bot Name'), 'type': 'string', 'map_to': 'user'}, 'to_user': {'name': _('To User ID'), 'type': 'string', 'prefix': '@', 'regex': ('^[A-Z0-9_]+$', 'i'), 'map_to': 'targets'}, 'to_channel': {'name': _('To Channel ID'), 'type': 'string', 'prefix': '#', 'regex': ('^[A-Z0-9_]+$', 'i'), 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'image': {'name': _('Include Image'), 'type': 'bool', 'default': True, 'map_to': 'include_image'}, 'to': {'alias_of': 'targets'}})

    def __init__(self, token, targets=None, include_image=True, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Initialize Flock Object\n        '
        super().__init__(**kwargs)
        self.targets = list()
        self.token = validate_regex(token, *self.template_tokens['token']['regex'])
        if not self.token:
            msg = 'An invalid Flock Access Key ({}) was specified.'.format(token)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.include_image = include_image
        has_error = False
        targets = parse_list(targets)
        for target in targets:
            result = IS_USER_RE.match(target)
            if result:
                self.targets.append('u:' + result.group('id'))
                continue
            result = IS_CHANNEL_RE.match(target)
            if result:
                self.targets.append('g:' + result.group('id'))
                continue
            has_error = True
            self.logger.warning('Ignoring invalid target ({}) specified.'.format(target))
        if has_error and (not self.targets):
            msg = 'No Flock targets to notify.'
            self.logger.warning(msg)
            raise TypeError(msg)
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            print('Hello World!')
        '\n        Perform Flock Notification\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json'}
        has_error = False
        if self.notify_format == NotifyFormat.HTML:
            body = '<flockml>{}</flockml>'.format(body)
        else:
            title = NotifyFlock.escape_html(title, whitespace=False)
            body = NotifyFlock.escape_html(body, whitespace=False)
            body = '<flockml>{}{}</flockml>'.format('' if not title else '<b>{}</b><br/>'.format(title), body)
        payload = {'token': self.token, 'flockml': body, 'sendAs': {'name': self.app_id if not self.user else self.user, 'profileImage': None if not self.include_image else self.image_url(notify_type)}}
        if len(self.targets):
            targets = list(self.targets)
            while len(targets) > 0:
                target = targets.pop(0)
                _payload = payload.copy()
                _payload['to'] = target
                if not self._post(self.notify_api, headers, _payload):
                    has_error = True
        else:
            url = '{}/{}'.format(self.notify_url, self.token)
            if not self._post(url, headers, payload):
                has_error = True
        return not has_error

    def _post(self, url, headers, payload):
        if False:
            print('Hello World!')
        '\n        A wrapper to the requests object\n        '
        has_error = False
        self.logger.debug('Flock POST URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
        self.logger.debug('Flock Payload: %s' % str(payload))
        self.throttle()
        try:
            r = requests.post(url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyFlock.http_response_code_lookup(r.status_code, FLOCK_HTTP_ERROR_MAP)
                self.logger.warning('Failed to send Flock notification : {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                has_error = True
            else:
                self.logger.info('Sent Flock notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending Flock notification.')
            self.logger.debug('Socket Exception: %s' % str(e))
            has_error = True
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            return 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'image': 'yes' if self.include_image else 'no'}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        return '{schema}://{token}/{targets}?{params}'.format(schema=self.secure_protocol, token=self.pprint(self.token, privacy, safe=''), targets='/'.join([NotifyFlock.quote(target, safe='') for target in self.targets]), params=NotifyFlock.urlencode(params))

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the number of targets associated with this notification\n        '
        targets = len(self.targets)
        return targets if targets > 0 else 1

    @staticmethod
    def parse_url(url):
        if False:
            return 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['targets'] = NotifyFlock.split_path(results['fullpath'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyFlock.parse_list(results['qsd']['to'])
        results['token'] = NotifyFlock.unquote(results['host'])
        results['include_image'] = parse_bool(results['qsd'].get('image', True))
        return results

    @staticmethod
    def parse_native_url(url):
        if False:
            for i in range(10):
                print('nop')
        '\n        Support https://api.flock.com/hooks/sendMessage/TOKEN\n        '
        result = re.match('^https?://api\\.flock\\.com/hooks/sendMessage/(?P<token>[a-z0-9-]{24})/?(?P<params>\\?.+)?$', url, re.I)
        if result:
            return NotifyFlock.parse_url('{schema}://{token}/{params}'.format(schema=NotifyFlock.secure_protocol, token=result.group('token'), params='' if not result.group('params') else result.group('params')))
        return None