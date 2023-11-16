import requests
import re
from json import dumps
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyType
from ..common import NotifyImageSize
from ..utils import validate_regex
from ..utils import parse_list
from ..utils import parse_bool
from ..AppriseLocale import gettext_lazy as _
TARGET_LIST_DELIM = re.compile('[ \\t\\r\\n,#\\\\/]+')

class NotifyLine(NotifyBase):
    """
    A wrapper for Line Notifications
    """
    service_name = 'Line'
    service_url = 'https://line.me/'
    secure_protocol = 'line'
    notify_url = 'https://api.line.me/v2/bot/message/push'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_line'
    title_maxlen = 0
    body_maxlen = 5000
    image_size = NotifyImageSize.XY_128
    templates = ('{schema}://{token}/{targets}',)
    template_tokens = dict(NotifyBase.template_tokens, **{'token': {'name': _('Access Token'), 'type': 'string', 'private': True, 'required': True}, 'target_user': {'name': _('Target User'), 'type': 'string', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string', 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}, 'image': {'name': _('Include Image'), 'type': 'bool', 'default': True, 'map_to': 'include_image'}})

    def __init__(self, token, targets=None, include_image=True, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Initialize Line Object\n        '
        super().__init__(**kwargs)
        self.token = validate_regex(token)
        if not self.token:
            msg = 'An invalid Access Token ({}) was specified.'.format(token)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.include_image = include_image
        self.targets = parse_list(targets)
        self.__cached_users = dict()
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Send our Line Notification\n        '
        if len(self.targets) == 0:
            self.logger.warning('There were no Line targets to notify.')
            return False
        has_error = False
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json', 'Authorization': 'Bearer {}'.format(self.token)}
        payload = {'to': None, 'messages': [{'type': 'text', 'text': body, 'sender': {'name': self.app_id}}]}
        image_url = None if not self.include_image else self.image_url(notify_type)
        if image_url:
            payload['messages'][0]['sender']['iconUrl'] = image_url
        targets = list(self.targets)
        while len(targets):
            target = targets.pop(0)
            payload['to'] = target
            self.logger.debug('Line POST URL: %s (cert_verify=%r)' % (self.notify_url, self.verify_certificate))
            self.logger.debug('Line Payload: %s' % str(payload))
            self.throttle()
            try:
                r = requests.post(self.notify_url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code != requests.codes.ok:
                    status_str = NotifyLine.http_response_code_lookup(r.status_code)
                    self.logger.warning('Failed to send Line notification to {}: {}{}error={}.'.format(target, status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                else:
                    self.logger.info('Sent Line notification to {}.'.format(target))
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending Line notification to {}.'.format(target))
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
                continue
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            return 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'image': 'yes' if self.include_image else 'no'}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        return '{schema}://{token}/{targets}?{params}'.format(schema=self.secure_protocol, token=self.pprint(self.token, privacy, mode=PrivacyMode.Secret, safe=''), targets='/'.join([self.pprint(x, privacy, safe='') for x in self.targets]), params=NotifyLine.urlencode(params))

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the number of targets associated with this notification\n        '
        return len(self.targets)

    @staticmethod
    def parse_url(url):
        if False:
            return 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['targets'] = NotifyLine.split_path(results['fullpath'])
        if 'token' in results['qsd'] and len(results['qsd']['token']):
            results['token'] = NotifyLine.unquote(results['qsd']['token'])
        else:
            results['token'] = NotifyLine.unquote(results['host'])
            if not results['token'].endswith('='):
                for (index, entry) in enumerate(list(results['targets']), start=1):
                    if entry.endswith('='):
                        results['token'] += '/' + '/'.join(results['targets'][0:index])
                        results['targets'] = results['targets'][index:]
                        break
        results['include_image'] = parse_bool(results['qsd'].get('image', True))
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += [x for x in filter(bool, TARGET_LIST_DELIM.split(NotifyLine.unquote(results['qsd']['to'])))]
        return results