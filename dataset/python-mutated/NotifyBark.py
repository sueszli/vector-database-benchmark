import requests
import json
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyImageSize
from ..common import NotifyType
from ..utils import parse_list
from ..utils import parse_bool
from ..AppriseLocale import gettext_lazy as _
BARK_SOUNDS = ('alarm.caf', 'anticipate.caf', 'bell.caf', 'birdsong.caf', 'bloom.caf', 'calypso.caf', 'chime.caf', 'choo.caf', 'descent.caf', 'electronic.caf', 'fanfare.caf', 'glass.caf', 'gotosleep.caf', 'healthnotification.caf', 'horn.caf', 'ladder.caf', 'mailsent.caf', 'minuet.caf', 'multiwayinvitation.caf', 'newmail.caf', 'newsflash.caf', 'noir.caf', 'paymentsuccess.caf', 'shake.caf', 'sherwoodforest.caf', 'silence.caf', 'spell.caf', 'suspense.caf', 'telegraph.caf', 'tiptoes.caf', 'typewriters.caf', 'update.caf')

class NotifyBarkLevel:
    """
    Defines the Bark Level options
    """
    ACTIVE = 'active'
    TIME_SENSITIVE = 'timeSensitive'
    PASSIVE = 'passive'
BARK_LEVELS = (NotifyBarkLevel.ACTIVE, NotifyBarkLevel.TIME_SENSITIVE, NotifyBarkLevel.PASSIVE)

class NotifyBark(NotifyBase):
    """
    A wrapper for Notify Bark Notifications
    """
    service_name = 'Bark'
    service_url = 'https://github.com/Finb/Bark'
    protocol = 'bark'
    secure_protocol = 'barks'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_bark'
    image_size = NotifyImageSize.XY_128
    templates = ('{schema}://{host}/{targets}', '{schema}://{host}:{port}/{targets}', '{schema}://{user}:{password}@{host}/{targets}', '{schema}://{user}:{password}@{host}:{port}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'host': {'name': _('Hostname'), 'type': 'string', 'required': True}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}, 'user': {'name': _('Username'), 'type': 'string'}, 'password': {'name': _('Password'), 'type': 'string', 'private': True}, 'target_device': {'name': _('Target Device'), 'type': 'string', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string', 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}, 'sound': {'name': _('Sound'), 'type': 'choice:string', 'values': BARK_SOUNDS}, 'level': {'name': _('Level'), 'type': 'choice:string', 'values': BARK_LEVELS}, 'click': {'name': _('Click'), 'type': 'string'}, 'badge': {'name': _('Badge'), 'type': 'int', 'min': 0}, 'category': {'name': _('Category'), 'type': 'string'}, 'group': {'name': _('Group'), 'type': 'string'}, 'image': {'name': _('Include Image'), 'type': 'bool', 'default': True, 'map_to': 'include_image'}})

    def __init__(self, targets=None, include_image=True, sound=None, category=None, group=None, level=None, click=None, badge=None, **kwargs):
        if False:
            return 10
        '\n        Initialize Notify Bark Object\n        '
        super().__init__(**kwargs)
        self.notify_url = '%s://%s%s/push' % ('https' if self.secure else 'http', self.host, ':{}'.format(self.port) if self.port and isinstance(self.port, int) else '')
        self.category = category if isinstance(category, str) else None
        self.group = group if isinstance(group, str) else None
        self.targets = parse_list(targets)
        self.include_image = include_image
        self.click = click
        try:
            self.badge = int(badge)
            if self.badge < 0:
                raise ValueError()
        except TypeError:
            self.badge = None
        except ValueError:
            self.badge = None
            self.logger.warning('The specified Bark badge ({}) is not valid ', badge)
        self.sound = None if not sound else next((f for f in BARK_SOUNDS if f.startswith(sound.lower())), None)
        if sound and (not self.sound):
            self.logger.warning('The specified Bark sound ({}) was not found ', sound)
        self.level = None if not level else next((f for f in BARK_LEVELS if f[0] == level[0]), None)
        if level and (not self.level):
            self.logger.warning('The specified Bark level ({}) is not valid ', level)
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform Bark Notification\n        '
        has_error = False
        if not self.targets:
            self.logger.warning('There are no Bark devices to notify')
            return False
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json; charset=utf-8'}
        payload = {'title': title if title else self.app_desc, 'body': body}
        image_url = None if not self.include_image else self.image_url(notify_type)
        if image_url:
            payload['icon'] = image_url
        if self.sound:
            payload['sound'] = self.sound
        if self.click:
            payload['url'] = self.click
        if self.badge:
            payload['badge'] = self.badge
        if self.level:
            payload['level'] = self.level
        if self.category:
            payload['category'] = self.category
        if self.group:
            payload['group'] = self.group
        auth = None
        if self.user:
            auth = (self.user, self.password)
        targets = list(self.targets)
        while len(targets) > 0:
            target = targets.pop()
            payload['device_key'] = target
            self.logger.debug('Bark POST URL: %s (cert_verify=%r)' % (self.notify_url, self.verify_certificate))
            self.logger.debug('Bark Payload: %s' % str(payload))
            self.throttle()
            try:
                r = requests.post(self.notify_url, data=json.dumps(payload), headers=headers, auth=auth, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code != requests.codes.ok:
                    status_str = NotifyBark.http_response_code_lookup(r.status_code)
                    self.logger.warning('Failed to send Bark notification to {}: {}{}error={}.'.format(target, status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                else:
                    self.logger.info('Sent Bark notification to {}.'.format(target))
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending Bark notification to {}.'.format(target))
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
                continue
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'image': 'yes' if self.include_image else 'no'}
        if self.sound:
            params['sound'] = self.sound
        if self.click:
            params['click'] = self.click
        if self.badge:
            params['badge'] = str(self.badge)
        if self.level:
            params['level'] = self.level
        if self.category:
            params['category'] = self.category
        if self.group:
            params['group'] = self.group
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        auth = ''
        if self.user and self.password:
            auth = '{user}:{password}@'.format(user=NotifyBark.quote(self.user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
        elif self.user:
            auth = '{user}@'.format(user=NotifyBark.quote(self.user, safe=''))
        default_port = 443 if self.secure else 80
        return '{schema}://{auth}{hostname}{port}/{targets}?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, auth=auth, hostname=self.host, port='' if self.port is None or self.port == default_port else ':{}'.format(self.port), targets='/'.join([NotifyBark.quote('{}'.format(x)) for x in self.targets]), params=NotifyBark.urlencode(params))

    def __len__(self):
        if False:
            print('Hello World!')
        '\n        Returns the number of targets associated with this notification\n        '
        return len(self.targets)

    @staticmethod
    def parse_url(url):
        if False:
            return 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url)
        if not results:
            return results
        results['targets'] = NotifyBark.split_path(results['fullpath'])
        if 'category' in results['qsd'] and results['qsd']['category']:
            results['category'] = NotifyBark.unquote(results['qsd']['category'].strip())
        if 'group' in results['qsd'] and results['qsd']['group']:
            results['group'] = NotifyBark.unquote(results['qsd']['group'].strip())
        if 'badge' in results['qsd'] and results['qsd']['badge']:
            results['badge'] = NotifyBark.unquote(results['qsd']['badge'].strip())
        if 'level' in results['qsd'] and results['qsd']['level']:
            results['level'] = NotifyBark.unquote(results['qsd']['level'].strip())
        if 'click' in results['qsd'] and results['qsd']['click']:
            results['click'] = NotifyBark.unquote(results['qsd']['click'].strip())
        if 'sound' in results['qsd'] and results['qsd']['sound']:
            results['sound'] = NotifyBark.unquote(results['qsd']['sound'].strip())
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyBark.parse_list(results['qsd']['to'])
        results['include_image'] = parse_bool(results['qsd'].get('image', True))
        return results