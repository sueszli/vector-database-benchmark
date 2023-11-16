import re
import requests
from itertools import chain
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..common import NotifyFormat
from ..conversion import convert_between
from ..utils import parse_list
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
from ..attachment.AttachBase import AttachBase
PUSHOVER_SEND_TO_ALL = 'ALL_DEVICES'
VALIDATE_DEVICE = re.compile('^\\s*(?P<device>[a-z0-9_-]{1,25})\\s*$', re.I)

class PushoverPriority:
    LOW = -2
    MODERATE = -1
    NORMAL = 0
    HIGH = 1
    EMERGENCY = 2

class PushoverSound:
    PUSHOVER = 'pushover'
    BIKE = 'bike'
    BUGLE = 'bugle'
    CASHREGISTER = 'cashregister'
    CLASSICAL = 'classical'
    COSMIC = 'cosmic'
    FALLING = 'falling'
    GAMELAN = 'gamelan'
    INCOMING = 'incoming'
    INTERMISSION = 'intermission'
    MAGIC = 'magic'
    MECHANICAL = 'mechanical'
    PIANOBAR = 'pianobar'
    SIREN = 'siren'
    SPACEALARM = 'spacealarm'
    TUGBOAT = 'tugboat'
    ALIEN = 'alien'
    CLIMB = 'climb'
    PERSISTENT = 'persistent'
    ECHO = 'echo'
    UPDOWN = 'updown'
    NONE = 'none'
PUSHOVER_SOUNDS = (PushoverSound.PUSHOVER, PushoverSound.BIKE, PushoverSound.BUGLE, PushoverSound.CASHREGISTER, PushoverSound.CLASSICAL, PushoverSound.COSMIC, PushoverSound.FALLING, PushoverSound.GAMELAN, PushoverSound.INCOMING, PushoverSound.INTERMISSION, PushoverSound.MAGIC, PushoverSound.MECHANICAL, PushoverSound.PIANOBAR, PushoverSound.SIREN, PushoverSound.SPACEALARM, PushoverSound.TUGBOAT, PushoverSound.ALIEN, PushoverSound.CLIMB, PushoverSound.PERSISTENT, PushoverSound.ECHO, PushoverSound.UPDOWN, PushoverSound.NONE)
PUSHOVER_PRIORITIES = {PushoverPriority.LOW: 'low', PushoverPriority.MODERATE: 'moderate', PushoverPriority.NORMAL: 'normal', PushoverPriority.HIGH: 'high', PushoverPriority.EMERGENCY: 'emergency'}
PUSHOVER_PRIORITY_MAP = {'l': PushoverPriority.LOW, 'm': PushoverPriority.MODERATE, 'n': PushoverPriority.NORMAL, 'h': PushoverPriority.HIGH, 'e': PushoverPriority.EMERGENCY, '-2': PushoverPriority.LOW, '-1': PushoverPriority.MODERATE, '0': PushoverPriority.NORMAL, '1': PushoverPriority.HIGH, '2': PushoverPriority.EMERGENCY}
PUSHOVER_HTTP_ERROR_MAP = {401: 'Unauthorized - Invalid Token.'}

class NotifyPushover(NotifyBase):
    """
    A wrapper for Pushover Notifications
    """
    service_name = 'Pushover'
    service_url = 'https://pushover.net/'
    secure_protocol = 'pover'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_pushover'
    notify_url = 'https://api.pushover.net/1/messages.json'
    attachment_support = True
    body_maxlen = 1024
    default_pushover_sound = PushoverSound.PUSHOVER
    attach_max_size_bytes = 2621440
    attach_supported_mime_type = '^image/.*'
    templates = ('{schema}://{user_key}@{token}', '{schema}://{user_key}@{token}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'user_key': {'name': _('User Key'), 'type': 'string', 'private': True, 'required': True}, 'token': {'name': _('Access Token'), 'type': 'string', 'private': True, 'required': True}, 'target_device': {'name': _('Target Device'), 'type': 'string', 'regex': ('^[a-z0-9_-]{1,25}$', 'i'), 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'priority': {'name': _('Priority'), 'type': 'choice:int', 'values': PUSHOVER_PRIORITIES, 'default': PushoverPriority.NORMAL}, 'sound': {'name': _('Sound'), 'type': 'string', 'regex': ('^[a-z]{1,12}$', 'i'), 'default': PushoverSound.PUSHOVER}, 'url': {'name': _('URL'), 'map_to': 'supplemental_url', 'type': 'string'}, 'url_title': {'name': _('URL Title'), 'map_to': 'supplemental_url_title', 'type': 'string'}, 'retry': {'name': _('Retry'), 'type': 'int', 'min': 30, 'default': 900}, 'expire': {'name': _('Expire'), 'type': 'int', 'min': 0, 'max': 10800, 'default': 3600}, 'to': {'alias_of': 'targets'}})

    def __init__(self, user_key, token, targets=None, priority=None, sound=None, retry=None, expire=None, supplemental_url=None, supplemental_url_title=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Initialize Pushover Object\n        '
        super().__init__(**kwargs)
        self.token = validate_regex(token)
        if not self.token:
            msg = 'An invalid Pushover Access Token ({}) was specified.'.format(token)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.user_key = validate_regex(user_key)
        if not self.user_key:
            msg = 'An invalid Pushover User Key ({}) was specified.'.format(user_key)
            self.logger.warning(msg)
            raise TypeError(msg)
        targets = parse_list(targets)
        self.invalid_targets = list()
        if len(targets) == 0:
            self.targets = (PUSHOVER_SEND_TO_ALL,)
        else:
            self.targets = []
            for target in targets:
                result = VALIDATE_DEVICE.match(target)
                if result:
                    self.targets.append(result.group('device'))
                    continue
                self.logger.warning('Dropped invalid Pushover device ({}) specified.'.format(target))
                self.invalid_targets.append(target)
        self.supplemental_url = supplemental_url
        self.supplemental_url_title = supplemental_url_title
        self.sound = NotifyPushover.default_pushover_sound if not isinstance(sound, str) else sound.lower()
        if self.sound and self.sound not in PUSHOVER_SOUNDS:
            msg = 'Using custom sound specified ({}). '.format(sound)
            self.logger.debug(msg)
        self.priority = int(NotifyPushover.template_args['priority']['default'] if priority is None else next((v for (k, v) in PUSHOVER_PRIORITY_MAP.items() if str(priority).lower().startswith(k)), NotifyPushover.template_args['priority']['default']))
        if self.priority == PushoverPriority.EMERGENCY:
            self.retry = self.template_args['retry']['default']
            try:
                self.retry = int(retry)
            except (ValueError, TypeError):
                pass
            self.expire = self.template_args['expire']['default']
            try:
                self.expire = int(expire)
            except (ValueError, TypeError):
                pass
            if self.retry < 30:
                msg = 'Pushover retry must be at least 30 seconds.'
                self.logger.warning(msg)
                raise TypeError(msg)
            if self.expire < 0 or self.expire > 10800:
                msg = 'Pushover expire must reside in the range of 0 to 10800 seconds.'
                self.logger.warning(msg)
                raise TypeError(msg)
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, attach=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Perform Pushover Notification\n        '
        if not self.targets:
            self.logger.warning('There were no Pushover targets to notify.')
            return False
        payload = {'token': self.token, 'user': self.user_key, 'priority': str(self.priority), 'title': title if title else self.app_desc, 'message': body, 'device': ','.join(self.targets), 'sound': self.sound}
        if self.supplemental_url:
            payload['url'] = self.supplemental_url
        if self.supplemental_url_title:
            payload['url_title'] = self.supplemental_url_title
        if self.notify_format == NotifyFormat.HTML:
            payload['html'] = 1
        elif self.notify_format == NotifyFormat.MARKDOWN:
            payload['message'] = convert_between(NotifyFormat.MARKDOWN, NotifyFormat.HTML, body)
            payload['html'] = 1
        if self.priority == PushoverPriority.EMERGENCY:
            payload.update({'retry': self.retry, 'expire': self.expire})
        if attach and self.attachment_support:
            _payload = payload.copy()
            for (no, attachment) in enumerate(attach):
                if no or not body:
                    _payload['message'] = attachment.name
                if not self._send(_payload, attachment):
                    return False
                _payload['title'] = ''
                _payload['sound'] = PushoverSound.NONE
        else:
            return self._send(payload)
        return True

    def _send(self, payload, attach=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Wrapper to the requests (post) object\n        '
        if isinstance(attach, AttachBase):
            if not attach:
                self.logger.error('Could not access attachment {}.'.format(attach.url(privacy=True)))
                return False
            if not re.match(self.attach_supported_mime_type, attach.mimetype, re.I):
                self.logger.debug('Ignored unsupported Pushover attachment ({}): {}'.format(attach.mimetype, attach.url(privacy=True)))
                attach = None
            else:
                file_size = len(attach)
                if not (file_size > 0 and file_size <= self.attach_max_size_bytes):
                    self.logger.warning('Pushover attachment size ({}B) exceeds limit: {}'.format(file_size, attach.url(privacy=True)))
                    return False
                self.logger.debug('Posting Pushover attachment {}'.format(attach.url(privacy=True)))
        headers = {'User-Agent': self.app_id}
        auth = (self.token, '')
        files = None
        self.logger.debug('Pushover POST URL: %s (cert_verify=%r)' % (self.notify_url, self.verify_certificate))
        self.logger.debug('Pushover Payload: %s' % str(payload))
        self.throttle()
        try:
            if attach:
                files = {'attachment': (attach.name, open(attach.path, 'rb'))}
            r = requests.post(self.notify_url, data=payload, headers=headers, files=files, auth=auth, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyPushover.http_response_code_lookup(r.status_code, PUSHOVER_HTTP_ERROR_MAP)
                self.logger.warning('Failed to send Pushover notification to {}: {}{}error={}.'.format(payload['device'], status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent Pushover notification to %s.' % payload['device'])
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending Pushover:%s ' % payload['device'] + 'notification.')
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        except (OSError, IOError) as e:
            self.logger.warning('An I/O error occurred while reading {}.'.format(attach.name if attach else 'attachment'))
            self.logger.debug('I/O Exception: %s' % str(e))
            return False
        finally:
            if files:
                files['attachment'][1].close()
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'priority': PUSHOVER_PRIORITIES[self.template_args['priority']['default']] if self.priority not in PUSHOVER_PRIORITIES else PUSHOVER_PRIORITIES[self.priority]}
        if self.priority == PushoverPriority.EMERGENCY:
            params.update({'expire': self.expire, 'retry': self.retry})
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        devices = '/'.join([NotifyPushover.quote(x, safe='') for x in chain(self.targets, self.invalid_targets)])
        if devices == PUSHOVER_SEND_TO_ALL:
            devices = ''
        return '{schema}://{user_key}@{token}/{devices}/?{params}'.format(schema=self.secure_protocol, user_key=self.pprint(self.user_key, privacy, safe=''), token=self.pprint(self.token, privacy, safe=''), devices=devices, params=NotifyPushover.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            i = 10
            return i + 15
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        if 'priority' in results['qsd'] and len(results['qsd']['priority']):
            results['priority'] = NotifyPushover.unquote(results['qsd']['priority'])
        results['targets'] = NotifyPushover.split_path(results['fullpath'])
        results['user_key'] = NotifyPushover.unquote(results['user'])
        if 'sound' in results['qsd'] and len(results['qsd']['sound']):
            results['sound'] = NotifyPushover.unquote(results['qsd']['sound'])
        if 'url' in results['qsd'] and len(results['qsd']['url']):
            results['supplemental_url'] = NotifyPushover.unquote(results['qsd']['url'])
        if 'url_title' in results['qsd'] and len(results['qsd']['url_title']):
            results['supplemental_url_title'] = results['qsd']['url_title']
        if 'expire' in results['qsd'] and len(results['qsd']['expire']):
            results['expire'] = results['qsd']['expire']
        if 'retry' in results['qsd'] and len(results['qsd']['retry']):
            results['retry'] = results['qsd']['retry']
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyPushover.parse_list(results['qsd']['to'])
        results['token'] = NotifyPushover.unquote(results['host'])
        return results