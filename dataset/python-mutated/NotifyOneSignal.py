import requests
from json import dumps
from itertools import chain
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..common import NotifyImageSize
from ..utils import validate_regex
from ..utils import parse_list
from ..utils import parse_bool
from ..utils import is_email
from ..AppriseLocale import gettext_lazy as _

class OneSignalCategory:
    """
    We define the different category types that we can notify via OneSignal
    """
    PLAYER = 'include_player_ids'
    EMAIL = 'include_email_tokens'
    USER = 'include_external_user_ids'
    SEGMENT = 'included_segments'
ONESIGNAL_CATEGORIES = (OneSignalCategory.PLAYER, OneSignalCategory.EMAIL, OneSignalCategory.USER, OneSignalCategory.SEGMENT)

class NotifyOneSignal(NotifyBase):
    """
    A wrapper for OneSignal Notifications
    """
    service_name = 'OneSignal'
    service_url = 'https://onesignal.com'
    secure_protocol = 'onesignal'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_onesignal'
    notify_url = 'https://onesignal.com/api/v1/notifications'
    image_size = NotifyImageSize.XY_72
    default_batch_size = 2000
    templates = ('{schema}://{app}@{apikey}/{targets}', '{schema}://{template}:{app}@{apikey}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'app': {'name': _('App ID'), 'type': 'string', 'private': True, 'required': True}, 'template': {'name': _('Template'), 'type': 'string', 'private': True}, 'apikey': {'name': _('API Key'), 'type': 'string', 'private': True, 'required': True}, 'target_player': {'name': _('Target Player ID'), 'type': 'string', 'map_to': 'targets'}, 'target_email': {'name': _('Target Email'), 'type': 'string', 'map_to': 'targets'}, 'target_user': {'name': _('Target User'), 'type': 'string', 'prefix': '@', 'map_to': 'targets'}, 'target_segment': {'name': _('Include Segment'), 'type': 'string', 'prefix': '#', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string', 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}, 'image': {'name': _('Include Image'), 'type': 'bool', 'default': True, 'map_to': 'include_image'}, 'batch': {'name': _('Batch Mode'), 'type': 'bool', 'default': False}, 'template': {'alias_of': 'template'}, 'subtitle': {'name': _('Subtitle'), 'type': 'string'}, 'language': {'name': _('Language'), 'type': 'string', 'default': 'en'}})

    def __init__(self, app, apikey, targets=None, include_image=True, template=None, subtitle=None, language=None, batch=False, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Initialize OneSignal\n\n        '
        super().__init__(**kwargs)
        self.apikey = validate_regex(apikey)
        if not self.apikey:
            msg = 'An invalid OneSignal API key ({}) was specified.'.format(apikey)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.app = validate_regex(app)
        if not self.app:
            msg = 'An invalid OneSignal Application ID ({}) was specified.'.format(app)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.batch_size = self.default_batch_size if batch else 1
        self.include_image = include_image
        self.targets = {OneSignalCategory.PLAYER: [], OneSignalCategory.EMAIL: [], OneSignalCategory.USER: [], OneSignalCategory.SEGMENT: []}
        self.template_id = template
        self.subtitle = subtitle
        self.language = language.strip().lower()[0:2] if language else NotifyOneSignal.template_args['language']['default']
        if not self.language or len(self.language) != 2:
            msg = 'An invalid OneSignal Language ({}) was specified.'.format(language)
            self.logger.warning(msg)
            raise TypeError(msg)
        for _target in parse_list(targets):
            target = _target.strip()
            if len(target) < 2:
                self.logger.debug('Ignoring OneSignal Entry: %s' % target)
                continue
            if target.startswith(NotifyOneSignal.template_tokens['target_user']['prefix']):
                self.targets[OneSignalCategory.USER].append(target)
                self.logger.debug('Detected OneSignal UserID: %s' % self.targets[OneSignalCategory.USER][-1])
                continue
            if target.startswith(NotifyOneSignal.template_tokens['target_segment']['prefix']):
                self.targets[OneSignalCategory.SEGMENT].append(target)
                self.logger.debug('Detected OneSignal Include Segment: %s' % self.targets[OneSignalCategory.SEGMENT][-1])
                continue
            result = is_email(target)
            if result:
                self.targets[OneSignalCategory.EMAIL].append(result['full_email'])
                self.logger.debug('Detected OneSignal Email: %s' % self.targets[OneSignalCategory.EMAIL][-1])
            else:
                self.targets[OneSignalCategory.PLAYER].append(target)
                self.logger.debug('Detected OneSignal Player ID: %s' % self.targets[OneSignalCategory.PLAYER][-1])
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Perform OneSignal Notification\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json; charset=utf-8', 'Authorization': 'Basic {}'.format(self.apikey)}
        has_error = False
        sent_count = 0
        payload = {'app_id': self.app, 'headings': {self.language: title if title else self.app_desc}, 'contents': {self.language: body}, 'content_available': True}
        if self.subtitle:
            payload.update({'subtitle': {self.language: self.subtitle}})
        if self.template_id:
            payload['template_id'] = self.template_id
        image_url = None if not self.include_image else self.image_url(notify_type)
        if image_url:
            payload['large_icon'] = image_url
        image_url = None if not self.include_image else self.image_url(notify_type, image_size=NotifyImageSize.XY_32)
        if image_url:
            payload['small_icon'] = image_url
        for category in ONESIGNAL_CATEGORIES:
            targets = self.targets[category]
            for index in range(0, len(targets), self.batch_size):
                payload[category] = targets[index:index + self.batch_size]
                sent_count += len(payload[category])
                self.logger.debug('OneSignal POST URL: %s (cert_verify=%r)' % (self.notify_url, self.verify_certificate))
                self.logger.debug('OneSignal Payload: %s' % str(payload))
                self.throttle()
                try:
                    r = requests.post(self.notify_url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                    if r.status_code not in (requests.codes.ok, requests.codes.no_content):
                        status_str = NotifyOneSignal.http_response_code_lookup(r.status_code)
                        self.logger.warning('Failed to send OneSignal notification: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                        self.logger.debug('Response Details:\r\n%s', r.content)
                        has_error = True
                    else:
                        self.logger.info('Sent OneSignal notification.')
                except requests.RequestException as e:
                    self.logger.warning('A Connection error occurred sending OneSignal notification.')
                    self.logger.debug('Socket Exception: %s', str(e))
                    has_error = True
        if not sent_count:
            self.logger.warning('There are no OneSignal targets to notify')
            return False
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'image': 'yes' if self.include_image else 'no', 'batch': 'yes' if self.batch_size > 1 else 'no'}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        return '{schema}://{tp_id}{app}@{apikey}/{targets}?{params}'.format(schema=self.secure_protocol, tp_id='{}:'.format(self.pprint(self.template_id, privacy, safe='')) if self.template_id else '', app=self.pprint(self.app, privacy, safe=''), apikey=self.pprint(self.apikey, privacy, safe=''), targets='/'.join(chain([NotifyOneSignal.quote(x) for x in self.targets[OneSignalCategory.PLAYER]], [NotifyOneSignal.quote(x) for x in self.targets[OneSignalCategory.EMAIL]], [NotifyOneSignal.quote('{}{}'.format(NotifyOneSignal.template_tokens['target_user']['prefix'], x), safe='') for x in self.targets[OneSignalCategory.USER]], [NotifyOneSignal.quote('{}{}'.format(NotifyOneSignal.template_tokens['target_segment']['prefix'], x), safe='') for x in self.targets[OneSignalCategory.SEGMENT]])), params=NotifyOneSignal.urlencode(params))

    def __len__(self):
        if False:
            return 10
        '\n        Returns the number of targets associated with this notification\n        '
        if self.batch_size > 1:
            total_targets = 0
            for (k, m) in self.targets.items():
                targets = len(m)
                total_targets += int(targets / self.batch_size) + (1 if targets % self.batch_size else 0)
            return total_targets
        return sum([len(m) for (_, m) in self.targets.items()])

    @staticmethod
    def parse_url(url):
        if False:
            i = 10
            return i + 15
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        if not results.get('password'):
            results['app'] = NotifyOneSignal.unquote(results['user'])
        else:
            results['app'] = NotifyOneSignal.unquote(results['password'])
            results['template'] = NotifyOneSignal.unquote(results['user'])
        results['include_image'] = parse_bool(results['qsd'].get('image', NotifyOneSignal.template_args['image']['default']))
        results['batch'] = parse_bool(results['qsd'].get('batch', NotifyOneSignal.template_args['batch']['default']))
        results['apikey'] = NotifyOneSignal.unquote(results['host'])
        results['targets'] = NotifyOneSignal.split_path(results['fullpath'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyOneSignal.parse_list(results['qsd']['to'])
        if 'app' in results['qsd'] and len(results['qsd']['app']):
            results['app'] = NotifyOneSignal.unquote(results['qsd']['app'])
        if 'apikey' in results['qsd'] and len(results['qsd']['apikey']):
            results['apikey'] = NotifyOneSignal.unquote(results['qsd']['apikey'])
        if 'template' in results['qsd'] and len(results['qsd']['template']):
            results['template'] = NotifyOneSignal.unquote(results['qsd']['template'])
        if 'subtitle' in results['qsd'] and len(results['qsd']['subtitle']):
            results['subtitle'] = NotifyOneSignal.unquote(results['qsd']['subtitle'])
        if 'lang' in results['qsd'] and len(results['qsd']['lang']):
            results['language'] = NotifyOneSignal.unquote(results['qsd']['lang'])
        return results