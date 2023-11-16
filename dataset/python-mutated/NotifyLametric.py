import re
import requests
from json import dumps
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
from ..utils import is_hostname
from ..utils import is_ipaddr
LAMETRIC_APP_ID_DETECTOR_RE = re.compile('(com\\.lametric\\.)?(?P<app_id>[0-9a-z.-]{1,64})(/(?P<app_ver>[1-9][0-9]*))?', re.I)
LAMETRIC_IS_APP_TOKEN = re.compile('^[a-z0-9]{80,}==$', re.I)

class LametricMode:
    """
    Define Lametric Notification Modes
    """
    CLOUD = 'cloud'
    DEVICE = 'device'
LAMETRIC_MODES = (LametricMode.CLOUD, LametricMode.DEVICE)

class LametricPriority:
    """
    Priority of the message
    """
    INFO = 'info'
    WARNING = 'warning'
    CRITICAL = 'critical'
LAMETRIC_PRIORITIES = (LametricPriority.INFO, LametricPriority.WARNING, LametricPriority.CRITICAL)

class LametricIconType:
    """
    Represents the nature of notification.
    """
    INFO = 'info'
    ALERT = 'alert'
    NONE = 'none'
LAMETRIC_ICON_TYPES = (LametricIconType.INFO, LametricIconType.ALERT, LametricIconType.NONE)

class LametricSoundCategory:
    """
    Define Sound Categories
    """
    NOTIFICATIONS = 'notifications'
    ALARMS = 'alarms'

class LametricSound:
    """
    There are 2 categories of sounds, to make things simple we just lump them
    all togther in one class object.

    Syntax is (Category, (AlarmID, Alias1, Alias2, ...))
    """
    ALARM01 = (LametricSoundCategory.ALARMS, ('alarm1', 'a1', 'a01'))
    ALARM02 = (LametricSoundCategory.ALARMS, ('alarm2', 'a2', 'a02'))
    ALARM03 = (LametricSoundCategory.ALARMS, ('alarm3', 'a3', 'a03'))
    ALARM04 = (LametricSoundCategory.ALARMS, ('alarm4', 'a4', 'a04'))
    ALARM05 = (LametricSoundCategory.ALARMS, ('alarm5', 'a5', 'a05'))
    ALARM06 = (LametricSoundCategory.ALARMS, ('alarm6', 'a6', 'a06'))
    ALARM07 = (LametricSoundCategory.ALARMS, ('alarm7', 'a7', 'a07'))
    ALARM08 = (LametricSoundCategory.ALARMS, ('alarm8', 'a8', 'a08'))
    ALARM09 = (LametricSoundCategory.ALARMS, ('alarm9', 'a9', 'a09'))
    ALARM10 = (LametricSoundCategory.ALARMS, ('alarm10', 'a10'))
    ALARM11 = (LametricSoundCategory.ALARMS, ('alarm11', 'a11'))
    ALARM12 = (LametricSoundCategory.ALARMS, ('alarm12', 'a12'))
    ALARM13 = (LametricSoundCategory.ALARMS, ('alarm13', 'a13'))
    BICYCLE = (LametricSoundCategory.NOTIFICATIONS, ('bicycle', 'bike'))
    CAR = (LametricSoundCategory.NOTIFICATIONS, ('car',))
    CASH = (LametricSoundCategory.NOTIFICATIONS, ('cash',))
    CAT = (LametricSoundCategory.NOTIFICATIONS, ('cat',))
    DOG01 = (LametricSoundCategory.NOTIFICATIONS, ('dog', 'dog1', 'dog01'))
    DOG02 = (LametricSoundCategory.NOTIFICATIONS, ('dog2', 'dog02'))
    ENERGY = (LametricSoundCategory.NOTIFICATIONS, ('energy',))
    KNOCK = (LametricSoundCategory.NOTIFICATIONS, ('knock-knock', 'knock'))
    EMAIL = (LametricSoundCategory.NOTIFICATIONS, ('letter_email', 'letter', 'email'))
    LOSE01 = (LametricSoundCategory.NOTIFICATIONS, ('lose1', 'lose01', 'lose'))
    LOSE02 = (LametricSoundCategory.NOTIFICATIONS, ('lose2', 'lose02'))
    NEGATIVE01 = (LametricSoundCategory.NOTIFICATIONS, ('negative1', 'negative01', 'neg01', 'neg1', '-'))
    NEGATIVE02 = (LametricSoundCategory.NOTIFICATIONS, ('negative2', 'negative02', 'neg02', 'neg2', '--'))
    NEGATIVE03 = (LametricSoundCategory.NOTIFICATIONS, ('negative3', 'negative03', 'neg03', 'neg3', '---'))
    NEGATIVE04 = (LametricSoundCategory.NOTIFICATIONS, ('negative4', 'negative04', 'neg04', 'neg4', '----'))
    NEGATIVE05 = (LametricSoundCategory.NOTIFICATIONS, ('negative5', 'negative05', 'neg05', 'neg5', '-----'))
    NOTIFICATION01 = (LametricSoundCategory.NOTIFICATIONS, ('notification', 'notification1', 'notification01', 'not01', 'not1'))
    NOTIFICATION02 = (LametricSoundCategory.NOTIFICATIONS, ('notification2', 'notification02', 'not02', 'not2'))
    NOTIFICATION03 = (LametricSoundCategory.NOTIFICATIONS, ('notification3', 'notification03', 'not03', 'not3'))
    NOTIFICATION04 = (LametricSoundCategory.NOTIFICATIONS, ('notification4', 'notification04', 'not04', 'not4'))
    OPEN_DOOR = (LametricSoundCategory.NOTIFICATIONS, ('open_door', 'open', 'door'))
    POSITIVE01 = (LametricSoundCategory.NOTIFICATIONS, ('positive1', 'positive01', 'pos01', 'p1', '+'))
    POSITIVE02 = (LametricSoundCategory.NOTIFICATIONS, ('positive2', 'positive02', 'pos02', 'p2', '++'))
    POSITIVE03 = (LametricSoundCategory.NOTIFICATIONS, ('positive3', 'positive03', 'pos03', 'p3', '+++'))
    POSITIVE04 = (LametricSoundCategory.NOTIFICATIONS, ('positive4', 'positive04', 'pos04', 'p4', '++++'))
    POSITIVE05 = (LametricSoundCategory.NOTIFICATIONS, ('positive5', 'positive05', 'pos05', 'p5', '+++++'))
    POSITIVE06 = (LametricSoundCategory.NOTIFICATIONS, ('positive6', 'positive06', 'pos06', 'p6', '++++++'))
    STATISTIC = (LametricSoundCategory.NOTIFICATIONS, ('statistic', 'stat'))
    THUNDER = (LametricSoundCategory.NOTIFICATIONS, 'thunder')
    WATER01 = (LametricSoundCategory.NOTIFICATIONS, ('water1', 'water01'))
    WATER02 = (LametricSoundCategory.NOTIFICATIONS, ('water2', 'water02'))
    WIN01 = (LametricSoundCategory.NOTIFICATIONS, ('win', 'win01', 'win1'))
    WIN02 = (LametricSoundCategory.NOTIFICATIONS, ('win2', 'win02'))
    WIND = (LametricSoundCategory.NOTIFICATIONS, ('wind',))
    WIND_SHORT = (LametricSoundCategory.NOTIFICATIONS, ('wind_short',))
LAMETRIC_SOUNDS = (LametricSound.ALARM13, LametricSound.ALARM12, LametricSound.ALARM11, LametricSound.ALARM10, LametricSound.ALARM09, LametricSound.ALARM08, LametricSound.ALARM07, LametricSound.ALARM06, LametricSound.ALARM05, LametricSound.ALARM04, LametricSound.ALARM03, LametricSound.ALARM02, LametricSound.ALARM01, LametricSound.BICYCLE, LametricSound.CAR, LametricSound.CASH, LametricSound.CAT, LametricSound.DOG02, LametricSound.DOG01, LametricSound.ENERGY, LametricSound.KNOCK, LametricSound.EMAIL, LametricSound.LOSE02, LametricSound.LOSE01, LametricSound.NEGATIVE01, LametricSound.NEGATIVE02, LametricSound.NEGATIVE03, LametricSound.NEGATIVE04, LametricSound.NEGATIVE05, LametricSound.NOTIFICATION04, LametricSound.NOTIFICATION03, LametricSound.NOTIFICATION02, LametricSound.NOTIFICATION01, LametricSound.OPEN_DOOR, LametricSound.POSITIVE01, LametricSound.POSITIVE02, LametricSound.POSITIVE03, LametricSound.POSITIVE04, LametricSound.POSITIVE05, LametricSound.POSITIVE01, LametricSound.STATISTIC, LametricSound.THUNDER, LametricSound.WATER02, LametricSound.WATER01, LametricSound.WIND, LametricSound.WIND_SHORT, LametricSound.WIN01, LametricSound.WIN02)

class NotifyLametric(NotifyBase):
    """
    A wrapper for LaMetric Notifications
    """
    service_name = 'LaMetric'
    service_url = 'https://lametric.com'
    protocol = 'lametric'
    secure_protocol = 'lametrics'
    request_rate_per_sec = 0.2
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_lametric'
    title_maxlen = 0
    cloud_notify_url = 'https://developer.lametric.com/api/v1/dev/widget/update/com.lametric.{app_id}/{app_ver}'
    device_notify_url = '{schema}://{host}{port}/api/v2/device/notifications'
    default_device_user = 'dev'
    lametric_icon_id_mapping = {NotifyType.INFO: 'i620', NotifyType.SUCCESS: 'i9182', NotifyType.WARNING: 'i9183', NotifyType.FAILURE: 'i9184'}
    templates = ('{schema}://{app_token}@{app_id}', '{schema}://{app_token}@{app_id}/{app_ver}', '{schema}://{apikey}@{host}', '{schema}://{user}:{apikey}@{host}', '{schema}://{apikey}@{host}:{port}', '{schema}://{user}:{apikey}@{host}:{port}')
    template_tokens = dict(NotifyBase.template_tokens, **{'apikey': {'name': _('Device API Key'), 'type': 'string', 'private': True}, 'app_id': {'name': _('App ID'), 'type': 'string', 'private': True}, 'app_ver': {'name': _('App Version'), 'type': 'string', 'regex': ('^[1-9][0-9]*$', ''), 'default': '1'}, 'app_token': {'name': _('App Access Token'), 'type': 'string', 'regex': ('^[A-Z0-9]{80,}==$', 'i')}, 'host': {'name': _('Hostname'), 'type': 'string'}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535, 'default': 8080}, 'user': {'name': _('Username'), 'type': 'string'}})
    template_args = dict(NotifyBase.template_args, **{'apikey': {'alias_of': 'apikey'}, 'app_id': {'alias_of': 'app_id'}, 'app_ver': {'alias_of': 'app_ver'}, 'app_token': {'alias_of': 'app_token'}, 'priority': {'name': _('Priority'), 'type': 'choice:string', 'values': LAMETRIC_PRIORITIES, 'default': LametricPriority.INFO}, 'icon': {'name': _('Custom Icon'), 'type': 'string'}, 'icon_type': {'name': _('Icon Type'), 'type': 'choice:string', 'values': LAMETRIC_ICON_TYPES, 'default': LametricIconType.NONE}, 'mode': {'name': _('Mode'), 'type': 'choice:string', 'values': LAMETRIC_MODES, 'default': LametricMode.DEVICE}, 'sound': {'name': _('Sound'), 'type': 'string'}, 'cycles': {'name': _('Cycles'), 'type': 'int', 'min': 0, 'default': 1}})

    def __init__(self, apikey=None, app_token=None, app_id=None, app_ver=None, priority=None, icon=None, icon_type=None, sound=None, mode=None, cycles=None, **kwargs):
        if False:
            return 10
        '\n        Initialize LaMetric Object\n        '
        super().__init__(**kwargs)
        self.mode = mode.strip().lower() if isinstance(mode, str) else self.template_args['mode']['default']
        self.lametric_app_id = None
        self.lametric_app_ver = None
        self.lametric_app_access_token = None
        self.lametric_apikey = None
        if self.mode not in LAMETRIC_MODES:
            msg = 'An invalid LaMetric Mode ({}) was specified.'.format(mode)
            self.logger.warning(msg)
            raise TypeError(msg)
        if self.mode == LametricMode.CLOUD:
            try:
                results = LAMETRIC_APP_ID_DETECTOR_RE.match(app_id)
            except TypeError:
                msg = 'An invalid LaMetric Application ID ({}) was specified.'.format(app_id)
                self.logger.warning(msg)
                raise TypeError(msg)
            self.lametric_app_access_token = validate_regex(app_token, *self.template_tokens['app_token']['regex'])
            if not self.lametric_app_access_token:
                msg = 'An invalid LaMetric Application Access Token ({}) was specified.'.format(app_token)
                self.logger.warning(msg)
                raise TypeError(msg)
            if app_ver:
                self.lametric_app_ver = validate_regex(app_ver, *self.template_tokens['app_ver']['regex'])
                if not self.lametric_app_ver:
                    msg = 'An invalid LaMetric Application Version ({}) was specified.'.format(app_ver)
                    self.logger.warning(msg)
                    raise TypeError(msg)
            else:
                self.lametric_app_ver = results.group('app_ver') if results.group('app_ver') else self.template_tokens['app_ver']['default']
            self.lametric_app_id = results.group('app_id')
        if self.mode == LametricMode.DEVICE:
            self.lametric_apikey = validate_regex(apikey)
            if not self.lametric_apikey:
                msg = 'An invalid LaMetric Device API Key ({}) was specified.'.format(apikey)
                self.logger.warning(msg)
                raise TypeError(msg)
        if priority not in LAMETRIC_PRIORITIES:
            self.priority = self.template_args['priority']['default']
        else:
            self.priority = priority
        self.icon = re.search('[#\\s]*(?P<value>.+?)\\s*$', icon).group('value') if isinstance(icon, str) else None
        if icon_type not in LAMETRIC_ICON_TYPES:
            self.icon_type = self.template_args['icon_type']['default']
        else:
            self.icon_type = icon_type
        self.cycles = self.template_args['cycles']['default'] if not (isinstance(cycles, int) and cycles > self.template_args['cycles']['min']) else cycles
        self.sound = None
        if isinstance(sound, str):
            self.sound = self.sound_lookup(sound.strip().lower())
            if self.sound is None:
                self.logger.warning('An invalid LaMetric sound ({}) was specified.'.format(sound))
        return

    @staticmethod
    def sound_lookup(lookup):
        if False:
            i = 10
            return i + 15
        '\n        A simple match function that takes string and returns the\n        LametricSound object it was found in.\n\n        '
        for x in LAMETRIC_SOUNDS:
            match = next((f for f in x[1] if f.startswith(lookup)), None)
            if match:
                return x
        return None

    def _cloud_notification_payload(self, body, notify_type, headers):
        if False:
            i = 10
            return i + 15
        '\n        Return URL and payload for cloud directed requests\n        '
        headers.update({'X-Access-Token': self.lametric_apikey})
        if self.sound:
            self.logger.warning('LaMetric sound setting is unavailable in Cloud mode')
        if self.priority != self.template_args['priority']['default']:
            self.logger.warning('LaMetric priority setting is unavailable in Cloud mode')
        if self.icon_type != self.template_args['icon_type']['default']:
            self.logger.warning('LaMetric icon_type setting is unavailable in Cloud mode')
        if self.cycles != self.template_args['cycles']['default']:
            self.logger.warning('LaMetric cycle settings is unavailable in Cloud mode')
        icon = self.icon if self.icon else self.lametric_icon_id_mapping[notify_type]
        payload = {'frames': [{'icon': icon, 'text': body, 'index': 0}]}
        notify_url = self.cloud_notify_url.format(app_id=self.lametric_app_id, app_ver=self.lametric_app_ver)
        return (notify_url, None, payload)

    def _device_notification_payload(self, body, notify_type, headers):
        if False:
            return 10
        '\n        Return URL and Payload for Device directed requests\n        '
        icon = self.icon if self.icon else self.lametric_icon_id_mapping[notify_type]
        payload = {'priority': self.priority, 'icon_type': self.icon_type, 'lifetime': 120000, 'model': {'cycles': self.cycles, 'frames': [{'icon': icon, 'text': body}]}}
        if self.sound:
            payload['model']['sound'] = {'category': self.sound[0], 'id': self.sound[1][0], 'repeat': 1}
        if not self.user:
            self.user = self.default_device_user
        auth = (self.user, self.password)
        notify_url = self.device_notify_url.format(schema='https' if self.secure else 'http', host=self.host, port=':{}'.format(self.port if self.port else self.template_tokens['port']['default']))
        return (notify_url, auth, payload)

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Perform LaMetric Notification\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json', 'Accept': 'application/json', 'Cache-Control': 'no-cache'}
        (notify_url, auth, payload) = getattr(self, '_{}_notification_payload'.format(self.mode))(body=body, notify_type=notify_type, headers=headers)
        self.logger.debug('LaMetric POST URL: %s (cert_verify=%r)' % (notify_url, self.verify_certificate))
        self.logger.debug('LaMetric Payload: %s' % str(payload))
        self.throttle()
        try:
            r = requests.post(notify_url, data=dumps(payload), headers=headers, auth=auth, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code not in (requests.codes.created, requests.codes.ok):
                status_str = NotifyLametric.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send LaMetric notification: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent LaMetric notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending LaMetric notification to %s.' % self.host)
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'mode': self.mode}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        if self.icon:
            params['icon'] = self.icon
        if self.mode == LametricMode.CLOUD:
            return '{schema}://{token}@{app_id}/{app_ver}/?{params}'.format(schema=self.protocol, token=self.pprint(self.lametric_app_access_token, privacy, safe=''), app_id=self.pprint(self.lametric_app_id, privacy, safe=''), app_ver=NotifyLametric.quote(self.lametric_app_ver, safe=''), params=NotifyLametric.urlencode(params))
        if self.priority != self.template_args['priority']['default']:
            params['priority'] = self.priority
        if self.icon_type != self.template_args['icon_type']['default']:
            params['icon_type'] = self.icon_type
        if self.cycles != self.template_args['cycles']['default']:
            params['cycles'] = self.cycles
        if self.sound:
            params['sound'] = self.sound[1][0]
        auth = ''
        if self.user and self.password:
            auth = '{user}:{apikey}@'.format(user=NotifyLametric.quote(self.user, safe=''), apikey=self.pprint(self.lametric_apikey, privacy, safe=''))
        else:
            auth = '{apikey}@'.format(apikey=self.pprint(self.lametric_apikey, privacy, safe=''))
        return '{schema}://{auth}{hostname}{port}/?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, auth=auth, hostname=self.host, port='' if self.port is None or self.port == self.template_tokens['port']['default'] else ':{}'.format(self.port), params=NotifyLametric.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            return 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        if results.get('user') and (not results.get('password')):
            results['password'] = results['user']
            results['user'] = None
        if 'priority' in results['qsd'] and results['qsd']['priority']:
            results['priority'] = NotifyLametric.unquote(results['qsd']['priority'].strip().lower())
        if 'icon' in results['qsd'] and results['qsd']['icon']:
            results['icon'] = NotifyLametric.unquote(results['qsd']['icon'].strip().lower())
        if 'icon_type' in results['qsd'] and results['qsd']['icon_type']:
            results['icon_type'] = NotifyLametric.unquote(results['qsd']['icon_type'].strip().lower())
        if 'sound' in results['qsd'] and results['qsd']['sound']:
            results['sound'] = NotifyLametric.unquote(results['qsd']['sound'].strip().lower())
        if 'apikey' in results['qsd'] and results['qsd']['apikey']:
            results['apikey'] = NotifyLametric.unquote(results['qsd']['apikey'])
        if 'app' in results['qsd'] and results['qsd']['app']:
            results['app_id'] = NotifyLametric.unquote(results['qsd']['app'])
        if 'app_ver' in results['qsd'] and results['qsd']['app_ver']:
            results['app_ver'] = NotifyLametric.unquote(results['qsd']['app_ver'])
        if 'token' in results['qsd'] and results['qsd']['token']:
            results['app_token'] = NotifyLametric.unquote(results['qsd']['token'])
        if 'mode' in results['qsd'] and results['qsd']['mode']:
            results['mode'] = NotifyLametric.unquote(results['qsd']['mode'].strip().lower())
        else:
            results['mode'] = LametricMode.DEVICE if (is_hostname(results['host']) or is_ipaddr(results['host'])) and (results['password'] and (not LAMETRIC_IS_APP_TOKEN.match(results['password']))) and (next((f for f in results.keys() if f.startswith('app_')), None) is None) else LametricMode.CLOUD
        if results['mode'] == LametricMode.DEVICE:
            if 'apikey' not in results:
                results['apikey'] = NotifyLametric.unquote(results['password'])
        else:
            if 'app_id' not in results:
                results['app_id'] = NotifyLametric.unquote(results['host'])
            if 'app_token' not in results:
                results['app_token'] = NotifyLametric.unquote(results['password'])
        try:
            results['cycles'] = abs(int(results['qsd'].get('cycles')))
        except (TypeError, ValueError):
            pass
        return results

    @staticmethod
    def parse_native_url(url):
        if False:
            while True:
                i = 10
        '\n        Support\n           https://developer.lametric.com/api/v1/dev/                   widget/update/com.lametric.{APP_ID}/1\n\n           https://developer.lametric.com/api/v1/dev/                   widget/update/com.lametric.{APP_ID}/{APP_VER}\n        '
        result = re.match('^http(?P<secure>s)?://(?P<host>[^/]+)/api/(?P<api_ver>v[1-9]*[0-9]+)/dev/widget/update/com\\.lametric\\.(?P<app_id>[0-9a-z.-]{1,64})(/(?P<app_ver>[1-9][0-9]*))?/?(?P<params>\\?.+)?$', url, re.I)
        if result:
            return NotifyLametric.parse_url('{schema}://{app_id}{app_ver}/{params}'.format(schema=NotifyLametric.secure_protocol if result.group('secure') else NotifyLametric.protocol, app_id=result.group('app_id'), app_ver='/{}'.format(result.group('app_ver')) if result.group('app_ver') else '', params='' if not result.group('params') else result.group('params')))
        return None