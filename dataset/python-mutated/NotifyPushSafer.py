import base64
import requests
from json import loads
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import parse_list
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _

class PushSaferSound:
    """
    Defines all of the supported PushSafe sounds
    """
    SILENT = 0
    AHEM = 1
    APPLAUSE = 2
    ARROW = 3
    BABY = 4
    BELL = 5
    BICYCLE = 6
    BOING = 7
    BUZZER = 8
    CAMERA = 9
    CAR_HORN = 10
    CASH_REGISTER = 11
    CHIME = 12
    CREAKY_DOOR = 13
    CUCKOO_CLOCK = 14
    DISCONNECT = 15
    DOG = 16
    DOORBELL = 17
    FANFARE = 18
    GUN_SHOT = 19
    HONK = 20
    JAW_HARP = 21
    MORSE = 22
    ELECTRICITY = 23
    RADIO_TURNER = 24
    SIRENS = 25
    MILITARY_TRUMPETS = 26
    UFO = 27
    LONG_WHAH = 28
    GOODBYE = 29
    HELLO = 30
    NO = 31
    OKAY = 32
    OOOHHHWEEE = 33
    WARNING = 34
    WELCOME = 35
    YEAH = 36
    YES = 37
    BEEP1 = 38
    WEEE = 39
    CUTINOUT = 40
    FLICK_GLASS = 41
    SHORT_WHAH = 42
    LASER = 43
    WIND_CHIME = 44
    ECHO = 45
    ZIPPER = 46
    HIHAT = 47
    BEEP2 = 48
    BEEP3 = 49
    BEEP4 = 50
    ALARM_ARMED = 51
    ALARM_DISARMED = 52
    BACKUP_READY = 53
    DOOR_CLOSED = 54
    DOOR_OPENED = 55
    WINDOW_CLOSED = 56
    WINDOW_OPEN = 57
    LIGHT_ON = 58
    LIGHT_OFF = 59
    DOORBELL_RANG = 60
PUSHSAFER_SOUND_MAP = {'silent': PushSaferSound.SILENT, 'ahem': PushSaferSound.AHEM, 'applause': PushSaferSound.APPLAUSE, 'arrow': PushSaferSound.ARROW, 'baby': PushSaferSound.BABY, 'bell': PushSaferSound.BELL, 'bicycle': PushSaferSound.BICYCLE, 'bike': PushSaferSound.BICYCLE, 'boing': PushSaferSound.BOING, 'buzzer': PushSaferSound.BUZZER, 'camera': PushSaferSound.CAMERA, 'carhorn': PushSaferSound.CAR_HORN, 'horn': PushSaferSound.CAR_HORN, 'cashregister': PushSaferSound.CASH_REGISTER, 'chime': PushSaferSound.CHIME, 'creakydoor': PushSaferSound.CREAKY_DOOR, 'cuckooclock': PushSaferSound.CUCKOO_CLOCK, 'cuckoo': PushSaferSound.CUCKOO_CLOCK, 'disconnect': PushSaferSound.DISCONNECT, 'dog': PushSaferSound.DOG, 'doorbell': PushSaferSound.DOORBELL, 'fanfare': PushSaferSound.FANFARE, 'gunshot': PushSaferSound.GUN_SHOT, 'honk': PushSaferSound.HONK, 'jawharp': PushSaferSound.JAW_HARP, 'morse': PushSaferSound.MORSE, 'electric': PushSaferSound.ELECTRICITY, 'radiotuner': PushSaferSound.RADIO_TURNER, 'sirens': PushSaferSound.SIRENS, 'militarytrumpets': PushSaferSound.MILITARY_TRUMPETS, 'military': PushSaferSound.MILITARY_TRUMPETS, 'trumpets': PushSaferSound.MILITARY_TRUMPETS, 'ufo': PushSaferSound.UFO, 'whahwhah': PushSaferSound.LONG_WHAH, 'whah': PushSaferSound.SHORT_WHAH, 'goodye': PushSaferSound.GOODBYE, 'hello': PushSaferSound.HELLO, 'no': PushSaferSound.NO, 'okay': PushSaferSound.OKAY, 'ok': PushSaferSound.OKAY, 'ooohhhweee': PushSaferSound.OOOHHHWEEE, 'warn': PushSaferSound.WARNING, 'warning': PushSaferSound.WARNING, 'welcome': PushSaferSound.WELCOME, 'yeah': PushSaferSound.YEAH, 'yes': PushSaferSound.YES, 'beep': PushSaferSound.BEEP1, 'beep1': PushSaferSound.BEEP1, 'weee': PushSaferSound.WEEE, 'wee': PushSaferSound.WEEE, 'cutinout': PushSaferSound.CUTINOUT, 'flickglass': PushSaferSound.FLICK_GLASS, 'laser': PushSaferSound.LASER, 'windchime': PushSaferSound.WIND_CHIME, 'echo': PushSaferSound.ECHO, 'zipper': PushSaferSound.ZIPPER, 'hihat': PushSaferSound.HIHAT, 'beep2': PushSaferSound.BEEP2, 'beep3': PushSaferSound.BEEP3, 'beep4': PushSaferSound.BEEP4, 'alarmarmed': PushSaferSound.ALARM_ARMED, 'armed': PushSaferSound.ALARM_ARMED, 'alarmdisarmed': PushSaferSound.ALARM_DISARMED, 'disarmed': PushSaferSound.ALARM_DISARMED, 'backupready': PushSaferSound.BACKUP_READY, 'dooropen': PushSaferSound.DOOR_OPENED, 'dopen': PushSaferSound.DOOR_OPENED, 'doorclosed': PushSaferSound.DOOR_CLOSED, 'dclosed': PushSaferSound.DOOR_CLOSED, 'windowopen': PushSaferSound.WINDOW_OPEN, 'wopen': PushSaferSound.WINDOW_OPEN, 'windowclosed': PushSaferSound.WINDOW_CLOSED, 'wclosed': PushSaferSound.WINDOW_CLOSED, 'lighton': PushSaferSound.LIGHT_ON, 'lon': PushSaferSound.LIGHT_ON, 'lightoff': PushSaferSound.LIGHT_OFF, 'loff': PushSaferSound.LIGHT_OFF, 'doorbellrang': PushSaferSound.DOORBELL_RANG}

class PushSaferPriority:
    LOW = -2
    MODERATE = -1
    NORMAL = 0
    HIGH = 1
    EMERGENCY = 2
PUSHSAFER_PRIORITIES = (PushSaferPriority.LOW, PushSaferPriority.MODERATE, PushSaferPriority.NORMAL, PushSaferPriority.HIGH, PushSaferPriority.EMERGENCY)
PUSHSAFER_PRIORITY_MAP = {'low': PushSaferPriority.LOW, 'medium': PushSaferPriority.MODERATE, 'normal': PushSaferPriority.NORMAL, 'high': PushSaferPriority.HIGH, 'emergency': PushSaferPriority.EMERGENCY}
DEFAULT_PRIORITY = 'normal'

class PushSaferVibration:
    """
    Defines the acceptable vibration settings for notification
    """
    LOW = 1
    NORMAL = 2
    HIGH = 3
PUSHSAFER_VIBRATIONS = (PushSaferVibration.LOW, PushSaferVibration.NORMAL, PushSaferVibration.HIGH)
PICTURE_PARAMETER = ('p', 'p2', 'p3')
PUSHSAFER_SEND_TO_ALL = 'a'

class NotifyPushSafer(NotifyBase):
    """
    A wrapper for PushSafer Notifications
    """
    service_name = 'Pushsafer'
    service_url = 'https://www.pushsafer.com/'
    protocol = 'psafer'
    secure_protocol = 'psafers'
    attachment_support = True
    request_rate_per_sec = 1.2
    default_pushsafer_icon = 25
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_pushsafer'
    notify_url = '{schema}://www.pushsafer.com/api'
    templates = ('{schema}://{privatekey}', '{schema}://{privatekey}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'privatekey': {'name': _('Private Key'), 'type': 'string', 'private': True, 'required': True}, 'target_device': {'name': _('Target Device'), 'type': 'string', 'map_to': 'targets'}, 'target_email': {'name': _('Target Email'), 'type': 'string', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'priority': {'name': _('Priority'), 'type': 'choice:int', 'values': PUSHSAFER_PRIORITIES}, 'sound': {'name': _('Sound'), 'type': 'choice:string', 'values': PUSHSAFER_SOUND_MAP}, 'vibration': {'name': _('Vibration'), 'type': 'choice:int', 'values': PUSHSAFER_VIBRATIONS}, 'to': {'alias_of': 'targets'}})

    def __init__(self, privatekey, targets=None, priority=None, sound=None, vibration=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialize PushSafer Object\n        '
        super().__init__(**kwargs)
        try:
            self.priority = int(priority)
        except TypeError:
            self.priority = None
        except ValueError:
            priority = priority.lower().strip()
            match = next((key for key in PUSHSAFER_PRIORITY_MAP.keys() if key.startswith(priority)), None) if priority else None
            if not match:
                msg = 'An invalid PushSafer priority ({}) was specified.'.format(priority)
                self.logger.warning(msg)
                raise TypeError(msg)
            self.priority = PUSHSAFER_PRIORITY_MAP[match]
        if self.priority is not None and self.priority not in PUSHSAFER_PRIORITY_MAP.values():
            msg = 'An invalid PushSafer priority ({}) was specified.'.format(priority)
            self.logger.warning(msg)
            raise TypeError(msg)
        try:
            self.sound = int(sound)
        except TypeError:
            self.sound = None
        except ValueError:
            sound = sound.lower().strip()
            match = next((key for key in PUSHSAFER_SOUND_MAP.keys() if key.startswith(sound)), None) if sound else None
            if not match:
                msg = 'An invalid PushSafer sound ({}) was specified.'.format(sound)
                self.logger.warning(msg)
                raise TypeError(msg)
            self.sound = PUSHSAFER_SOUND_MAP[match]
        if self.sound is not None and self.sound not in PUSHSAFER_SOUND_MAP.values():
            msg = 'An invalid PushSafer sound ({}) was specified.'.format(sound)
            self.logger.warning(msg)
            raise TypeError(msg)
        try:
            self.vibration = int(vibration)
        except TypeError:
            self.vibration = None
        except ValueError:
            msg = 'An invalid PushSafer vibration ({}) was specified.'.format(vibration)
            self.logger.warning(msg)
            raise TypeError(msg)
        if self.vibration and self.vibration not in PUSHSAFER_VIBRATIONS:
            msg = 'An invalid PushSafer vibration ({}) was specified.'.format(vibration)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.privatekey = validate_regex(privatekey)
        if not self.privatekey:
            msg = 'An invalid PushSafer Private Key ({}) was specified.'.format(privatekey)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.targets = parse_list(targets)
        if len(self.targets) == 0:
            self.targets = (PUSHSAFER_SEND_TO_ALL,)
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, attach=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform PushSafer Notification\n        '
        has_error = False
        attachments = []
        if attach and self.attachment_support:
            for attachment in attach:
                if not attachment:
                    self.logger.error('Could not access attachment {}.'.format(attachment.url(privacy=True)))
                    return False
                if not attachment.mimetype.startswith('image/'):
                    self.logger.debug('Ignoring unsupported PushSafer attachment {}.'.format(attachment.url(privacy=True)))
                    continue
                self.logger.debug('Posting PushSafer attachment {}'.format(attachment.url(privacy=True)))
                try:
                    with open(attachment.path, 'rb') as f:
                        attachment = (attachment.name, 'data:{};base64,{}'.format(attachment.mimetype, base64.b64encode(f.read())))
                except (OSError, IOError) as e:
                    self.logger.warning('An I/O error occurred while reading {}.'.format(attachment.name if attachment else 'attachment'))
                    self.logger.debug('I/O Exception: %s' % str(e))
                    return False
                attachments.append(attachment)
        targets = list(self.targets)
        while len(targets):
            recipient = targets.pop(0)
            payload = {'t': title, 'm': body, 'i': self.default_pushsafer_icon, 'c': self.color(notify_type), 'd': recipient}
            if self.sound is not None:
                payload['s'] = str(self.sound)
            if self.vibration is not None:
                payload['v'] = str(self.vibration)
            if not attachments:
                (okay, response) = self._send(payload)
                if not okay:
                    has_error = True
                    continue
                self.logger.info('Sent PushSafer notification to "%s".' % recipient)
            else:
                _payload = payload.copy()
                for idx in range(0, len(attachments), len(PICTURE_PARAMETER)):
                    for (c, attachment) in enumerate(attachments[idx:idx + len(PICTURE_PARAMETER)]):
                        (filename, dataurl) = attachment
                        _payload.update({PICTURE_PARAMETER[c]: dataurl})
                        self.logger.debug('Added attachment (%s) to "%s".' % (filename, recipient))
                    (okay, response) = self._send(_payload)
                    if not okay:
                        has_error = True
                        continue
                    self.logger.info('Sent PushSafer attachment (%s) to "%s".' % (filename, recipient))
                    _payload = payload.copy()
                    _payload['t'] = ''
                    _payload['m'] = '...'
        return not has_error

    def _send(self, payload, **kwargs):
        if False:
            return 10
        '\n        Wrapper to the requests (post) object\n        '
        headers = {'User-Agent': self.app_id}
        notify_url = self.notify_url.format(schema='https' if self.secure else 'http')
        payload['k'] = self.privatekey
        self.logger.debug('PushSafer POST URL: %s (cert_verify=%r)' % (notify_url, self.verify_certificate))
        self.logger.debug('PushSafer Payload: %s' % str(payload))
        self.throttle()
        response = None
        _code = None
        _str = 'Unknown'
        try:
            r = requests.post(notify_url, data=payload, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            try:
                response = loads(r.content)
                _code = response.get('status')
                _str = response.get('success', _str) if _code == 1 else response.get('error', _str)
            except (AttributeError, TypeError, ValueError):
                response = r.content
            if r.status_code not in (requests.codes.ok, requests.codes.no_content):
                status_str = NotifyPushSafer.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to deliver payload to PushSafer:{}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return (False, response)
            elif _code != 1:
                self.logger.warning('Failed to deliver payload to PushSafer; error={}.'.format(_str))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return (False, response)
            return (True, response)
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred communicating with PushSafer.')
            self.logger.debug('Socket Exception: %s' % str(e))
            return (False, response)

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = self.url_parameters(*args, privacy=privacy, **kwargs)
        if self.priority is not None:
            params['priority'] = next((key for (key, value) in PUSHSAFER_PRIORITY_MAP.items() if value == self.priority), DEFAULT_PRIORITY)
        if self.sound is not None:
            params['sound'] = next((key for (key, value) in PUSHSAFER_SOUND_MAP.items() if value == self.sound), '')
        if self.vibration is not None:
            params['vibration'] = str(self.vibration)
        targets = '/'.join([NotifyPushSafer.quote(x) for x in self.targets])
        if targets == PUSHSAFER_SEND_TO_ALL:
            targets = ''
        return '{schema}://{privatekey}/{targets}?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, privatekey=self.pprint(self.privatekey, privacy, safe=''), targets=targets, params=NotifyPushSafer.urlencode(params))

    def __len__(self):
        if False:
            i = 10
            return i + 15
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
        results['targets'] = NotifyPushSafer.split_path(results['fullpath'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyPushSafer.parse_list(results['qsd']['to'])
        results['privatekey'] = NotifyPushSafer.unquote(results['host'])
        if 'priority' in results['qsd'] and len(results['qsd']['priority']):
            results['priority'] = NotifyPushSafer.unquote(results['qsd']['priority'])
        if 'sound' in results['qsd'] and len(results['qsd']['sound']):
            results['sound'] = NotifyPushSafer.unquote(results['qsd']['sound'])
        if 'vibration' in results['qsd'] and len(results['qsd']['vibration']):
            results['vibration'] = NotifyPushSafer.unquote(results['qsd']['vibration'])
        return results