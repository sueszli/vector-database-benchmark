import re
import time
import hmac
import hashlib
import base64
import requests
from json import dumps
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyFormat
from ..common import NotifyType
from ..utils import parse_list
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
IS_PHONE_NO = re.compile('^\\+?(?P<phone>[0-9\\s)(+-]+)\\s*$')

class NotifyDingTalk(NotifyBase):
    """
    A wrapper for DingTalk Notifications
    """
    service_name = 'DingTalk'
    service_url = 'https://www.dingtalk.com/'
    secure_protocol = 'dingtalk'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_dingtalk'
    notify_url = 'https://oapi.dingtalk.com/robot/send?access_token={token}'
    templates = ('{schema}://{token}/', '{schema}://{token}/{targets}/', '{schema}://{secret}@{token}/', '{schema}://{secret}@{token}/{targets}/')
    template_tokens = dict(NotifyBase.template_tokens, **{'token': {'name': _('Token'), 'type': 'string', 'private': True, 'required': True, 'regex': ('^[a-z0-9]+$', 'i')}, 'secret': {'name': _('Secret'), 'type': 'string', 'private': True, 'regex': ('^[a-z0-9]+$', 'i')}, 'target_phone_no': {'name': _('Target Phone No'), 'type': 'string', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}, 'token': {'alias_of': 'token'}, 'secret': {'alias_of': 'secret'}})

    def __init__(self, token, targets=None, secret=None, **kwargs):
        if False:
            return 10
        '\n        Initialize DingTalk Object\n        '
        super().__init__(**kwargs)
        self.token = validate_regex(token, *self.template_tokens['token']['regex'])
        if not self.token:
            msg = 'An invalid DingTalk API Token ({}) was specified.'.format(token)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.secret = None
        if secret:
            self.secret = validate_regex(secret, *self.template_tokens['secret']['regex'])
            if not self.secret:
                msg = 'An invalid DingTalk Secret ({}) was specified.'.format(token)
                self.logger.warning(msg)
                raise TypeError(msg)
        self.targets = list()
        for target in parse_list(targets):
            result = IS_PHONE_NO.match(target)
            if result:
                result = ''.join(re.findall('\\d+', result.group('phone')))
                if len(result) < 11 or len(result) > 14:
                    self.logger.warning('Dropped invalid phone # ({}) specified.'.format(target))
                    continue
                self.targets.append(result)
                continue
            self.logger.warning('Dropped invalid phone # ({}) specified.'.format(target))
        return

    def get_signature(self):
        if False:
            i = 10
            return i + 15
        '\n        Calculates time-based signature so that we can send arbitrary messages.\n        '
        timestamp = str(round(time.time() * 1000))
        secret_enc = self.secret.encode('utf-8')
        str_to_sign_enc = '{}\n{}'.format(timestamp, self.secret).encode('utf-8')
        hmac_code = hmac.new(secret_enc, str_to_sign_enc, digestmod=hashlib.sha256).digest()
        signature = NotifyDingTalk.quote(base64.b64encode(hmac_code), safe='')
        return (timestamp, signature)

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            return 10
        '\n        Perform DingTalk Notification\n        '
        payload = {'msgtype': 'text', 'at': {'atMobiles': self.targets, 'isAtAll': False}}
        if self.notify_format == NotifyFormat.MARKDOWN:
            payload['markdown'] = {'title': title, 'text': body}
        else:
            payload['text'] = {'content': body}
        notify_url = self.notify_url.format(token=self.token)
        params = None
        if self.secret:
            (timestamp, signature) = self.get_signature()
            params = {'timestamp': timestamp, 'sign': signature}
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json'}
        self.logger.debug('DingTalk URL: {} (cert_verify={})'.format(notify_url, self.verify_certificate))
        self.logger.debug('DingTalk Payload: {}'.format(payload))
        self.throttle()
        try:
            r = requests.post(notify_url, data=dumps(payload), headers=headers, params=params, verify=self.verify_certificate)
            if r.status_code != requests.codes.ok:
                status_str = NotifyDingTalk.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send DingTalk notification: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent DingTalk notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occured sending DingTalk notification.')
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    @property
    def title_maxlen(self):
        if False:
            print('Hello World!')
        "\n        The title isn't used when not in markdown mode.\n        "
        return NotifyBase.title_maxlen if self.notify_format == NotifyFormat.MARKDOWN else 0

    def url(self, privacy=False, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        args = {'format': self.notify_format, 'overflow': self.overflow_mode, 'verify': 'yes' if self.verify_certificate else 'no'}
        return '{schema}://{secret}{token}/{targets}/?{args}'.format(schema=self.secure_protocol, secret='' if not self.secret else '{}@'.format(self.pprint(self.secret, privacy, mode=PrivacyMode.Secret, safe='')), token=self.pprint(self.token, privacy, safe=''), targets='/'.join([NotifyDingTalk.quote(x, safe='') for x in self.targets]), args=NotifyDingTalk.urlencode(args))

    def __len__(self):
        if False:
            return 10
        '\n        Returns the number of targets associated with this notification\n        '
        targets = len(self.targets)
        return targets if targets > 0 else 1

    @staticmethod
    def parse_url(url):
        if False:
            print('Hello World!')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to substantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['token'] = NotifyDingTalk.unquote(results['host'])
        if results.get('user'):
            results['secret'] = results.get('user')
        results['targets'] = NotifyDingTalk.split_path(results['fullpath'])
        if 'token' in results['qsd'] and len(results['qsd']['token']):
            results['token'] = NotifyDingTalk.unquote(results['qsd']['token'])
        if 'secret' in results['qsd'] and len(results['qsd']['secret']):
            results['secret'] = NotifyDingTalk.unquote(results['qsd']['secret'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyDingTalk.parse_list(results['qsd']['to'])
        return results