import re
import requests
from json import dumps
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
TARGET_LIST_DELIM = re.compile('[ \\t\\r\\n,\\\\/]+')

class ParsePlatformDevice:
    ALL = 'all'
    IOS = 'ios'
    ANDROID = 'android'
PARSE_PLATFORM_DEVICES = (ParsePlatformDevice.ALL, ParsePlatformDevice.IOS, ParsePlatformDevice.ANDROID)

class NotifyParsePlatform(NotifyBase):
    """
    A wrapper for Parse Platform Notifications
    """
    service_name = 'Parse Platform'
    service_url = ' https://parseplatform.org/'
    protocol = 'parsep'
    secure_protocol = 'parseps'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_parseplatform'
    templates = ('{schema}://{app_id}:{master_key}@{host}', '{schema}://{app_id}:{master_key}@{host}:{port}')
    template_tokens = dict(NotifyBase.template_tokens, **{'host': {'name': _('Hostname'), 'type': 'string', 'required': True}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}, 'app_id': {'name': _('App ID'), 'type': 'string', 'private': True, 'required': True}, 'master_key': {'name': _('Master Key'), 'type': 'string', 'private': True, 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'device': {'name': _('Device'), 'type': 'choice:string', 'values': PARSE_PLATFORM_DEVICES, 'default': ParsePlatformDevice.ALL}, 'app_id': {'alias_of': 'app_id'}, 'master_key': {'alias_of': 'master_key'}})

    def __init__(self, app_id, master_key, device=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize Parse Platform Object\n        '
        super().__init__(**kwargs)
        self.fullpath = kwargs.get('fullpath')
        if not isinstance(self.fullpath, str):
            self.fullpath = '/'
        self.application_id = validate_regex(app_id)
        if not self.application_id:
            msg = 'An invalid Parse Platform Application ID ({}) was specified.'.format(app_id)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.master_key = validate_regex(master_key)
        if not self.master_key:
            msg = 'An invalid Parse Platform Master Key ({}) was specified.'.format(master_key)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.devices = []
        if device:
            self.device = device.lower()
            if device not in PARSE_PLATFORM_DEVICES:
                msg = 'An invalid Parse Platform device ({}) was specified.'.format(device)
                self.logger.warning(msg)
                raise TypeError(msg)
        else:
            self.device = self.template_args['device']['default']
        if self.device == ParsePlatformDevice.ALL:
            self.devices = [d for d in PARSE_PLATFORM_DEVICES if d != ParsePlatformDevice.ALL]
        else:
            self.devices.append(device)
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Perform Parse Platform Notification\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json', 'X-Parse-Application-Id': self.application_id, 'X-Parse-Master-Key': self.master_key}
        payload = {'where': {'deviceType': {'$in': self.devices}}, 'data': {'title': title, 'alert': body}}
        schema = 'https' if self.secure else 'http'
        url = '%s://%s' % (schema, self.host)
        if isinstance(self.port, int):
            url += ':%d' % self.port
        url += self.fullpath.rstrip('/') + '/parse/push/'
        self.logger.debug('Parse Platform POST URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
        self.logger.debug('Parse Platform Payload: %s' % str(payload))
        self.throttle()
        try:
            r = requests.post(url, data=dumps(payload), headers=headers, verify=self.verify_certificate)
            if r.status_code != requests.codes.ok:
                status_str = NotifyParsePlatform.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send Parse Platform notification: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent Parse Platform notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occured sending Parse Platform notification to %s.' % self.host)
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'device': self.device}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        default_port = 443 if self.secure else 80
        return '{schema}://{app_id}:{master_key}@{hostname}{port}{fullpath}/?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, app_id=self.pprint(self.application_id, privacy, safe=''), master_key=self.pprint(self.master_key, privacy, safe=''), hostname=NotifyParsePlatform.quote(self.host, safe=''), port='' if self.port is None or self.port == default_port else ':{}'.format(self.port), fullpath=NotifyParsePlatform.quote(self.fullpath, safe='/'), params=NotifyParsePlatform.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            print('Hello World!')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to substantiate this object.\n\n        '
        results = NotifyBase.parse_url(url)
        if not results:
            return results
        results['app_id'] = NotifyParsePlatform.unquote(results['user'])
        results['master_key'] = NotifyParsePlatform.unquote(results['password'])
        if 'device' in results['qsd'] and len(results['qsd']['device']):
            results['device'] = results['qsd']['device']
        if 'app_id' in results['qsd'] and len(results['qsd']['app_id']):
            results['app_id'] = results['qsd']['app_id']
        if 'master_key' in results['qsd'] and len(results['qsd']['master_key']):
            results['master_key'] = results['qsd']['master_key']
        return results