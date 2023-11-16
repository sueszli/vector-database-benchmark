import requests
from json import dumps
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyType
from ..utils import parse_list
from ..AppriseLocale import gettext_lazy as _

class NotifyNextcloudTalk(NotifyBase):
    """
    A wrapper for Nextcloud Talk Notifications
    """
    service_name = _('Nextcloud Talk')
    service_url = 'https://nextcloud.com/talk'
    protocol = 'nctalk'
    secure_protocol = 'nctalks'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_nextcloudtalk'
    title_maxlen = 255
    body_maxlen = 4000
    templates = ('{schema}://{user}:{password}@{host}/{targets}', '{schema}://{user}:{password}@{host}:{port}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'host': {'name': _('Hostname'), 'type': 'string', 'required': True}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}, 'user': {'name': _('Username'), 'type': 'string', 'required': True}, 'password': {'name': _('Password'), 'type': 'string', 'private': True, 'required': True}, 'target_room_id': {'name': _('Room ID'), 'type': 'string', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string', 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'url_prefix': {'name': _('URL Prefix'), 'type': 'string'}})
    template_kwargs = {'headers': {'name': _('HTTP Header'), 'prefix': '+'}}

    def __init__(self, targets=None, headers=None, url_prefix=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Initialize Nextcloud Talk Object\n        '
        super().__init__(**kwargs)
        if self.user is None or self.password is None:
            msg = 'User and password have to be specified.'
            self.logger.warning(msg)
            raise TypeError(msg)
        self.targets = parse_list(targets)
        self.url_prefix = '' if not url_prefix else url_prefix.strip('/')
        self.headers = {}
        if headers:
            self.headers.update(headers)
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Perform Nextcloud Talk Notification\n        '
        if len(self.targets) == 0:
            self.logger.warning('There were no Nextcloud Talk targets to notify.')
            return False
        headers = {'User-Agent': self.app_id, 'OCS-APIRequest': 'true', 'Accept': 'application/json', 'Content-Type': 'application/json'}
        headers.update(self.headers)
        has_error = False
        targets = list(self.targets)
        while len(targets):
            target = targets.pop(0)
            if not body:
                payload = {'message': title if title else self.app_desc}
            else:
                payload = {'message': title + '\r\n' + body if title else self.app_desc + '\r\n' + body}
            notify_url = '{schema}://{host}/{url_prefix}/ocs/v2.php/apps/spreed/api/v1/chat/{target}'
            notify_url = notify_url.format(schema='https' if self.secure else 'http', host=self.host if not isinstance(self.port, int) else '{}:{}'.format(self.host, self.port), url_prefix=self.url_prefix, target=target)
            self.logger.debug('Nextcloud Talk POST URL: %s (cert_verify=%r)', notify_url, self.verify_certificate)
            self.logger.debug('Nextcloud Talk Payload: %s', str(payload))
            self.throttle()
            try:
                r = requests.post(notify_url, data=dumps(payload), headers=headers, auth=(self.user, self.password), verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code not in (requests.codes.created, requests.codes.ok):
                    status_str = NotifyNextcloudTalk.http_response_code_lookup(r.status_code)
                    self.logger.warning('Failed to send Nextcloud Talk notification:{}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                else:
                    self.logger.info('Sent Nextcloud Talk notification.')
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending Nextcloud Talk notification.')
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
                continue
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = self.url_parameters(*args, privacy=privacy, **kwargs)
        params.update({'+{}'.format(k): v for (k, v) in self.headers.items()})
        if self.url_prefix:
            params['url_prefix'] = self.url_prefix
        auth = '{user}:{password}@'.format(user=NotifyNextcloudTalk.quote(self.user, safe=''), password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''))
        default_port = 443 if self.secure else 80
        return '{schema}://{auth}{hostname}{port}/{targets}?{params}'.format(schema=self.secure_protocol if self.secure else self.protocol, auth=auth, hostname=self.host, port='' if self.port is None or self.port == default_port else ':{}'.format(self.port), targets='/'.join([NotifyNextcloudTalk.quote(x) for x in self.targets]), params=NotifyNextcloudTalk.urlencode(params))

    def __len__(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the number of targets associated with this notification\n        '
        targets = len(self.targets)
        return targets if targets else 1

    @staticmethod
    def parse_url(url):
        if False:
            while True:
                i = 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url)
        if not results:
            return results
        results['targets'] = NotifyNextcloudTalk.split_path(results['fullpath'])
        if 'url_prefix' in results['qsd'] and len(results['qsd']['url_prefix']):
            results['url_prefix'] = NotifyNextcloudTalk.unquote(results['qsd']['url_prefix'])
        results['headers'] = {NotifyNextcloudTalk.unquote(x): NotifyNextcloudTalk.unquote(y) for (x, y) in results['qsd+'].items()}
        return results