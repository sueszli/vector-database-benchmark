import re
import requests
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import parse_list
from ..utils import validate_regex
from ..utils import is_email
from ..utils import remove_suffix
from ..AppriseLocale import gettext_lazy as _
VALIDATE_BOTNAME = re.compile('(?P<name>[A-Z0-9_-]{1,32})', re.I)
VALIDATE_ORG = re.compile('(?P<org>[A-Z0-9_-]{1,32})(\\.(?P<hostname>[^\\s]+))?', re.I)
ZULIP_HTTP_ERROR_MAP = {401: 'Unauthorized - Invalid Token.'}
TARGET_LIST_DELIM = re.compile('[ \\t\\r\\n,#\\\\/]+')
IS_VALID_TARGET_RE = re.compile('#?(?P<stream>[A-Z0-9_]{1,32})', re.I)

class NotifyZulip(NotifyBase):
    """
    A wrapper for Zulip Notifications
    """
    service_name = 'Zulip'
    service_url = 'https://zulipchat.com/'
    secure_protocol = 'zulip'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_zulip'
    notify_url = 'https://{org}.{hostname}/api/v1/messages'
    title_maxlen = 60
    body_maxlen = 10000
    templates = ('{schema}://{botname}@{organization}/{token}', '{schema}://{botname}@{organization}/{token}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'botname': {'name': _('Bot Name'), 'type': 'string', 'regex': ('^[A-Z0-9_-]{1,32}$', 'i'), 'required': True}, 'organization': {'name': _('Organization'), 'type': 'string', 'required': True, 'regex': ('^[A-Z0-9_-]{1,32})$', 'i')}, 'token': {'name': _('Token'), 'type': 'string', 'required': True, 'private': True, 'regex': ('^[A-Z0-9]{32}$', 'i')}, 'target_user': {'name': _('Target User'), 'type': 'string', 'map_to': 'targets'}, 'target_stream': {'name': _('Target Stream'), 'type': 'string', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}})
    default_hostname = 'zulipchat.com'
    default_notification_stream = 'general'

    def __init__(self, botname, organization, token, targets=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Initialize Zulip Object\n        '
        super().__init__(**kwargs)
        self.hostname = self.default_hostname
        try:
            match = VALIDATE_BOTNAME.match(botname.strip())
            if not match:
                raise TypeError
            botname = match.group('name')
            botname = remove_suffix(botname, '-bot')
            self.botname = botname
        except (TypeError, AttributeError):
            msg = 'The Zulip botname specified ({}) is invalid.'.format(botname)
            self.logger.warning(msg)
            raise TypeError(msg)
        try:
            match = VALIDATE_ORG.match(organization.strip())
            if not match:
                raise TypeError
            self.organization = match.group('org')
            if match.group('hostname'):
                self.hostname = match.group('hostname')
        except (TypeError, AttributeError):
            msg = 'The Zulip organization specified ({}) is invalid.'.format(organization)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.token = validate_regex(token, *self.template_tokens['token']['regex'])
        if not self.token:
            msg = 'The Zulip token specified ({}) is invalid.'.format(token)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.targets = parse_list(targets)
        if len(self.targets) == 0:
            self.targets.append(self.default_notification_stream)

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            print('Hello World!')
        '\n        Perform Zulip Notification\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8'}
        has_error = False
        url = self.notify_url.format(org=self.organization, hostname=self.hostname)
        payload = {'subject': title, 'content': body}
        auth = ('{botname}-bot@{org}.{hostname}'.format(botname=self.botname, org=self.organization, hostname=self.hostname), self.token)
        targets = list(self.targets)
        while len(targets):
            target = targets.pop(0)
            result = is_email(target)
            if result:
                payload['type'] = 'private'
            else:
                payload['type'] = 'stream'
            payload['to'] = target if not result else result['full_email']
            self.logger.debug('Zulip POST URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
            self.logger.debug('Zulip Payload: %s' % str(payload))
            self.throttle()
            try:
                r = requests.post(url, data=payload, headers=headers, auth=auth, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code != requests.codes.ok:
                    status_str = NotifyZulip.http_response_code_lookup(r.status_code, ZULIP_HTTP_ERROR_MAP)
                    self.logger.warning('Failed to send Zulip notification to {}: {}{}error={}.'.format(target, status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                else:
                    self.logger.info('Sent Zulip notification to {}.'.format(target))
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending Zulip notification to {}.'.format(target))
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
                continue
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = self.url_parameters(*args, privacy=privacy, **kwargs)
        organization = '{}{}'.format(self.organization, '.{}'.format(self.hostname) if self.hostname != self.default_hostname else '')
        return '{schema}://{botname}@{org}/{token}/{targets}?{params}'.format(schema=self.secure_protocol, botname=NotifyZulip.quote(self.botname, safe=''), org=NotifyZulip.quote(organization, safe=''), token=self.pprint(self.token, privacy, safe=''), targets='/'.join([NotifyZulip.quote(x, safe='') for x in self.targets]), params=NotifyZulip.urlencode(params))

    def __len__(self):
        if False:
            print('Hello World!')
        '\n        Returns the number of targets associated with this notification\n        '
        return len(self.targets)

    @staticmethod
    def parse_url(url):
        if False:
            while True:
                i = 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['botname'] = NotifyZulip.unquote(results['user'])
        results['organization'] = NotifyZulip.unquote(results['host'])
        try:
            results['token'] = NotifyZulip.split_path(results['fullpath'])[0]
        except IndexError:
            results['token'] = None
        results['targets'] = NotifyZulip.split_path(results['fullpath'])[1:]
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += [x for x in filter(bool, TARGET_LIST_DELIM.split(NotifyZulip.unquote(results['qsd']['to'])))]
        return results