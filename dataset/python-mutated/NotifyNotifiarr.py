import re
import requests
from json import dumps
from itertools import chain
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..AppriseLocale import gettext_lazy as _
from ..common import NotifyImageSize
from ..utils import parse_list, parse_bool
from ..utils import validate_regex
CHANNEL_LIST_DELIM = re.compile('[ \\t\\r\\n,#\\\\/]+')
CHANNEL_REGEX = re.compile('^\\s*(\\#|\\%35)?(?P<channel>[0-9]+)', re.I)

class NotifyNotifiarr(NotifyBase):
    """
    A wrapper for Notifiarr Notifications
    """
    service_name = 'Notifiarr'
    service_url = 'https://notifiarr.com/'
    secure_protocol = 'notifiarr'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_notifiarr'
    notify_url = 'https://notifiarr.com/api/v1/notification/apprise'
    request_rate_per_sec = 0.04
    image_size = NotifyImageSize.XY_256
    templates = ('{schema}://{apikey}/{targets}',)
    template_tokens = dict(NotifyBase.template_tokens, **{'apikey': {'name': _('Token'), 'type': 'string', 'required': True, 'private': True}, 'target_channel': {'name': _('Target Channel'), 'type': 'string', 'prefix': '#', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string', 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'key': {'alias_of': 'apikey'}, 'apikey': {'alias_of': 'apikey'}, 'discord_user': {'name': _('Ping Discord User'), 'type': 'int'}, 'discord_role': {'name': _('Ping Discord Role'), 'type': 'int'}, 'event': {'name': _('Discord Event ID'), 'type': 'int'}, 'image': {'name': _('Include Image'), 'type': 'bool', 'default': False, 'map_to': 'include_image'}, 'source': {'name': _('Source'), 'type': 'string'}, 'from': {'alias_of': 'source'}, 'to': {'alias_of': 'targets'}})

    def __init__(self, apikey=None, include_image=None, discord_user=None, discord_role=None, event=None, targets=None, source=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Initialize Notifiarr Object\n\n        headers can be a dictionary of key/value pairs that you want to\n        additionally include as part of the server headers to post with\n\n        '
        super().__init__(**kwargs)
        self.apikey = apikey
        if not self.apikey:
            msg = 'An invalid Notifiarr APIKey ({}) was specified.'.format(apikey)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.include_image = include_image if isinstance(include_image, bool) else self.template_args['image']['default']
        self.discord_user = 0
        if discord_user:
            try:
                self.discord_user = int(discord_user)
            except (ValueError, TypeError):
                msg = 'An invalid Notifiarr User ID ({}) was specified.'.format(discord_user)
                self.logger.warning(msg)
                raise TypeError(msg)
        self.discord_role = 0
        if discord_role:
            try:
                self.discord_role = int(discord_role)
            except (ValueError, TypeError):
                msg = 'An invalid Notifiarr Role ID ({}) was specified.'.format(discord_role)
                self.logger.warning(msg)
                raise TypeError(msg)
        self.source = validate_regex(source)
        self.event = 0
        if event:
            try:
                self.event = int(event)
            except (ValueError, TypeError):
                msg = 'An invalid Notifiarr Discord Event ID ({}) was specified.'.format(event)
                self.logger.warning(msg)
                raise TypeError(msg)
        self.targets = {'channels': [], 'invalid': []}
        for target in parse_list(targets):
            result = CHANNEL_REGEX.match(target)
            if result:
                self.targets['channels'].append(int(result.group('channel')))
                continue
            self.logger.warning('Dropped invalid channel ({}) specified.'.format(target))
            self.targets['invalid'].append(target)
        return

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'image': 'yes' if self.include_image else 'no'}
        if self.source:
            params['source'] = self.source
        if self.discord_user:
            params['discord_user'] = self.discord_user
        if self.discord_role:
            params['discord_role'] = self.discord_role
        if self.event:
            params['event'] = self.event
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        return '{schema}://{apikey}/{targets}?{params}'.format(schema=self.secure_protocol, apikey=self.pprint(self.apikey, privacy, safe=''), targets='/'.join([NotifyNotifiarr.quote(x, safe='+#@') for x in chain(['#{}'.format(x) for x in self.targets['channels']], self.targets['invalid'])]), params=NotifyNotifiarr.urlencode(params))

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Perform Notifiarr Notification\n        '
        if not self.targets['channels']:
            self.logger.warning('There were no Notifiarr channels to notify.')
            return False
        has_error = False
        image_url = self.image_url(notify_type)
        for (idx, channel) in enumerate(self.targets['channels']):
            payload = {'source': self.source if self.source else self.app_id, 'type': notify_type, 'notification': {'update': True if self.event else False, 'name': self.app_id, 'event': str(self.event) if self.event else ''}, 'discord': {'color': self.color(notify_type), 'ping': {'pingUser': self.discord_user if not idx and self.discord_user else 0, 'pingRole': self.discord_role if not idx and self.discord_role else 0}, 'text': {'title': title, 'content': '', 'description': body, 'footer': self.app_desc}, 'ids': {'channel': channel}}}
            if self.include_image and image_url:
                payload['discord']['text']['icon'] = image_url
                payload['discord']['images'] = {'thumbnail': image_url}
            if not self._send(payload):
                has_error = True
        return not has_error

    def _send(self, payload):
        if False:
            print('Hello World!')
        '\n        Send notification\n        '
        self.logger.debug('Notifiarr POST URL: %s (cert_verify=%r)' % (self.notify_url, self.verify_certificate))
        self.logger.debug('Notifiarr Payload: %s' % str(payload))
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json', 'Accept': 'text/plain', 'X-api-Key': self.apikey}
        self.throttle()
        try:
            r = requests.post(self.notify_url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code < 200 or r.status_code >= 300:
                status_str = NotifyNotifiarr.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send Notifiarr %s notification: %serror=%s.', status_str, ', ' if status_str else '', str(r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent Notifiarr notification.')
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending Notifiarr Chat notification to %s.' % self.host)
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        return True

    def __len__(self):
        if False:
            print('Hello World!')
        '\n        Returns the number of targets associated with this notification\n        '
        targets = len(self.targets['channels']) + len(self.targets['invalid'])
        return targets if targets > 0 else 1

    @staticmethod
    def parse_url(url):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['targets'] = NotifyNotifiarr.split_path(results['fullpath'])
        if 'discord_user' in results['qsd'] and len(results['qsd']['discord_user']):
            results['discord_user'] = NotifyNotifiarr.unquote(results['qsd']['discord_user'])
        if 'discord_role' in results['qsd'] and len(results['qsd']['discord_role']):
            results['discord_role'] = NotifyNotifiarr.unquote(results['qsd']['discord_role'])
        if 'event' in results['qsd'] and len(results['qsd']['event']):
            results['event'] = NotifyNotifiarr.unquote(results['qsd']['event'])
        results['include_image'] = parse_bool(results['qsd'].get('image', False))
        host_is_potential_target = False
        if 'source' in results['qsd'] and len(results['qsd']['source']):
            results['source'] = NotifyNotifiarr.unquote(results['qsd']['source'])
        elif 'from' in results['qsd'] and len(results['qsd']['from']):
            results['source'] = NotifyNotifiarr.unquote(results['qsd']['from'])
        if 'apikey' in results['qsd'] and len(results['qsd']['apikey']):
            results['apikey'] = NotifyNotifiarr.unquote(results['qsd']['apikey'])
            host_is_potential_target = True
        elif 'key' in results['qsd'] and len(results['qsd']['key']):
            results['apikey'] = NotifyNotifiarr.unquote(results['qsd']['key'])
            host_is_potential_target = True
        else:
            results['apikey'] = NotifyNotifiarr.unquote(results['host'])
        if host_is_potential_target is True and results['host']:
            results['targets'].append(NotifyNotifiarr.unquote(results['host']))
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += [x for x in filter(bool, CHANNEL_LIST_DELIM.split(NotifyNotifiarr.unquote(results['qsd']['to'])))]
        return results