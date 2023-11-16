import re
import requests
from json import loads
from itertools import chain
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyFormat
from ..common import NotifyType
from ..utils import parse_list
from ..utils import is_email
from ..AppriseLocale import gettext_lazy as _
IS_CHANNEL = re.compile('^#?(?P<name>((?P<workspace>[A-Za-z0-9_-]+):)?(?P<channel>[^\\s]{1,64}))$')
IS_CHANNEL_ID = re.compile('^(?P<name>((?P<workspace>[0-9]+):)?(?P<channel>[0-9]+))$')
LIST_DELIM = re.compile('[ \\t\\r\\n,\\\\/]+')

class NotifyTwist(NotifyBase):
    """
    A wrapper for Notify Twist Notifications
    """
    service_name = 'Twist'
    service_url = 'https://twist.com'
    secure_protocol = 'twist'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_twist'
    body_maxlen = 1000
    notify_format = NotifyFormat.MARKDOWN
    api_url = 'https://api.twist.com/api/v3/'
    request_rate_per_sec = 0.2
    default_notification_channel = 'general'
    templates = ('{schema}://{password}:{email}', '{schema}://{password}:{email}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'password': {'name': _('Password'), 'type': 'string', 'private': True, 'required': True}, 'email': {'name': _('Email'), 'type': 'string', 'required': True}, 'target_channel': {'name': _('Target Channel'), 'type': 'string', 'prefix': '#', 'map_to': 'targets'}, 'target_channel_id': {'name': _('Target Channel ID'), 'type': 'string', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}})

    def __init__(self, email=None, targets=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize Notify Twist Object\n        '
        super().__init__(**kwargs)
        self.channels = set()
        self.channel_ids = set()
        self.token = None
        self.default_workspace = None
        self._cached_workspaces = set()
        self._cached_channels = dict()
        self.email = email if email else '{}@{}'.format(self.user, self.host)
        result = is_email(self.email)
        if not result:
            msg = 'The Twist Auth email specified ({}) is invalid.'.format(self.email)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.email = result['full_email']
        if email:
            self.user = result['user']
            self.host = result['domain']
        if not self.password:
            msg = 'No Twist password was specified with account: {}'.format(self.email)
            self.logger.warning(msg)
            raise TypeError(msg)
        for recipient in parse_list(targets):
            result = IS_CHANNEL_ID.match(recipient)
            if result:
                self.channel_ids.add(result.group('name'))
                continue
            result = IS_CHANNEL.match(recipient)
            if result:
                self.channels.add(result.group('name').lower())
                continue
            self.logger.warning('Dropped invalid channel/id ({}) specified.'.format(recipient))
        if len(self.channels) + len(self.channel_ids) == 0:
            self.channels.add(self.default_notification_channel)
            self.logger.warning('Added default notification channel {}'.format(self.default_notification_channel))
        return

    def url(self, privacy=False, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = self.url_parameters(*args, privacy=privacy, **kwargs)
        return '{schema}://{password}:{user}@{host}/{targets}/?{params}'.format(schema=self.secure_protocol, password=self.pprint(self.password, privacy, mode=PrivacyMode.Secret, safe=''), user=self.quote(self.user, safe=''), host=self.host, targets='/'.join([NotifyTwist.quote(x, safe='') for x in chain(['#{}'.format(x) for x in self.channels], self.channel_ids)]), params=NotifyTwist.urlencode(params))

    def __len__(self):
        if False:
            print('Hello World!')
        '\n        Returns the number of targets associated with this notification\n        '
        return len(self.channels) + len(self.channel_ids)

    def login(self):
        if False:
            print('Hello World!')
        '\n        A simple wrapper to authenticate with the Twist Server\n        '
        payload = {'email': self.email, 'password': self.password}
        self.default_workspace = None
        self._cached_workspaces = set()
        self._cached_channels = dict()
        (postokay, response) = self._fetch('users/login', payload=payload, login=True)
        if not postokay or not response:
            self.token = False
            return False
        self.default_workspace = response.get('default_workspace')
        self.token = response.get('token')
        self.logger.info('Authenticated to Twist as {}'.format(self.email))
        return True

    def logout(self):
        if False:
            print('Hello World!')
        '\n        A simple wrapper to log out of the server\n        '
        if not self.token:
            return True
        (postokay, response) = self._fetch('users/logout')
        self.token = None
        return True

    def get_workspaces(self):
        if False:
            print('Hello World!')
        "\n        Returns all workspaces associated with this user account as a set\n\n        This returned object is either an empty dictionary or one that\n        looks like this:\n           {\n             'workspace': <workspace_id>,\n             'workspace': <workspace_id>,\n             'workspace': <workspace_id>,\n           }\n\n        All workspaces are made lowercase for comparison purposes\n        "
        if not self.token and (not self.login()):
            return dict()
        (postokay, response) = self._fetch('workspaces/get')
        if not postokay or not response:
            return dict()
        result = {}
        for entry in response:
            result[entry.get('name', '').lower()] = entry.get('id', '')
        return result

    def get_channels(self, wid):
        if False:
            print('Hello World!')
        "\n        Simply returns the channel objects associated with the specified\n        workspace id.\n\n        This returned object is either an empty dictionary or one that\n        looks like this:\n           {\n             'channel1': <channel_id>,\n             'channel2': <channel_id>,\n             'channel3': <channel_id>,\n           }\n\n        All channels are made lowercase for comparison purposes\n        "
        if not self.token and (not self.login()):
            return {}
        payload = {'workspace_id': wid}
        (postokay, response) = self._fetch('channels/get', payload=payload)
        if not postokay or not isinstance(response, list):
            return {}
        result = {}
        for entry in response:
            result[entry.get('name', '').lower()] = entry.get('id', '')
        return result

    def _channel_migration(self):
        if False:
            i = 10
            return i + 15
        '\n        A simple wrapper to get all of the current workspaces including\n        the default one.  This plays a role in what channel(s) get notified\n        and where.\n\n        A cache lookup has overhead, and is only required to be preformed\n        if the user specified channels by their string value\n        '
        if not self.token and (not self.login()):
            return False
        if not len(self.channels):
            return True
        if self.default_workspace and self.default_workspace not in self._cached_channels:
            self._cached_channels[self.default_workspace] = self.get_channels(self.default_workspace)
        has_error = False
        while len(self.channels):
            result = IS_CHANNEL.match(self.channels.pop())
            workspace = result.group('workspace')
            channel = result.group('channel').lower()
            if workspace:
                workspace = workspace.lower()
                if not len(self._cached_workspaces):
                    self._cached_workspaces = self.get_workspaces()
                if workspace not in self._cached_workspaces:
                    self.logger.warning('The Twist User {} is not associated with the Team {}'.format(self.email, workspace))
                    has_error = True
                    continue
                workspace_id = self._cached_workspaces[workspace]
            else:
                workspace_id = self.default_workspace
            if workspace_id in self._cached_channels and channel in self._cached_channels[workspace_id]:
                self.channel_ids.add('{}:{}'.format(workspace_id, self._cached_channels[workspace_id][channel]))
                continue
            self.logger.warning('The Channel #{} was not found{}.'.format(channel, '' if not workspace else ' with Team {}'.format(workspace)))
            has_error = True
            continue
        return not has_error

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Perform Twist Notification\n        '
        has_error = False
        if not self.token and (not self.login()):
            return False
        if len(self.channels) > 0:
            self._channel_migration()
        if not len(self.channel_ids):
            self.logger.warning('There are no Twist targets to notify')
            return False
        ids = list(self.channel_ids)
        while len(ids) > 0:
            result = IS_CHANNEL_ID.match(ids.pop())
            channel_id = int(result.group('channel'))
            payload = {'channel_id': channel_id, 'title': title, 'content': body}
            (postokay, response) = self._fetch('threads/add', payload=payload)
            if not postokay:
                has_error = True
                continue
            self.logger.info('Sent Twist notification to {}.'.format(result.group('name')))
        return not has_error

    def _fetch(self, url, payload=None, method='POST', login=False):
        if False:
            i = 10
            return i + 15
        '\n        Wrapper to Twist API requests object\n        '
        headers = {'User-Agent': self.app_id}
        headers['Content-Type'] = 'application/x-www-form-urlencoded; charset=utf-8'
        if self.token:
            headers['Authorization'] = 'Bearer {}'.format(self.token)
        api_url = '{}{}'.format(self.api_url, url)
        self.logger.debug('Twist {} URL: {} (cert_verify={})'.format(method, api_url, self.verify_certificate))
        self.logger.debug('Twist Payload: %s' % str(payload))
        self.throttle()
        content = {}
        fn = requests.post if method == 'POST' else requests.get
        try:
            r = fn(api_url, data=payload, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            try:
                content = loads(r.content)
            except (TypeError, ValueError, AttributeError):
                content = {}
            if r.status_code != requests.codes.ok and login is False and isinstance(content, dict) and (content.get('error_code') in (120, 200)):
                if self.login():
                    r = fn(api_url, data=payload, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                    try:
                        content = loads(r.content)
                    except (TypeError, ValueError, AttributeError):
                        content = {}
            if r.status_code != requests.codes.ok:
                status_str = NotifyTwist.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send Twist {} to {}: {}error={}.'.format(method, api_url, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return (False, content)
        except requests.RequestException as e:
            self.logger.warning('Exception received when sending Twist {} to {}: '.format(method, api_url))
            self.logger.debug('Socket Exception: %s' % str(e))
            return (False, content)
        return (True, content)

    @staticmethod
    def parse_url(url):
        if False:
            i = 10
            return i + 15
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url)
        if not results:
            return results
        if not results.get('user'):
            return None
        results['targets'] = NotifyTwist.split_path(results['fullpath'])
        if not results.get('password'):
            if len(results['targets']) == 0:
                return None
            results['password'] = NotifyTwist.quote(results['targets'].pop(0), safe='')
        else:
            _password = results['user']
            results['user'] = results['password']
            results['password'] = _password
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyTwist.parse_list(results['qsd']['to'])
        return results

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Destructor\n        '
        try:
            self.logout()
        except LookupError:
            pass
        except ImportError:
            pass