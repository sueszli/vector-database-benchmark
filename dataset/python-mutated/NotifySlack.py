import re
import requests
from json import dumps
from json import loads
from time import time
from .NotifyBase import NotifyBase
from ..common import NotifyImageSize
from ..common import NotifyType
from ..common import NotifyFormat
from ..utils import is_email
from ..utils import parse_bool
from ..utils import parse_list
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
SLACK_HTTP_ERROR_MAP = {401: 'Unauthorized - Invalid Token.'}
CHANNEL_LIST_DELIM = re.compile('[ \\t\\r\\n,#\\\\/]+')

class SlackMode:
    """
    Tracks the mode of which we're using Slack
    """
    WEBHOOK = 'webhook'
    BOT = 'bot'
SLACK_MODES = (SlackMode.WEBHOOK, SlackMode.BOT)

class NotifySlack(NotifyBase):
    """
    A wrapper for Slack Notifications
    """
    service_name = 'Slack'
    service_url = 'https://slack.com/'
    secure_protocol = 'slack'
    request_rate_per_sec = 1.2
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_slack'
    attachment_support = True
    webhook_url = 'https://hooks.slack.com/services'
    api_url = 'https://slack.com/api/{}'
    image_size = NotifyImageSize.XY_72
    body_maxlen = 35000
    notify_format = NotifyFormat.MARKDOWN
    default_notification_channel = '#general'
    templates = ('{schema}://{token_a}/{token_b}/{token_c}', '{schema}://{botname}@{token_a}/{token_b}{token_c}', '{schema}://{token_a}/{token_b}/{token_c}/{targets}', '{schema}://{botname}@{token_a}/{token_b}/{token_c}/{targets}', '{schema}://{access_token}/', '{schema}://{access_token}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'botname': {'name': _('Bot Name'), 'type': 'string', 'map_to': 'user'}, 'access_token': {'name': _('OAuth Access Token'), 'type': 'string', 'private': True, 'required': True, 'regex': ('^xox[abp]-[A-Z0-9-]+$', 'i')}, 'token_a': {'name': _('Token A'), 'type': 'string', 'private': True, 'regex': ('^[A-Z0-9]+$', 'i')}, 'token_b': {'name': _('Token B'), 'type': 'string', 'private': True, 'regex': ('^[A-Z0-9]+$', 'i')}, 'token_c': {'name': _('Token C'), 'type': 'string', 'private': True, 'regex': ('^[A-Za-z0-9]+$', 'i')}, 'target_encoded_id': {'name': _('Target Encoded ID'), 'type': 'string', 'prefix': '+', 'map_to': 'targets'}, 'target_email': {'name': _('Target Email'), 'type': 'string', 'map_to': 'targets'}, 'target_user': {'name': _('Target User'), 'type': 'string', 'prefix': '@', 'map_to': 'targets'}, 'target_channels': {'name': _('Target Channel'), 'type': 'string', 'prefix': '#', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'image': {'name': _('Include Image'), 'type': 'bool', 'default': True, 'map_to': 'include_image'}, 'footer': {'name': _('Include Footer'), 'type': 'bool', 'default': True, 'map_to': 'include_footer'}, 'blocks': {'name': _('Use Blocks'), 'type': 'bool', 'default': False, 'map_to': 'use_blocks'}, 'to': {'alias_of': 'targets'}, 'token': {'name': _('Token'), 'alias_of': ('access_token', 'token_a', 'token_b', 'token_c')}})

    def __init__(self, access_token=None, token_a=None, token_b=None, token_c=None, targets=None, include_image=True, include_footer=True, use_blocks=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Initialize Slack Object\n        '
        super().__init__(**kwargs)
        self.mode = SlackMode.BOT if access_token else SlackMode.WEBHOOK
        if self.mode is SlackMode.WEBHOOK:
            self.token_a = validate_regex(token_a, *self.template_tokens['token_a']['regex'])
            if not self.token_a:
                msg = 'An invalid Slack (first) Token ({}) was specified.'.format(token_a)
                self.logger.warning(msg)
                raise TypeError(msg)
            self.token_b = validate_regex(token_b, *self.template_tokens['token_b']['regex'])
            if not self.token_b:
                msg = 'An invalid Slack (second) Token ({}) was specified.'.format(token_b)
                self.logger.warning(msg)
                raise TypeError(msg)
            self.token_c = validate_regex(token_c, *self.template_tokens['token_c']['regex'])
            if not self.token_c:
                msg = 'An invalid Slack (third) Token ({}) was specified.'.format(token_c)
                self.logger.warning(msg)
                raise TypeError(msg)
        else:
            self.access_token = validate_regex(access_token, *self.template_tokens['access_token']['regex'])
            if not self.access_token:
                msg = 'An invalid Slack OAuth Access Token ({}) was specified.'.format(access_token)
                self.logger.warning(msg)
                raise TypeError(msg)
        self._lookup_users = {}
        self.use_blocks = parse_bool(use_blocks, self.template_args['blocks']['default']) if use_blocks is not None else self.template_args['blocks']['default']
        self.channels = parse_list(targets)
        if len(self.channels) == 0:
            self.channels.append(None if self.mode is SlackMode.WEBHOOK else self.default_notification_channel)
        self._re_formatting_map = {'\\r\\*\\n': '\\n', '&': '&amp;', '<': '&lt;', '>': '&gt;'}
        self._re_channel_support = re.compile('(?P<match>(?:<|\\&lt;)?[ \\t]*!(?P<channel>[^| \\n]+)(?:[ \\t]*\\|[ \\t]*(?:(?P<val>[^\\n]+?)[ \\t]*)?(?:>|\\&gt;)|(?:>|\\&gt;)))', re.IGNORECASE)
        self._re_url_support = re.compile('(?P<match>(?:<|\\&lt;)?[ \\t]*(?P<url>(?:https?|mailto)://[^| \\n]+)(?:[ \\t]*\\|[ \\t]*(?:(?P<val>[^\\n]+?)[ \\t]*)?(?:>|\\&gt;)|(?:>|\\&gt;)))', re.IGNORECASE)
        self._re_formatting_rules = re.compile('(' + '|'.join(self._re_formatting_map.keys()) + ')', re.IGNORECASE)
        self.include_image = include_image
        self.include_footer = include_footer
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, attach=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Perform Slack Notification\n        '
        has_error = False
        if self.use_blocks:
            _slack_format = 'mrkdwn' if self.notify_format == NotifyFormat.MARKDOWN else 'plain_text'
            payload = {'username': self.user if self.user else self.app_id, 'attachments': [{'blocks': [{'type': 'section', 'text': {'type': _slack_format, 'text': body}}], 'color': self.color(notify_type)}]}
            if title:
                payload['attachments'][0]['blocks'].insert(0, {'type': 'header', 'text': {'type': 'plain_text', 'text': title, 'emoji': True}})
            if self.include_footer:
                image_url = None if not self.include_image else self.image_url(notify_type)
                _footer = {'type': 'context', 'elements': [{'type': _slack_format, 'text': self.app_id}]}
                if image_url:
                    payload['icon_url'] = image_url
                    _footer['elements'].insert(0, {'type': 'image', 'image_url': image_url, 'alt_text': notify_type})
                payload['attachments'][0]['blocks'].append(_footer)
        else:
            if self.notify_format == NotifyFormat.MARKDOWN:
                body = self._re_formatting_rules.sub(lambda x: self._re_formatting_map[x.group()], body)
                for match in self._re_channel_support.findall(body):
                    channel = match[1].strip()
                    desc = match[2].strip()
                    body = re.sub(re.escape(match[0]), '<!{channel}|{desc}>'.format(channel=channel, desc=desc) if desc else '<!{channel}>'.format(channel=channel), body, re.IGNORECASE)
                for match in self._re_url_support.findall(body):
                    url = match[1].replace('&amp;', '&')
                    desc = match[2].strip()
                    body = re.sub(re.escape(match[0]), '<{url}|{desc}>'.format(url=url, desc=desc) if desc else '<{url}>'.format(url=url), body, re.IGNORECASE)
            title = self._re_formatting_rules.sub(lambda x: self._re_formatting_map[x.group()], title)
            payload = {'username': self.user if self.user else self.app_id, 'mrkdwn': self.notify_format == NotifyFormat.MARKDOWN, 'attachments': [{'title': title, 'text': body, 'color': self.color(notify_type), 'ts': time()}]}
            image_url = None if not self.include_image else self.image_url(notify_type)
            if image_url:
                payload['icon_url'] = image_url
            if self.include_footer:
                if image_url:
                    payload['attachments'][0]['footer_icon'] = image_url
                payload['attachments'][0]['footer'] = self.app_id
        if attach and self.attachment_support and (self.mode is SlackMode.WEBHOOK):
            self.logger.warning('Slack Webhooks do not support attachments.')
        if self.mode is SlackMode.WEBHOOK:
            url = '{}/{}/{}/{}'.format(self.webhook_url, self.token_a, self.token_b, self.token_c)
        else:
            url = self.api_url.format('chat.postMessage')
        channels = list(self.channels)
        attach_channel_list = []
        while len(channels):
            channel = channels.pop(0)
            if channel is not None:
                channel = validate_regex(channel, '[+#@]?[A-Z0-9_]{1,32}')
                if not channel:
                    self.logger.warning('The specified target {} is invalid;skipping.'.format(channel))
                    has_error = True
                    continue
                if channel[0] == '+':
                    payload['channel'] = channel[1:]
                elif channel[0] == '@':
                    payload['channel'] = channel
                else:
                    email = is_email(channel)
                    if email:
                        payload['channel'] = self.lookup_userid(email['full_email'])
                        if not payload['channel']:
                            has_error = True
                            continue
                    else:
                        payload['channel'] = channel if channel[0] == '#' else '#{}'.format(channel)
                attach_channel_list.append(payload['channel'])
            response = self._send(url, payload)
            if not response:
                has_error = True
                continue
            self.logger.info('Sent Slack notification{}.'.format(' to {}'.format(channel) if channel is not None else ''))
        if attach and self.attachment_support and (self.mode is SlackMode.BOT) and attach_channel_list:
            for attachment in attach:
                if not attachment:
                    self.logger.error('Could not access attachment {}.'.format(attachment.url(privacy=True)))
                    return False
                self.logger.debug('Posting Slack attachment {}'.format(attachment.url(privacy=True)))
                _payload = {'filename': attachment.name, 'channels': ','.join(attach_channel_list)}
                _url = self.api_url.format('files.upload')
                response = self._send(_url, _payload, attach=attachment)
                if not (response and response.get('file') and response['file'].get('url_private')):
                    return False
        return not has_error

    def lookup_userid(self, email):
        if False:
            print('Hello World!')
        "\n        Takes an email address and attempts to resolve/acquire it's user\n        id for notification purposes.\n        "
        if email in self._lookup_users:
            return self._lookup_users[email]
        if self.mode is not SlackMode.BOT:
            self.logger.warning('Emails can not be resolved to Slack User IDs unless you have a bot configured.')
            return None
        lookup_url = self.api_url.format('users.lookupByEmail')
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/x-www-form-urlencoded', 'Authorization': 'Bearer {}'.format(self.access_token)}
        params = {'email': email}
        self.logger.debug('Slack User Lookup POST URL: %s (cert_verify=%r)' % (lookup_url, self.verify_certificate))
        self.logger.debug('Slack User Lookup Parameters: %s' % str(params))
        response = {'ok': False}
        user_id = None
        self.throttle()
        try:
            r = requests.get(lookup_url, headers=headers, params=params, verify=self.verify_certificate, timeout=self.request_timeout)
            try:
                response = loads(r.content)
            except (AttributeError, TypeError, ValueError):
                pass
            if r.status_code != requests.codes.ok or not (response and response.get('ok', False)):
                status_str = NotifySlack.http_response_code_lookup(r.status_code, SLACK_HTTP_ERROR_MAP)
                self.logger.warning('Failed to send Slack User Lookup:{}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            user_id = response['user']['id']
            self._lookup_users[email] = user_id
            self.logger.info('Email %s resolves to the Slack User ID: %s.', email, user_id)
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred looking up Slack User.')
            self.logger.debug('Socket Exception: %s' % str(e))
            return None
        return user_id

    def _send(self, url, payload, attach=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Wrapper to the requests (post) object\n        '
        self.logger.debug('Slack POST URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
        self.logger.debug('Slack Payload: %s' % str(payload))
        headers = {'User-Agent': self.app_id, 'Accept': 'application/json'}
        if not attach:
            headers['Content-Type'] = 'application/json; charset=utf-8'
        if self.mode is SlackMode.BOT:
            headers['Authorization'] = 'Bearer {}'.format(self.access_token)
        response = {'ok': False}
        self.throttle()
        files = None
        try:
            if attach:
                files = {'file': (attach.name, open(attach.path, 'rb'))}
            r = requests.post(url, data=payload if attach else dumps(payload), headers=headers, files=files, verify=self.verify_certificate, timeout=self.request_timeout)
            try:
                response = loads(r.content)
            except (AttributeError, TypeError, ValueError):
                pass
            status_okay = response and response.get('ok', False) if self.mode is SlackMode.BOT else r.content == b'ok'
            if r.status_code != requests.codes.ok or not status_okay:
                status_str = NotifySlack.http_response_code_lookup(r.status_code, SLACK_HTTP_ERROR_MAP)
                self.logger.warning('Failed to send {}to Slack: {}{}error={}.'.format(attach.name if attach else '', status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred posting {}to Slack.'.format(attach.name if attach else ''))
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        except (OSError, IOError) as e:
            self.logger.warning('An I/O error occurred while reading {}.'.format(attach.name if attach else 'attachment'))
            self.logger.debug('I/O Exception: %s' % str(e))
            return False
        finally:
            if files:
                files['file'][1].close()
        return response

    def url(self, privacy=False, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'image': 'yes' if self.include_image else 'no', 'footer': 'yes' if self.include_footer else 'no', 'blocks': 'yes' if self.use_blocks else 'no'}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        botname = ''
        if self.user:
            botname = '{botname}@'.format(botname=NotifySlack.quote(self.user, safe=''))
        if self.mode == SlackMode.WEBHOOK:
            return '{schema}://{botname}{token_a}/{token_b}/{token_c}/{targets}/?{params}'.format(schema=self.secure_protocol, botname=botname, token_a=self.pprint(self.token_a, privacy, safe=''), token_b=self.pprint(self.token_b, privacy, safe=''), token_c=self.pprint(self.token_c, privacy, safe=''), targets='/'.join([NotifySlack.quote(x, safe='') for x in self.channels]), params=NotifySlack.urlencode(params))
        return '{schema}://{botname}{access_token}/{targets}/?{params}'.format(schema=self.secure_protocol, botname=botname, access_token=self.pprint(self.access_token, privacy, safe=''), targets='/'.join([NotifySlack.quote(x, safe='') for x in self.channels]), params=NotifySlack.urlencode(params))

    def __len__(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the number of targets associated with this notification\n        '
        return len(self.channels)

    @staticmethod
    def parse_url(url):
        if False:
            i = 10
            return i + 15
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        token = NotifySlack.unquote(results['host'])
        entries = NotifySlack.split_path(results['fullpath'])
        if token.startswith('xo'):
            results['access_token'] = token
        else:
            results['token_a'] = token
            results['token_b'] = entries.pop(0) if entries else None
            results['token_c'] = entries.pop(0) if entries else None
        results['targets'] = entries
        if 'token' in results['qsd'] and len(results['qsd']['token']):
            entries = [x for x in filter(bool, CHANNEL_LIST_DELIM.split(NotifySlack.unquote(results['qsd']['token'])))]
            if entries and entries[0].startswith('xo'):
                results['access_token'] = entries[0]
                results['token_a'] = None
                results['token_b'] = None
                results['token_c'] = None
            else:
                results['access_token'] = None
                results['token_a'] = entries.pop(0) if entries else None
                results['token_b'] = entries.pop(0) if entries else None
                results['token_c'] = entries.pop(0) if entries else None
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += [x for x in filter(bool, CHANNEL_LIST_DELIM.split(NotifySlack.unquote(results['qsd']['to'])))]
        results['include_image'] = parse_bool(results['qsd'].get('image', True))
        if 'blocks' in results['qsd'] and len(results['qsd']['blocks']):
            results['use_blocks'] = parse_bool(results['qsd']['blocks'])
        results['include_footer'] = parse_bool(results['qsd'].get('footer', True))
        return results

    @staticmethod
    def parse_native_url(url):
        if False:
            return 10
        '\n        Support https://hooks.slack.com/services/TOKEN_A/TOKEN_B/TOKEN_C\n        '
        result = re.match('^https?://hooks\\.slack\\.com/services/(?P<token_a>[A-Z0-9]+)/(?P<token_b>[A-Z0-9]+)/(?P<token_c>[A-Z0-9]+)/?(?P<params>\\?.+)?$', url, re.I)
        if result:
            return NotifySlack.parse_url('{schema}://{token_a}/{token_b}/{token_c}/{params}'.format(schema=NotifySlack.secure_protocol, token_a=result.group('token_a'), token_b=result.group('token_b'), token_c=result.group('token_c'), params='' if not result.group('params') else result.group('params')))
        return None