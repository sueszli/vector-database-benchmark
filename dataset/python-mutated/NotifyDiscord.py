import re
import requests
from json import dumps
from datetime import timedelta
from datetime import datetime
from datetime import timezone
from .NotifyBase import NotifyBase
from ..common import NotifyImageSize
from ..common import NotifyFormat
from ..common import NotifyType
from ..utils import parse_bool
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
from ..attachment.AttachBase import AttachBase

class NotifyDiscord(NotifyBase):
    """
    A wrapper to Discord Notifications

    """
    service_name = 'Discord'
    service_url = 'https://discord.com/'
    secure_protocol = 'discord'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_discord'
    notify_url = 'https://discord.com/api/webhooks'
    attachment_support = True
    image_size = NotifyImageSize.XY_256
    request_rate_per_sec = 0
    clock_skew = timedelta(seconds=10)
    body_maxlen = 2000
    discord_max_fields = 10
    templates = ('{schema}://{webhook_id}/{webhook_token}', '{schema}://{botname}@{webhook_id}/{webhook_token}')
    template_tokens = dict(NotifyBase.template_tokens, **{'botname': {'name': _('Bot Name'), 'type': 'string', 'map_to': 'user'}, 'webhook_id': {'name': _('Webhook ID'), 'type': 'string', 'private': True, 'required': True}, 'webhook_token': {'name': _('Webhook Token'), 'type': 'string', 'private': True, 'required': True}})
    template_args = dict(NotifyBase.template_args, **{'tts': {'name': _('Text To Speech'), 'type': 'bool', 'default': False}, 'avatar': {'name': _('Avatar Image'), 'type': 'bool', 'default': True}, 'avatar_url': {'name': _('Avatar URL'), 'type': 'string'}, 'href': {'name': _('URL'), 'type': 'string'}, 'url': {'alias_of': 'href'}, 'thread': {'name': _('Thread ID'), 'type': 'string'}, 'footer': {'name': _('Display Footer'), 'type': 'bool', 'default': False}, 'footer_logo': {'name': _('Footer Logo'), 'type': 'bool', 'default': True}, 'fields': {'name': _('Use Fields'), 'type': 'bool', 'default': True}, 'image': {'name': _('Include Image'), 'type': 'bool', 'default': False, 'map_to': 'include_image'}})

    def __init__(self, webhook_id, webhook_token, tts=False, avatar=True, footer=False, footer_logo=True, include_image=False, fields=True, avatar_url=None, href=None, thread=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Initialize Discord Object\n\n        '
        super().__init__(**kwargs)
        self.webhook_id = validate_regex(webhook_id)
        if not self.webhook_id:
            msg = 'An invalid Discord Webhook ID ({}) was specified.'.format(webhook_id)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.webhook_token = validate_regex(webhook_token)
        if not self.webhook_token:
            msg = 'An invalid Discord Webhook Token ({}) was specified.'.format(webhook_token)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.tts = tts
        self.avatar = avatar
        self.footer = footer
        self.footer_logo = footer_logo
        self.include_image = include_image
        self.fields = fields
        self.thread_id = thread
        self.avatar_url = avatar_url
        self.href = href
        self.ratelimit_reset = datetime.now(timezone.utc).replace(tzinfo=None)
        self.ratelimit_remaining = 1.0
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, attach=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Perform Discord Notification\n        '
        payload = {'tts': self.tts, 'wait': self.tts is False}
        image_url = self.image_url(notify_type)
        if self.avatar and (image_url or self.avatar_url):
            payload['avatar_url'] = self.avatar_url if self.avatar_url else image_url
        if self.user:
            payload['username'] = self.user
        params = {'thread_id': self.thread_id} if self.thread_id else None
        if body:
            fields = []
            if self.notify_format == NotifyFormat.MARKDOWN:
                payload['embeds'] = [{'author': {'name': self.app_id, 'url': self.app_url}, 'title': title, 'description': body, 'color': self.color(notify_type, int)}]
                if self.href:
                    payload['embeds'][0]['url'] = self.href
                if self.footer:
                    logo_url = self.image_url(notify_type, logo=True)
                    payload['embeds'][0]['footer'] = {'text': self.app_desc}
                    if self.footer_logo and logo_url:
                        payload['embeds'][0]['footer']['icon_url'] = logo_url
                if self.include_image and image_url:
                    payload['embeds'][0]['thumbnail'] = {'url': image_url, 'height': 256, 'width': 256}
                if self.fields:
                    (description, fields) = self.extract_markdown_sections(body)
                    payload['embeds'][0]['description'] = description
                    if fields:
                        payload['embeds'][0]['fields'] = fields[:self.discord_max_fields]
                        fields = fields[self.discord_max_fields:]
            else:
                payload['content'] = body if not title else '{}\r\n{}'.format(title, body)
            if not self._send(payload, params=params):
                return False
            if fields:
                payload['embeds'][0]['description'] = ''
                for i in range(0, len(fields), self.discord_max_fields):
                    payload['embeds'][0]['fields'] = fields[i:i + self.discord_max_fields]
                    if not self._send(payload):
                        return False
        if attach and self.attachment_support:
            payload.update({'tts': False, 'wait': True})
            if 'embeds' in payload:
                del payload['embeds']
            if 'content' in payload:
                del payload['content']
            for attachment in attach:
                self.logger.info('Posting Discord Attachment {}'.format(attachment.name))
                if not self._send(payload, params=params, attach=attachment):
                    return False
        return True

    def _send(self, payload, attach=None, params=None, rate_limit=1, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Wrapper to the requests (post) object\n        '
        headers = {'User-Agent': self.app_id}
        notify_url = '{0}/{1}/{2}'.format(self.notify_url, self.webhook_id, self.webhook_token)
        self.logger.debug('Discord POST URL: %s (cert_verify=%r)' % (notify_url, self.verify_certificate))
        self.logger.debug('Discord Payload: %s' % str(payload))
        wait = None
        if self.ratelimit_remaining <= 0.0:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            if now < self.ratelimit_reset:
                wait = abs((self.ratelimit_reset - now + self.clock_skew).total_seconds())
        self.throttle(wait=wait)
        if isinstance(attach, AttachBase):
            if not attach:
                self.logger.error('Could not access attachment {}.'.format(attach.url(privacy=True)))
                return False
            self.logger.debug('Posting Discord attachment {}'.format(attach.url(privacy=True)))
        files = None
        try:
            if attach:
                files = {'file': (attach.name, open(attach.path, 'rb'))}
            else:
                headers['Content-Type'] = 'application/json; charset=utf-8'
            r = requests.post(notify_url, params=params, data=payload if files else dumps(payload), headers=headers, files=files, verify=self.verify_certificate, timeout=self.request_timeout)
            try:
                self.ratelimit_remaining = float(r.headers.get('X-RateLimit-Remaining'))
                self.ratelimit_reset = datetime.fromtimestamp(int(r.headers.get('X-RateLimit-Reset')), timezone.utc).replace(tzinfo=None)
            except (TypeError, ValueError):
                pass
            if r.status_code not in (requests.codes.ok, requests.codes.no_content):
                status_str = NotifyBase.http_response_code_lookup(r.status_code)
                if r.status_code == requests.codes.too_many_requests and rate_limit > 0:
                    self.logger.warning('Discord rate limiting in effect; blocking for %.2f second(s)', self.ratelimit_remaining)
                    return self._send(payload=payload, attach=attach, params=params, rate_limit=rate_limit - 1, **kwargs)
                self.logger.warning('Failed to send {}to Discord notification: {}{}error={}.'.format(attach.name if attach else '', status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return False
            else:
                self.logger.info('Sent Discord {}.'.format('attachment' if attach else 'notification'))
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred posting {}to Discord.'.format(attach.name if attach else ''))
            self.logger.debug('Socket Exception: %s' % str(e))
            return False
        except (OSError, IOError) as e:
            self.logger.warning('An I/O error occurred while reading {}.'.format(attach.name if attach else 'attachment'))
            self.logger.debug('I/O Exception: %s' % str(e))
            return False
        finally:
            if files:
                files['file'][1].close()
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'tts': 'yes' if self.tts else 'no', 'avatar': 'yes' if self.avatar else 'no', 'footer': 'yes' if self.footer else 'no', 'footer_logo': 'yes' if self.footer_logo else 'no', 'image': 'yes' if self.include_image else 'no', 'fields': 'yes' if self.fields else 'no'}
        if self.avatar_url:
            params['avatar_url'] = self.avatar_url
        if self.href:
            params['href'] = self.href
        if self.thread_id:
            params['thread'] = self.thread_id
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        return '{schema}://{webhook_id}/{webhook_token}/?{params}'.format(schema=self.secure_protocol, webhook_id=self.pprint(self.webhook_id, privacy, safe=''), webhook_token=self.pprint(self.webhook_token, privacy, safe=''), params=NotifyDiscord.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            return 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        Syntax:\n          discord://webhook_id/webhook_token\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        webhook_id = NotifyDiscord.unquote(results['host'])
        try:
            webhook_token = NotifyDiscord.split_path(results['fullpath'])[0]
        except IndexError:
            webhook_token = None
        results['webhook_id'] = webhook_id
        results['webhook_token'] = webhook_token
        results['tts'] = parse_bool(results['qsd'].get('tts', False))
        results['fields'] = parse_bool(results['qsd'].get('fields', True))
        results['footer'] = parse_bool(results['qsd'].get('footer', False))
        results['footer_logo'] = parse_bool(results['qsd'].get('footer_logo', True))
        results['avatar'] = parse_bool(results['qsd'].get('avatar', True))
        results['include_image'] = parse_bool(results['qsd'].get('image', NotifyDiscord.template_args['image']['default']))
        if 'avatar_url' in results['qsd']:
            results['avatar_url'] = NotifyDiscord.unquote(results['qsd']['avatar_url'])
        if 'href' in results['qsd']:
            results['href'] = NotifyDiscord.unquote(results['qsd']['href'])
        elif 'url' in results['qsd']:
            results['href'] = NotifyDiscord.unquote(results['qsd']['url'])
            results['format'] = NotifyFormat.MARKDOWN
        if 'thread' in results['qsd']:
            results['thread'] = NotifyDiscord.unquote(results['qsd']['thread'])
            results['format'] = NotifyFormat.MARKDOWN
        return results

    @staticmethod
    def parse_native_url(url):
        if False:
            for i in range(10):
                print('nop')
        '\n        Support https://discord.com/api/webhooks/WEBHOOK_ID/WEBHOOK_TOKEN\n        Support Legacy URL as well:\n            https://discordapp.com/api/webhooks/WEBHOOK_ID/WEBHOOK_TOKEN\n        '
        result = re.match('^https?://discord(app)?\\.com/api/webhooks/(?P<webhook_id>[0-9]+)/(?P<webhook_token>[A-Z0-9_-]+)/?(?P<params>\\?.+)?$', url, re.I)
        if result:
            return NotifyDiscord.parse_url('{schema}://{webhook_id}/{webhook_token}/{params}'.format(schema=NotifyDiscord.secure_protocol, webhook_id=result.group('webhook_id'), webhook_token=result.group('webhook_token'), params='' if not result.group('params') else result.group('params')))
        return None

    @staticmethod
    def extract_markdown_sections(markdown):
        if False:
            print('Hello World!')
        '\n        Takes a string in a markdown type format and extracts\n        the headers and their corresponding sections into individual\n        fields that get passed as an embed entry to Discord.\n\n        '
        match = re.match('^\\s*(?P<desc>[^\\s#]+.*?)(?=\\s*$|[\\r\\n]+\\s*#)', markdown, flags=re.S)
        description = match.group('desc').strip() if match else ''
        if description:
            markdown = re.sub(re.escape(description), '', markdown, count=1)
        regex = re.compile('\\s*#[# \\t\\v]*(?P<name>[^\\n]+)(\\n|\\s*$)\\s*((?P<value>[^#].+?)(?=\\s*$|[\\r\\n]+\\s*#))?', flags=re.S)
        common = regex.finditer(markdown)
        fields = list()
        for el in common:
            d = el.groupdict()
            fields.append({'name': d.get('name', '').strip('#`* \r\n\t\x0b'), 'value': '```{}\n{}```'.format('md' if d.get('value') else '', d.get('value').strip() + '\n' if d.get('value') else '')})
        return (description, fields)