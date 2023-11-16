import requests
import re
import os
from json import loads
from json import dumps
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..common import NotifyImageSize
from ..common import NotifyFormat
from ..utils import parse_bool
from ..utils import parse_list
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
from ..attachment.AttachBase import AttachBase
TELEGRAM_IMAGE_XY = NotifyImageSize.XY_256
IS_CHAT_ID_RE = re.compile('^(@*(?P<idno>-?[0-9]{1,32})|(?P<name>[a-z_-][a-z0-9_-]+))$', re.IGNORECASE)

class TelegramContentPlacement:
    """
    The Telegram Content Placement
    """
    BEFORE = 'before'
    AFTER = 'after'
TELEGRAM_CONTENT_PLACEMENT = (TelegramContentPlacement.BEFORE, TelegramContentPlacement.AFTER)

class NotifyTelegram(NotifyBase):
    """
    A wrapper for Telegram Notifications
    """
    service_name = 'Telegram'
    service_url = 'https://telegram.org/'
    secure_protocol = 'tgram'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_telegram'
    notify_format = NotifyFormat.HTML
    notify_url = 'https://api.telegram.org/bot'
    attachment_support = True
    image_size = NotifyImageSize.XY_256
    body_maxlen = 4096
    title_maxlen = 0
    request_rate_per_sec = 0.001
    templates = ('{schema}://{bot_token}', '{schema}://{bot_token}/{targets}')
    mime_lookup = ({'regex': re.compile('^(image/gif|video/H264)', re.I), 'function_name': 'sendAnimation', 'key': 'animation'}, {'regex': re.compile('^image/.*', re.I), 'function_name': 'sendPhoto', 'key': 'photo'}, {'regex': re.compile('^video/mp4', re.I), 'function_name': 'sendVideo', 'key': 'video'}, {'regex': re.compile('^(application|audio)/ogg', re.I), 'function_name': 'sendVoice', 'key': 'voice'}, {'regex': re.compile('^audio/(mpeg|mp4a-latm)', re.I), 'function_name': 'sendAudio', 'key': 'audio'}, {'regex': re.compile('.*', re.I), 'function_name': 'sendDocument', 'key': 'document'})
    __telegram_escape_html_entries = ((re.compile('\\s*<!.+?-->\\s*', re.I | re.M | re.S), '', {}), (re.compile('\\s*<\\s*(!?DOCTYPE|p|div|span|body|script|link|meta|html|font|head|label|form|input|textarea|select|iframe|source|script)([^a-z0-9>][^>]*)?>\\s*', re.I | re.M | re.S), '', {}), (re.compile('\\s*<\\s*/(span|body|script|meta|html|font|head|label|form|input|textarea|select|ol|ul|link|iframe|source|script)([^a-z0-9>][^>]*)?>\\s*', re.I | re.M | re.S), '', {}), (re.compile('<\\s*(strong)([^a-z0-9>][^>]*)?>', re.I | re.M | re.S), '<b>', {}), (re.compile('<\\s*/\\s*(strong)([^a-z0-9>][^>]*)?>', re.I | re.M | re.S), '</b>', {}), (re.compile('\\s*<\\s*(h[1-6]|title)([^a-z0-9>][^>]*)?>\\s*', re.I | re.M | re.S), '{}<b>', {'html': '\r\n'}), (re.compile('\\s*<\\s*/\\s*(h[1-6]|title)([^a-z0-9>][^>]*)?>\\s*', re.I | re.M | re.S), '</b>{}', {'html': '<br/>'}), (re.compile('<\\s*(caption|em)([^a-z0-9>][^>]*)?>', re.I | re.M | re.S), '<i>', {}), (re.compile('<\\s*/\\s*(caption|em)([^a-z0-9>][^>]*)?>', re.I | re.M | re.S), '</i>', {}), (re.compile('<\\s*li([^a-z0-9>][^>]*)?>\\s*', re.I | re.M | re.S), ' -', {}), (re.compile('<\\s*pre([^a-z0-9>][^>]*)?>', re.I | re.M | re.S), '{}<code>', {'html': '\r\n'}), (re.compile('<\\s*/\\s*pre([^a-z0-9>][^>]*)?>', re.I | re.M | re.S), '</code>{}', {'html': '\r\n'}), (re.compile('\\s*<\\s*/?\\s*(ol|ul|br|hr)\\s*/?>\\s*', re.I | re.M | re.S), '\r\n', {}), (re.compile('\\s*<\\s*/\\s*(br|p|hr|li|div)([^a-z0-9>][^>]*)?>\\s*', re.I | re.M | re.S), '\r\n', {}), (re.compile('\\&nbsp;?', re.I), ' ', {}), (re.compile('\\&emsp;?', re.I), '   ', {}), (re.compile('\\&apos;?', re.I), "'", {}), (re.compile('\\&quot;?', re.I), '"', {}), (re.compile('\\r*\\n[\\r\\n]+', re.I), '\r\n', {}))
    template_tokens = dict(NotifyBase.template_tokens, **{'bot_token': {'name': _('Bot Token'), 'type': 'string', 'private': True, 'required': True, 'regex': ('^(bot)?(?P<key>[0-9]+:[a-z0-9_-]+)$', 'i')}, 'target_user': {'name': _('Target Chat ID'), 'type': 'string', 'map_to': 'targets', 'map_to': 'targets', 'regex': ('^((-?[0-9]{1,32})|([a-z_-][a-z0-9_-]+))$', 'i')}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'image': {'name': _('Include Image'), 'type': 'bool', 'default': False, 'map_to': 'include_image'}, 'detect': {'name': _('Detect Bot Owner'), 'type': 'bool', 'default': True, 'map_to': 'detect_owner'}, 'silent': {'name': _('Silent Notification'), 'type': 'bool', 'default': False}, 'preview': {'name': _('Web Page Preview'), 'type': 'bool', 'default': False}, 'topic': {'name': _('Topic Thread ID'), 'type': 'int'}, 'to': {'alias_of': 'targets'}, 'content': {'name': _('Content Placement'), 'type': 'choice:string', 'values': TELEGRAM_CONTENT_PLACEMENT, 'default': TelegramContentPlacement.BEFORE}})

    def __init__(self, bot_token, targets, detect_owner=True, include_image=False, silent=None, preview=None, topic=None, content=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Initialize Telegram Object\n        '
        super().__init__(**kwargs)
        self.bot_token = validate_regex(bot_token, *self.template_tokens['bot_token']['regex'], fmt='{key}')
        if not self.bot_token:
            err = 'The Telegram Bot Token specified ({}) is invalid.'.format(bot_token)
            self.logger.warning(err)
            raise TypeError(err)
        self.targets = parse_list(targets)
        self.silent = self.template_args['silent']['default'] if silent is None else bool(silent)
        self.preview = self.template_args['preview']['default'] if preview is None else bool(preview)
        self.content = self.template_args['content']['default'] if not isinstance(content, str) else content.lower()
        if self.content and self.content not in TELEGRAM_CONTENT_PLACEMENT:
            msg = 'The content placement specified ({}) is invalid.'.format(content)
            self.logger.warning(msg)
            raise TypeError(msg)
        if topic:
            try:
                self.topic = int(topic)
            except (TypeError, ValueError):
                err = 'The Telegram Topic ID specified ({}) is invalid.'.format(topic)
                self.logger.warning(err)
                raise TypeError(err)
        else:
            self.topic = None
        self.detect_owner = detect_owner
        if self.user:
            self.targets.append(self.user)
        self.include_image = include_image

    def send_media(self, chat_id, notify_type, attach=None):
        if False:
            while True:
                i = 10
        '\n        Sends a sticker based on the specified notify type\n\n        '
        headers = {'User-Agent': self.app_id}
        function_name = 'SendPhoto'
        key = 'photo'
        path = None
        if isinstance(attach, AttachBase):
            if not attach:
                self.logger.error('Could not access attachment {}.'.format(attach.url(privacy=True)))
                return False
            self.logger.debug('Posting Telegram attachment {}'.format(attach.url(privacy=True)))
            path = attach.path
            file_name = attach.name
            mimetype = attach.mimetype
            (function_name, key) = next(((x['function_name'], x['key']) for x in self.mime_lookup if x['regex'].match(mimetype)))
        else:
            attach = self.image_path(notify_type) if attach is None else attach
            if attach is None:
                return True
            path = attach
            file_name = os.path.basename(path)
        url = '%s%s/%s' % (self.notify_url, self.bot_token, function_name)
        self.throttle()
        payload = {'chat_id': chat_id}
        if self.topic:
            payload['message_thread_id'] = self.topic
        try:
            with open(path, 'rb') as f:
                files = {key: (file_name, f)}
                self.logger.debug('Telegram attachment POST URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
                r = requests.post(url, headers=headers, files=files, data=payload, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code != requests.codes.ok:
                    status_str = NotifyTelegram.http_response_code_lookup(r.status_code)
                    self.logger.warning('Failed to send Telegram attachment: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    return False
                return True
        except requests.RequestException as e:
            self.logger.warning('A connection error occurred posting Telegram attachment.')
            self.logger.debug('Socket Exception: %s' % str(e))
        except (IOError, OSError):
            self.logger.error('File can not be opened for read: {}'.format(path))
        return False

    def detect_bot_owner(self):
        if False:
            i = 10
            return i + 15
        "\n        Takes a bot and attempts to detect it's chat id from that\n\n        "
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json'}
        url = '%s%s/%s' % (self.notify_url, self.bot_token, 'getUpdates')
        self.logger.debug('Telegram User Detection POST URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
        response = None
        try:
            r = requests.post(url, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifyTelegram.http_response_code_lookup(r.status_code)
                try:
                    error_msg = loads(r.content).get('description', 'unknown')
                except (AttributeError, TypeError, ValueError):
                    error_msg = None
                if error_msg:
                    self.logger.warning('Failed to detect the Telegram user: (%s) %s.' % (r.status_code, error_msg))
                else:
                    self.logger.warning('Failed to detect the Telegram user: {}{}error={}.'.format(status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return 0
            response = loads(r.content)
        except (AttributeError, TypeError, ValueError):
            self.logger.warning('A communication error occurred detecting the Telegram User.')
            return 0
        except requests.RequestException as e:
            self.logger.warning('A connection error occurred detecting the Telegram User.')
            self.logger.debug('Socket Exception: %s' % str(e))
            return 0
        if response.get('ok', False):
            for entry in response.get('result', []):
                if 'message' in entry and 'from' in entry['message']:
                    _id = entry['message']['from'].get('id', 0)
                    _user = entry['message']['from'].get('first_name')
                    self.logger.info('Detected Telegram user %s (userid=%d)' % (_user, _id))
                    return _id
        self.logger.warning('Failed to detect a Telegram user; try sending your bot a message first.')
        return 0

    def send(self, body, title='', notify_type=NotifyType.INFO, attach=None, body_format=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Perform Telegram Notification\n        '
        if len(self.targets) == 0 and self.detect_owner:
            _id = self.detect_bot_owner()
            if _id:
                self.targets.append(str(_id))
                self.logger.info('Update your Telegram Apprise URL to read: {}'.format(self.url(privacy=True)))
        if len(self.targets) == 0:
            self.logger.warning('There were not Telegram chat_ids to notify.')
            return False
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/json'}
        has_error = False
        url = '%s%s/%s' % (self.notify_url, self.bot_token, 'sendMessage')
        payload = {'disable_notification': self.silent, 'disable_web_page_preview': not self.preview}
        if self.topic:
            payload['message_thread_id'] = self.topic
        if self.notify_format == NotifyFormat.MARKDOWN:
            payload['parse_mode'] = 'MARKDOWN'
            payload['text'] = body
        else:
            payload['parse_mode'] = 'HTML'
            for (r, v, m) in self.__telegram_escape_html_entries:
                if 'html' in m:
                    v = v.format(m['html'] if body_format in (NotifyFormat.HTML, NotifyFormat.MARKDOWN) else '')
                body = r.sub(v, body)
            payload['text'] = body
        attach_content = TelegramContentPlacement.AFTER if not body else self.content
        targets = list(self.targets)
        while len(targets):
            chat_id = targets.pop(0)
            chat_id = IS_CHAT_ID_RE.match(chat_id)
            if not chat_id:
                self.logger.warning("The specified chat_id '%s' is invalid; skipping." % (chat_id,))
                has_error = True
                continue
            if chat_id.group('name') is not None:
                payload['chat_id'] = '@%s' % chat_id.group('name')
            else:
                payload['chat_id'] = int(chat_id.group('idno'))
            if self.include_image is True:
                if not self.send_media(payload['chat_id'], notify_type):
                    notify_type
                    self.logger.warning('Failed to send Telegram type image to {}.', payload['chat_id'])
            if attach and self.attachment_support and (attach_content == TelegramContentPlacement.AFTER):
                if not self._send_attachments(chat_id=payload['chat_id'], notify_type=notify_type, attach=attach):
                    has_error = True
                    continue
                if not body:
                    continue
            self.throttle()
            self.logger.debug('Telegram POST URL: %s (cert_verify=%r)' % (url, self.verify_certificate))
            self.logger.debug('Telegram Payload: %s' % str(payload))
            try:
                r = requests.post(url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code != requests.codes.ok:
                    status_str = NotifyTelegram.http_response_code_lookup(r.status_code)
                    try:
                        error_msg = loads(r.content).get('description', 'unknown')
                    except (AttributeError, TypeError, ValueError):
                        error_msg = None
                    self.logger.warning('Failed to send Telegram notification to {}: {}, error={}.'.format(payload['chat_id'], error_msg if error_msg else status_str, r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
            except requests.RequestException as e:
                self.logger.warning('A connection error occurred sending Telegram:%s ' % payload['chat_id'] + 'notification.')
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
                continue
            self.logger.info('Sent Telegram notification.')
            if attach and self.attachment_support and (attach_content == TelegramContentPlacement.BEFORE):
                if not self._send_attachments(chat_id=payload['chat_id'], notify_type=notify_type, attach=attach):
                    has_error = True
                    continue
        return not has_error

    def _send_attachments(self, chat_id, notify_type, attach):
        if False:
            i = 10
            return i + 15
        '\n        Sends our attachments\n        '
        has_error = False
        for attachment in attach:
            if not self.send_media(chat_id, notify_type, attach=attachment):
                has_error = True
                break
            self.logger.info('Sent Telegram attachment: {}.'.format(attachment))
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'image': self.include_image, 'detect': 'yes' if self.detect_owner else 'no', 'silent': 'yes' if self.silent else 'no', 'preview': 'yes' if self.preview else 'no', 'content': self.content}
        if self.topic:
            params['topic'] = self.topic
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        return '{schema}://{bot_token}/{targets}/?{params}'.format(schema=self.secure_protocol, bot_token=self.pprint(self.bot_token, privacy, safe=''), targets='/'.join([NotifyTelegram.quote('@{}'.format(x)) for x in self.targets]), params=NotifyTelegram.urlencode(params))

    def __len__(self):
        if False:
            print('Hello World!')
        '\n        Returns the number of targets associated with this notification\n        '
        return len(self.targets)

    @staticmethod
    def parse_url(url):
        if False:
            print('Hello World!')
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        try:
            tgram = re.match('(?P<protocol>{schema}://)(bot)?(?P<prefix>([a-z0-9_-]+)(:[a-z0-9_-]+)?@)?(?P<btoken_a>[0-9]+)(:|%3A)+(?P<remaining>.*)$'.format(schema=NotifyTelegram.secure_protocol), url, re.I)
        except (TypeError, AttributeError):
            tgram = None
        if not tgram:
            return None
        if tgram.group('prefix'):
            results = NotifyBase.parse_url('%s%s%s/%s' % (tgram.group('protocol'), tgram.group('prefix'), tgram.group('btoken_a'), tgram.group('remaining')), verify_host=False)
        else:
            results = NotifyBase.parse_url('%s%s/%s' % (tgram.group('protocol'), tgram.group('btoken_a'), tgram.group('remaining')), verify_host=False)
        bot_token_a = NotifyTelegram.unquote(results['host'])
        entries = NotifyTelegram.split_path(results['fullpath'])
        bot_token_b = entries.pop(0)
        bot_token = '%s:%s' % (bot_token_a, bot_token_b)
        results['targets'] = entries
        if 'content' in results['qsd'] and len(results['qsd']['content']):
            results['content'] = results['qsd']['content']
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyTelegram.parse_list(results['qsd']['to'])
        results['bot_token'] = bot_token
        if 'topic' in results['qsd'] and len(results['qsd']['topic']):
            results['topic'] = results['qsd']['topic']
        results['silent'] = parse_bool(results['qsd'].get('silent', False))
        results['preview'] = parse_bool(results['qsd'].get('preview', False))
        results['include_image'] = parse_bool(results['qsd'].get('image', False))
        results['detect_owner'] = parse_bool(results['qsd'].get('detect', True))
        return results