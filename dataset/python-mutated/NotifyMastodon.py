import re
import requests
from copy import deepcopy
from json import dumps, loads
from datetime import datetime
from datetime import timezone
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyImageSize
from ..common import NotifyFormat
from ..common import NotifyType
from ..utils import parse_list
from ..utils import parse_bool
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
from ..attachment.AttachBase import AttachBase
IS_USER = re.compile('^\\s*@?(?P<user>[A-Z0-9_]+(?:@(?P<host>[A-Z0-9_.-]+))?)$', re.I)
USER_DETECTION_RE = re.compile('(@[A-Z0-9_]+(?:@[A-Z0-9_.-]+)?)(?=$|[\\s,.&()\\[\\]]+)', re.I)

class MastodonMessageVisibility:
    """
    The visibility of any status message made
    """
    DEFAULT = 'default'
    DIRECT = 'direct'
    PRIVATE = 'private'
    UNLISTED = 'unlisted'
    PUBLIC = 'public'
MASTODON_MESSAGE_VISIBILITIES = (MastodonMessageVisibility.DEFAULT, MastodonMessageVisibility.DIRECT, MastodonMessageVisibility.PRIVATE, MastodonMessageVisibility.UNLISTED, MastodonMessageVisibility.PUBLIC)

class NotifyMastodon(NotifyBase):
    """
    A wrapper for Notify Mastodon Notifications
    """
    service_name = 'Mastodon'
    service_url = 'https://joinmastodon.org'
    protocol = ('mastodon', 'toot')
    secure_protocol = ('mastodons', 'toots')
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_mastodon'
    attachment_support = True
    image_size = NotifyImageSize.XY_128
    __toot_non_gif_images_batch = 4
    mastodon_whoami = '/api/v1/accounts/verify_credentials'
    mastodon_media = '/api/v1/media'
    mastodon_toot = '/api/v1/statuses'
    mastodon_dm = '/api/v1/dm'
    title_maxlen = 0
    body_maxlen = 500
    notify_format = NotifyFormat.TEXT
    request_rate_per_sec = 0
    ratelimit_reset = datetime.now(timezone.utc).replace(tzinfo=None)
    ratelimit_remaining = 1
    templates = ('{schema}://{token}@{host}', '{schema}://{token}@{host}:{port}', '{schema}://{token}@{host}/{targets}', '{schema}://{token}@{host}:{port}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'host': {'name': _('Hostname'), 'type': 'string', 'required': True}, 'token': {'name': _('Access Token'), 'type': 'string', 'required': True}, 'port': {'name': _('Port'), 'type': 'int', 'min': 1, 'max': 65535}, 'target_user': {'name': _('Target User'), 'type': 'string', 'prefix': '@', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'token': {'alias_of': 'token'}, 'visibility': {'name': _('Visibility'), 'type': 'choice:string', 'values': MASTODON_MESSAGE_VISIBILITIES, 'default': MastodonMessageVisibility.DEFAULT}, 'cache': {'name': _('Cache Results'), 'type': 'bool', 'default': True}, 'batch': {'name': _('Batch Mode'), 'type': 'bool', 'default': True}, 'sensitive': {'name': _('Sensitive Attachments'), 'type': 'bool', 'default': False}, 'spoiler': {'name': _('Spoiler Text'), 'type': 'string'}, 'key': {'name': _('Idempotency-Key'), 'type': 'string'}, 'language': {'name': _('Language Code'), 'type': 'string'}, 'to': {'alias_of': 'targets'}})

    def __init__(self, token=None, targets=None, batch=True, sensitive=None, spoiler=None, visibility=None, cache=True, key=None, language=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize Notify Mastodon Object\n        '
        super().__init__(**kwargs)
        self.schema = 'https' if self.secure else 'http'
        self._whoami_cache = None
        self.token = validate_regex(token)
        if not self.token:
            msg = 'An invalid Mastodon Access Token was specified.'
            self.logger.warning(msg)
            raise TypeError(msg)
        if visibility:
            vis = 'invalid' if not isinstance(visibility, str) else visibility.lower().strip()
            self.visibility = next((v for v in MASTODON_MESSAGE_VISIBILITIES if v.startswith(vis)), None)
            if self.visibility not in MASTODON_MESSAGE_VISIBILITIES:
                msg = 'The Mastodon visibility specified ({}) is invalid.'.format(visibility)
                self.logger.warning(msg)
                raise TypeError(msg)
        else:
            self.visibility = self.template_args['visibility']['default']
        self.api_url = '%s://%s' % (self.schema, self.host)
        if isinstance(self.port, int):
            self.api_url += ':%d' % self.port
        self.cache = cache
        self.batch = self.template_args['batch']['default'] if batch is None else batch
        self.sensitive = self.template_args['sensitive']['default'] if sensitive is None else sensitive
        self.spoiler = spoiler if isinstance(spoiler, str) else None
        self.idempotency_key = key if isinstance(key, str) else None
        self.language = language if isinstance(language, str) else None
        self.targets = []
        has_error = False
        for target in parse_list(targets):
            match = IS_USER.match(target)
            if match and match.group('user'):
                self.targets.append('@' + match.group('user'))
                continue
            has_error = True
            self.logger.warning('Dropped invalid Mastodon user ({}) specified.'.format(target))
        if has_error and (not self.targets):
            msg = 'No Mastodon targets to notify.'
            self.logger.warning(msg)
            raise TypeError(msg)
        return

    def url(self, privacy=False, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'visibility': self.visibility, 'batch': 'yes' if self.batch else 'no', 'sensitive': 'yes' if self.sensitive else 'no'}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        if self.spoiler:
            params['spoiler'] = self.spoiler
        if self.idempotency_key:
            params['key'] = self.idempotency_key
        if self.language:
            params['language'] = self.language
        default_port = 443 if self.secure else 80
        return '{schema}://{token}@{host}{port}/{targets}?{params}'.format(schema=self.secure_protocol[0] if self.secure else self.protocol[0], token=self.pprint(self.token, privacy, mode=PrivacyMode.Secret, safe=''), host=self.host, port='' if self.port is None or self.port == default_port else ':{}'.format(self.port), targets='/'.join([NotifyMastodon.quote(x, safe='@') for x in self.targets]), params=NotifyMastodon.urlencode(params))

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the number of targets associated with this notification\n        '
        targets = len(self.targets)
        return targets if targets > 0 else 1

    def send(self, body, title='', notify_type=NotifyType.INFO, attach=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        wrapper to _send since we can alert more then one channel\n        '
        attachments = []
        users = set(USER_DETECTION_RE.findall(body))
        targets = users - set(self.targets.copy())
        if not self.targets and self.visibility == MastodonMessageVisibility.DIRECT:
            result = self._whoami()
            if not result:
                return False
            myself = '@' + next(iter(result.keys()))
            if myself in users:
                targets.remove(myself)
            else:
                targets.add(myself)
        if attach and self.attachment_support:
            for attachment in attach:
                if not attachment:
                    self.logger.error('Could not access attachment {}.'.format(attachment.url(privacy=True)))
                    return False
                if not re.match('^(image|video|audio)/.*', attachment.mimetype, re.I):
                    self.logger.warning('Ignoring unsupported Mastodon attachment {}.'.format(attachment.url(privacy=True)))
                    continue
                self.logger.debug('Preparing Mastodon attachment {}'.format(attachment.url(privacy=True)))
                (postokay, response) = self._request(self.mastodon_media, payload=attachment)
                if not postokay:
                    if response and 'authorized scopes' in response.get('error', ''):
                        self.logger.warning('Failed to Send Attachment to Mastodon: missing scope: write:media')
                    return False
                if not (isinstance(response, dict) and response.get('id')):
                    self.logger.debug('Could not attach the file to Mastodon: %s (mime=%s)', attachment.name, attachment.mimetype)
                    continue
                response.update({'file_name': attachment.name, 'file_mime': attachment.mimetype, 'file_path': attachment.path})
                attachments.append(response)
        payload = {'status': '{} {}'.format(' '.join(targets), body) if targets else body, 'sensitive': self.sensitive}
        if self.visibility != MastodonMessageVisibility.DEFAULT:
            payload['visibility'] = self.visibility
        if self.spoiler:
            payload['spoiler_text'] = self.spoiler
        if self.idempotency_key:
            payload['Idempotency-Key'] = self.idempotency_key
        if self.language:
            payload['language'] = self.language
        payloads = []
        if not attachments:
            payloads.append(payload)
        else:
            batch_size = 1 if not self.batch else self.__toot_non_gif_images_batch
            batches = []
            batch = []
            for attachment in attachments:
                batch.append(attachment['id'])
                if not re.match('^image/(png|jpe?g)', attachment['file_mime'], re.I) or len(batch) >= batch_size:
                    batches.append(batch)
                    batch = []
            if batch:
                batches.append(batch)
            for (no, media_ids) in enumerate(batches):
                _payload = deepcopy(payload)
                _payload['media_ids'] = media_ids
                if no or not body:
                    _payload['status'] = '{:02d}/{:02d}'.format(no + 1, len(batches))
                    _payload['sensitive'] = False
                    if self.idempotency_key:
                        _payload['Idempotency-Key'] = '{}-part{:02d}'.format(self.idempotency_key, no)
                payloads.append(_payload)
        has_error = False
        for (no, payload) in enumerate(payloads, start=1):
            (postokay, response) = self._request(self.mastodon_toot, payload)
            if not postokay:
                has_error = True
                if response and 'authorized scopes' in response.get('error', ''):
                    self.logger.warning('Failed to Send Status to Mastodon: missing scope: write:statuses')
                continue
            try:
                url = '{}/web/@{}'.format(self.api_url, response['account']['username'])
            except (KeyError, TypeError):
                url = 'unknown'
            self.logger.debug('Mastodon [%.2d/%.2d] (%d attached) delivered to %s', no, len(payloads), len(payload.get('media_ids', [])), url)
            self.logger.info('Sent [%.2d/%.2d] Mastodon notification as public toot.', no, len(payloads))
        return not has_error

    def _whoami(self, lazy=True):
        if False:
            while True:
                i = 10
        '\n        Looks details of current authenticated user\n\n        '
        if lazy and self._whoami_cache is not None:
            return self._whoami_cache
        (postokay, response) = self._request(self.mastodon_whoami, method='GET')
        if postokay:
            try:
                self._whoami_cache = {response['username']: response['id']}
            except (TypeError, KeyError):
                pass
        elif response and 'authorized scopes' in response.get('error', ''):
            self.logger.warning('Failed to lookup Mastodon Auth details; missing scope: read:accounts')
        return self._whoami_cache if postokay else {}

    def _request(self, path, payload=None, method='POST'):
        if False:
            while True:
                i = 10
        '\n        Wrapper to Mastodon API requests object\n        '
        headers = {'User-Agent': self.app_id, 'Authorization': f'Bearer {self.token}'}
        data = None
        files = None
        url = '{}{}'.format(self.api_url, path)
        self.logger.debug('Mastodon {} URL: {} (cert_verify={})'.format(method, url, self.verify_certificate))
        if isinstance(payload, AttachBase):
            files = {'file': (payload.name, open(payload.path, 'rb'), 'application/octet-stream')}
            data = {'description': payload.name}
        else:
            headers['Content-Type'] = 'application/json'
            data = dumps(payload)
            self.logger.debug('Mastodon Payload: %s' % str(payload))
        content = {}
        wait = None
        if self.ratelimit_remaining == 0:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            if now < self.ratelimit_reset:
                wait = (self.ratelimit_reset - now).total_seconds() + 0.5
        self.throttle(wait=wait)
        fn = requests.post if method == 'POST' else requests.get
        try:
            r = fn(url, data=data, files=files, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            try:
                content = loads(r.content)
            except (AttributeError, TypeError, ValueError):
                content = {}
            if r.status_code not in (requests.codes.ok, requests.codes.created, requests.codes.accepted):
                status_str = NotifyMastodon.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send Mastodon {} to {}: {}error={}.'.format(method, url, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return (False, content)
            try:
                self.ratelimit_remaining = int(r.headers.get('X-RateLimit-Remaining'))
                self.ratelimit_reset = datetime.fromtimestamp(int(r.headers.get('X-RateLimit-Limit')), timezone.utc).replace(tzinfo=None)
            except (TypeError, ValueError):
                pass
        except requests.RequestException as e:
            self.logger.warning('Exception received when sending Mastodon {} to {}: '.format(method, url))
            self.logger.debug('Socket Exception: %s' % str(e))
            return (False, content)
        except (OSError, IOError) as e:
            self.logger.warning('An I/O error occurred while handling {}.'.format(payload.name if isinstance(payload, AttachBase) else payload))
            self.logger.debug('I/O Exception: %s' % str(e))
            return (False, content)
        finally:
            if files:
                files['file'][1].close()
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
        if 'token' in results['qsd'] and len(results['qsd']['token']):
            results['token'] = NotifyMastodon.unquote(results['qsd']['token'])
        elif not results['password'] and results['user']:
            results['token'] = NotifyMastodon.unquote(results['user'])
        results['targets'] = NotifyMastodon.split_path(results['fullpath'])
        if 'visibility' in results['qsd'] and len(results['qsd']['visibility']):
            results['visibility'] = NotifyMastodon.unquote(results['qsd']['visibility'])
        elif results['schema'].startswith('toot'):
            results['visibility'] = MastodonMessageVisibility.PUBLIC
        if 'key' in results['qsd'] and len(results['qsd']['key']):
            results['key'] = NotifyMastodon.unquote(results['qsd']['key'])
        if 'spoiler' in results['qsd'] and len(results['qsd']['spoiler']):
            results['spoiler'] = NotifyMastodon.unquote(results['qsd']['spoiler'])
        if 'language' in results['qsd'] and len(results['qsd']['language']):
            results['language'] = NotifyMastodon.unquote(results['qsd']['language'])
        results['sensitive'] = parse_bool(results['qsd'].get('sensitive', NotifyMastodon.template_args['sensitive']['default']))
        results['batch'] = parse_bool(results['qsd'].get('batch', NotifyMastodon.template_args['batch']['default']))
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyMastodon.parse_list(results['qsd']['to'])
        return results