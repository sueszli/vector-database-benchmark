import re
import requests
from copy import deepcopy
from datetime import datetime
from datetime import timezone
from requests_oauthlib import OAuth1
from json import dumps
from json import loads
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyType
from ..utils import parse_list
from ..utils import parse_bool
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
from ..attachment.AttachBase import AttachBase
IS_USER = re.compile('^\\s*@?(?P<user>[A-Z0-9_]+)$', re.I)

class TwitterMessageMode:
    """
    Twitter Message Mode
    """
    DM = 'dm'
    TWEET = 'tweet'
TWITTER_MESSAGE_MODES = (TwitterMessageMode.DM, TwitterMessageMode.TWEET)

class NotifyTwitter(NotifyBase):
    """
    A wrapper to Twitter Notifications

    """
    service_name = 'Twitter'
    service_url = 'https://twitter.com/'
    secure_protocol = ('x', 'twitter', 'tweet')
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_twitter'
    attachment_support = True
    title_maxlen = 0
    twitter_lookup = 'https://api.twitter.com/1.1/users/lookup.json'
    twitter_whoami = 'https://api.twitter.com/1.1/account/verify_credentials.json'
    twitter_dm = 'https://api.twitter.com/1.1/direct_messages/events/new.json'
    twitter_tweet = 'https://api.twitter.com/1.1/statuses/update.json'
    __tweet_non_gif_images_batch = 4
    twitter_media = 'https://upload.twitter.com/1.1/media/upload.json'
    request_rate_per_sec = 0
    ratelimit_reset = datetime.now(timezone.utc).replace(tzinfo=None)
    ratelimit_remaining = 1
    templates = ('{schema}://{ckey}/{csecret}/{akey}/{asecret}', '{schema}://{ckey}/{csecret}/{akey}/{asecret}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'ckey': {'name': _('Consumer Key'), 'type': 'string', 'private': True, 'required': True}, 'csecret': {'name': _('Consumer Secret'), 'type': 'string', 'private': True, 'required': True}, 'akey': {'name': _('Access Key'), 'type': 'string', 'private': True, 'required': True}, 'asecret': {'name': _('Access Secret'), 'type': 'string', 'private': True, 'required': True}, 'target_user': {'name': _('Target User'), 'type': 'string', 'prefix': '@', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'mode': {'name': _('Message Mode'), 'type': 'choice:string', 'values': TWITTER_MESSAGE_MODES, 'default': TwitterMessageMode.DM}, 'cache': {'name': _('Cache Results'), 'type': 'bool', 'default': True}, 'to': {'alias_of': 'targets'}, 'batch': {'name': _('Batch Mode'), 'type': 'bool', 'default': True}})

    def __init__(self, ckey, csecret, akey, asecret, targets=None, mode=TwitterMessageMode.DM, cache=True, batch=True, **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialize Twitter Object\n\n        '
        super().__init__(**kwargs)
        self.ckey = validate_regex(ckey)
        if not self.ckey:
            msg = 'An invalid Twitter Consumer Key was specified.'
            self.logger.warning(msg)
            raise TypeError(msg)
        self.csecret = validate_regex(csecret)
        if not self.csecret:
            msg = 'An invalid Twitter Consumer Secret was specified.'
            self.logger.warning(msg)
            raise TypeError(msg)
        self.akey = validate_regex(akey)
        if not self.akey:
            msg = 'An invalid Twitter Access Key was specified.'
            self.logger.warning(msg)
            raise TypeError(msg)
        self.asecret = validate_regex(asecret)
        if not self.asecret:
            msg = 'An invalid Access Secret was specified.'
            self.logger.warning(msg)
            raise TypeError(msg)
        self.mode = self.template_args['mode']['default'] if not isinstance(mode, str) else mode.lower()
        if self.mode not in TWITTER_MESSAGE_MODES:
            msg = 'The Twitter message mode specified ({}) is invalid.'.format(mode)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.cache = cache
        self.batch = batch
        has_error = False
        self.targets = []
        for target in parse_list(targets):
            match = IS_USER.match(target)
            if match and match.group('user'):
                self.targets.append(match.group('user'))
                continue
            has_error = True
            self.logger.warning('Dropped invalid Twitter user ({}) specified.'.format(target))
        if has_error and (not self.targets):
            msg = 'No Twitter targets to notify.'
            self.logger.warning(msg)
            raise TypeError(msg)
        self._whoami_cache = None
        self._user_cache = {}
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, attach=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform Twitter Notification\n        '
        attachments = []
        if attach and self.attachment_support:
            for attachment in attach:
                if not attachment:
                    self.logger.error('Could not access attachment {}.'.format(attachment.url(privacy=True)))
                    return False
                if not re.match('^image/.*', attachment.mimetype, re.I):
                    self.logger.warning('Ignoring unsupported Twitter attachment {}.'.format(attachment.url(privacy=True)))
                    continue
                self.logger.debug('Preparing Twitter attachment {}'.format(attachment.url(privacy=True)))
                (postokay, response) = self._fetch(self.twitter_media, payload=attachment)
                if not postokay:
                    return False
                if not (isinstance(response, dict) and response.get('media_id')):
                    self.logger.debug('Could not attach the file to Twitter: %s (mime=%s)', attachment.name, attachment.mimetype)
                    continue
                response.update({'file_name': attachment.name, 'file_mime': attachment.mimetype, 'file_path': attachment.path})
                attachments.append(response)
        return getattr(self, '_send_{}'.format(self.mode))(body=body, title=title, notify_type=notify_type, attachments=attachments, **kwargs)

    def _send_tweet(self, body, title='', notify_type=NotifyType.INFO, attachments=None, **kwargs):
        if False:
            return 10
        '\n        Twitter Public Tweet\n        '
        has_error = False
        payload = {'status': body}
        payloads = []
        if not attachments:
            payloads.append(payload)
        else:
            batch_size = 1 if not self.batch else self.__tweet_non_gif_images_batch
            batches = []
            batch = []
            for attachment in attachments:
                batch.append(str(attachment['media_id']))
                if not re.match('^image/(png|jpe?g)', attachment['file_mime'], re.I) or len(batch) >= batch_size:
                    batches.append(','.join(batch))
                    batch = []
            if batch:
                batches.append(','.join(batch))
            for (no, media_ids) in enumerate(batches):
                _payload = deepcopy(payload)
                _payload['media_ids'] = media_ids
                if no or not body:
                    _payload['status'] = '{:02d}/{:02d}'.format(no + 1, len(batches))
                payloads.append(_payload)
        for (no, payload) in enumerate(payloads, start=1):
            (postokay, response) = self._fetch(self.twitter_tweet, payload=payload, json=False)
            if not postokay:
                has_error = True
                errors = []
                try:
                    errors = ['Error Code {}: {}'.format(e.get('code', 'unk'), e.get('message')) for e in response['errors']]
                except (KeyError, TypeError):
                    pass
                for error in errors:
                    self.logger.debug('Tweet [%.2d/%.2d] Details: %s', no, len(payloads), error)
                continue
            try:
                url = 'https://twitter.com/{}/status/{}'.format(response['user']['screen_name'], response['id_str'])
            except (KeyError, TypeError):
                url = 'unknown'
            self.logger.debug('Tweet [%.2d/%.2d] Details: %s', no, len(payloads), url)
            self.logger.info('Sent [%.2d/%.2d] Twitter notification as public tweet.', no, len(payloads))
        return not has_error

    def _send_dm(self, body, title='', notify_type=NotifyType.INFO, attachments=None, **kwargs):
        if False:
            return 10
        '\n        Twitter Direct Message\n        '
        has_error = False
        payload = {'event': {'type': 'message_create', 'message_create': {'target': {'recipient_id': None}, 'message_data': {'text': body}}}}
        targets = self._whoami(lazy=self.cache) if not len(self.targets) else self._user_lookup(self.targets, lazy=self.cache)
        if not targets:
            self.logger.warning('Failed to acquire user(s) to Direct Message via Twitter')
            return False
        payloads = []
        if not attachments:
            payloads.append(payload)
        else:
            for (no, attachment) in enumerate(attachments):
                _payload = deepcopy(payload)
                _data = _payload['event']['message_create']['message_data']
                _data['attachment'] = {'type': 'media', 'media': {'id': attachment['media_id']}, 'additional_owners': ','.join([str(x) for x in targets.values()])}
                if no or not body:
                    _data['text'] = '{:02d}/{:02d}'.format(no + 1, len(attachments))
                payloads.append(_payload)
        for (no, payload) in enumerate(payloads, start=1):
            for (screen_name, user_id) in targets.items():
                target = payload['event']['message_create']['target']
                target['recipient_id'] = user_id
                (postokay, response) = self._fetch(self.twitter_dm, payload=payload)
                if not postokay:
                    has_error = True
                    continue
                self.logger.info('Sent [{:02d}/{:02d}] Twitter DM notification to @{}.'.format(no, len(payloads), screen_name))
        return not has_error

    def _whoami(self, lazy=True):
        if False:
            i = 10
            return i + 15
        '\n        Looks details of current authenticated user\n\n        '
        if lazy and self._whoami_cache is not None:
            return self._whoami_cache
        results = {}
        (postokay, response) = self._fetch(self.twitter_whoami, method='GET', json=False)
        if postokay:
            try:
                results[response['screen_name']] = response['id']
                self._whoami_cache = {response['screen_name']: response['id']}
                self._user_cache.update(results)
            except (TypeError, KeyError):
                pass
        return results

    def _user_lookup(self, screen_name, lazy=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Looks up a screen name and returns the user id\n\n        the screen_name can be a list/set/tuple as well\n        '
        results = {}
        names = parse_list(screen_name)
        if lazy and self._user_cache:
            results = {k: v for (k, v) in self._user_cache.items() if k in names}
            names = [name for name in names if name not in results]
        if not len(names):
            return results
        for i in range(0, len(names), 100):
            (postokay, response) = self._fetch(self.twitter_lookup, payload={'screen_name': names[i:i + 100]}, json=False)
            if not postokay or not isinstance(response, list):
                continue
            for entry in response:
                try:
                    results[entry['screen_name']] = entry['id']
                except (TypeError, KeyError):
                    pass
        self._user_cache.update(results)
        return results

    def _fetch(self, url, payload=None, method='POST', json=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Wrapper to Twitter API requests object\n        '
        headers = {'User-Agent': self.app_id}
        data = None
        files = None
        if isinstance(payload, AttachBase):
            files = {'media': (payload.name, open(payload.path, 'rb'))}
        elif json:
            headers['Content-Type'] = 'application/json'
            data = dumps(payload)
        else:
            data = payload
        auth = OAuth1(self.ckey, client_secret=self.csecret, resource_owner_key=self.akey, resource_owner_secret=self.asecret)
        self.logger.debug('Twitter {} URL: {} (cert_verify={})'.format(method, url, self.verify_certificate))
        self.logger.debug('Twitter Payload: %s' % str(payload))
        wait = None
        if self.ratelimit_remaining == 0:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            if now < self.ratelimit_reset:
                wait = (self.ratelimit_reset - now).total_seconds() + 0.5
        content = {}
        self.throttle(wait=wait)
        fn = requests.post if method == 'POST' else requests.get
        try:
            r = fn(url, data=data, files=files, headers=headers, auth=auth, verify=self.verify_certificate, timeout=self.request_timeout)
            try:
                content = loads(r.content)
            except (AttributeError, TypeError, ValueError):
                content = {}
            if r.status_code != requests.codes.ok:
                status_str = NotifyTwitter.http_response_code_lookup(r.status_code)
                self.logger.warning('Failed to send Twitter {} to {}: {}error={}.'.format(method, url, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return (False, content)
            try:
                self.ratelimit_remaining = int(r.headers.get('x-rate-limit-remaining'))
                self.ratelimit_reset = datetime.fromtimestamp(int(r.headers.get('x-rate-limit-reset')), timezone.utc).replace(tzinfo=None)
            except (TypeError, ValueError):
                pass
        except requests.RequestException as e:
            self.logger.warning('Exception received when sending Twitter {} to {}: '.format(method, url))
            self.logger.debug('Socket Exception: %s' % str(e))
            return (False, content)
        except (OSError, IOError) as e:
            self.logger.warning('An I/O error occurred while handling {}.'.format(payload.name if isinstance(payload, AttachBase) else payload))
            self.logger.debug('I/O Exception: %s' % str(e))
            return (False, content)
        finally:
            if files:
                files['media'][1].close()
        return (True, content)

    @property
    def body_maxlen(self):
        if False:
            return 10
        '\n        The maximum allowable characters allowed in the body per message\n        This is used during a Private DM Message Size (not Public Tweets\n        which are limited to 280 characters)\n        '
        return 10000 if self.mode == TwitterMessageMode.DM else 280

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'mode': self.mode, 'batch': 'yes' if self.batch else 'no', 'cache': 'yes' if self.cache else 'no'}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        return '{schema}://{ckey}/{csecret}/{akey}/{asecret}/{targets}/?{params}'.format(schema=self.secure_protocol[0], ckey=self.pprint(self.ckey, privacy, safe=''), csecret=self.pprint(self.csecret, privacy, mode=PrivacyMode.Secret, safe=''), akey=self.pprint(self.akey, privacy, safe=''), asecret=self.pprint(self.asecret, privacy, mode=PrivacyMode.Secret, safe=''), targets='/'.join([NotifyTwitter.quote('@{}'.format(target), safe='@') for target in self.targets]), params=NotifyTwitter.urlencode(params))

    def __len__(self):
        if False:
            print('Hello World!')
        '\n        Returns the number of targets associated with this notification\n        '
        targets = len(self.targets)
        return targets if targets > 0 else 1

    @staticmethod
    def parse_url(url):
        if False:
            i = 10
            return i + 15
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        tokens = NotifyTwitter.split_path(results['fullpath'])
        results['ckey'] = NotifyTwitter.unquote(results['host'])
        results['csecret'] = tokens.pop(0) if tokens else None
        results['akey'] = tokens.pop(0) if tokens else None
        results['asecret'] = tokens.pop(0) if tokens else None
        if 'mode' in results['qsd'] and len(results['qsd']['mode']):
            results['mode'] = NotifyTwitter.unquote(results['qsd']['mode'])
        elif results['schema'].startswith('tweet'):
            results['mode'] = TwitterMessageMode.TWEET
        results['targets'] = []
        if results.get('user'):
            results['targets'].append(results.get('user'))
        results['targets'].extend(tokens)
        if 'cache' in results['qsd'] and len(results['qsd']['cache']):
            results['cache'] = parse_bool(results['qsd']['cache'], True)
        results['batch'] = parse_bool(results['qsd'].get('batch', NotifyTwitter.template_args['batch']['default']))
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'] += NotifyTwitter.parse_list(results['qsd']['to'])
        return results