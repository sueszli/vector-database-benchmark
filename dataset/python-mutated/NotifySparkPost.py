import requests
import base64
from json import loads
from json import dumps
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..common import NotifyFormat
from ..utils import is_email
from email.utils import formataddr
from ..utils import validate_regex
from ..utils import parse_emails
from ..utils import parse_bool
from ..AppriseLocale import gettext_lazy as _
SPARKPOST_HTTP_ERROR_MAP = {400: 'A bad request was made to the server', 401: 'Invalid User ID and/or Unauthorized User', 403: 'Permission Denied; the provided API Key was not valid', 404: 'There is a problem with the server query URI.', 405: 'Invalid HTTP method', 420: 'Sending limit reached.', 422: 'Invalid data/format/type/length', 429: 'To many requests per sec; rate limit'}

class SparkPostRegion:
    """
    Regions
    """
    US = 'us'
    EU = 'eu'
SPARKPOST_API_LOOKUP = {SparkPostRegion.US: 'https://api.sparkpost.com/api/v1', SparkPostRegion.EU: 'https://api.eu.sparkpost.com/api/v1'}
SPARKPOST_REGIONS = (SparkPostRegion.US, SparkPostRegion.EU)

class NotifySparkPost(NotifyBase):
    """
    A wrapper for SparkPost Notifications
    """
    service_name = 'SparkPost'
    service_url = 'https://sparkpost.com/'
    attachment_support = True
    secure_protocol = 'sparkpost'
    request_rate_per_sec = 0.2
    sparkpost_retry_wait_sec = 5
    sparkpost_retry_attempts = 3
    default_batch_size = 2000
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_sparkpost'
    notify_format = NotifyFormat.HTML
    templates = ('{schema}://{user}@{host}:{apikey}/', '{schema}://{user}@{host}:{apikey}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'user': {'name': _('User Name'), 'type': 'string', 'required': True}, 'host': {'name': _('Domain'), 'type': 'string', 'required': True}, 'apikey': {'name': _('API Key'), 'type': 'string', 'private': True, 'required': True}, 'targets': {'name': _('Target Emails'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'name': {'name': _('From Name'), 'type': 'string', 'map_to': 'from_name'}, 'region': {'name': _('Region Name'), 'type': 'choice:string', 'values': SPARKPOST_REGIONS, 'default': SparkPostRegion.US, 'map_to': 'region_name'}, 'to': {'alias_of': 'targets'}, 'cc': {'name': _('Carbon Copy'), 'type': 'list:string'}, 'bcc': {'name': _('Blind Carbon Copy'), 'type': 'list:string'}, 'batch': {'name': _('Batch Mode'), 'type': 'bool', 'default': False}})
    template_kwargs = {'headers': {'name': _('Email Header'), 'prefix': '+'}, 'tokens': {'name': _('Template Tokens'), 'prefix': ':'}}

    def __init__(self, apikey, targets, cc=None, bcc=None, from_name=None, region_name=None, headers=None, tokens=None, batch=None, **kwargs):
        if False:
            return 10
        '\n        Initialize SparkPost Object\n        '
        super().__init__(**kwargs)
        self.apikey = validate_regex(apikey)
        if not self.apikey:
            msg = 'An invalid SparkPost API Key ({}) was specified.'.format(apikey)
            self.logger.warning(msg)
            raise TypeError(msg)
        if not self.user:
            msg = 'No SparkPost username was specified.'
            self.logger.warning(msg)
            raise TypeError(msg)
        self.targets = list()
        self.cc = set()
        self.bcc = set()
        self.names = {}
        try:
            self.region_name = self.template_args['region']['default'] if region_name is None else region_name.lower()
            if self.region_name not in SPARKPOST_REGIONS:
                raise
        except:
            msg = 'The SparkPost region specified ({}) is invalid.'.format(region_name)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.from_name = from_name
        self.from_addr = '{user}@{host}'.format(user=self.user, host=self.host)
        if not is_email(self.from_addr):
            msg = 'Invalid ~From~ email format: {}'.format(self.from_addr)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.headers = {}
        if headers:
            self.headers.update(headers)
        self.tokens = {}
        if tokens:
            self.tokens.update(tokens)
        self.batch = self.template_args['batch']['default'] if batch is None else batch
        if targets:
            for recipient in parse_emails(targets):
                result = is_email(recipient)
                if result:
                    self.targets.append((result['name'] if result['name'] else False, result['full_email']))
                    continue
                self.logger.warning('Dropped invalid To email ({}) specified.'.format(recipient))
        else:
            self.targets.append((self.from_name if self.from_name else False, self.from_addr))
        for recipient in parse_emails(cc):
            email = is_email(recipient)
            if email:
                self.cc.add(email['full_email'])
                self.names[email['full_email']] = email['name'] if email['name'] else False
                continue
            self.logger.warning('Dropped invalid Carbon Copy email ({}) specified.'.format(recipient))
        for recipient in parse_emails(bcc):
            email = is_email(recipient)
            if email:
                self.bcc.add(email['full_email'])
                self.names[email['full_email']] = email['name'] if email['name'] else False
                continue
            self.logger.warning('Dropped invalid Blind Carbon Copy email ({}) specified.'.format(recipient))

    def __post(self, payload, retry):
        if False:
            while True:
                i = 10
        '\n        Performs the actual post and returns the response\n\n        '
        headers = {'User-Agent': self.app_id, 'Accept': 'application/json', 'Content-Type': 'application/json', 'Authorization': self.apikey}
        url = '{}/transmissions/'.format(SPARKPOST_API_LOOKUP[self.region_name])
        self.logger.debug('SparkPost POST URL: {} (cert_verify={})'.format(url, self.verify_certificate))
        if 'attachments' in payload['content']:
            log_payload = {k: v for (k, v) in payload.items() if k != 'content'}
            log_payload['content'] = {k: v for (k, v) in payload['content'].items() if k != 'attachments'}
            log_payload['content']['attachments'] = [{k: v for (k, v) in x.items() if k != 'data'} for x in payload['content']['attachments']]
        else:
            log_payload = payload
        self.logger.debug('SparkPost Payload: {}'.format(log_payload))
        wait = None
        verbose_dest = ', '.join([x['address']['email'] for x in payload['recipients']]) if len(payload['recipients']) <= 3 else '{} recipients'.format(len(payload['recipients']))
        json_response = {}
        status_code = -1
        while 1:
            self.throttle(wait=wait)
            try:
                r = requests.post(url, data=dumps(payload), headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
                try:
                    json_response = loads(r.content)
                except (AttributeError, TypeError, ValueError):
                    pass
                status_code = r.status_code
                payload['recipients'] = list()
                if status_code == requests.codes.ok:
                    self.logger.info('Sent SparkPost notification to {}.'.format(verbose_dest))
                    return (status_code, json_response)
                status_str = NotifyBase.http_response_code_lookup(status_code, SPARKPOST_API_LOOKUP)
                self.logger.warning('Failed to send SparkPost notification to {}: {}{}error={}.'.format(verbose_dest, status_str, ', ' if status_str else '', status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                if status_code == requests.codes.too_many_requests and retry:
                    retry = retry - 1
                    if retry > 0:
                        wait = self.sparkpost_retry_wait_sec
                        continue
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending SparkPost notification')
                self.logger.debug('Socket Exception: %s' % str(e))
            return (status_code, json_response)

    def send(self, body, title='', notify_type=NotifyType.INFO, attach=None, **kwargs):
        if False:
            return 10
        '\n        Perform SparkPost Notification\n        '
        if not self.targets:
            self.logger.warning('There are no Email recipients to notify')
            return False
        has_error = False
        batch_size = 1 if not self.batch else self.default_batch_size
        reply_to = formataddr((self.from_name if self.from_name else False, self.from_addr), charset='utf-8')
        payload = {'options': {'open_tracking': False, 'click_tracking': False}, 'content': {'from': {'name': self.from_name if self.from_name else self.app_desc, 'email': self.from_addr}, 'subject': title if title.strip() else '.', 'reply_to': reply_to}}
        if self.notify_format == NotifyFormat.HTML:
            payload['content']['html'] = body
        else:
            payload['content']['text'] = body
        if attach and self.attachment_support:
            payload['content']['attachments'] = []
            for attachment in attach:
                if not attachment:
                    self.logger.error('Could not access attachment {}.'.format(attachment.url(privacy=True)))
                    return False
                self.logger.debug('Preparing SparkPost attachment {}'.format(attachment.url(privacy=True)))
                try:
                    with open(attachment.path, 'rb') as fp:
                        payload['content']['attachments'].append({'name': attachment.name, 'type': attachment.mimetype, 'data': base64.b64encode(fp.read()).decode('ascii')})
                except (OSError, IOError) as e:
                    self.logger.warning('An I/O error occurred while reading {}.'.format(attachment.name if attachment else 'attachment'))
                    self.logger.debug('I/O Exception: %s' % str(e))
                    return False
        tokens = self.tokens.copy()
        tokens['app_body'] = body
        tokens['app_title'] = title
        tokens['app_type'] = notify_type
        tokens['app_id'] = self.app_id
        tokens['app_desc'] = self.app_desc
        tokens['app_color'] = self.color(notify_type)
        tokens['app_url'] = self.app_url
        payload['substitution_data'] = self.tokens
        emails = list(self.targets)
        for index in range(0, len(emails), batch_size):
            payload['recipients'] = list()
            cc = self.cc - self.bcc
            bcc = set(self.bcc)
            headers = self.headers.copy()
            for addr in self.targets[index:index + batch_size]:
                entry = {'address': {'email': addr[1]}}
                cc = cc - set([addr[1]])
                bcc = bcc - set([addr[1]])
                if addr[0]:
                    entry['address']['name'] = addr[0]
                payload['recipients'].append(entry)
            if cc:
                for addr in cc:
                    entry = {'address': {'email': addr, 'header_to': self.targets[index:index + batch_size][0][1]}}
                    if self.names.get(addr):
                        entry['address']['name'] = self.names[addr]
                    payload['recipients'].append(entry)
                headers['CC'] = ','.join(cc)
            for addr in bcc:
                payload['recipients'].append({'address': {'email': addr, 'header_to': self.targets[index:index + batch_size][0][1]}})
            if headers:
                payload['content']['headers'] = headers
            (status_code, response) = self.__post(payload, self.sparkpost_retry_attempts)
            if status_code != requests.codes.ok:
                has_error = True
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            return 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'region': self.region_name, 'batch': 'yes' if self.batch else 'no'}
        params.update({'+{}'.format(k): v for (k, v) in self.headers.items()})
        params.update({':{}'.format(k): v for (k, v) in self.tokens.items()})
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        if self.from_name is not None:
            params['name'] = self.from_name
        if self.cc:
            params['cc'] = ','.join(['{}{}'.format('' if not e not in self.names else '{}:'.format(self.names[e]), e) for e in self.cc])
        if self.bcc:
            params['bcc'] = ','.join(self.bcc)
        has_targets = not (len(self.targets) == 1 and self.targets[0][1] == self.from_addr)
        return '{schema}://{user}@{host}/{apikey}/{targets}/?{params}'.format(schema=self.secure_protocol, host=self.host, user=NotifySparkPost.quote(self.user, safe=''), apikey=self.pprint(self.apikey, privacy, safe=''), targets='' if not has_targets else '/'.join([NotifySparkPost.quote('{}{}'.format('' if not e[0] else '{}:'.format(e[0]), e[1]), safe='') for e in self.targets]), params=NotifySparkPost.urlencode(params))

    def __len__(self):
        if False:
            print('Hello World!')
        '\n        Returns the number of targets associated with this notification\n        '
        batch_size = 1 if not self.batch else self.default_batch_size
        targets = len(self.targets)
        if batch_size > 1:
            targets = int(targets / batch_size) + (1 if targets % batch_size else 0)
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
        results['targets'] = NotifySparkPost.split_path(results['fullpath'])
        try:
            results['apikey'] = results['targets'].pop(0)
        except IndexError:
            results['apikey'] = None
        if 'name' in results['qsd'] and len(results['qsd']['name']):
            results['from_name'] = NotifySparkPost.unquote(results['qsd']['name'])
        if 'region' in results['qsd'] and len(results['qsd']['region']):
            results['region_name'] = NotifySparkPost.unquote(results['qsd']['region'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'].append(results['qsd']['to'])
        if 'cc' in results['qsd'] and len(results['qsd']['cc']):
            results['cc'] = results['qsd']['cc']
        if 'bcc' in results['qsd'] and len(results['qsd']['bcc']):
            results['bcc'] = results['qsd']['bcc']
        results['headers'] = {NotifyBase.unquote(x): NotifyBase.unquote(y) for (x, y) in results['qsd+'].items()}
        results['tokens'] = {NotifyBase.unquote(x): NotifyBase.unquote(y) for (x, y) in results['qsd:'].items()}
        results['batch'] = parse_bool(results['qsd'].get('batch', NotifySparkPost.template_args['batch']['default']))
        return results