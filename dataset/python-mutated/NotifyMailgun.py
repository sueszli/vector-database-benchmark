import requests
from email.utils import formataddr
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..common import NotifyFormat
from ..utils import parse_emails
from ..utils import parse_bool
from ..utils import is_email
from ..utils import validate_regex
from ..logger import logger
from ..AppriseLocale import gettext_lazy as _
MAILGUN_HTTP_ERROR_MAP = {400: 'A bad request was made to the server.', 401: 'The provided API Key was not valid.', 402: 'The request failed for a reason out of your control.', 404: 'The requested API query is not valid.', 413: 'Provided attachment is to big.'}

class MailgunRegion:
    US = 'us'
    EU = 'eu'
MAILGUN_API_LOOKUP = {MailgunRegion.US: 'https://api.mailgun.net/v3/', MailgunRegion.EU: 'https://api.eu.mailgun.net/v3/'}
MAILGUN_REGIONS = (MailgunRegion.US, MailgunRegion.EU)

class NotifyMailgun(NotifyBase):
    """
    A wrapper for Mailgun Notifications
    """
    service_name = 'Mailgun'
    service_url = 'https://www.mailgun.com/'
    secure_protocol = 'mailgun'
    request_rate_per_sec = 0.2
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_mailgun'
    attachment_support = True
    notify_format = NotifyFormat.HTML
    default_batch_size = 2000
    templates = ('{schema}://{user}@{host}:{apikey}/', '{schema}://{user}@{host}:{apikey}/{targets}')
    template_tokens = dict(NotifyBase.template_tokens, **{'user': {'name': _('User Name'), 'type': 'string', 'required': True}, 'host': {'name': _('Domain'), 'type': 'string', 'required': True}, 'apikey': {'name': _('API Key'), 'type': 'string', 'private': True, 'required': True}, 'target_email': {'name': _('Target Email'), 'type': 'string', 'map_to': 'targets'}, 'targets': {'name': _('Targets'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'name': {'name': _('From Name'), 'type': 'string', 'map_to': 'from_addr'}, 'from': {'alias_of': 'name'}, 'region': {'name': _('Region Name'), 'type': 'choice:string', 'values': MAILGUN_REGIONS, 'default': MailgunRegion.US, 'map_to': 'region_name'}, 'to': {'alias_of': 'targets'}, 'cc': {'name': _('Carbon Copy'), 'type': 'list:string'}, 'bcc': {'name': _('Blind Carbon Copy'), 'type': 'list:string'}, 'batch': {'name': _('Batch Mode'), 'type': 'bool', 'default': False}})
    template_kwargs = {'headers': {'name': _('Email Header'), 'prefix': '+'}, 'tokens': {'name': _('Template Tokens'), 'prefix': ':'}}

    def __init__(self, apikey, targets, cc=None, bcc=None, from_addr=None, region_name=None, headers=None, tokens=None, batch=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize Mailgun Object\n        '
        super().__init__(**kwargs)
        self.apikey = validate_regex(apikey)
        if not self.apikey:
            msg = 'An invalid Mailgun API Key ({}) was specified.'.format(apikey)
            self.logger.warning(msg)
            raise TypeError(msg)
        if not self.user:
            msg = 'No Mailgun username was specified.'
            self.logger.warning(msg)
            raise TypeError(msg)
        self.targets = list()
        self.cc = set()
        self.bcc = set()
        self.names = {}
        self.headers = {}
        if headers:
            self.headers.update(headers)
        self.tokens = {}
        if tokens:
            self.tokens.update(tokens)
        self.batch = batch
        try:
            self.region_name = NotifyMailgun.template_args['region']['default'] if region_name is None else region_name.lower()
            if self.region_name not in MAILGUN_REGIONS:
                raise
        except:
            msg = 'The Mailgun region specified ({}) is invalid.'.format(region_name)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.from_addr = [self.app_id, '{user}@{host}'.format(user=self.user, host=self.host)]
        if from_addr:
            result = is_email(from_addr)
            if result:
                self.from_addr = (result['name'] if result['name'] else False, result['full_email'])
            else:
                self.from_addr[0] = from_addr
        if not is_email(self.from_addr[1]):
            msg = 'Invalid ~From~ email format: {}'.format(self.from_addr)
            self.logger.warning(msg)
            raise TypeError(msg)
        if targets:
            for recipient in parse_emails(targets):
                result = is_email(recipient)
                if result:
                    self.targets.append((result['name'] if result['name'] else False, result['full_email']))
                    continue
                self.logger.warning('Dropped invalid To email ({}) specified.'.format(recipient))
        else:
            self.targets.append((False, self.from_addr[1]))
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

    def send(self, body, title='', notify_type=NotifyType.INFO, attach=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Perform Mailgun Notification\n        '
        if not self.targets:
            self.logger.warning('There are no Email recipients to notify')
            return False
        has_error = False
        batch_size = 1 if not self.batch else self.default_batch_size
        headers = {'User-Agent': self.app_id, 'Accept': 'application/json'}
        files = {}
        if attach and self.attachment_support:
            for (idx, attachment) in enumerate(attach):
                if not attachment:
                    self.logger.error('Could not access attachment {}.'.format(attachment.url(privacy=True)))
                    return False
                self.logger.debug('Preparing Mailgun attachment {}'.format(attachment.url(privacy=True)))
                try:
                    files['attachment[{}]'.format(idx)] = (attachment.name, open(attachment.path, 'rb'))
                except (OSError, IOError) as e:
                    self.logger.warning('An I/O error occurred while opening {}.'.format(attachment.name if attachment else 'attachment'))
                    self.logger.debug('I/O Exception: %s' % str(e))
                    for entry in files.values():
                        self.logger.trace('Closing attachment {}'.format(entry[0]))
                        entry[1].close()
                    return False
        reply_to = formataddr(self.from_addr, charset='utf-8')
        payload = {'o:skip-verification': not self.verify_certificate, 'from': reply_to, 'subject': title}
        if self.notify_format == NotifyFormat.HTML:
            payload['html'] = body
        else:
            payload['text'] = body
        url = '{}{}/messages'.format(MAILGUN_API_LOOKUP[self.region_name], self.host)
        emails = list(self.targets)
        for index in range(0, len(emails), batch_size):
            cc = self.cc - self.bcc
            bcc = set(self.bcc)
            to = list()
            for entry in files.values():
                try:
                    self.logger.trace('Seeking to head of attachment {}'.format(entry[0]))
                    entry[1].seek(0)
                except (OSError, IOError) as e:
                    self.logger.warning('An I/O error occurred seeking to head of attachment {}.'.format(entry[0]))
                    self.logger.debug('I/O Exception: %s' % str(e))
                    for entry in files.values():
                        self.logger.trace('Closing attachment {}'.format(entry[0]))
                        entry[1].close()
                    return False
            for to_addr in self.targets[index:index + batch_size]:
                cc = cc - set([to_addr[1]])
                bcc = bcc - set([to_addr[1]])
                to.append(formataddr(to_addr, charset='utf-8'))
            payload['to'] = ','.join(to)
            if cc:
                payload['cc'] = ','.join([formataddr((self.names.get(addr, False), addr), charset='utf-8') for addr in cc])
            if bcc:
                payload['bcc'] = ','.join(bcc)
            if self.tokens:
                payload.update({'v:{}'.format(k): v for (k, v) in self.tokens.items()})
            if self.headers:
                payload.update({'h:{}'.format(k): v for (k, v) in self.headers.items()})
            self.logger.debug('Mailgun POST URL: {} (cert_verify={})'.format(url, self.verify_certificate))
            self.logger.debug('Mailgun Payload: {}'.format(payload))
            verbose_dest = ', '.join([x[1] for x in self.targets[index:index + batch_size]]) if len(self.targets[index:index + batch_size]) <= 3 else '{} recipients'.format(len(self.targets[index:index + batch_size]))
            self.throttle()
            try:
                r = requests.post(url, auth=('api', self.apikey), data=payload, headers=headers, files=None if not files else files, verify=self.verify_certificate, timeout=self.request_timeout)
                if r.status_code != requests.codes.ok:
                    status_str = NotifyBase.http_response_code_lookup(r.status_code, MAILGUN_HTTP_ERROR_MAP)
                    self.logger.warning('Failed to send Mailgun notification to {}: {}{}error={}.'.format(verbose_dest, status_str, ', ' if status_str else '', r.status_code))
                    self.logger.debug('Response Details:\r\n{}'.format(r.content))
                    has_error = True
                    continue
                else:
                    self.logger.info('Sent Mailgun notification to {}.'.format(verbose_dest))
            except requests.RequestException as e:
                self.logger.warning('A Connection error occurred sending Mailgun:%s ' % verbose_dest + 'notification.')
                self.logger.debug('Socket Exception: %s' % str(e))
                has_error = True
                continue
            except (OSError, IOError) as e:
                self.logger.warning('An I/O error occurred while reading attachments')
                self.logger.debug('I/O Exception: %s' % str(e))
                has_error = True
                continue
        for entry in files.values():
            self.logger.trace('Closing attachment {}'.format(entry[0]))
            entry[1].close()
        return not has_error

    def url(self, privacy=False, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'region': self.region_name, 'batch': 'yes' if self.batch else 'no'}
        params.update({'+{}'.format(k): v for (k, v) in self.headers.items()})
        params.update({':{}'.format(k): v for (k, v) in self.tokens.items()})
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        if self.from_addr[0]:
            params['name'] = self.from_addr[0]
        if self.cc:
            params['cc'] = ','.join(['{}{}'.format('' if not e not in self.names else '{}:'.format(self.names[e]), e) for e in self.cc])
        if self.bcc:
            params['bcc'] = ','.join(self.bcc)
        has_targets = not (len(self.targets) == 1 and self.targets[0][1] == self.from_addr)
        return '{schema}://{user}@{host}/{apikey}/{targets}/?{params}'.format(schema=self.secure_protocol, host=self.host, user=NotifyMailgun.quote(self.user, safe=''), apikey=self.pprint(self.apikey, privacy, safe=''), targets='' if not has_targets else '/'.join([NotifyMailgun.quote('{}{}'.format('' if not e[0] else '{}:'.format(e[0]), e[1]), safe='') for e in self.targets]), params=NotifyMailgun.urlencode(params))

    def __len__(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the number of targets associated with this notification\n        '
        batch_size = 1 if not self.batch else self.default_batch_size
        targets = len(self.targets)
        if batch_size > 1:
            targets = int(targets / batch_size) + (1 if targets % batch_size else 0)
        return targets if targets > 0 else 1

    @staticmethod
    def parse_url(url):
        if False:
            return 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        results['targets'] = NotifyMailgun.split_path(results['fullpath'])
        try:
            results['apikey'] = results['targets'].pop(0)
        except IndexError:
            results['apikey'] = None
        if 'from' in results['qsd'] and len(results['qsd']['from']):
            results['from_addr'] = NotifyMailgun.unquote(results['qsd']['from'])
            if 'name' in results['qsd'] and len(results['qsd']['name']):
                results['from_addr'] = formataddr((NotifyMailgun.unquote(results['qsd']['name']), results['from_addr']), charset='utf-8')
                logger.warning('Mailgun name= and from= are synonymous; use one or the other.')
        elif 'name' in results['qsd'] and len(results['qsd']['name']):
            results['from_addr'] = NotifyMailgun.unquote(results['qsd']['name'])
        if 'region' in results['qsd'] and len(results['qsd']['region']):
            results['region_name'] = NotifyMailgun.unquote(results['qsd']['region'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'].append(results['qsd']['to'])
        if 'cc' in results['qsd'] and len(results['qsd']['cc']):
            results['cc'] = results['qsd']['cc']
        if 'bcc' in results['qsd'] and len(results['qsd']['bcc']):
            results['bcc'] = results['qsd']['bcc']
        results['headers'] = {NotifyBase.unquote(x): NotifyBase.unquote(y) for (x, y) in results['qsd+'].items()}
        results['tokens'] = {NotifyBase.unquote(x): NotifyBase.unquote(y) for (x, y) in results['qsd:'].items()}
        results['batch'] = parse_bool(results['qsd'].get('batch', NotifyMailgun.template_args['batch']['default']))
        return results