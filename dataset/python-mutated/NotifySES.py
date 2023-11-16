import re
import hmac
import base64
import requests
from hashlib import sha256
from datetime import datetime
from datetime import timezone
from collections import OrderedDict
from xml.etree import ElementTree
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
from email.header import Header
from urllib.parse import quote
from .NotifyBase import NotifyBase
from ..URLBase import PrivacyMode
from ..common import NotifyFormat
from ..common import NotifyType
from ..utils import parse_emails
from ..utils import validate_regex
from ..AppriseLocale import gettext_lazy as _
from ..utils import is_email
IS_REGION = re.compile('^\\s*(?P<country>[a-z]{2})-(?P<area>[a-z-]+?)-(?P<no>[0-9]+)\\s*$', re.I)
AWS_HTTP_ERROR_MAP = {403: 'Unauthorized - Invalid Access/Secret Key Combination.'}

class NotifySES(NotifyBase):
    """
    A wrapper for AWS SES (Amazon Simple Email Service)
    """
    service_name = 'AWS Simple Email Service (SES)'
    service_url = 'https://aws.amazon.com/ses/'
    secure_protocol = 'ses'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_ses'
    attachment_support = True
    request_rate_per_sec = 2.5
    notify_format = NotifyFormat.HTML
    templates = ('{schema}://{from_email}/{access_key_id}/{secret_access_key}/{region}/{targets}', '{schema}://{from_email}/{access_key_id}/{secret_access_key}/{region}')
    template_tokens = dict(NotifyBase.template_tokens, **{'from_email': {'name': _('From Email'), 'type': 'string', 'map_to': 'from_addr', 'required': True}, 'access_key_id': {'name': _('Access Key ID'), 'type': 'string', 'private': True, 'required': True}, 'secret_access_key': {'name': _('Secret Access Key'), 'type': 'string', 'private': True, 'required': True}, 'region': {'name': _('Region'), 'type': 'string', 'regex': ('^[a-z]{2}-[a-z-]+?-[0-9]+$', 'i'), 'required': True, 'map_to': 'region_name'}, 'targets': {'name': _('Target Emails'), 'type': 'list:string'}})
    template_args = dict(NotifyBase.template_args, **{'to': {'alias_of': 'targets'}, 'from': {'alias_of': 'from_email'}, 'reply': {'name': _('Reply To Email'), 'type': 'string', 'map_to': 'reply_to'}, 'name': {'name': _('From Name'), 'type': 'string', 'map_to': 'from_name'}, 'cc': {'name': _('Carbon Copy'), 'type': 'list:string'}, 'bcc': {'name': _('Blind Carbon Copy'), 'type': 'list:string'}, 'access': {'alias_of': 'access_key_id'}, 'secret': {'alias_of': 'secret_access_key'}, 'region': {'alias_of': 'region'}})

    def __init__(self, access_key_id, secret_access_key, region_name, reply_to=None, from_addr=None, from_name=None, targets=None, cc=None, bcc=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Initialize Notify AWS SES Object\n        '
        super().__init__(**kwargs)
        self.aws_access_key_id = validate_regex(access_key_id)
        if not self.aws_access_key_id:
            msg = 'An invalid AWS Access Key ID was specified.'
            self.logger.warning(msg)
            raise TypeError(msg)
        self.aws_secret_access_key = validate_regex(secret_access_key)
        if not self.aws_secret_access_key:
            msg = 'An invalid AWS Secret Access Key ({}) was specified.'.format(secret_access_key)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.aws_region_name = validate_regex(region_name, *self.template_tokens['region']['regex'])
        if not self.aws_region_name:
            msg = 'An invalid AWS Region ({}) was specified.'.format(region_name)
            self.logger.warning(msg)
            raise TypeError(msg)
        self.targets = list()
        self.cc = set()
        self.bcc = set()
        self.names = {}
        self.notify_url = 'https://email.{}.amazonaws.com'.format(self.aws_region_name)
        self.aws_service_name = 'ses'
        self.aws_canonical_uri = '/'
        self.aws_auth_version = 'AWS4'
        self.aws_auth_algorithm = 'AWS4-HMAC-SHA256'
        self.aws_auth_request = 'aws4_request'
        self.from_name = from_name
        if from_addr:
            self.from_addr = from_addr
        else:
            self.from_addr = '{user}@{host}'.format(user=self.user, host=self.host) if self.user else None
        if not (self.from_addr and is_email(self.from_addr)):
            msg = 'An invalid AWS From ({}) was specified.'.format('{user}@{host}'.format(user=self.user, host=self.host))
            self.logger.warning(msg)
            raise TypeError(msg)
        self.reply_to = None
        if reply_to:
            result = is_email(reply_to)
            if not result:
                msg = 'An invalid AWS Reply To ({}) was specified.'.format('{user}@{host}'.format(user=self.user, host=self.host))
                self.logger.warning(msg)
                raise TypeError(msg)
            self.reply_to = (result['name'] if result['name'] else False, result['full_email'])
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
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, attach=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        wrapper to send_notification since we can alert more then one channel\n        '
        if not self.targets:
            self.logger.warning('There are no SES email recipients to notify')
            return False
        has_error = False
        from_name = self.from_name if self.from_name else self.reply_to[0] if self.reply_to and self.reply_to[0] else self.app_desc
        reply_to = (from_name, self.from_addr if not self.reply_to else self.reply_to[1])
        emails = list(self.targets)
        while len(emails):
            (to_name, to_addr) = emails.pop(0)
            cc = self.cc - self.bcc - set([to_addr])
            bcc = self.bcc - set([to_addr])
            cc = [formataddr((self.names.get(addr, False), addr), charset='utf-8') for addr in cc]
            bcc = [formataddr((self.names.get(addr, False), addr), charset='utf-8') for addr in bcc]
            self.logger.debug('Email From: {} <{}>'.format(quote(reply_to[0], ' '), quote(reply_to[1], '@ ')))
            self.logger.debug('Email To: {}'.format(to_addr))
            if cc:
                self.logger.debug('Email Cc: {}'.format(', '.join(cc)))
            if bcc:
                self.logger.debug('Email Bcc: {}'.format(', '.join(bcc)))
            if self.notify_format == NotifyFormat.HTML:
                content = MIMEText(body, 'html', 'utf-8')
            else:
                content = MIMEText(body, 'plain', 'utf-8')
            base = MIMEMultipart() if attach and self.attachment_support else content
            base['Subject'] = Header(title, 'utf-8')
            base['From'] = formataddr((from_name if from_name else False, self.from_addr), charset='utf-8')
            base['To'] = formataddr((to_name, to_addr), charset='utf-8')
            if reply_to[1] != self.from_addr:
                base['Reply-To'] = formataddr(reply_to, charset='utf-8')
            base['Cc'] = ','.join(cc)
            base['Date'] = datetime.now(timezone.utc).strftime('%a, %d %b %Y %H:%M:%S +0000')
            base['X-Application'] = self.app_id
            if attach and self.attachment_support:
                base.attach(content)
                for attachment in attach:
                    if not attachment:
                        self.logger.error('Could not access attachment {}.'.format(attachment.url(privacy=True)))
                        return False
                    self.logger.debug('Preparing Email attachment {}'.format(attachment.url(privacy=True)))
                    with open(attachment.path, 'rb') as abody:
                        app = MIMEApplication(abody.read())
                        app.set_type(attachment.mimetype)
                        app.add_header('Content-Disposition', 'attachment; filename="{}"'.format(Header(attachment.name, 'utf-8')))
                        base.attach(app)
            payload = {'Action': 'SendRawEmail', 'Version': '2010-12-01', 'RawMessage.Data': base64.b64encode(base.as_string().encode('utf-8')).decode('utf-8')}
            for (no, email) in enumerate([to_addr] + bcc + cc, start=1):
                payload['Destinations.member.{}'.format(no)] = email
            payload['Source'] = '{} <{}>'.format(quote(from_name, ' '), quote(self.from_addr, '@ '))
            (result, response) = self._post(payload=payload, to=to_addr)
            if not result:
                has_error = True
                continue
        return not has_error

    def _post(self, payload, to):
        if False:
            return 10
        "\n        Wrapper to request.post() to manage it's response better and make\n        the send() function cleaner and easier to maintain.\n\n        This function returns True if the _post was successful and False\n        if it wasn't.\n        "
        self.throttle()
        payload = NotifySES.urlencode(payload)
        headers = self.aws_prepare_request(payload)
        self.logger.debug('AWS SES POST URL: %s (cert_verify=%r)' % (self.notify_url, self.verify_certificate))
        self.logger.debug('AWS SES Payload (%d bytes)', len(payload))
        try:
            r = requests.post(self.notify_url, data=payload, headers=headers, verify=self.verify_certificate, timeout=self.request_timeout)
            if r.status_code != requests.codes.ok:
                status_str = NotifySES.http_response_code_lookup(r.status_code, AWS_HTTP_ERROR_MAP)
                self.logger.warning('Failed to send AWS SES notification to {}: {}{}error={}.'.format(to, status_str, ', ' if status_str else '', r.status_code))
                self.logger.debug('Response Details:\r\n{}'.format(r.content))
                return (False, NotifySES.aws_response_to_dict(r.text))
            else:
                self.logger.info('Sent AWS SES notification to "%s".' % to)
        except requests.RequestException as e:
            self.logger.warning('A Connection error occurred sending AWS SES notification to "%s".' % to)
            self.logger.debug('Socket Exception: %s' % str(e))
            return (False, NotifySES.aws_response_to_dict(None))
        return (True, NotifySES.aws_response_to_dict(r.text))

    def aws_prepare_request(self, payload, reference=None):
        if False:
            print('Hello World!')
        '\n        Takes the intended payload and returns the headers for it.\n\n        The payload is presumed to have been already urlencoded()\n\n        '
        headers = {'User-Agent': self.app_id, 'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8', 'Content-Length': 0, 'Authorization': None, 'X-Amz-Date': None}
        reference = datetime.now(timezone.utc)
        headers['Content-Length'] = str(len(payload))
        amzdate = reference.strftime('%Y%m%dT%H%M%SZ')
        headers['X-Amz-Date'] = amzdate
        scope = '{date}/{region}/{service}/{request}'.format(date=reference.strftime('%Y%m%d'), region=self.aws_region_name, service=self.aws_service_name, request=self.aws_auth_request)
        signed_headers = OrderedDict([('content-type', headers['Content-Type']), ('host', 'email.{region}.amazonaws.com'.format(region=self.aws_region_name)), ('x-amz-date', headers['X-Amz-Date'])])
        canonical_request = '\n'.join([u'POST', self.aws_canonical_uri, '', '\n'.join(['%s:%s' % (k, v) for (k, v) in signed_headers.items()]) + '\n', ';'.join(signed_headers.keys()), sha256(payload.encode('utf-8')).hexdigest()])
        to_sign = '\n'.join([self.aws_auth_algorithm, amzdate, scope, sha256(canonical_request.encode('utf-8')).hexdigest()])
        headers['Authorization'] = ', '.join(['{algorithm} Credential={key}/{scope}'.format(algorithm=self.aws_auth_algorithm, key=self.aws_access_key_id, scope=scope), 'SignedHeaders={signed_headers}'.format(signed_headers=';'.join(signed_headers.keys())), 'Signature={signature}'.format(signature=self.aws_auth_signature(to_sign, reference))])
        return headers

    def aws_auth_signature(self, to_sign, reference):
        if False:
            i = 10
            return i + 15
        '\n        Generates a AWS v4 signature based on provided payload\n        which should be in the form of a string.\n        '

        def _sign(key, msg, to_hex=False):
            if False:
                while True:
                    i = 10
            '\n            Perform AWS Signing\n            '
            if to_hex:
                return hmac.new(key, msg.encode('utf-8'), sha256).hexdigest()
            return hmac.new(key, msg.encode('utf-8'), sha256).digest()
        _date = _sign((self.aws_auth_version + self.aws_secret_access_key).encode('utf-8'), reference.strftime('%Y%m%d'))
        _region = _sign(_date, self.aws_region_name)
        _service = _sign(_region, self.aws_service_name)
        _signed = _sign(_service, self.aws_auth_request)
        return _sign(_signed, to_sign, to_hex=True)

    @staticmethod
    def aws_response_to_dict(aws_response):
        if False:
            i = 10
            return i + 15
        '\n        Takes an AWS Response object as input and returns it as a dictionary\n        but not befor extracting out what is useful to us first.\n\n        eg:\n          IN:\n\n            <SendRawEmailResponse\n                 xmlns="http://ses.amazonaws.com/doc/2010-12-01/">\n              <SendRawEmailResult>\n                <MessageId>\n                   010f017d87656ee2-a2ea291f-79ea-\n                   44f3-9d25-00d041de3007-000000</MessageId>\n              </SendRawEmailResult>\n              <ResponseMetadata>\n                <RequestId>7abb454e-904b-4e46-a23c-2f4d2fc127a6</RequestId>\n              </ResponseMetadata>\n            </SendRawEmailResponse>\n\n          OUT:\n           {\n             \'type\': \'SendRawEmailResponse\',\n              \'message_id\': \'010f017d87656ee2-a2ea291f-79ea-\n                             44f3-9d25-00d041de3007-000000\',\n              \'request_id\': \'7abb454e-904b-4e46-a23c-2f4d2fc127a6\',\n           }\n        '
        aws_keep_map = {'RequestId': 'request_id', 'MessageId': 'message_id', 'Type': 'error_type', 'Code': 'error_code', 'Message': 'error_message'}
        response = {'type': None, 'request_id': None, 'message_id': None}
        try:
            root = ElementTree.fromstring(re.sub(' xmlns="[^"]+"', '', aws_response, count=1))
            response['type'] = str(root.tag)

            def _xml_iter(root, response):
                if False:
                    for i in range(10):
                        print('nop')
                if len(root) > 0:
                    for child in root:
                        _xml_iter(child, response)
                elif root.tag in aws_keep_map.keys():
                    response[aws_keep_map[root.tag]] = root.text.strip()
            _xml_iter(root, response)
        except (ElementTree.ParseError, TypeError):
            pass
        return response

    def url(self, privacy=False, *args, **kwargs):
        if False:
            return 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = self.url_parameters(*args, privacy=privacy, **kwargs)
        if self.from_name is not None:
            params['name'] = self.from_name
        if self.cc:
            params['cc'] = ','.join(['{}{}'.format('' if not e not in self.names else '{}:'.format(self.names[e]), e) for e in self.cc])
        if self.bcc:
            params['bcc'] = ','.join(self.bcc)
        if self.reply_to:
            params['reply'] = '{} <{}>'.format(*self.reply_to) if self.reply_to[0] else self.reply_to[1]
        has_targets = not (len(self.targets) == 1 and self.targets[0][1] == self.from_addr)
        return '{schema}://{from_addr}/{key_id}/{key_secret}/{region}/{targets}/?{params}'.format(schema=self.secure_protocol, from_addr=NotifySES.quote(self.from_addr, safe='@'), key_id=self.pprint(self.aws_access_key_id, privacy, safe=''), key_secret=self.pprint(self.aws_secret_access_key, privacy, mode=PrivacyMode.Secret, safe=''), region=NotifySES.quote(self.aws_region_name, safe=''), targets='' if not has_targets else '/'.join([NotifySES.quote('{}{}'.format('' if not e[0] else '{}:'.format(e[0]), e[1]), safe='') for e in self.targets]), params=NotifySES.urlencode(params))

    def __len__(self):
        if False:
            return 10
        '\n        Returns the number of targets associated with this notification\n        '
        targets = len(self.targets)
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
        entries = NotifySES.split_path(results['fullpath'])
        access_key_id = entries.pop(0) if entries else None
        secret_access_key = None
        region_name = None
        secret_access_key_parts = list()
        index = 0
        for (index, entry) in enumerate(entries, start=1):
            result = IS_REGION.match(entry)
            if result:
                region_name = '{country}-{area}-{no}'.format(country=result.group('country').lower(), area=result.group('area').lower(), no=result.group('no'))
                break
            elif is_email(entry):
                index -= 1
                break
            secret_access_key_parts.append(entry)
        secret_access_key = '/'.join(secret_access_key_parts) if secret_access_key_parts else None
        results['targets'] = entries[index:]
        if 'name' in results['qsd'] and len(results['qsd']['name']):
            results['from_name'] = NotifySES.unquote(results['qsd']['name'])
        if 'to' in results['qsd'] and len(results['qsd']['to']):
            results['targets'].append(results['qsd']['to'])
        if 'cc' in results['qsd'] and len(results['qsd']['cc']):
            results['cc'] = NotifySES.parse_list(results['qsd']['cc'])
        if 'bcc' in results['qsd'] and len(results['qsd']['bcc']):
            results['bcc'] = NotifySES.parse_list(results['qsd']['bcc'])
        if 'from' in results['qsd'] and len(results['qsd']['from']):
            results['from_addr'] = NotifySES.unquote(results['qsd']['from'])
        if 'reply' in results['qsd'] and len(results['qsd']['reply']):
            results['reply_to'] = NotifySES.unquote(results['qsd']['reply'])
        if 'secret' in results['qsd'] and len(results['qsd']['secret']):
            results['secret_access_key'] = NotifySES.unquote(results['qsd']['secret'])
        else:
            results['secret_access_key'] = secret_access_key
        if 'access' in results['qsd'] and len(results['qsd']['access']):
            results['access_key_id'] = NotifySES.unquote(results['qsd']['access'])
        else:
            results['access_key_id'] = access_key_id
        if 'region' in results['qsd'] and len(results['qsd']['region']):
            results['region_name'] = NotifySES.unquote(results['qsd']['region'])
        else:
            results['region_name'] = region_name
        return results