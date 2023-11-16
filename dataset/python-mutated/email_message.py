"""A convenience class around email.Message and email.MIMEMultipart."""
from __future__ import absolute_import
from email import Header, Message, MIMEMultipart, MIMEText, Utils
from bzrlib import __version__ as _bzrlib_version
from bzrlib.osutils import safe_unicode
from bzrlib.smtp_connection import SMTPConnection

class EmailMessage(object):
    """An email message.

    The constructor needs an origin address, a destination address or addresses
    and a subject, and accepts a body as well. Add additional parts to the
    message with add_inline_attachment(). Retrieve the entire formatted message
    with as_string().

    Headers can be accessed with get() and msg[], and modified with msg[] =.
    """

    def __init__(self, from_address, to_address, subject, body=None):
        if False:
            while True:
                i = 10
        "Create an email message.\n\n        :param from_address: The origin address, to be put on the From header.\n        :param to_address: The destination address of the message, to be put in\n            the To header. Can also be a list of addresses.\n        :param subject: The subject of the message.\n        :param body: If given, the body of the message.\n\n        All four parameters can be unicode strings or byte strings, but for the\n        addresses and subject byte strings must be encoded in UTF-8. For the\n        body any byte string will be accepted; if it's not ASCII or UTF-8,\n        it'll be sent with charset=8-bit.\n        "
        self._headers = {}
        self._body = body
        self._parts = []
        if isinstance(to_address, basestring):
            to_address = [to_address]
        to_addresses = []
        for addr in to_address:
            to_addresses.append(self.address_to_encoded_header(addr))
        self._headers['To'] = ', '.join(to_addresses)
        self._headers['From'] = self.address_to_encoded_header(from_address)
        self._headers['Subject'] = Header.Header(safe_unicode(subject))
        self._headers['User-Agent'] = 'Bazaar (%s)' % _bzrlib_version

    def add_inline_attachment(self, body, filename=None, mime_subtype='plain'):
        if False:
            return 10
        "Add an inline attachment to the message.\n\n        :param body: A text to attach. Can be an unicode string or a byte\n            string, and it'll be sent as ascii, utf-8, or 8-bit, in that\n            preferred order.\n        :param filename: The name for the attachment. This will give a default\n            name for email programs to save the attachment.\n        :param mime_subtype: MIME subtype of the attachment (eg. 'plain' for\n            text/plain [default]).\n\n        The attachment body will be displayed inline, so do not use this\n        function to attach binary attachments.\n        "
        if self._body is not None:
            self._parts.append((self._body, None, 'plain'))
            self._body = None
        self._parts.append((body, filename, mime_subtype))

    def as_string(self, boundary=None):
        if False:
            while True:
                i = 10
        'Return the entire formatted message as a string.\n\n        :param boundary: The boundary to use between MIME parts, if applicable.\n            Used for tests.\n        '
        if not self._parts:
            msgobj = Message.Message()
            if self._body is not None:
                (body, encoding) = self.string_with_encoding(self._body)
                msgobj.set_payload(body, encoding)
        else:
            msgobj = MIMEMultipart.MIMEMultipart()
            if boundary is not None:
                msgobj.set_boundary(boundary)
            for (body, filename, mime_subtype) in self._parts:
                (body, encoding) = self.string_with_encoding(body)
                payload = MIMEText.MIMEText(body, mime_subtype, encoding)
                if filename is not None:
                    content_type = payload['Content-Type']
                    content_type += '; name="%s"' % filename
                    payload.replace_header('Content-Type', content_type)
                payload['Content-Disposition'] = 'inline'
                msgobj.attach(payload)
        for (header, value) in sorted(self._headers.items()):
            msgobj[header] = value
        return msgobj.as_string()
    __str__ = as_string

    def get(self, header, failobj=None):
        if False:
            for i in range(10):
                print('nop')
        'Get a header from the message, returning failobj if not present.'
        return self._headers.get(header, failobj)

    def __getitem__(self, header):
        if False:
            return 10
        'Get a header from the message, returning None if not present.\n\n        This method intentionally does not raise KeyError to mimic the behavior\n        of __getitem__ in email.Message.\n        '
        return self._headers.get(header, None)

    def __setitem__(self, header, value):
        if False:
            print('Hello World!')
        return self._headers.__setitem__(header, value)

    @staticmethod
    def send(config, from_address, to_address, subject, body, attachment=None, attachment_filename=None, attachment_mime_subtype='plain'):
        if False:
            while True:
                i = 10
        'Create an email message and send it with SMTPConnection.\n\n        :param config: config object to pass to SMTPConnection constructor.\n\n        See EmailMessage.__init__() and EmailMessage.add_inline_attachment()\n        for an explanation of the rest of parameters.\n        '
        msg = EmailMessage(from_address, to_address, subject, body)
        if attachment is not None:
            msg.add_inline_attachment(attachment, attachment_filename, attachment_mime_subtype)
        SMTPConnection(config).send_email(msg)

    @staticmethod
    def address_to_encoded_header(address):
        if False:
            while True:
                i = 10
        'RFC2047-encode an address if necessary.\n\n        :param address: An unicode string, or UTF-8 byte string.\n        :return: A possibly RFC2047-encoded string.\n        '
        (user, email) = Utils.parseaddr(address)
        if not user:
            return email
        else:
            return Utils.formataddr((str(Header.Header(safe_unicode(user))), email))

    @staticmethod
    def string_with_encoding(string_):
        if False:
            print('Hello World!')
        "Return a str object together with an encoding.\n\n        :param string\\_: A str or unicode object.\n        :return: A tuple (str, encoding), where encoding is one of 'ascii',\n            'utf-8', or '8-bit', in that preferred order.\n        "
        if isinstance(string_, unicode):
            try:
                return (string_.encode('ascii'), 'ascii')
            except UnicodeEncodeError:
                return (string_.encode('utf-8'), 'utf-8')
        else:
            try:
                string_.decode('ascii')
                return (string_, 'ascii')
            except UnicodeDecodeError:
                try:
                    string_.decode('utf-8')
                    return (string_, 'utf-8')
                except UnicodeDecodeError:
                    return (string_, '8-bit')