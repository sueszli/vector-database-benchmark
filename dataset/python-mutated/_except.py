"""
Exceptions in L{twisted.mail}.
"""
from typing import Optional

class IMAP4Exception(Exception):
    pass

class IllegalClientResponse(IMAP4Exception):
    pass

class IllegalOperation(IMAP4Exception):
    pass

class IllegalMailboxEncoding(IMAP4Exception):
    pass

class MailboxException(IMAP4Exception):
    pass

class MailboxCollision(MailboxException):

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'Mailbox named %s already exists' % self.args

class NoSuchMailbox(MailboxException):

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        return 'No mailbox named %s exists' % self.args

class ReadOnlyMailbox(MailboxException):

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'Mailbox open in read-only state'

class UnhandledResponse(IMAP4Exception):
    pass

class NegativeResponse(IMAP4Exception):
    pass

class NoSupportedAuthentication(IMAP4Exception):

    def __init__(self, serverSupports, clientSupports):
        if False:
            i = 10
            return i + 15
        IMAP4Exception.__init__(self, 'No supported authentication schemes available')
        self.serverSupports = serverSupports
        self.clientSupports = clientSupports

    def __str__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return IMAP4Exception.__str__(self) + ': Server supports {!r}, client supports {!r}'.format(self.serverSupports, self.clientSupports)

class IllegalServerResponse(IMAP4Exception):
    pass

class IllegalIdentifierError(IMAP4Exception):
    pass

class IllegalQueryError(IMAP4Exception):
    pass

class MismatchedNesting(IMAP4Exception):
    pass

class MismatchedQuoting(IMAP4Exception):
    pass

class SMTPError(Exception):
    pass

class SMTPClientError(SMTPError):
    """
    Base class for SMTP client errors.
    """

    def __init__(self, code: int, resp: bytes, log: Optional[bytes]=None, addresses: Optional[object]=None, isFatal: bool=False, retry: bool=False):
        if False:
            return 10
        '\n        @param code: The SMTP response code associated with this error.\n        @param resp: The string response associated with this error.\n        @param log: A string log of the exchange leading up to and including\n            the error.\n        @param isFatal: A boolean indicating whether this connection can\n            proceed or not. If True, the connection will be dropped.\n        @param retry: A boolean indicating whether the delivery should be\n            retried. If True and the factory indicates further retries are\n            desirable, they will be attempted, otherwise the delivery will be\n            failed.\n        '
        if isinstance(resp, str):
            resp = resp.encode('utf-8')
        if isinstance(log, str):
            log = log.encode('utf-8')
        self.code = code
        self.resp = resp
        self.log = log
        self.addresses = addresses
        self.isFatal = isFatal
        self.retry = retry

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        return self.__bytes__().decode('utf-8')

    def __bytes__(self) -> bytes:
        if False:
            while True:
                i = 10
        if self.code > 0:
            res = [f'{self.code:03d} '.encode() + self.resp]
        else:
            res = [self.resp]
        if self.log:
            res.append(self.log)
            res.append(b'')
        return b'\n'.join(res)

class ESMTPClientError(SMTPClientError):
    """
    Base class for ESMTP client errors.
    """

class EHLORequiredError(ESMTPClientError):
    """
    The server does not support EHLO.

    This is considered a non-fatal error (the connection will not be dropped).
    """

class AUTHRequiredError(ESMTPClientError):
    """
    Authentication was required but the server does not support it.

    This is considered a non-fatal error (the connection will not be dropped).
    """

class TLSRequiredError(ESMTPClientError):
    """
    Transport security was required but the server does not support it.

    This is considered a non-fatal error (the connection will not be dropped).
    """

class AUTHDeclinedError(ESMTPClientError):
    """
    The server rejected our credentials.

    Either the username, password, or challenge response
    given to the server was rejected.

    This is considered a non-fatal error (the connection will not be
    dropped).
    """

class AuthenticationError(ESMTPClientError):
    """
    An error occurred while authenticating.

    Either the server rejected our request for authentication or the
    challenge received was malformed.

    This is considered a non-fatal error (the connection will not be
    dropped).
    """

class SMTPTLSError(ESMTPClientError):
    """
    An error occurred while negiotiating for transport security.

    This is considered a non-fatal error (the connection will not be dropped).
    """

class SMTPConnectError(SMTPClientError):
    """
    Failed to connect to the mail exchange host.

    This is considered a fatal error.  A retry will be made.
    """

    def __init__(self, code, resp, log=None, addresses=None, isFatal=True, retry=True):
        if False:
            i = 10
            return i + 15
        SMTPClientError.__init__(self, code, resp, log, addresses, isFatal, retry)

class SMTPTimeoutError(SMTPClientError):
    """
    Failed to receive a response from the server in the expected time period.

    This is considered a fatal error.  A retry will be made.
    """

    def __init__(self, code, resp, log=None, addresses=None, isFatal=True, retry=True):
        if False:
            i = 10
            return i + 15
        SMTPClientError.__init__(self, code, resp, log, addresses, isFatal, retry)

class SMTPProtocolError(SMTPClientError):
    """
    The server sent a mangled response.

    This is considered a fatal error.  A retry will not be made.
    """

    def __init__(self, code, resp, log=None, addresses=None, isFatal=True, retry=False):
        if False:
            i = 10
            return i + 15
        SMTPClientError.__init__(self, code, resp, log, addresses, isFatal, retry)

class SMTPDeliveryError(SMTPClientError):
    """
    Indicates that a delivery attempt has had an error.
    """

class SMTPServerError(SMTPError):

    def __init__(self, code, resp):
        if False:
            while True:
                i = 10
        self.code = code
        self.resp = resp

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        return '%.3d %s' % (self.code, self.resp)

class SMTPAddressError(SMTPServerError):

    def __init__(self, addr, code, resp):
        if False:
            while True:
                i = 10
        from twisted.mail.smtp import Address
        SMTPServerError.__init__(self, code, resp)
        self.addr = Address(addr)

    def __str__(self) -> str:
        if False:
            while True:
                i = 10
        return '%.3d <%s>... %s' % (self.code, self.addr, self.resp)

class SMTPBadRcpt(SMTPAddressError):

    def __init__(self, addr, code=550, resp='Cannot receive for specified address'):
        if False:
            print('Hello World!')
        SMTPAddressError.__init__(self, addr, code, resp)

class SMTPBadSender(SMTPAddressError):

    def __init__(self, addr, code=550, resp='Sender not acceptable'):
        if False:
            i = 10
            return i + 15
        SMTPAddressError.__init__(self, addr, code, resp)

class AddressError(SMTPError):
    """
    Parse error in address
    """

class POP3Error(Exception):
    """
    The base class for POP3 errors.
    """
    pass

class _POP3MessageDeleted(Exception):
    """
    An internal control-flow error which indicates that a deleted message was
    requested.
    """

class POP3ClientError(Exception):
    """
    The base class for all exceptions raised by POP3Client.
    """

class InsecureAuthenticationDisallowed(POP3ClientError):
    """
    An error indicating secure authentication was required but no mechanism
    could be found.
    """

class TLSError(POP3ClientError):
    """
    An error indicating secure authentication was required but either the
    transport does not support TLS or no TLS context factory was supplied.
    """

class TLSNotSupportedError(POP3ClientError):
    """
    An error indicating secure authentication was required but the server does
    not support TLS.
    """

class ServerErrorResponse(POP3ClientError):
    """
    An error indicating that the server returned an error response to a
    request.

    @ivar consumer: See L{__init__}
    """

    def __init__(self, reason, consumer=None):
        if False:
            while True:
                i = 10
        '\n        @type reason: L{bytes}\n        @param reason: The server response minus the status indicator.\n\n        @type consumer: callable that takes L{object}\n        @param consumer: The function meant to handle the values for a\n            multi-line response.\n        '
        POP3ClientError.__init__(self, reason)
        self.consumer = consumer

class LineTooLong(POP3ClientError):
    """
    An error indicating that the server sent a line which exceeded the
    maximum line length (L{LineOnlyReceiver.MAX_LENGTH}).
    """