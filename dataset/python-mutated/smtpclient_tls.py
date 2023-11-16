"""
Demonstrate sending mail via SMTP while employing TLS and performing
authentication.
"""
import sys
from twisted.internet import reactor
from twisted.internet.defer import Deferred
from twisted.internet.ssl import optionsForClientTLS
from twisted.mail.smtp import ESMTPSenderFactory
from twisted.python.usage import Options, UsageError

def sendmail(authenticationUsername, authenticationSecret, fromAddress, toAddress, messageFile, smtpHost, smtpPort=25):
    if False:
        while True:
            i = 10
    '\n    @param authenticationUsername: The username with which to authenticate.\n    @param authenticationSecret: The password with which to authenticate.\n    @param fromAddress: The SMTP reverse path (ie, MAIL FROM)\n    @param toAddress: The SMTP forward path (ie, RCPT TO)\n    @param messageFile: A file-like object containing the headers and body of\n    the message to send.\n    @param smtpHost: The MX host to which to connect.\n    @param smtpPort: The port number to which to connect.\n\n    @return: A Deferred which will be called back when the message has been\n    sent or which will errback if it cannot be sent.\n    '
    contextFactory = optionsForClientTLS(smtpHost.decode('utf8'))
    resultDeferred = Deferred()
    senderFactory = ESMTPSenderFactory(authenticationUsername, authenticationSecret, fromAddress, toAddress, messageFile, resultDeferred, contextFactory=contextFactory)
    reactor.connectTCP(smtpHost, smtpPort, senderFactory)
    return resultDeferred

class SendmailOptions(Options):
    synopsis = 'smtpclient_tls.py [options]'
    optParameters = [('username', 'u', None, 'The username with which to authenticate to the SMTP server.'), ('password', 'p', None, 'The password with which to authenticate to the SMTP server.'), ('from-address', 'f', None, 'The address from which to send the message.'), ('to-address', 't', None, 'The address to which to send the message.'), ('message', 'm', None, 'The filename which contains the message to send.'), ('smtp-host', 'h', None, 'The host through which to send the message.'), ('smtp-port', None, '25', 'The port number on smtp-host to which to connect.')]

    def postOptions(self):
        if False:
            while True:
                i = 10
        '\n        Parse integer parameters, open the message file, and make sure all\n        required parameters have been specified.\n        '
        try:
            self['smtp-port'] = int(self['smtp-port'])
        except ValueError:
            raise UsageError('--smtp-port argument must be an integer.')
        if self['username'] is None:
            raise UsageError('Must specify authentication username with --username')
        if self['password'] is None:
            raise UsageError('Must specify authentication password with --password')
        if self['from-address'] is None:
            raise UsageError('Must specify from address with --from-address')
        if self['to-address'] is None:
            raise UsageError('Must specify from address with --to-address')
        if self['smtp-host'] is None:
            raise UsageError('Must specify smtp host with --smtp-host')
        if self['message'] is None:
            raise UsageError('Must specify a message file to send with --message')
        try:
            self['message'] = open(self['message'])
        except Exception as e:
            raise UsageError(e)

def cbSentMessage(result):
    if False:
        i = 10
        return i + 15
    '\n    Called when the message has been sent.\n\n    Report success to the user and then stop the reactor.\n    '
    print('Message sent')
    reactor.stop()

def ebSentMessage(err):
    if False:
        for i in range(10):
            print('nop')
    '\n    Called if the message cannot be sent.\n\n    Report the failure to the user and then stop the reactor.\n    '
    err.printTraceback()
    reactor.stop()

def main(args=None):
    if False:
        i = 10
        return i + 15
    '\n    Parse arguments and send an email based on them.\n    '
    o = SendmailOptions()
    try:
        o.parseOptions(args)
    except UsageError as e:
        raise SystemExit(e)
    else:
        from twisted.python import log
        log.startLogging(sys.stdout)
        result = sendmail(o['username'], o['password'], o['from-address'], o['to-address'], o['message'], o['smtp-host'], o['smtp-port'])
        result.addCallbacks(cbSentMessage, ebSentMessage)
        reactor.run()
if __name__ == '__main__':
    main(sys.argv[1:])