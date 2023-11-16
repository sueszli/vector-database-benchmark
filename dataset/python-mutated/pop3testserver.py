import sys
from twisted.internet import reactor
from twisted.internet.protocol import Factory
from twisted.protocols import basic
USER = 'test'
PASS = 'twisted'
PORT = 1100
SSL_SUPPORT = True
UIDL_SUPPORT = True
INVALID_SERVER_RESPONSE = False
INVALID_CAPABILITY_RESPONSE = False
INVALID_LOGIN_RESPONSE = False
DENY_CONNECTION = False
DROP_CONNECTION = False
BAD_TLS_RESPONSE = False
TIMEOUT_RESPONSE = False
TIMEOUT_DEFERRED = False
SLOW_GREETING = False
'Commands'
CONNECTION_MADE = b'+OK POP3 localhost v2003.83 server ready'
CAPABILITIES = [b'TOP', b'LOGIN-DELAY 180', b'USER', b'SASL LOGIN']
CAPABILITIES_SSL = b'STLS'
CAPABILITIES_UIDL = b'UIDL'
INVALID_RESPONSE = b'-ERR Unknown request'
VALID_RESPONSE = b'+OK Command Completed'
AUTH_DECLINED = b'-ERR LOGIN failed'
AUTH_ACCEPTED = b'+OK Mailbox open, 0 messages'
TLS_ERROR = b'-ERR server side error start TLS handshake'
LOGOUT_COMPLETE = b'+OK quit completed'
NOT_LOGGED_IN = b'-ERR Unknown AUHORIZATION state command'
STAT = b'+OK 0 0'
UIDL = b'+OK Unique-ID listing follows\r\n.'
LIST = b'+OK Mailbox scan listing follows\r\n.'
CAP_START = b'+OK Capability list follows:'

class POP3TestServer(basic.LineReceiver):

    def __init__(self, contextFactory=None):
        if False:
            print('Hello World!')
        self.loggedIn = False
        self.caps = None
        self.tmpUser = None
        self.ctx = contextFactory

    def sendSTATResp(self, req):
        if False:
            for i in range(10):
                print('nop')
        self.sendLine(STAT)

    def sendUIDLResp(self, req):
        if False:
            for i in range(10):
                print('nop')
        self.sendLine(UIDL)

    def sendLISTResp(self, req):
        if False:
            for i in range(10):
                print('nop')
        self.sendLine(LIST)

    def sendCapabilities(self):
        if False:
            i = 10
            return i + 15
        if self.caps is None:
            self.caps = [CAP_START]
        if UIDL_SUPPORT:
            self.caps.append(CAPABILITIES_UIDL)
        if SSL_SUPPORT:
            self.caps.append(CAPABILITIES_SSL)
        for cap in CAPABILITIES:
            self.caps.append(cap)
        resp = b'\r\n'.join(self.caps)
        resp += b'\r\n.'
        self.sendLine(resp)

    def connectionMade(self):
        if False:
            return 10
        if DENY_CONNECTION:
            self.disconnect()
            return
        if SLOW_GREETING:
            reactor.callLater(20, self.sendGreeting)
        else:
            self.sendGreeting()

    def sendGreeting(self):
        if False:
            for i in range(10):
                print('nop')
        self.sendLine(CONNECTION_MADE)

    def lineReceived(self, line):
        if False:
            while True:
                i = 10
        'Error Conditions'
        uline = line.upper()
        find = lambda s: uline.find(s) != -1
        if TIMEOUT_RESPONSE:
            return
        if DROP_CONNECTION:
            self.disconnect()
            return
        elif find(b'CAPA'):
            if INVALID_CAPABILITY_RESPONSE:
                self.sendLine(INVALID_RESPONSE)
            else:
                self.sendCapabilities()
        elif find(b'STLS') and SSL_SUPPORT:
            self.startTLS()
        elif find(b'USER'):
            if INVALID_LOGIN_RESPONSE:
                self.sendLine(INVALID_RESPONSE)
                return
            resp = None
            try:
                self.tmpUser = line.split(' ')[1]
                resp = VALID_RESPONSE
            except BaseException:
                resp = AUTH_DECLINED
            self.sendLine(resp)
        elif find(b'PASS'):
            resp = None
            try:
                pwd = line.split(' ')[1]
                if self.tmpUser is None or pwd is None:
                    resp = AUTH_DECLINED
                elif self.tmpUser == USER and pwd == PASS:
                    resp = AUTH_ACCEPTED
                    self.loggedIn = True
                else:
                    resp = AUTH_DECLINED
            except BaseException:
                resp = AUTH_DECLINED
            self.sendLine(resp)
        elif find(b'QUIT'):
            self.loggedIn = False
            self.sendLine(LOGOUT_COMPLETE)
            self.disconnect()
        elif INVALID_SERVER_RESPONSE:
            self.sendLine(INVALID_RESPONSE)
        elif not self.loggedIn:
            self.sendLine(NOT_LOGGED_IN)
        elif find(b'NOOP'):
            self.sendLine(VALID_RESPONSE)
        elif find(b'STAT'):
            if TIMEOUT_DEFERRED:
                return
            self.sendLine(STAT)
        elif find(b'LIST'):
            if TIMEOUT_DEFERRED:
                return
            self.sendLine(LIST)
        elif find(b'UIDL'):
            if TIMEOUT_DEFERRED:
                return
            elif not UIDL_SUPPORT:
                self.sendLine(INVALID_RESPONSE)
                return
            self.sendLine(UIDL)

    def startTLS(self):
        if False:
            while True:
                i = 10
        if SSL_SUPPORT and self.ctx is not None:
            self.sendLine(b'+OK Begin TLS negotiation now')
            self.transport.startTLS(self.ctx)
        else:
            self.sendLine(b'-ERR TLS not available')

    def disconnect(self):
        if False:
            print('Hello World!')
        self.transport.loseConnection()
usage = "popServer.py [arg] (default is Standard POP Server with no messages)\nno_ssl  - Start with no SSL support\nno_uidl - Start with no UIDL support\nbad_resp - Send a non-RFC compliant response to the Client\nbad_cap_resp - send a non-RFC compliant response when the Client sends a 'CAPABILITY' request\nbad_login_resp - send a non-RFC compliant response when the Client sends a 'LOGIN' request\ndeny - Deny the connection\ndrop - Drop the connection after sending the greeting\nbad_tls - Send a bad response to a STARTTLS\ntimeout - Do not return a response to a Client request\nto_deferred - Do not return a response on a 'Select' request. This\n              will test Deferred callback handling\nslow - Wait 20 seconds after the connection is made to return a Server Greeting\n"

def printMessage(msg):
    if False:
        while True:
            i = 10
    print('Server Starting in %s mode' % msg)

def processArg(arg):
    if False:
        print('Hello World!')
    if arg.lower() == 'no_ssl':
        global SSL_SUPPORT
        SSL_SUPPORT = False
        printMessage('NON-SSL')
    elif arg.lower() == 'no_uidl':
        global UIDL_SUPPORT
        UIDL_SUPPORT = False
        printMessage('NON-UIDL')
    elif arg.lower() == 'bad_resp':
        global INVALID_SERVER_RESPONSE
        INVALID_SERVER_RESPONSE = True
        printMessage('Invalid Server Response')
    elif arg.lower() == 'bad_cap_resp':
        global INVALID_CAPABILITY_RESPONSE
        INVALID_CAPABILITY_RESPONSE = True
        printMessage('Invalid Capability Response')
    elif arg.lower() == 'bad_login_resp':
        global INVALID_LOGIN_RESPONSE
        INVALID_LOGIN_RESPONSE = True
        printMessage('Invalid Capability Response')
    elif arg.lower() == 'deny':
        global DENY_CONNECTION
        DENY_CONNECTION = True
        printMessage('Deny Connection')
    elif arg.lower() == 'drop':
        global DROP_CONNECTION
        DROP_CONNECTION = True
        printMessage('Drop Connection')
    elif arg.lower() == 'bad_tls':
        global BAD_TLS_RESPONSE
        BAD_TLS_RESPONSE = True
        printMessage('Bad TLS Response')
    elif arg.lower() == 'timeout':
        global TIMEOUT_RESPONSE
        TIMEOUT_RESPONSE = True
        printMessage('Timeout Response')
    elif arg.lower() == 'to_deferred':
        global TIMEOUT_DEFERRED
        TIMEOUT_DEFERRED = True
        printMessage('Timeout Deferred Response')
    elif arg.lower() == 'slow':
        global SLOW_GREETING
        SLOW_GREETING = True
        printMessage('Slow Greeting')
    elif arg.lower() == '--help':
        print(usage)
        sys.exit()
    else:
        print(usage)
        sys.exit()

def main():
    if False:
        print('Hello World!')
    if len(sys.argv) < 2:
        printMessage('POP3 with no messages')
    else:
        args = sys.argv[1:]
        for arg in args:
            processArg(arg)
    f = Factory()
    f.protocol = POP3TestServer
    reactor.listenTCP(PORT, f)
    reactor.run()
if __name__ == '__main__':
    main()