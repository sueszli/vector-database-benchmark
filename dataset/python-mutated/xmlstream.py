"""
XMPP XML Streams

Building blocks for setting up XML Streams, including helping classes for
doing authentication on either client or server side, and working with XML
Stanzas.

@var STREAM_AUTHD_EVENT: Token dispatched by L{Authenticator} when the
    stream has been completely initialized
@type STREAM_AUTHD_EVENT: L{str}.

@var INIT_FAILED_EVENT: Token dispatched by L{Authenticator} when the
    stream has failed to be initialized
@type INIT_FAILED_EVENT: L{str}.

@var Reset: Token to signal that the XML stream has been reset.
@type Reset: Basic object.
"""
from binascii import hexlify
from hashlib import sha1
from sys import intern
from typing import Optional, Tuple
from zope.interface import directlyProvides, implementer
from twisted.internet import defer, protocol
from twisted.internet.error import ConnectionLost
from twisted.python import failure, log, randbytes
from twisted.words.protocols.jabber import error, ijabber, jid
from twisted.words.xish import domish, xmlstream
from twisted.words.xish.xmlstream import STREAM_CONNECTED_EVENT, STREAM_END_EVENT, STREAM_ERROR_EVENT, STREAM_START_EVENT
try:
    from twisted.internet import ssl as _ssl
except ImportError:
    ssl = None
else:
    if not _ssl.supported:
        ssl = None
    else:
        ssl = _ssl
STREAM_AUTHD_EVENT = intern('//event/stream/authd')
INIT_FAILED_EVENT = intern('//event/xmpp/initfailed')
NS_STREAMS = 'http://etherx.jabber.org/streams'
NS_XMPP_TLS = 'urn:ietf:params:xml:ns:xmpp-tls'
Reset = object()

def hashPassword(sid, password):
    if False:
        print('Hello World!')
    '\n    Create a SHA1-digest string of a session identifier and password.\n\n    @param sid: The stream session identifier.\n    @type sid: C{unicode}.\n    @param password: The password to be hashed.\n    @type password: C{unicode}.\n    '
    if not isinstance(sid, str):
        raise TypeError('The session identifier must be a unicode object')
    if not isinstance(password, str):
        raise TypeError('The password must be a unicode object')
    input = f'{sid}{password}'
    return sha1(input.encode('utf-8')).hexdigest()

class Authenticator:
    """
    Base class for business logic of initializing an XmlStream

    Subclass this object to enable an XmlStream to initialize and authenticate
    to different types of stream hosts (such as clients, components, etc.).

    Rules:
      1. The Authenticator MUST dispatch a L{STREAM_AUTHD_EVENT} when the
         stream has been completely initialized.
      2. The Authenticator SHOULD reset all state information when
         L{associateWithStream} is called.
      3. The Authenticator SHOULD override L{streamStarted}, and start
         initialization there.

    @type xmlstream: L{XmlStream}
    @ivar xmlstream: The XmlStream that needs authentication

    @note: the term authenticator is historical. Authenticators perform
           all steps required to prepare the stream for the exchange
           of XML stanzas.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.xmlstream = None

    def connectionMade(self):
        if False:
            print('Hello World!')
        "\n        Called by the XmlStream when the underlying socket connection is\n        in place.\n\n        This allows the Authenticator to send an initial root element, if it's\n        connecting, or wait for an inbound root from the peer if it's accepting\n        the connection.\n\n        Subclasses can use self.xmlstream.send() to send any initial data to\n        the peer.\n        "

    def streamStarted(self, rootElement):
        if False:
            while True:
                i = 10
        '\n        Called by the XmlStream when the stream has started.\n\n        A stream is considered to have started when the start tag of the root\n        element has been received.\n\n        This examines C{rootElement} to see if there is a version attribute.\n        If absent, C{0.0} is assumed per RFC 3920. Subsequently, the\n        minimum of the version from the received stream header and the\n        value stored in L{xmlstream} is taken and put back in L{xmlstream}.\n\n        Extensions of this method can extract more information from the\n        stream header and perform checks on them, optionally sending\n        stream errors and closing the stream.\n        '
        if rootElement.hasAttribute('version'):
            version = rootElement['version'].split('.')
            try:
                version = (int(version[0]), int(version[1]))
            except (IndexError, ValueError):
                version = (0, 0)
        else:
            version = (0, 0)
        self.xmlstream.version = min(self.xmlstream.version, version)

    def associateWithStream(self, xmlstream):
        if False:
            i = 10
            return i + 15
        '\n        Called by the XmlStreamFactory when a connection has been made\n        to the requested peer, and an XmlStream object has been\n        instantiated.\n\n        The default implementation just saves a handle to the new\n        XmlStream.\n\n        @type xmlstream: L{XmlStream}\n        @param xmlstream: The XmlStream that will be passing events to this\n                          Authenticator.\n\n        '
        self.xmlstream = xmlstream

class ConnectAuthenticator(Authenticator):
    """
    Authenticator for initiating entities.
    """
    namespace: Optional[str] = None

    def __init__(self, otherHost):
        if False:
            while True:
                i = 10
        self.otherHost = otherHost

    def connectionMade(self):
        if False:
            print('Hello World!')
        self.xmlstream.namespace = self.namespace
        self.xmlstream.otherEntity = jid.internJID(self.otherHost)
        self.xmlstream.sendHeader()

    def initializeStream(self):
        if False:
            while True:
                i = 10
        '\n        Perform stream initialization procedures.\n\n        An L{XmlStream} holds a list of initializer objects in its\n        C{initializers} attribute. This method calls these initializers in\n        order and dispatches the L{STREAM_AUTHD_EVENT} event when the list has\n        been successfully processed. Otherwise it dispatches the\n        C{INIT_FAILED_EVENT} event with the failure.\n\n        Initializers may return the special L{Reset} object to halt the\n        initialization processing. It signals that the current initializer was\n        successfully processed, but that the XML Stream has been reset. An\n        example is the TLSInitiatingInitializer.\n        '

        def remove_first(result):
            if False:
                return 10
            self.xmlstream.initializers.pop(0)
            return result

        def do_next(result):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Take the first initializer and process it.\n\n            On success, the initializer is removed from the list and\n            then next initializer will be tried.\n            '
            if result is Reset:
                return None
            try:
                init = self.xmlstream.initializers[0]
            except IndexError:
                self.xmlstream.dispatch(self.xmlstream, STREAM_AUTHD_EVENT)
                return None
            else:
                d = defer.maybeDeferred(init.initialize)
                d.addCallback(remove_first)
                d.addCallback(do_next)
                return d
        d = defer.succeed(None)
        d.addCallback(do_next)
        d.addErrback(self.xmlstream.dispatch, INIT_FAILED_EVENT)

    def streamStarted(self, rootElement):
        if False:
            return 10
        '\n        Called by the XmlStream when the stream has started.\n\n        This extends L{Authenticator.streamStarted} to extract further stream\n        headers from C{rootElement}, optionally wait for stream features being\n        received and then call C{initializeStream}.\n        '
        Authenticator.streamStarted(self, rootElement)
        self.xmlstream.sid = rootElement.getAttribute('id')
        if rootElement.hasAttribute('from'):
            self.xmlstream.otherEntity = jid.internJID(rootElement['from'])
        if self.xmlstream.version >= (1, 0):

            def onFeatures(element):
                if False:
                    for i in range(10):
                        print('nop')
                features = {}
                for feature in element.elements():
                    features[feature.uri, feature.name] = feature
                self.xmlstream.features = features
                self.initializeStream()
            self.xmlstream.addOnetimeObserver('/features[@xmlns="%s"]' % NS_STREAMS, onFeatures)
        else:
            self.initializeStream()

class ListenAuthenticator(Authenticator):
    """
    Authenticator for receiving entities.
    """
    namespace: Optional[str] = None

    def associateWithStream(self, xmlstream):
        if False:
            i = 10
            return i + 15
        '\n        Called by the XmlStreamFactory when a connection has been made.\n\n        Extend L{Authenticator.associateWithStream} to set the L{XmlStream}\n        to be non-initiating.\n        '
        Authenticator.associateWithStream(self, xmlstream)
        self.xmlstream.initiating = False

    def streamStarted(self, rootElement):
        if False:
            return 10
        '\n        Called by the XmlStream when the stream has started.\n\n        This extends L{Authenticator.streamStarted} to extract further\n        information from the stream headers from C{rootElement}.\n        '
        Authenticator.streamStarted(self, rootElement)
        self.xmlstream.namespace = rootElement.defaultUri
        if rootElement.hasAttribute('to'):
            self.xmlstream.thisEntity = jid.internJID(rootElement['to'])
        self.xmlstream.prefixes = {}
        for (prefix, uri) in rootElement.localPrefixes.items():
            self.xmlstream.prefixes[uri] = prefix
        self.xmlstream.sid = hexlify(randbytes.secureRandom(8)).decode('ascii')

class FeatureNotAdvertized(Exception):
    """
    Exception indicating a stream feature was not advertized, while required by
    the initiating entity.
    """

@implementer(ijabber.IInitiatingInitializer)
class BaseFeatureInitiatingInitializer:
    """
    Base class for initializers with a stream feature.

    This assumes the associated XmlStream represents the initiating entity
    of the connection.

    @cvar feature: tuple of (uri, name) of the stream feature root element.
    @type feature: tuple of (C{str}, C{str})

    @ivar required: whether the stream feature is required to be advertized
                    by the receiving entity.
    @type required: C{bool}
    """
    feature: Optional[Tuple[str, str]] = None

    def __init__(self, xs, required=False):
        if False:
            while True:
                i = 10
        self.xmlstream = xs
        self.required = required

    def initialize(self):
        if False:
            while True:
                i = 10
        '\n        Initiate the initialization.\n\n        Checks if the receiving entity advertizes the stream feature. If it\n        does, the initialization is started. If it is not advertized, and the\n        C{required} instance variable is C{True}, it raises\n        L{FeatureNotAdvertized}. Otherwise, the initialization silently\n        succeeds.\n        '
        if self.feature in self.xmlstream.features:
            return self.start()
        elif self.required:
            raise FeatureNotAdvertized
        else:
            return None

    def start(self):
        if False:
            while True:
                i = 10
        '\n        Start the actual initialization.\n\n        May return a deferred for asynchronous initialization.\n        '

class TLSError(Exception):
    """
    TLS base exception.
    """

class TLSFailed(TLSError):
    """
    Exception indicating failed TLS negotiation
    """

class TLSRequired(TLSError):
    """
    Exception indicating required TLS negotiation.

    This exception is raised when the receiving entity requires TLS
    negotiation and the initiating does not desire to negotiate TLS.
    """

class TLSNotSupported(TLSError):
    """
    Exception indicating missing TLS support.

    This exception is raised when the initiating entity wants and requires to
    negotiate TLS when the OpenSSL library is not available.
    """

class TLSInitiatingInitializer(BaseFeatureInitiatingInitializer):
    """
    TLS stream initializer for the initiating entity.

    It is strongly required to include this initializer in the list of
    initializers for an XMPP stream. By default it will try to negotiate TLS.
    An XMPP server may indicate that TLS is required. If TLS is not desired,
    set the C{wanted} attribute to False instead of removing it from the list
    of initializers, so a proper exception L{TLSRequired} can be raised.

    @ivar wanted: indicates if TLS negotiation is wanted.
    @type wanted: C{bool}
    """
    feature = (NS_XMPP_TLS, 'starttls')
    wanted = True
    _deferred = None
    _configurationForTLS = None

    def __init__(self, xs, required=True, configurationForTLS=None):
        if False:
            print('Hello World!')
        '\n        @param configurationForTLS: An object which creates appropriately\n            configured TLS connections. This is passed to C{startTLS} on the\n            transport and is preferably created using\n            L{twisted.internet.ssl.optionsForClientTLS}.  If C{None}, the\n            default is to verify the server certificate against the trust roots\n            as provided by the platform. See\n            L{twisted.internet._sslverify.platformTrust}.\n        @type configurationForTLS: L{IOpenSSLClientConnectionCreator} or\n            C{None}\n        '
        super().__init__(xs, required=required)
        self._configurationForTLS = configurationForTLS

    def onProceed(self, obj):
        if False:
            return 10
        '\n        Proceed with TLS negotiation and reset the XML stream.\n        '
        self.xmlstream.removeObserver('/failure', self.onFailure)
        if self._configurationForTLS:
            ctx = self._configurationForTLS
        else:
            ctx = ssl.optionsForClientTLS(self.xmlstream.otherEntity.host)
        self.xmlstream.transport.startTLS(ctx)
        self.xmlstream.reset()
        self.xmlstream.sendHeader()
        self._deferred.callback(Reset)

    def onFailure(self, obj):
        if False:
            return 10
        self.xmlstream.removeObserver('/proceed', self.onProceed)
        self._deferred.errback(TLSFailed())

    def start(self):
        if False:
            i = 10
            return i + 15
        '\n        Start TLS negotiation.\n\n        This checks if the receiving entity requires TLS, the SSL library is\n        available and uses the C{required} and C{wanted} instance variables to\n        determine what to do in the various different cases.\n\n        For example, if the SSL library is not available, and wanted and\n        required by the user, it raises an exception. However if it is not\n        required by both parties, initialization silently succeeds, moving\n        on to the next step.\n        '
        if self.wanted:
            if ssl is None:
                if self.required:
                    return defer.fail(TLSNotSupported())
                else:
                    return defer.succeed(None)
            else:
                pass
        elif self.xmlstream.features[self.feature].required:
            return defer.fail(TLSRequired())
        else:
            return defer.succeed(None)
        self._deferred = defer.Deferred()
        self.xmlstream.addOnetimeObserver('/proceed', self.onProceed)
        self.xmlstream.addOnetimeObserver('/failure', self.onFailure)
        self.xmlstream.send(domish.Element((NS_XMPP_TLS, 'starttls')))
        return self._deferred

class XmlStream(xmlstream.XmlStream):
    """
    XMPP XML Stream protocol handler.

    @ivar version: XML stream version as a tuple (major, minor). Initially,
                   this is set to the minimally supported version. Upon
                   receiving the stream header of the peer, it is set to the
                   minimum of that value and the version on the received
                   header.
    @type version: (C{int}, C{int})
    @ivar namespace: default namespace URI for stream
    @type namespace: C{unicode}
    @ivar thisEntity: JID of this entity
    @type thisEntity: L{JID}
    @ivar otherEntity: JID of the peer entity
    @type otherEntity: L{JID}
    @ivar sid: session identifier
    @type sid: C{unicode}
    @ivar initiating: True if this is the initiating stream
    @type initiating: C{bool}
    @ivar features: map of (uri, name) to stream features element received from
                    the receiving entity.
    @type features: C{dict} of (C{unicode}, C{unicode}) to L{domish.Element}.
    @ivar prefixes: map of URI to prefixes that are to appear on stream
                    header.
    @type prefixes: C{dict} of C{unicode} to C{unicode}
    @ivar initializers: list of stream initializer objects
    @type initializers: C{list} of objects that provide L{IInitializer}
    @ivar authenticator: associated authenticator that uses C{initializers} to
                         initialize the XML stream.
    """
    version = (1, 0)
    namespace = 'invalid'
    thisEntity = None
    otherEntity = None
    sid = None
    initiating = True
    _headerSent = False

    def __init__(self, authenticator):
        if False:
            for i in range(10):
                print('nop')
        xmlstream.XmlStream.__init__(self)
        self.prefixes = {NS_STREAMS: 'stream'}
        self.authenticator = authenticator
        self.initializers = []
        self.features = {}
        authenticator.associateWithStream(self)

    def _callLater(self, *args, **kwargs):
        if False:
            print('Hello World!')
        from twisted.internet import reactor
        return reactor.callLater(*args, **kwargs)

    def reset(self):
        if False:
            i = 10
            return i + 15
        '\n        Reset XML Stream.\n\n        Resets the XML Parser for incoming data. This is to be used after\n        successfully negotiating a new layer, e.g. TLS and SASL. Note that\n        registered event observers will continue to be in place.\n        '
        self._headerSent = False
        self._initializeStream()

    def onStreamError(self, errelem):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called when a stream:error element has been received.\n\n        Dispatches a L{STREAM_ERROR_EVENT} event with the error element to\n        allow for cleanup actions and drops the connection.\n\n        @param errelem: The received error element.\n        @type errelem: L{domish.Element}\n        '
        self.dispatch(failure.Failure(error.exceptionFromStreamError(errelem)), STREAM_ERROR_EVENT)
        self.transport.loseConnection()

    def sendHeader(self):
        if False:
            while True:
                i = 10
        '\n        Send stream header.\n        '
        localPrefixes = {}
        for (uri, prefix) in self.prefixes.items():
            if uri != NS_STREAMS:
                localPrefixes[prefix] = uri
        rootElement = domish.Element((NS_STREAMS, 'stream'), self.namespace, localPrefixes=localPrefixes)
        if self.otherEntity:
            rootElement['to'] = self.otherEntity.userhost()
        if self.thisEntity:
            rootElement['from'] = self.thisEntity.userhost()
        if not self.initiating and self.sid:
            rootElement['id'] = self.sid
        if self.version >= (1, 0):
            rootElement['version'] = '%d.%d' % self.version
        self.send(rootElement.toXml(prefixes=self.prefixes, closeElement=0))
        self._headerSent = True

    def sendFooter(self):
        if False:
            while True:
                i = 10
        '\n        Send stream footer.\n        '
        self.send('</stream:stream>')

    def sendStreamError(self, streamError):
        if False:
            while True:
                i = 10
        "\n        Send stream level error.\n\n        If we are the receiving entity, and haven't sent the header yet,\n        we sent one first.\n\n        After sending the stream error, the stream is closed and the transport\n        connection dropped.\n\n        @param streamError: stream error instance\n        @type streamError: L{error.StreamError}\n        "
        if not self._headerSent and (not self.initiating):
            self.sendHeader()
        if self._headerSent:
            self.send(streamError.getElement())
            self.sendFooter()
        self.transport.loseConnection()

    def send(self, obj):
        if False:
            i = 10
            return i + 15
        "\n        Send data over the stream.\n\n        This overrides L{xmlstream.XmlStream.send} to use the default namespace\n        of the stream header when serializing L{domish.IElement}s. It is\n        assumed that if you pass an object that provides L{domish.IElement},\n        it represents a direct child of the stream's root element.\n        "
        if domish.IElement.providedBy(obj):
            obj = obj.toXml(prefixes=self.prefixes, defaultUri=self.namespace, prefixesInScope=list(self.prefixes.values()))
        xmlstream.XmlStream.send(self, obj)

    def connectionMade(self):
        if False:
            return 10
        '\n        Called when a connection is made.\n\n        Notifies the authenticator when a connection has been made.\n        '
        xmlstream.XmlStream.connectionMade(self)
        self.authenticator.connectionMade()

    def onDocumentStart(self, rootElement):
        if False:
            for i in range(10):
                print('nop')
        "\n        Called when the stream header has been received.\n\n        Extracts the header's C{id} and C{version} attributes from the root\n        element. The C{id} attribute is stored in our C{sid} attribute and the\n        C{version} attribute is parsed and the minimum of the version we sent\n        and the parsed C{version} attribute is stored as a tuple (major, minor)\n        in this class' C{version} attribute. If no C{version} attribute was\n        present, we assume version 0.0.\n\n        If appropriate (we are the initiating stream and the minimum of our and\n        the other party's version is at least 1.0), a one-time observer is\n        registered for getting the stream features. The registered function is\n        C{onFeatures}.\n\n        Ultimately, the authenticator's C{streamStarted} method will be called.\n\n        @param rootElement: The root element.\n        @type rootElement: L{domish.Element}\n        "
        xmlstream.XmlStream.onDocumentStart(self, rootElement)
        self.addOnetimeObserver("/error[@xmlns='%s']" % NS_STREAMS, self.onStreamError)
        self.authenticator.streamStarted(rootElement)

class XmlStreamFactory(xmlstream.XmlStreamFactory):
    """
    Factory for Jabber XmlStream objects as a reconnecting client.

    Note that this differs from L{xmlstream.XmlStreamFactory} in that
    it generates Jabber specific L{XmlStream} instances that have
    authenticators.
    """
    protocol = XmlStream

    def __init__(self, authenticator):
        if False:
            while True:
                i = 10
        xmlstream.XmlStreamFactory.__init__(self, authenticator)
        self.authenticator = authenticator

class XmlStreamServerFactory(xmlstream.BootstrapMixin, protocol.ServerFactory):
    """
    Factory for Jabber XmlStream objects as a server.

    @since: 8.2.
    @ivar authenticatorFactory: Factory callable that takes no arguments, to
                                create a fresh authenticator to be associated
                                with the XmlStream.
    """
    protocol = XmlStream

    def __init__(self, authenticatorFactory):
        if False:
            print('Hello World!')
        xmlstream.BootstrapMixin.__init__(self)
        self.authenticatorFactory = authenticatorFactory

    def buildProtocol(self, addr):
        if False:
            print('Hello World!')
        '\n        Create an instance of XmlStream.\n\n        A new authenticator instance will be created and passed to the new\n        XmlStream. Registered bootstrap event observers are installed as well.\n        '
        authenticator = self.authenticatorFactory()
        xs = self.protocol(authenticator)
        xs.factory = self
        self.installBootstraps(xs)
        return xs

class TimeoutError(Exception):
    """
    Exception raised when no IQ response has been received before the
    configured timeout.
    """

def upgradeWithIQResponseTracker(xs):
    if False:
        return 10
    '\n    Enhances an XmlStream for iq response tracking.\n\n    This makes an L{XmlStream} object provide L{IIQResponseTracker}. When a\n    response is an error iq stanza, the deferred has its errback invoked with a\n    failure that holds a L{StanzaError<error.StanzaError>} that is\n    easier to examine.\n    '

    def callback(iq):
        if False:
            while True:
                i = 10
        '\n        Handle iq response by firing associated deferred.\n        '
        if getattr(iq, 'handled', False):
            return
        try:
            d = xs.iqDeferreds[iq['id']]
        except KeyError:
            pass
        else:
            del xs.iqDeferreds[iq['id']]
            iq.handled = True
            if iq['type'] == 'error':
                d.errback(error.exceptionFromStanza(iq))
            else:
                d.callback(iq)

    def disconnected(_):
        if False:
            while True:
                i = 10
        "\n        Make sure deferreds do not linger on after disconnect.\n\n        This errbacks all deferreds of iq's for which no response has been\n        received with a L{ConnectionLost} failure. Otherwise, the deferreds\n        will never be fired.\n        "
        iqDeferreds = xs.iqDeferreds
        xs.iqDeferreds = {}
        for d in iqDeferreds.values():
            d.errback(ConnectionLost())
    xs.iqDeferreds = {}
    xs.iqDefaultTimeout = getattr(xs, 'iqDefaultTimeout', None)
    xs.addObserver(xmlstream.STREAM_END_EVENT, disconnected)
    xs.addObserver('/iq[@type="result"]', callback)
    xs.addObserver('/iq[@type="error"]', callback)
    directlyProvides(xs, ijabber.IIQResponseTracker)

class IQ(domish.Element):
    """
    Wrapper for an iq stanza.

    Iq stanzas are used for communications with a request-response behaviour.
    Each iq request is associated with an XML stream and has its own unique id
    to be able to track the response.

    @ivar timeout: if set, a timeout period after which the deferred returned
                   by C{send} will have its errback called with a
                   L{TimeoutError} failure.
    @type timeout: C{float}
    """
    timeout = None

    def __init__(self, xmlstream, stanzaType='set'):
        if False:
            while True:
                i = 10
        "\n        @type xmlstream: L{xmlstream.XmlStream}\n        @param xmlstream: XmlStream to use for transmission of this IQ\n\n        @type stanzaType: C{str}\n        @param stanzaType: IQ type identifier ('get' or 'set')\n        "
        domish.Element.__init__(self, (None, 'iq'))
        self.addUniqueId()
        self['type'] = stanzaType
        self._xmlstream = xmlstream

    def send(self, to=None):
        if False:
            i = 10
            return i + 15
        '\n        Send out this iq.\n\n        Returns a deferred that is fired when an iq response with the same id\n        is received. Result responses will be passed to the deferred callback.\n        Error responses will be transformed into a\n        L{StanzaError<error.StanzaError>} and result in the errback of the\n        deferred being invoked.\n\n        @rtype: L{defer.Deferred}\n        '
        if to is not None:
            self['to'] = to
        if not ijabber.IIQResponseTracker.providedBy(self._xmlstream):
            upgradeWithIQResponseTracker(self._xmlstream)
        d = defer.Deferred()
        self._xmlstream.iqDeferreds[self['id']] = d
        timeout = self.timeout or self._xmlstream.iqDefaultTimeout
        if timeout is not None:

            def onTimeout():
                if False:
                    return 10
                del self._xmlstream.iqDeferreds[self['id']]
                d.errback(TimeoutError('IQ timed out'))
            call = self._xmlstream._callLater(timeout, onTimeout)

            def cancelTimeout(result):
                if False:
                    print('Hello World!')
                if call.active():
                    call.cancel()
                return result
            d.addBoth(cancelTimeout)
        self._xmlstream.send(self)
        return d

def toResponse(stanza, stanzaType=None):
    if False:
        print('Hello World!')
    '\n    Create a response stanza from another stanza.\n\n    This takes the addressing and id attributes from a stanza to create a (new,\n    empty) response stanza. The addressing attributes are swapped and the id\n    copied. Optionally, the stanza type of the response can be specified.\n\n    @param stanza: the original stanza\n    @type stanza: L{domish.Element}\n    @param stanzaType: optional response stanza type\n    @type stanzaType: C{str}\n    @return: the response stanza.\n    @rtype: L{domish.Element}\n    '
    toAddr = stanza.getAttribute('from')
    fromAddr = stanza.getAttribute('to')
    stanzaID = stanza.getAttribute('id')
    response = domish.Element((None, stanza.name))
    if toAddr:
        response['to'] = toAddr
    if fromAddr:
        response['from'] = fromAddr
    if stanzaID:
        response['id'] = stanzaID
    if stanzaType:
        response['type'] = stanzaType
    return response

@implementer(ijabber.IXMPPHandler)
class XMPPHandler:
    """
    XMPP protocol handler.

    Classes derived from this class implement (part of) one or more XMPP
    extension protocols, and are referred to as a subprotocol implementation.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.parent = None
        self.xmlstream = None

    def setHandlerParent(self, parent):
        if False:
            return 10
        self.parent = parent
        self.parent.addHandler(self)

    def disownHandlerParent(self, parent):
        if False:
            print('Hello World!')
        self.parent.removeHandler(self)
        self.parent = None

    def makeConnection(self, xs):
        if False:
            for i in range(10):
                print('nop')
        self.xmlstream = xs
        self.connectionMade()

    def connectionMade(self):
        if False:
            i = 10
            return i + 15
        '\n        Called after a connection has been established.\n\n        Can be overridden to perform work before stream initialization.\n        '

    def connectionInitialized(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The XML stream has been initialized.\n\n        Can be overridden to perform work after stream initialization, e.g. to\n        set up observers and start exchanging XML stanzas.\n        '

    def connectionLost(self, reason):
        if False:
            while True:
                i = 10
        '\n        The XML stream has been closed.\n\n        This method can be extended to inspect the C{reason} argument and\n        act on it.\n        '
        self.xmlstream = None

    def send(self, obj):
        if False:
            return 10
        '\n        Send data over the managed XML stream.\n\n        @note: The stream manager maintains a queue for data sent using this\n               method when there is no current initialized XML stream. This\n               data is then sent as soon as a new stream has been established\n               and initialized. Subsequently, L{connectionInitialized} will be\n               called again. If this queueing is not desired, use C{send} on\n               C{self.xmlstream}.\n\n        @param obj: data to be sent over the XML stream. This is usually an\n                    object providing L{domish.IElement}, or serialized XML. See\n                    L{xmlstream.XmlStream} for details.\n        '
        self.parent.send(obj)

@implementer(ijabber.IXMPPHandlerCollection)
class XMPPHandlerCollection:
    """
    Collection of XMPP subprotocol handlers.

    This allows for grouping of subprotocol handlers, but is not an
    L{XMPPHandler} itself, so this is not recursive.

    @ivar handlers: List of protocol handlers.
    @type handlers: C{list} of objects providing
                      L{IXMPPHandler}
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.handlers = []

    def __iter__(self):
        if False:
            while True:
                i = 10
        '\n        Act as a container for handlers.\n        '
        return iter(self.handlers)

    def addHandler(self, handler):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add protocol handler.\n\n        Protocol handlers are expected to provide L{ijabber.IXMPPHandler}.\n        '
        self.handlers.append(handler)

    def removeHandler(self, handler):
        if False:
            i = 10
            return i + 15
        '\n        Remove protocol handler.\n        '
        self.handlers.remove(handler)

class StreamManager(XMPPHandlerCollection):
    """
    Business logic representing a managed XMPP connection.

    This maintains a single XMPP connection and provides facilities for packet
    routing and transmission. Business logic modules are objects providing
    L{ijabber.IXMPPHandler} (like subclasses of L{XMPPHandler}), and added
    using L{addHandler}.

    @ivar xmlstream: currently managed XML stream
    @type xmlstream: L{XmlStream}
    @ivar logTraffic: if true, log all traffic.
    @type logTraffic: C{bool}
    @ivar _initialized: Whether the stream represented by L{xmlstream} has
                        been initialized. This is used when caching outgoing
                        stanzas.
    @type _initialized: C{bool}
    @ivar _packetQueue: internal buffer of unsent data. See L{send} for details.
    @type _packetQueue: C{list}
    """
    logTraffic = False

    def __init__(self, factory):
        if False:
            for i in range(10):
                print('nop')
        XMPPHandlerCollection.__init__(self)
        self.xmlstream = None
        self._packetQueue = []
        self._initialized = False
        factory.addBootstrap(STREAM_CONNECTED_EVENT, self._connected)
        factory.addBootstrap(STREAM_AUTHD_EVENT, self._authd)
        factory.addBootstrap(INIT_FAILED_EVENT, self.initializationFailed)
        factory.addBootstrap(STREAM_END_EVENT, self._disconnected)
        self.factory = factory

    def addHandler(self, handler):
        if False:
            while True:
                i = 10
        "\n        Add protocol handler.\n\n        When an XML stream has already been established, the handler's\n        C{connectionInitialized} will be called to get it up to speed.\n        "
        XMPPHandlerCollection.addHandler(self, handler)
        if self.xmlstream and self._initialized:
            handler.makeConnection(self.xmlstream)
            handler.connectionInitialized()

    def _connected(self, xs):
        if False:
            return 10
        "\n        Called when the transport connection has been established.\n\n        Here we optionally set up traffic logging (depending on L{logTraffic})\n        and call each handler's C{makeConnection} method with the L{XmlStream}\n        instance.\n        "

        def logDataIn(buf):
            if False:
                i = 10
                return i + 15
            log.msg('RECV: %r' % buf)

        def logDataOut(buf):
            if False:
                while True:
                    i = 10
            log.msg('SEND: %r' % buf)
        if self.logTraffic:
            xs.rawDataInFn = logDataIn
            xs.rawDataOutFn = logDataOut
        self.xmlstream = xs
        for e in self:
            e.makeConnection(xs)

    def _authd(self, xs):
        if False:
            while True:
                i = 10
        "\n        Called when the stream has been initialized.\n\n        Send out cached stanzas and call each handler's\n        C{connectionInitialized} method.\n        "
        for p in self._packetQueue:
            xs.send(p)
        self._packetQueue = []
        self._initialized = True
        for e in self:
            e.connectionInitialized()

    def initializationFailed(self, reason):
        if False:
            i = 10
            return i + 15
        "\n        Called when stream initialization has failed.\n\n        Stream initialization has halted, with the reason indicated by\n        C{reason}. It may be retried by calling the authenticator's\n        C{initializeStream}. See the respective authenticators for details.\n\n        @param reason: A failure instance indicating why stream initialization\n                       failed.\n        @type reason: L{failure.Failure}\n        "

    def _disconnected(self, reason):
        if False:
            for i in range(10):
                print('nop')
        "\n        Called when the stream has been closed.\n\n        From this point on, the manager doesn't interact with the\n        L{XmlStream} anymore and notifies each handler that the connection\n        was lost by calling its C{connectionLost} method.\n        "
        self.xmlstream = None
        self._initialized = False
        for e in self:
            e.connectionLost(reason)

    def send(self, obj):
        if False:
            while True:
                i = 10
        '\n        Send data over the XML stream.\n\n        When there is no established XML stream, the data is queued and sent\n        out when a new XML stream has been established and initialized.\n\n        @param obj: data to be sent over the XML stream. See\n                    L{xmlstream.XmlStream.send} for details.\n        '
        if self._initialized:
            self.xmlstream.send(obj)
        else:
            self._packetQueue.append(obj)
__all__ = ['Authenticator', 'BaseFeatureInitiatingInitializer', 'ConnectAuthenticator', 'FeatureNotAdvertized', 'INIT_FAILED_EVENT', 'IQ', 'ListenAuthenticator', 'NS_STREAMS', 'NS_XMPP_TLS', 'Reset', 'STREAM_AUTHD_EVENT', 'STREAM_CONNECTED_EVENT', 'STREAM_END_EVENT', 'STREAM_ERROR_EVENT', 'STREAM_START_EVENT', 'StreamManager', 'TLSError', 'TLSFailed', 'TLSInitiatingInitializer', 'TLSNotSupported', 'TLSRequired', 'TimeoutError', 'XMPPHandler', 'XMPPHandlerCollection', 'XmlStream', 'XmlStreamFactory', 'XmlStreamServerFactory', 'hashPassword', 'toResponse', 'upgradeWithIQResponseTracker']