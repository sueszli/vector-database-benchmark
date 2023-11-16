"""
Public Jabber Interfaces.
"""
from zope.interface import Attribute, Interface

class IInitializer(Interface):
    """
    Interface for XML stream initializers.

    Initializers perform a step in getting the XML stream ready to be
    used for the exchange of XML stanzas.
    """

class IInitiatingInitializer(IInitializer):
    """
    Interface for XML stream initializers for the initiating entity.
    """
    xmlstream = Attribute('The associated XML stream')

    def initialize():
        if False:
            print('Hello World!')
        '\n        Initiate the initialization step.\n\n        May return a deferred when the initialization is done asynchronously.\n        '

class IIQResponseTracker(Interface):
    """
    IQ response tracker interface.

    The XMPP stanza C{iq} has a request-response nature that fits
    naturally with deferreds. You send out a request and when the response
    comes back a deferred is fired.

    The L{twisted.words.protocols.jabber.client.IQ} class implements a C{send}
    method that returns a deferred. This deferred is put in a dictionary that
    is kept in an L{XmlStream} object, keyed by the request stanzas C{id}
    attribute.

    An object providing this interface (usually an instance of L{XmlStream}),
    keeps the said dictionary and sets observers on the iq stanzas of type
    C{result} and C{error} and lets the callback fire the associated deferred.
    """
    iqDeferreds = Attribute('Dictionary of deferreds waiting for an iq response')

class IXMPPHandler(Interface):
    """
    Interface for XMPP protocol handlers.

    Objects that provide this interface can be added to a stream manager to
    handle of (part of) an XMPP extension protocol.
    """
    parent = Attribute('XML stream manager for this handler')
    xmlstream = Attribute('The managed XML stream')

    def setHandlerParent(parent):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the parent of the handler.\n\n        @type parent: L{IXMPPHandlerCollection}\n        '

    def disownHandlerParent(parent):
        if False:
            print('Hello World!')
        '\n        Remove the parent of the handler.\n\n        @type parent: L{IXMPPHandlerCollection}\n        '

    def makeConnection(xs):
        if False:
            return 10
        '\n        A connection over the underlying transport of the XML stream has been\n        established.\n\n        At this point, no traffic has been exchanged over the XML stream\n        given in C{xs}.\n\n        This should setup L{xmlstream} and call L{connectionMade}.\n\n        @type xs:\n               L{twisted.words.protocols.jabber.xmlstream.XmlStream}\n        '

    def connectionMade():
        if False:
            print('Hello World!')
        '\n        Called after a connection has been established.\n\n        This method can be used to change properties of the XML Stream, its\n        authenticator or the stream manager prior to stream initialization\n        (including authentication).\n        '

    def connectionInitialized():
        if False:
            print('Hello World!')
        '\n        The XML stream has been initialized.\n\n        At this point, authentication was successful, and XML stanzas can be\n        exchanged over the XML stream L{xmlstream}. This method can be\n        used to setup observers for incoming stanzas.\n        '

    def connectionLost(reason):
        if False:
            for i in range(10):
                print('nop')
        '\n        The XML stream has been closed.\n\n        Subsequent use of C{parent.send} will result in data being queued\n        until a new connection has been established.\n\n        @type reason: L{twisted.python.failure.Failure}\n        '

class IXMPPHandlerCollection(Interface):
    """
    Collection of handlers.

    Contain several handlers and manage their connection.
    """

    def __iter__():
        if False:
            for i in range(10):
                print('nop')
        '\n        Get an iterator over all child handlers.\n        '

    def addHandler(handler):
        if False:
            return 10
        '\n        Add a child handler.\n\n        @type handler: L{IXMPPHandler}\n        '

    def removeHandler(handler):
        if False:
            for i in range(10):
                print('nop')
        '\n        Remove a child handler.\n\n        @type handler: L{IXMPPHandler}\n        '

class IService(Interface):
    """
    External server-side component service interface.

    Services that provide this interface can be added to L{ServiceManager} to
    implement (part of) the functionality of the server-side component.
    """

    def componentConnected(xs):
        if False:
            return 10
        '\n        Parent component has established a connection.\n\n        At this point, authentication was successful, and XML stanzas\n        can be exchanged over the XML stream C{xs}. This method can be used\n        to setup observers for incoming stanzas.\n\n        @param xs: XML Stream that represents the established connection.\n        @type xs: L{xmlstream.XmlStream}\n        '

    def componentDisconnected():
        if False:
            i = 10
            return i + 15
        '\n        Parent component has lost the connection to the Jabber server.\n\n        Subsequent use of C{self.parent.send} will result in data being\n        queued until a new connection has been established.\n        '

    def transportConnected(xs):
        if False:
            return 10
        "\n        Parent component has established a connection over the underlying\n        transport.\n\n        At this point, no traffic has been exchanged over the XML stream. This\n        method can be used to change properties of the XML Stream (in C{xs}),\n        the service manager or it's authenticator prior to stream\n        initialization (including authentication).\n        "