"""
Interface definitions for L{twisted.web}.

@var UNKNOWN_LENGTH: An opaque object which may be used as the value of
    L{IBodyProducer.length} to indicate that the length of the entity
    body is not known in advance.
"""
from typing import TYPE_CHECKING, Callable, List, Optional
from zope.interface import Attribute, Interface
from twisted.cred.credentials import IUsernameDigestHash
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IPushProducer
from twisted.web.http_headers import Headers
if TYPE_CHECKING:
    from twisted.web.template import Flattenable, Tag

class IRequest(Interface):
    """
    An HTTP request.

    @since: 9.0
    """
    method = Attribute('A L{bytes} giving the HTTP method that was used.')
    uri = Attribute('A L{bytes} giving the full encoded URI which was requested (including query arguments).')
    path = Attribute('A L{bytes} giving the encoded query path of the request URI (not including query arguments).')
    args = Attribute("A mapping of decoded query argument names as L{bytes} to corresponding query argument values as L{list}s of L{bytes}.  For example, for a URI with C{foo=bar&foo=baz&quux=spam} for its query part, C{args} will be C{{b'foo': [b'bar', b'baz'], b'quux': [b'spam']}}.")
    prepath = Attribute('The URL path segments which have been processed during resource traversal, as a list of L{bytes}.')
    postpath = Attribute('The URL path segments which have not (yet) been processed during resource traversal, as a list of L{bytes}.')
    requestHeaders = Attribute('A L{http_headers.Headers} instance giving all received HTTP request headers.')
    content = Attribute('A file-like object giving the request body.  This may be a file on disk, an L{io.BytesIO}, or some other type.  The implementation is free to decide on a per-request basis.')
    responseHeaders = Attribute('A L{http_headers.Headers} instance holding all HTTP response headers to be sent.')

    def getHeader(key):
        if False:
            return 10
        '\n        Get an HTTP request header.\n\n        @type key: L{bytes} or L{str}\n        @param key: The name of the header to get the value of.\n\n        @rtype: L{bytes} or L{str} or L{None}\n        @return: The value of the specified header, or L{None} if that header\n            was not present in the request. The string type of the result\n            matches the type of C{key}.\n        '

    def getCookie(key):
        if False:
            i = 10
            return i + 15
        '\n        Get a cookie that was sent from the network.\n\n        @type key: L{bytes}\n        @param key: The name of the cookie to get.\n\n        @rtype: L{bytes} or L{None}\n        @returns: The value of the specified cookie, or L{None} if that cookie\n            was not present in the request.\n        '

    def getAllHeaders():
        if False:
            print('Hello World!')
        '\n        Return dictionary mapping the names of all received headers to the last\n        value received for each.\n\n        Since this method does not return all header information,\n        C{requestHeaders.getAllRawHeaders()} may be preferred.\n        '

    def getRequestHostname():
        if False:
            print('Hello World!')
        "\n        Get the hostname that the HTTP client passed in to the request.\n\n        This will either use the C{Host:} header (if it is available; which,\n        for a spec-compliant request, it will be) or the IP address of the host\n        we are listening on if the header is unavailable.\n\n        @note: This is the I{host portion} of the requested resource, which\n            means that:\n\n                1. it might be an IPv4 or IPv6 address, not just a DNS host\n                   name,\n\n                2. there's no guarantee it's even a I{valid} host name or IP\n                   address, since the C{Host:} header may be malformed,\n\n                3. it does not include the port number.\n\n        @returns: the requested hostname\n\n        @rtype: L{bytes}\n        "

    def getHost():
        if False:
            i = 10
            return i + 15
        "\n        Get my originally requesting transport's host.\n\n        @return: An L{IAddress<twisted.internet.interfaces.IAddress>}.\n        "

    def getClientAddress():
        if False:
            return 10
        "\n        Return the address of the client who submitted this request.\n\n        The address may not be a network address.  Callers must check\n        its type before using it.\n\n        @since: 18.4\n\n        @return: the client's address.\n        @rtype: an L{IAddress} provider.\n        "

    def getClientIP():
        if False:
            print('Hello World!')
        '\n        Return the IP address of the client who submitted this request.\n\n        This method is B{deprecated}.  See L{getClientAddress} instead.\n\n        @returns: the client IP address or L{None} if the request was submitted\n            over a transport where IP addresses do not make sense.\n        @rtype: L{str} or L{None}\n        '

    def getUser():
        if False:
            print('Hello World!')
        '\n        Return the HTTP user sent with this request, if any.\n\n        If no user was supplied, return the empty string.\n\n        @returns: the HTTP user, if any\n        @rtype: L{str}\n        '

    def getPassword():
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the HTTP password sent with this request, if any.\n\n        If no password was supplied, return the empty string.\n\n        @returns: the HTTP password, if any\n        @rtype: L{str}\n        '

    def isSecure():
        if False:
            i = 10
            return i + 15
        "\n        Return True if this request is using a secure transport.\n\n        Normally this method returns True if this request's HTTPChannel\n        instance is using a transport that implements ISSLTransport.\n\n        This will also return True if setHost() has been called\n        with ssl=True.\n\n        @returns: True if this request is secure\n        @rtype: C{bool}\n        "

    def getSession(sessionInterface=None):
        if False:
            while True:
                i = 10
        '\n        Look up the session associated with this request or create a new one if\n        there is not one.\n\n        @return: The L{Session} instance identified by the session cookie in\n            the request, or the C{sessionInterface} component of that session\n            if C{sessionInterface} is specified.\n        '

    def URLPath():
        if False:
            while True:
                i = 10
        '\n        @return: A L{URLPath<twisted.python.urlpath.URLPath>} instance\n            which identifies the URL for which this request is.\n        '

    def prePathURL():
        if False:
            while True:
                i = 10
        '\n        At any time during resource traversal or resource rendering,\n        returns an absolute URL to the most nested resource which has\n        yet been reached.\n\n        @see: {twisted.web.server.Request.prepath}\n\n        @return: An absolute URL.\n        @rtype: L{bytes}\n        '

    def rememberRootURL():
        if False:
            print('Hello World!')
        '\n        Remember the currently-processed part of the URL for later\n        recalling.\n        '

    def getRootURL():
        if False:
            return 10
        '\n        Get a previously-remembered URL.\n\n        @return: An absolute URL.\n        @rtype: L{bytes}\n        '

    def finish():
        if False:
            for i in range(10):
                print('nop')
        '\n        Indicate that the response to this request is complete.\n        '

    def write(data):
        if False:
            return 10
        '\n        Write some data to the body of the response to this request.  Response\n        headers are written the first time this method is called, after which\n        new response headers may not be added.\n\n        @param data: Bytes of the response body.\n        @type data: L{bytes}\n        '

    def addCookie(k, v, expires=None, domain=None, path=None, max_age=None, comment=None, secure=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set an outgoing HTTP cookie.\n\n        In general, you should consider using sessions instead of cookies, see\n        L{twisted.web.server.Request.getSession} and the\n        L{twisted.web.server.Session} class for details.\n        '

    def setResponseCode(code, message=None):
        if False:
            while True:
                i = 10
        '\n        Set the HTTP response code.\n\n        @type code: L{int}\n        @type message: L{bytes}\n        '

    def setHeader(k, v):
        if False:
            while True:
                i = 10
        '\n        Set an HTTP response header.  Overrides any previously set values for\n        this header.\n\n        @type k: L{bytes} or L{str}\n        @param k: The name of the header for which to set the value.\n\n        @type v: L{bytes} or L{str}\n        @param v: The value to set for the named header. A L{str} will be\n            UTF-8 encoded, which may not interoperable with other\n            implementations. Avoid passing non-ASCII characters if possible.\n        '

    def redirect(url):
        if False:
            print('Hello World!')
        '\n        Utility function that does a redirect.\n\n        The request should have finish() called after this.\n        '

    def setLastModified(when):
        if False:
            i = 10
            return i + 15
        '\n        Set the C{Last-Modified} time for the response to this request.\n\n        If I am called more than once, I ignore attempts to set Last-Modified\n        earlier, only replacing the Last-Modified time if it is to a later\n        value.\n\n        If I am a conditional request, I may modify my response code to\n        L{NOT_MODIFIED<http.NOT_MODIFIED>} if appropriate for the time given.\n\n        @param when: The last time the resource being returned was modified, in\n            seconds since the epoch.\n        @type when: L{int} or L{float}\n\n        @return: If I am a C{If-Modified-Since} conditional request and the time\n            given is not newer than the condition, I return\n            L{CACHED<http.CACHED>} to indicate that you should write no body.\n            Otherwise, I return a false value.\n        '

    def setETag(etag):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set an C{entity tag} for the outgoing response.\n\n        That\'s "entity tag" as in the HTTP/1.1 I{ETag} header, "used for\n        comparing two or more entities from the same requested resource."\n\n        If I am a conditional request, I may modify my response code to\n        L{NOT_MODIFIED<http.NOT_MODIFIED>} or\n        L{PRECONDITION_FAILED<http.PRECONDITION_FAILED>}, if appropriate for the\n        tag given.\n\n        @param etag: The entity tag for the resource being returned.\n        @type etag: L{str}\n\n        @return: If I am a C{If-None-Match} conditional request and the tag\n            matches one in the request, I return L{CACHED<http.CACHED>} to\n            indicate that you should write no body.  Otherwise, I return a\n            false value.\n        '

    def setHost(host, port, ssl=0):
        if False:
            i = 10
            return i + 15
        "\n        Change the host and port the request thinks it's using.\n\n        This method is useful for working with reverse HTTP proxies (e.g.  both\n        Squid and Apache's mod_proxy can do this), when the address the HTTP\n        client is using is different than the one we're listening on.\n\n        For example, Apache may be listening on https://www.example.com, and\n        then forwarding requests to http://localhost:8080, but we don't want\n        HTML produced by Twisted to say 'http://localhost:8080', they should\n        say 'https://www.example.com', so we do::\n\n           request.setHost('www.example.com', 443, ssl=1)\n        "

class INonQueuedRequestFactory(Interface):
    """
    A factory of L{IRequest} objects that does not take a ``queued`` parameter.
    """

    def __call__(channel):
        if False:
            i = 10
            return i + 15
        '\n        Create an L{IRequest} that is operating on the given channel. There\n        must only be one L{IRequest} object processing at any given time on a\n        channel.\n\n        @param channel: A L{twisted.web.http.HTTPChannel} object.\n        @type channel: L{twisted.web.http.HTTPChannel}\n\n        @return: A request object.\n        @rtype: L{IRequest}\n        '

class IAccessLogFormatter(Interface):
    """
    An object which can represent an HTTP request as a line of text for
    inclusion in an access log file.
    """

    def __call__(timestamp, request):
        if False:
            return 10
        '\n        Generate a line for the access log.\n\n        @param timestamp: The time at which the request was completed in the\n            standard format for access logs.\n        @type timestamp: L{unicode}\n\n        @param request: The request object about which to log.\n        @type request: L{twisted.web.server.Request}\n\n        @return: One line describing the request without a trailing newline.\n        @rtype: L{unicode}\n        '

class ICredentialFactory(Interface):
    """
    A credential factory defines a way to generate a particular kind of
    authentication challenge and a way to interpret the responses to these
    challenges.  It creates
    L{ICredentials<twisted.cred.credentials.ICredentials>} providers from
    responses.  These objects will be used with L{twisted.cred} to authenticate
    an authorize requests.
    """
    scheme = Attribute("A L{str} giving the name of the authentication scheme with which this factory is associated.  For example, C{'basic'} or C{'digest'}.")

    def getChallenge(request):
        if False:
            i = 10
            return i + 15
        '\n        Generate a new challenge to be sent to a client.\n\n        @type request: L{twisted.web.http.Request}\n        @param request: The request the response to which this challenge will\n            be included.\n\n        @rtype: L{dict}\n        @return: A mapping from L{str} challenge fields to associated L{str}\n            values.\n        '

    def decode(response, request):
        if False:
            while True:
                i = 10
        '\n        Create a credentials object from the given response.\n\n        @type response: L{str}\n        @param response: scheme specific response string\n\n        @type request: L{twisted.web.http.Request}\n        @param request: The request being processed (from which the response\n            was taken).\n\n        @raise twisted.cred.error.LoginFailed: If the response is invalid.\n\n        @rtype: L{twisted.cred.credentials.ICredentials} provider\n        @return: The credentials represented by the given response.\n        '

class IBodyProducer(IPushProducer):
    """
    Objects which provide L{IBodyProducer} write bytes to an object which
    provides L{IConsumer<twisted.internet.interfaces.IConsumer>} by calling its
    C{write} method repeatedly.

    L{IBodyProducer} providers may start producing as soon as they have an
    L{IConsumer<twisted.internet.interfaces.IConsumer>} provider.  That is, they
    should not wait for a C{resumeProducing} call to begin writing data.

    L{IConsumer.unregisterProducer<twisted.internet.interfaces.IConsumer.unregisterProducer>}
    must not be called.  Instead, the
    L{Deferred<twisted.internet.defer.Deferred>} returned from C{startProducing}
    must be fired when all bytes have been written.

    L{IConsumer.write<twisted.internet.interfaces.IConsumer.write>} may
    synchronously invoke any of C{pauseProducing}, C{resumeProducing}, or
    C{stopProducing}.  These methods must be implemented with this in mind.

    @since: 9.0
    """
    length = Attribute('\n        C{length} is a L{int} indicating how many bytes in total this\n        L{IBodyProducer} will write to the consumer or L{UNKNOWN_LENGTH}\n        if this is not known in advance.\n        ')

    def startProducing(consumer):
        if False:
            for i in range(10):
                print('nop')
        '\n        Start producing to the given\n        L{IConsumer<twisted.internet.interfaces.IConsumer>} provider.\n\n        @return: A L{Deferred<twisted.internet.defer.Deferred>} which stops\n            production of data when L{Deferred.cancel} is called, and which\n            fires with L{None} when all bytes have been produced or with a\n            L{Failure<twisted.python.failure.Failure>} if there is any problem\n            before all bytes have been produced.\n        '

    def stopProducing():
        if False:
            while True:
                i = 10
        '\n        In addition to the standard behavior of\n        L{IProducer.stopProducing<twisted.internet.interfaces.IProducer.stopProducing>}\n        (stop producing data), make sure the\n        L{Deferred<twisted.internet.defer.Deferred>} returned by\n        C{startProducing} is never fired.\n        '

class IRenderable(Interface):
    """
    An L{IRenderable} is an object that may be rendered by the
    L{twisted.web.template} templating system.
    """

    def lookupRenderMethod(name: str) -> Callable[[Optional[IRequest], 'Tag'], 'Flattenable']:
        if False:
            i = 10
            return i + 15
        '\n        Look up and return the render method associated with the given name.\n\n        @param name: The value of a render directive encountered in the\n            document returned by a call to L{IRenderable.render}.\n\n        @return: A two-argument callable which will be invoked with the request\n            being responded to and the tag object on which the render directive\n            was encountered.\n        '

    def render(request: Optional[IRequest]) -> 'Flattenable':
        if False:
            print('Hello World!')
        '\n        Get the document for this L{IRenderable}.\n\n        @param request: The request in response to which this method is being\n            invoked.\n\n        @return: An object which can be flattened.\n        '

class ITemplateLoader(Interface):
    """
    A loader for templates; something usable as a value for
    L{twisted.web.template.Element}'s C{loader} attribute.
    """

    def load() -> List['Flattenable']:
        if False:
            print('Hello World!')
        '\n        Load a template suitable for rendering.\n\n        @return: a L{list} of flattenable objects, such as byte and unicode\n            strings, L{twisted.web.template.Element}s and L{IRenderable} providers.\n        '

class IResponse(Interface):
    """
    An object representing an HTTP response received from an HTTP server.

    @since: 11.1
    """
    version = Attribute("A three-tuple describing the protocol and protocol version of the response.  The first element is of type L{str}, the second and third are of type L{int}.  For example, C{(b'HTTP', 1, 1)}.")
    code = Attribute('The HTTP status code of this response, as a L{int}.')
    phrase = Attribute('The HTTP reason phrase of this response, as a L{str}.')
    headers = Attribute('The HTTP response L{Headers} of this response.')
    length = Attribute('The L{int} number of bytes expected to be in the body of this response or L{UNKNOWN_LENGTH} if the server did not indicate how many bytes to expect.  For I{HEAD} responses, this will be 0; if the response includes a I{Content-Length} header, it will be available in C{headers}.')
    request = Attribute('The L{IClientRequest} that resulted in this response.')
    previousResponse = Attribute('The previous L{IResponse} from a redirect, or L{None} if there was no previous response. This can be used to walk the response or request history for redirections.')

    def deliverBody(protocol):
        if False:
            for i in range(10):
                print('nop')
        "\n        Register an L{IProtocol<twisted.internet.interfaces.IProtocol>} provider\n        to receive the response body.\n\n        The protocol will be connected to a transport which provides\n        L{IPushProducer}.  The protocol's C{connectionLost} method will be\n        called with:\n\n            - ResponseDone, which indicates that all bytes from the response\n              have been successfully delivered.\n\n            - PotentialDataLoss, which indicates that it cannot be determined\n              if the entire response body has been delivered.  This only occurs\n              when making requests to HTTP servers which do not set\n              I{Content-Length} or a I{Transfer-Encoding} in the response.\n\n            - ResponseFailed, which indicates that some bytes from the response\n              were lost.  The C{reasons} attribute of the exception may provide\n              more specific indications as to why.\n        "

    def setPreviousResponse(response):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the reference to the previous L{IResponse}.\n\n        The value of the previous response can be read via\n        L{IResponse.previousResponse}.\n        '

class _IRequestEncoder(Interface):
    """
    An object encoding data passed to L{IRequest.write}, for example for
    compression purpose.

    @since: 12.3
    """

    def encode(data):
        if False:
            print('Hello World!')
        '\n        Encode the data given and return the result.\n\n        @param data: The content to encode.\n        @type data: L{str}\n\n        @return: The encoded data.\n        @rtype: L{str}\n        '

    def finish():
        if False:
            i = 10
            return i + 15
        '\n        Callback called when the request is closing.\n\n        @return: If necessary, the pending data accumulated from previous\n            C{encode} calls.\n        @rtype: L{str}\n        '

class _IRequestEncoderFactory(Interface):
    """
    A factory for returing L{_IRequestEncoder} instances.

    @since: 12.3
    """

    def encoderForRequest(request):
        if False:
            print('Hello World!')
        '\n        If applicable, returns a L{_IRequestEncoder} instance which will encode\n        the request.\n        '

class IClientRequest(Interface):
    """
    An object representing an HTTP request to make to an HTTP server.

    @since: 13.1
    """
    method = Attribute("The HTTP method for this request, as L{bytes}. For example: C{b'GET'}, C{b'HEAD'}, C{b'POST'}, etc.")
    absoluteURI = Attribute('The absolute URI of the requested resource, as L{bytes}; or L{None} if the absolute URI cannot be determined.')
    headers = Attribute('Headers to be sent to the server, as a L{twisted.web.http_headers.Headers} instance.')

class IAgent(Interface):
    """
    An agent makes HTTP requests.

    The way in which requests are issued is left up to each implementation.
    Some may issue them directly to the server indicated by the net location
    portion of the request URL.  Others may use a proxy specified by system
    configuration.

    Processing of responses is also left very widely specified.  An
    implementation may perform no special handling of responses, or it may
    implement redirect following or content negotiation, it may implement a
    cookie store or automatically respond to authentication challenges.  It may
    implement many other unforeseen behaviors as well.

    It is also intended that L{IAgent} implementations be composable.  An
    implementation which provides cookie handling features should re-use an
    implementation that provides connection pooling and this combination could
    be used by an implementation which adds content negotiation functionality.
    Some implementations will be completely self-contained, such as those which
    actually perform the network operations to send and receive requests, but
    most or all other implementations should implement a small number of new
    features (perhaps one new feature) and delegate the rest of the
    request/response machinery to another implementation.

    This allows for great flexibility in the behavior an L{IAgent} will
    provide.  For example, an L{IAgent} with web browser-like behavior could be
    obtained by combining a number of (hypothetical) implementations::

        baseAgent = Agent(reactor)
        decode = ContentDecoderAgent(baseAgent, [(b"gzip", GzipDecoder())])
        cookie = CookieAgent(decode, diskStore.cookie)
        authenticate = AuthenticateAgent(
            cookie, [diskStore.credentials, GtkAuthInterface()])
        cache = CacheAgent(authenticate, diskStore.cache)
        redirect = BrowserLikeRedirectAgent(cache, limit=10)

        doSomeRequests(cache)
    """

    def request(method: bytes, uri: bytes, headers: Optional[Headers]=None, bodyProducer: Optional[IBodyProducer]=None) -> Deferred[IResponse]:
        if False:
            i = 10
            return i + 15
        '\n        Request the resource at the given location.\n\n        @param method: The request method to use, such as C{b"GET"}, C{b"HEAD"},\n            C{b"PUT"}, C{b"POST"}, etc.\n\n        @param uri: The location of the resource to request.  This should be an\n            absolute URI but some implementations may support relative URIs\n            (with absolute or relative paths).  I{HTTP} and I{HTTPS} are the\n            schemes most likely to be supported but others may be as well.\n\n        @param headers: The headers to send with the request (or L{None} to\n            send no extra headers).  An implementation may add its own headers\n            to this (for example for client identification or content\n            negotiation).\n\n        @param bodyProducer: An object which can generate bytes to make up the\n            body of this request (for example, the properly encoded contents of\n            a file for a file upload).  Or, L{None} if the request is to have\n            no body.\n\n        @return: A L{Deferred} that fires with an L{IResponse} provider when\n            the header of the response has been received (regardless of the\n            response status code) or with a L{Failure} if there is any problem\n            which prevents that response from being received (including\n            problems that prevent the request from being sent).\n        '

class IPolicyForHTTPS(Interface):
    """
    An L{IPolicyForHTTPS} provides a policy for verifying the certificates of
    HTTPS connections, in the form of a L{client connection creator
    <twisted.internet.interfaces.IOpenSSLClientConnectionCreator>} per network
    location.

    @since: 14.0
    """

    def creatorForNetloc(hostname, port):
        if False:
            i = 10
            return i + 15
        '\n        Create a L{client connection creator\n        <twisted.internet.interfaces.IOpenSSLClientConnectionCreator>}\n        appropriate for the given URL "netloc"; i.e. hostname and port number\n        pair.\n\n        @param hostname: The name of the requested remote host.\n        @type hostname: L{bytes}\n\n        @param port: The number of the requested remote port.\n        @type port: L{int}\n\n        @return: A client connection creator expressing the security\n            requirements for the given remote host.\n        @rtype: L{client connection creator\n            <twisted.internet.interfaces.IOpenSSLClientConnectionCreator>}\n        '

class IAgentEndpointFactory(Interface):
    """
    An L{IAgentEndpointFactory} provides a way of constructing an endpoint
    used for outgoing Agent requests. This is useful in the case of needing to
    proxy outgoing connections, or to otherwise vary the transport used.

    @since: 15.0
    """

    def endpointForURI(uri):
        if False:
            i = 10
            return i + 15
        "\n        Construct and return an L{IStreamClientEndpoint} for the outgoing\n        request's connection.\n\n        @param uri: The URI of the request.\n        @type uri: L{twisted.web.client.URI}\n\n        @return: An endpoint which will have its C{connect} method called to\n            issue the request.\n        @rtype: an L{IStreamClientEndpoint} provider\n\n        @raises twisted.internet.error.SchemeNotSupported: If the given\n            URI's scheme cannot be handled by this factory.\n        "
UNKNOWN_LENGTH = 'twisted.web.iweb.UNKNOWN_LENGTH'
__all__ = ['IUsernameDigestHash', 'ICredentialFactory', 'IRequest', 'IBodyProducer', 'IRenderable', 'IResponse', '_IRequestEncoder', '_IRequestEncoderFactory', 'IClientRequest', 'UNKNOWN_LENGTH']