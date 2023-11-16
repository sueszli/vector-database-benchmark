"""
Test HTTP/2 support.
"""
import itertools
from zope.interface import directlyProvides, providedBy
from twisted.internet import defer, error, reactor, task
from twisted.internet.address import IPv4Address
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.test.test_internet import DummyProducer
from twisted.trial import unittest
from twisted.web import http
from twisted.web.test.test_http import DelayedHTTPHandler, DelayedHTTPHandlerProxy, DummyHTTPHandler, DummyHTTPHandlerProxy, DummyPullProducerHandlerProxy, _IDeprecatedHTTPChannelToRequestInterfaceProxy, _makeRequestProxyFactory
skipH2 = None
try:
    import h2
    import h2.errors
    import h2.exceptions
    import hyperframe
    import priority
    from hpack.hpack import Decoder, Encoder
    from twisted.web._http2 import H2Connection
except ImportError:
    skipH2 = 'HTTP/2 support not enabled'

class FrameFactory:
    """
    A class containing lots of helper methods and state to build frames. This
    allows test cases to easily build correct HTTP/2 frames to feed to
    hyper-h2.
    """

    def __init__(self):
        if False:
            return 10
        self.encoder = Encoder()

    def refreshEncoder(self):
        if False:
            print('Hello World!')
        self.encoder = Encoder()

    def clientConnectionPreface(self):
        if False:
            while True:
                i = 10
        return b'PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n'

    def buildHeadersFrame(self, headers, flags=[], streamID=1, **priorityKwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Builds a single valid headers frame out of the contained headers.\n        '
        f = hyperframe.frame.HeadersFrame(streamID)
        f.data = self.encoder.encode(headers)
        f.flags.add('END_HEADERS')
        for flag in flags:
            f.flags.add(flag)
        for (k, v) in priorityKwargs.items():
            setattr(f, k, v)
        return f

    def buildDataFrame(self, data, flags=None, streamID=1):
        if False:
            i = 10
            return i + 15
        '\n        Builds a single data frame out of a chunk of data.\n        '
        flags = set(flags) if flags is not None else set()
        f = hyperframe.frame.DataFrame(streamID)
        f.data = data
        f.flags = flags
        return f

    def buildSettingsFrame(self, settings, ack=False):
        if False:
            i = 10
            return i + 15
        '\n        Builds a single settings frame.\n        '
        f = hyperframe.frame.SettingsFrame(0)
        if ack:
            f.flags.add('ACK')
        f.settings = settings
        return f

    def buildWindowUpdateFrame(self, streamID, increment):
        if False:
            i = 10
            return i + 15
        '\n        Builds a single WindowUpdate frame.\n        '
        f = hyperframe.frame.WindowUpdateFrame(streamID)
        f.window_increment = increment
        return f

    def buildGoAwayFrame(self, lastStreamID, errorCode=0, additionalData=b''):
        if False:
            return 10
        '\n        Builds a single GOAWAY frame.\n        '
        f = hyperframe.frame.GoAwayFrame(0)
        f.error_code = errorCode
        f.last_stream_id = lastStreamID
        f.additional_data = additionalData
        return f

    def buildRstStreamFrame(self, streamID, errorCode=0):
        if False:
            print('Hello World!')
        '\n        Builds a single RST_STREAM frame.\n        '
        f = hyperframe.frame.RstStreamFrame(streamID)
        f.error_code = errorCode
        return f

    def buildPriorityFrame(self, streamID, weight, dependsOn=0, exclusive=False):
        if False:
            i = 10
            return i + 15
        '\n        Builds a single priority frame.\n        '
        f = hyperframe.frame.PriorityFrame(streamID)
        f.depends_on = dependsOn
        f.stream_weight = weight
        f.exclusive = exclusive
        return f

    def buildPushPromiseFrame(self, streamID, promisedStreamID, headers, flags=[]):
        if False:
            return 10
        '\n        Builds a single Push Promise frame.\n        '
        f = hyperframe.frame.PushPromiseFrame(streamID)
        f.promised_stream_id = promisedStreamID
        f.data = self.encoder.encode(headers)
        f.flags = set(flags)
        f.flags.add('END_HEADERS')
        return f

class FrameBuffer:
    """
    A test object that converts data received from Twisted's HTTP/2 stack and
    turns it into a sequence of hyperframe frame objects.

    This is primarily used to make it easier to write and debug tests: rather
    than have to serialize the expected frames and then do byte-level
    comparison (which can be unclear in debugging output), this object makes it
    possible to work with the frames directly.

    It also ensures that headers are properly decompressed.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.decoder = Decoder()
        self._data = b''

    def receiveData(self, data):
        if False:
            while True:
                i = 10
        self._data += data

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def next(self):
        if False:
            while True:
                i = 10
        if len(self._data) < 9:
            raise StopIteration()
        (frame, length) = hyperframe.frame.Frame.parse_frame_header(self._data[:9])
        if len(self._data) < length + 9:
            raise StopIteration()
        frame.parse_body(memoryview(self._data[9:9 + length]))
        self._data = self._data[9 + length:]
        if isinstance(frame, hyperframe.frame.HeadersFrame):
            frame.data = self.decoder.decode(frame.data, raw=True)
        return frame
    __next__ = next

def buildRequestFrames(headers, data, frameFactory=None, streamID=1):
    if False:
        i = 10
        return i + 15
    "\n    Provides a sequence of HTTP/2 frames that encode a single HTTP request.\n    This should be used when you want to control the serialization yourself,\n    e.g. because you want to interleave other frames with these. If that's not\n    necessary, prefer L{buildRequestBytes}.\n\n    @param headers: The HTTP/2 headers to send.\n    @type headers: L{list} of L{tuple} of L{bytes}\n\n    @param data: The HTTP data to send. Each list entry will be sent in its own\n    frame.\n    @type data: L{list} of L{bytes}\n\n    @param frameFactory: The L{FrameFactory} that will be used to construct the\n    frames.\n    @type frameFactory: L{FrameFactory}\n\n    @param streamID: The ID of the stream on which to send the request.\n    @type streamID: L{int}\n    "
    if frameFactory is None:
        frameFactory = FrameFactory()
    frames = []
    frames.append(frameFactory.buildHeadersFrame(headers=headers, streamID=streamID))
    frames.extend((frameFactory.buildDataFrame(chunk, streamID=streamID) for chunk in data))
    frames[-1].flags.add('END_STREAM')
    return frames

def buildRequestBytes(headers, data, frameFactory=None, streamID=1):
    if False:
        i = 10
        return i + 15
    '\n    Provides the byte sequence for a collection of HTTP/2 frames representing\n    the provided request.\n\n    @param headers: The HTTP/2 headers to send.\n    @type headers: L{list} of L{tuple} of L{bytes}\n\n    @param data: The HTTP data to send. Each list entry will be sent in its own\n    frame.\n    @type data: L{list} of L{bytes}\n\n    @param frameFactory: The L{FrameFactory} that will be used to construct the\n    frames.\n    @type frameFactory: L{FrameFactory}\n\n    @param streamID: The ID of the stream on which to send the request.\n    @type streamID: L{int}\n    '
    frames = buildRequestFrames(headers, data, frameFactory, streamID)
    return b''.join((f.serialize() for f in frames))

def framesFromBytes(data):
    if False:
        return 10
    "\n    Given a sequence of bytes, decodes them into frames.\n\n    Note that this method should almost always be called only once, before\n    making some assertions. This is because decoding HTTP/2 frames is extremely\n    stateful, and this function doesn't preserve any of that state between\n    calls.\n\n    @param data: The serialized HTTP/2 frames.\n    @type data: L{bytes}\n\n    @returns: A list of HTTP/2 frames.\n    @rtype: L{list} of L{hyperframe.frame.Frame} subclasses.\n    "
    buffer = FrameBuffer()
    buffer.receiveData(data)
    return list(buffer)

class ChunkedHTTPHandler(http.Request):
    """
    A HTTP request object that writes chunks of data back to the network based
    on the URL.

    Must be called with a path /chunked/<num_chunks>
    """
    chunkData = b'hello world!'

    def process(self):
        if False:
            while True:
                i = 10
        chunks = int(self.uri.split(b'/')[-1])
        self.setResponseCode(200)
        for _ in range(chunks):
            self.write(self.chunkData)
        self.finish()
ChunkedHTTPHandlerProxy = _makeRequestProxyFactory(ChunkedHTTPHandler)

class ConsumerDummyHandler(http.Request):
    """
    This is a HTTP request handler that works with the C{IPushProducer}
    implementation in the L{H2Stream} object. No current IRequest object does
    that, but in principle future implementations could: that codepath should
    therefore be tested.
    """

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        http.Request.__init__(self, *args, **kwargs)
        self.channel.pauseProducing()
        self._requestReceived = False
        self._data = None

    def acceptData(self):
        if False:
            i = 10
            return i + 15
        '\n        Start the data pipe.\n        '
        self.channel.resumeProducing()

    def requestReceived(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._requestReceived = True
        return http.Request.requestReceived(self, *args, **kwargs)

    def process(self):
        if False:
            i = 10
            return i + 15
        self.setResponseCode(200)
        self._data = self.content.read()
        returnData = b'this is a response from a consumer dummy handler'
        self.write(returnData)
        self.finish()
ConsumerDummyHandlerProxy = _makeRequestProxyFactory(ConsumerDummyHandler)

class AbortingConsumerDummyHandler(ConsumerDummyHandler):
    """
    This is a HTTP request handler that works with the C{IPushProducer}
    implementation in the L{H2Stream} object. The difference between this and
    the ConsumerDummyHandler is that after resuming production it immediately
    aborts it again.
    """

    def acceptData(self):
        if False:
            while True:
                i = 10
        '\n        Start and then immediately stop the data pipe.\n        '
        self.channel.resumeProducing()
        self.channel.stopProducing()
AbortingConsumerDummyHandlerProxy = _makeRequestProxyFactory(AbortingConsumerDummyHandler)

class DummyProducerHandler(http.Request):
    """
    An HTTP request handler that registers a dummy producer to serve the body.

    The owner must call C{finish} to complete the response.
    """

    def process(self):
        if False:
            while True:
                i = 10
        self.setResponseCode(200)
        self.registerProducer(DummyProducer(), True)
DummyProducerHandlerProxy = _makeRequestProxyFactory(DummyProducerHandler)

class NotifyingRequestFactory:
    """
    A L{http.Request} factory that calls L{http.Request.notifyFinish} on all
    L{http.Request} objects before it returns them, and squirrels the resulting
    L{defer.Deferred} away on the class for later use. This is done as early
    as possible to ensure that we always see the result.
    """

    def __init__(self, wrappedFactory):
        if False:
            while True:
                i = 10
        self.results = []
        self._wrappedFactory = wrappedFactory
        for interface in providedBy(self._wrappedFactory):
            directlyProvides(self, interface)

    def __call__(self, *args, **kwargs):
        if False:
            return 10
        req = self._wrappedFactory(*args, **kwargs)
        self.results.append(req.notifyFinish())
        return _IDeprecatedHTTPChannelToRequestInterfaceProxy(req)
NotifyingRequestFactoryProxy = _makeRequestProxyFactory(NotifyingRequestFactory)

class HTTP2TestHelpers:
    """
    A superclass that contains no tests but provides test helpers for HTTP/2
    tests.
    """
    if skipH2:
        skip = skipH2

    def assertAllStreamsBlocked(self, connection):
        if False:
            while True:
                i = 10
        '\n        Confirm that all streams are blocked: that is, the priority tree\n        believes that none of the streams have data ready to send.\n        '
        self.assertRaises(priority.DeadlockError, next, connection.priority)

class HTTP2ServerTests(unittest.TestCase, HTTP2TestHelpers):
    getRequestHeaders = [(b':method', b'GET'), (b':authority', b'localhost'), (b':path', b'/'), (b':scheme', b'https'), (b'user-agent', b'twisted-test-code'), (b'custom-header', b'1'), (b'custom-header', b'2')]
    postRequestHeaders = [(b':method', b'POST'), (b':authority', b'localhost'), (b':path', b'/post_endpoint'), (b':scheme', b'https'), (b'user-agent', b'twisted-test-code'), (b'content-length', b'25')]
    postRequestData = [b'hello ', b'world, ', b"it's ", b'http/2!']
    getResponseHeaders = [(b':status', b'200'), (b'request', b'/'), (b'command', b'GET'), (b'version', b'HTTP/2'), (b'content-length', b'13')]
    getResponseData = b"'''\nNone\n'''\n"
    postResponseHeaders = [(b':status', b'200'), (b'request', b'/post_endpoint'), (b'command', b'POST'), (b'version', b'HTTP/2'), (b'content-length', b'36')]
    postResponseData = b"'''\n25\nhello world, it's http/2!'''\n"

    def connectAndReceive(self, connection, headers, body):
        if False:
            i = 10
            return i + 15
        '\n        Takes a single L{H2Connection} object and connects it to a\n        L{StringTransport} using a brand new L{FrameFactory}.\n\n        @param connection: The L{H2Connection} object to connect.\n        @type connection: L{H2Connection}\n\n        @param headers: The headers to send on the first request.\n        @type headers: L{Iterable} of L{tuple} of C{(bytes, bytes)}\n\n        @param body: Chunks of body to send, if any.\n        @type body: L{Iterable} of L{bytes}\n\n        @return: A tuple of L{FrameFactory}, L{StringTransport}\n        '
        frameFactory = FrameFactory()
        transport = StringTransport()
        requestBytes = frameFactory.clientConnectionPreface()
        requestBytes += buildRequestBytes(headers, body, frameFactory)
        connection.makeConnection(transport)
        for byte in iterbytes(requestBytes):
            connection.dataReceived(byte)
        return (frameFactory, transport)

    def test_basicRequest(self):
        if False:
            print('Hello World!')
        '\n        Send request over a TCP connection and confirm that we get back the\n        expected data in the order and style we expect.\n        '
        connection = H2Connection()
        connection.requestFactory = DummyHTTPHandlerProxy
        (_, transport) = self.connectAndReceive(connection, self.getRequestHeaders, [])

        def validate(streamID):
            if False:
                print('Hello World!')
            frames = framesFromBytes(transport.value())
            self.assertEqual(len(frames), 4)
            self.assertTrue(all((f.stream_id == 1 for f in frames[1:])))
            self.assertTrue(isinstance(frames[1], hyperframe.frame.HeadersFrame))
            self.assertTrue(isinstance(frames[2], hyperframe.frame.DataFrame))
            self.assertTrue(isinstance(frames[3], hyperframe.frame.DataFrame))
            self.assertEqual(dict(frames[1].data), dict(self.getResponseHeaders))
            self.assertEqual(frames[2].data, self.getResponseData)
            self.assertEqual(frames[3].data, b'')
            self.assertTrue('END_STREAM' in frames[3].flags)
        return connection._streamCleanupCallbacks[1].addCallback(validate)

    def test_postRequest(self):
        if False:
            i = 10
            return i + 15
        '\n        Send a POST request and confirm that the data is safely transferred.\n        '
        connection = H2Connection()
        connection.requestFactory = DummyHTTPHandlerProxy
        (_, transport) = self.connectAndReceive(connection, self.postRequestHeaders, self.postRequestData)

        def validate(streamID):
            if False:
                print('Hello World!')
            frames = framesFromBytes(transport.value())
            self.assertEqual(len(frames), 4)
            self.assertTrue(all((f.stream_id == 1 for f in frames[-3:])))
            self.assertTrue(isinstance(frames[-3], hyperframe.frame.HeadersFrame))
            self.assertTrue(isinstance(frames[-2], hyperframe.frame.DataFrame))
            self.assertTrue(isinstance(frames[-1], hyperframe.frame.DataFrame))
            self.assertEqual(dict(frames[-3].data), dict(self.postResponseHeaders))
            self.assertEqual(frames[-2].data, self.postResponseData)
            self.assertEqual(frames[-1].data, b'')
            self.assertTrue('END_STREAM' in frames[-1].flags)
        return connection._streamCleanupCallbacks[1].addCallback(validate)

    def test_postRequestNoLength(self):
        if False:
            print('Hello World!')
        '\n        Send a POST request without length and confirm that the data is safely\n        transferred.\n        '
        postResponseHeaders = [(b':status', b'200'), (b'request', b'/post_endpoint'), (b'command', b'POST'), (b'version', b'HTTP/2'), (b'content-length', b'38')]
        postResponseData = b"'''\nNone\nhello world, it's http/2!'''\n"
        postRequestHeaders = [(x, y) for (x, y) in self.postRequestHeaders if x != b'content-length']
        connection = H2Connection()
        connection.requestFactory = DummyHTTPHandlerProxy
        (_, transport) = self.connectAndReceive(connection, postRequestHeaders, self.postRequestData)

        def validate(streamID):
            if False:
                return 10
            frames = framesFromBytes(transport.value())
            self.assertEqual(len(frames), 4)
            self.assertTrue(all((f.stream_id == 1 for f in frames[-3:])))
            self.assertTrue(isinstance(frames[-3], hyperframe.frame.HeadersFrame))
            self.assertTrue(isinstance(frames[-2], hyperframe.frame.DataFrame))
            self.assertTrue(isinstance(frames[-1], hyperframe.frame.DataFrame))
            self.assertEqual(dict(frames[-3].data), dict(postResponseHeaders))
            self.assertEqual(frames[-2].data, postResponseData)
            self.assertEqual(frames[-1].data, b'')
            self.assertTrue('END_STREAM' in frames[-1].flags)
        return connection._streamCleanupCallbacks[1].addCallback(validate)

    def test_interleavedRequests(self):
        if False:
            print('Hello World!')
        '\n        Many interleaved POST requests all get received and responded to\n        appropriately.\n        '
        REQUEST_COUNT = 40
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyHTTPHandlerProxy
        streamIDs = list(range(1, REQUEST_COUNT * 2, 2))
        frames = [buildRequestFrames(self.postRequestHeaders, self.postRequestData, f, streamID) for streamID in streamIDs]
        requestBytes = f.clientConnectionPreface()
        frames = itertools.chain.from_iterable(zip(*frames))
        requestBytes += b''.join((frame.serialize() for frame in frames))
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)

        def validate(results):
            if False:
                print('Hello World!')
            frames = framesFromBytes(b.value())
            self.assertEqual(len(frames), 1 + 3 * 40)
            for streamID in streamIDs:
                streamFrames = [f for f in frames if f.stream_id == streamID and (not isinstance(f, hyperframe.frame.WindowUpdateFrame))]
                self.assertEqual(len(streamFrames), 3)
                self.assertEqual(dict(streamFrames[0].data), dict(self.postResponseHeaders))
                self.assertEqual(streamFrames[1].data, self.postResponseData)
                self.assertEqual(streamFrames[2].data, b'')
                self.assertTrue('END_STREAM' in streamFrames[2].flags)
        return defer.DeferredList(list(a._streamCleanupCallbacks.values())).addCallback(validate)

    def test_sendAccordingToPriority(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Data in responses is interleaved according to HTTP/2 priorities.\n        '
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = ChunkedHTTPHandlerProxy
        getRequestHeaders = self.getRequestHeaders
        getRequestHeaders[2] = (':path', '/chunked/4')
        frames = [buildRequestFrames(getRequestHeaders, [], f, streamID) for streamID in [1, 3, 5]]
        frames[0][0].flags.add('PRIORITY')
        frames[0][0].stream_weight = 64
        frames[1][0].flags.add('PRIORITY')
        frames[1][0].stream_weight = 32
        priorityFrame = f.buildPriorityFrame(streamID=5, weight=16, dependsOn=1, exclusive=True)
        frames[2].insert(0, priorityFrame)
        frames = itertools.chain.from_iterable(frames)
        requestBytes = f.clientConnectionPreface()
        requestBytes += b''.join((frame.serialize() for frame in frames))
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)

        def validate(results):
            if False:
                i = 10
                return i + 15
            frames = framesFromBytes(b.value())
            self.assertEqual(len(frames), 19)
            streamIDs = [f.stream_id for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
            expectedOrder = [1, 3, 1, 1, 3, 1, 1, 3, 5, 3, 5, 3, 5, 5, 5]
            self.assertEqual(streamIDs, expectedOrder)
        return defer.DeferredList(list(a._streamCleanupCallbacks.values())).addCallback(validate)

    def test_protocolErrorTerminatesConnection(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A protocol error from the remote peer terminates the connection.\n        '
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyHTTPHandlerProxy
        requestBytes = f.clientConnectionPreface()
        requestBytes += buildRequestBytes(self.getRequestHeaders, [], f)
        requestBytes += f.buildPushPromiseFrame(streamID=1, promisedStreamID=2, headers=self.getRequestHeaders, flags=['END_HEADERS']).serialize()
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
            if b.disconnecting:
                break
        frames = framesFromBytes(b.value())
        self.assertEqual(len(frames), 3)
        self.assertTrue(isinstance(frames[-1], hyperframe.frame.GoAwayFrame))
        self.assertTrue(b.disconnecting)

    def test_streamProducingData(self):
        if False:
            return 10
        '\n        The H2Stream data implements IPushProducer, and can have its data\n        production controlled by the Request if the Request chooses to.\n        '
        connection = H2Connection()
        connection.requestFactory = ConsumerDummyHandlerProxy
        (_, transport) = self.connectAndReceive(connection, self.postRequestHeaders, self.postRequestData)
        request = connection.streams[1]._request.original
        self.assertFalse(request._requestReceived)
        frames = framesFromBytes(transport.value())
        self.assertEqual(len(frames), 1)
        request.acceptData()
        self.assertTrue(request._requestReceived)
        self.assertTrue(request._data, b"hello world, it's http/2!")
        frames = framesFromBytes(transport.value())
        self.assertEqual(len(frames), 2)

        def validate(streamID):
            if False:
                return 10
            frames = framesFromBytes(transport.value())
            self.assertEqual(len(frames), 4)
            self.assertTrue('END_STREAM' in frames[-1].flags)
        return connection._streamCleanupCallbacks[1].addCallback(validate)

    def test_abortStreamProducingData(self):
        if False:
            i = 10
            return i + 15
        '\n        The H2Stream data implements IPushProducer, and can have its data\n        production controlled by the Request if the Request chooses to.\n        When the production is stopped, that causes the stream connection to\n        be lost.\n        '
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = AbortingConsumerDummyHandlerProxy
        frames = buildRequestFrames(self.postRequestHeaders, self.postRequestData, f)
        frames[-1].flags = set()
        requestBytes = f.clientConnectionPreface()
        requestBytes += b''.join((f.serialize() for f in frames))
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
        request = a.streams[1]._request.original
        self.assertFalse(request._requestReceived)
        cleanupCallback = a._streamCleanupCallbacks[1]
        request.acceptData()

        def validate(streamID):
            if False:
                i = 10
                return i + 15
            frames = framesFromBytes(b.value())
            self.assertEqual(len(frames), 2)
            self.assertTrue(isinstance(frames[-1], hyperframe.frame.RstStreamFrame))
            self.assertEqual(frames[-1].stream_id, 1)
        return cleanupCallback.addCallback(validate)

    def test_terminatedRequest(self):
        if False:
            while True:
                i = 10
        '\n        When a RstStream frame is received, the L{H2Connection} and L{H2Stream}\n        objects tear down the L{http.Request} and swallow all outstanding\n        writes.\n        '
        connection = H2Connection()
        connection.requestFactory = DummyProducerHandlerProxy
        (frameFactory, transport) = self.connectAndReceive(connection, self.getRequestHeaders, [])
        request = connection.streams[1]._request.original
        request.write(b'first chunk')
        request.write(b'second chunk')
        cleanupCallback = connection._streamCleanupCallbacks[1]
        connection.dataReceived(frameFactory.buildRstStreamFrame(1, errorCode=1).serialize())
        self.assertTrue(request._disconnected)
        self.assertTrue(request.channel is None)
        request.write(b'third chunk')

        def validate(streamID):
            if False:
                print('Hello World!')
            frames = framesFromBytes(transport.value())
            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[1].stream_id, 1)
            self.assertTrue(isinstance(frames[1], hyperframe.frame.HeadersFrame))
        return cleanupCallback.addCallback(validate)

    def test_terminatedConnection(self):
        if False:
            return 10
        '\n        When a GoAway frame is received, the L{H2Connection} and L{H2Stream}\n        objects tear down all outstanding L{http.Request} objects and stop all\n        writing.\n        '
        connection = H2Connection()
        connection.requestFactory = DummyProducerHandlerProxy
        (frameFactory, transport) = self.connectAndReceive(connection, self.getRequestHeaders, [])
        request = connection.streams[1]._request.original
        request.write(b'first chunk')
        request.write(b'second chunk')
        cleanupCallback = connection._streamCleanupCallbacks[1]
        connection.dataReceived(frameFactory.buildGoAwayFrame(lastStreamID=0).serialize())
        self.assertTrue(request._disconnected)
        self.assertTrue(request.channel is None)
        self.assertFalse(connection._stillProducing)

        def validate(streamID):
            if False:
                print('Hello World!')
            frames = framesFromBytes(transport.value())
            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[1].stream_id, 1)
            self.assertTrue(isinstance(frames[1], hyperframe.frame.HeadersFrame))
        return cleanupCallback.addCallback(validate)

    def test_respondWith100Continue(self):
        if False:
            print('Hello World!')
        '\n        Requests containing Expect: 100-continue cause provisional 100\n        responses to be emitted.\n        '
        connection = H2Connection()
        connection.requestFactory = DummyHTTPHandlerProxy
        headers = self.getRequestHeaders + [(b'expect', b'100-continue')]
        (_, transport) = self.connectAndReceive(connection, headers, [])

        def validate(streamID):
            if False:
                print('Hello World!')
            frames = framesFromBytes(transport.value())
            self.assertEqual(len(frames), 5)
            self.assertTrue(all((f.stream_id == 1 for f in frames[1:])))
            self.assertTrue(isinstance(frames[1], hyperframe.frame.HeadersFrame))
            self.assertEqual(frames[1].data, [(b':status', b'100')])
            self.assertTrue('END_STREAM' in frames[-1].flags)
        return connection._streamCleanupCallbacks[1].addCallback(validate)

    def test_respondWith400(self):
        if False:
            print('Hello World!')
        '\n        Triggering the call to L{H2Stream._respondToBadRequestAndDisconnect}\n        leads to a 400 error being sent automatically and the stream being torn\n        down.\n        '
        connection = H2Connection()
        connection.requestFactory = DummyProducerHandlerProxy
        (_, transport) = self.connectAndReceive(connection, self.getRequestHeaders, [])
        stream = connection.streams[1]
        request = stream._request.original
        cleanupCallback = connection._streamCleanupCallbacks[1]
        stream._respondToBadRequestAndDisconnect()
        self.assertTrue(request._disconnected)
        self.assertTrue(request.channel is None)

        def validate(streamID):
            if False:
                print('Hello World!')
            frames = framesFromBytes(transport.value())
            self.assertEqual(len(frames), 2)
            self.assertTrue(isinstance(frames[1], hyperframe.frame.HeadersFrame))
            self.assertEqual(frames[1].data, [(b':status', b'400')])
            self.assertTrue('END_STREAM' in frames[-1].flags)
        return cleanupCallback.addCallback(validate)

    def test_loseH2StreamConnection(self):
        if False:
            i = 10
            return i + 15
        '\n        Calling L{Request.loseConnection} causes all data that has previously\n        been sent to be flushed, and then the stream cleanly closed.\n        '
        connection = H2Connection()
        connection.requestFactory = DummyProducerHandlerProxy
        (_, transport) = self.connectAndReceive(connection, self.getRequestHeaders, [])
        stream = connection.streams[1]
        request = stream._request.original
        dataChunks = [b'hello', b'world', b'here', b'are', b'some', b'writes']
        for chunk in dataChunks:
            request.write(chunk)
        request.loseConnection()

        def validate(streamID):
            if False:
                return 10
            frames = framesFromBytes(transport.value())
            self.assertEqual(len(frames), 9)
            self.assertTrue(all((f.stream_id == 1 for f in frames[1:])))
            self.assertTrue(isinstance(frames[1], hyperframe.frame.HeadersFrame))
            self.assertTrue('END_STREAM' in frames[-1].flags)
            receivedDataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
            self.assertEqual(receivedDataChunks, dataChunks + [b''])
        return connection._streamCleanupCallbacks[1].addCallback(validate)

    def test_cannotRegisterTwoProducers(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The L{H2Stream} object forbids registering two producers.\n        '
        connection = H2Connection()
        connection.requestFactory = DummyProducerHandlerProxy
        self.connectAndReceive(connection, self.getRequestHeaders, [])
        stream = connection.streams[1]
        request = stream._request.original
        self.assertRaises(ValueError, stream.registerProducer, request, True)

    def test_handlesPullProducer(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{Request} objects that have registered pull producers get blocked and\n        unblocked according to HTTP/2 flow control.\n        '
        connection = H2Connection()
        connection.requestFactory = DummyPullProducerHandlerProxy
        (_, transport) = self.connectAndReceive(connection, self.getRequestHeaders, [])
        stream = connection.streams[1]
        request = stream._request.original
        producerComplete = request._actualProducer.result
        producerComplete.addCallback(lambda x: request.finish())

        def validate(streamID):
            if False:
                while True:
                    i = 10
            frames = framesFromBytes(transport.value())
            self.assertTrue('END_STREAM' in frames[-1].flags)
            dataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
            self.assertEqual(dataChunks, [b'0', b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8', b'9', b''])
        return connection._streamCleanupCallbacks[1].addCallback(validate)

    def test_isSecureWorksProperly(self):
        if False:
            i = 10
            return i + 15
        '\n        L{Request} objects can correctly ask isSecure on HTTP/2.\n        '
        connection = H2Connection()
        connection.requestFactory = DelayedHTTPHandlerProxy
        self.connectAndReceive(connection, self.getRequestHeaders, [])
        request = connection.streams[1]._request.original
        self.assertFalse(request.isSecure())
        connection.streams[1].abortConnection()

    def test_lateCompletionWorks(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{H2Connection} correctly unblocks when a stream is ended.\n        '
        connection = H2Connection()
        connection.requestFactory = DelayedHTTPHandlerProxy
        (_, transport) = self.connectAndReceive(connection, self.getRequestHeaders, [])
        request = connection.streams[1]._request.original
        reactor.callLater(0.01, request.finish)

        def validateComplete(*args):
            if False:
                i = 10
                return i + 15
            frames = framesFromBytes(transport.value())
            self.assertEqual(len(frames), 3)
            self.assertTrue('END_STREAM' in frames[-1].flags)
        return connection._streamCleanupCallbacks[1].addCallback(validateComplete)

    def test_writeSequenceForChannels(self):
        if False:
            i = 10
            return i + 15
        '\n        L{H2Stream} objects can send a series of frames via C{writeSequence}.\n        '
        connection = H2Connection()
        connection.requestFactory = DelayedHTTPHandlerProxy
        (_, transport) = self.connectAndReceive(connection, self.getRequestHeaders, [])
        stream = connection.streams[1]
        request = stream._request.original
        request.setResponseCode(200)
        stream.writeSequence([b'Hello', b',', b'world!'])
        request.finish()
        completionDeferred = connection._streamCleanupCallbacks[1]

        def validate(streamID):
            if False:
                while True:
                    i = 10
            frames = framesFromBytes(transport.value())
            self.assertTrue('END_STREAM' in frames[-1].flags)
            dataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
            self.assertEqual(dataChunks, [b'Hello', b',', b'world!', b''])
        return completionDeferred.addCallback(validate)

    def test_delayWrites(self):
        if False:
            i = 10
            return i + 15
        "\n        Delaying writes from L{Request} causes the L{H2Connection} to block on\n        sending until data is available. However, data is *not* sent if there's\n        no room in the flow control window.\n        "
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DelayedHTTPHandlerProxy
        requestBytes = f.clientConnectionPreface()
        requestBytes += f.buildSettingsFrame({h2.settings.SettingCodes.INITIAL_WINDOW_SIZE: 5}).serialize()
        requestBytes += buildRequestBytes(self.getRequestHeaders, [], f)
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
        stream = a.streams[1]
        request = stream._request.original
        request.write(b'fiver')
        dataChunks = [b'here', b'are', b'some', b'writes']

        def write_chunks():
            if False:
                while True:
                    i = 10
            for chunk in dataChunks:
                request.write(chunk)
            request.finish()
        d = task.deferLater(reactor, 0.01, write_chunks)
        d.addCallback(lambda *args: a.dataReceived(f.buildWindowUpdateFrame(streamID=1, increment=50).serialize()))

        def validate(streamID):
            if False:
                return 10
            frames = framesFromBytes(b.value())
            self.assertEqual(len(frames), 9)
            self.assertTrue(all((f.stream_id == 1 for f in frames[2:])))
            self.assertTrue(isinstance(frames[2], hyperframe.frame.HeadersFrame))
            self.assertTrue('END_STREAM' in frames[-1].flags)
            receivedDataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
            self.assertEqual(receivedDataChunks, [b'fiver'] + dataChunks + [b''])
        return a._streamCleanupCallbacks[1].addCallback(validate)

    def test_resetAfterBody(self):
        if False:
            return 10
        '\n        A client that immediately resets after sending the body causes Twisted\n        to send no response.\n        '
        frameFactory = FrameFactory()
        transport = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyHTTPHandlerProxy
        requestBytes = frameFactory.clientConnectionPreface()
        requestBytes += buildRequestBytes(headers=self.getRequestHeaders, data=[], frameFactory=frameFactory)
        requestBytes += frameFactory.buildRstStreamFrame(streamID=1).serialize()
        a.makeConnection(transport)
        a.dataReceived(requestBytes)
        frames = framesFromBytes(transport.value())
        self.assertEqual(len(frames), 1)
        self.assertNotIn(1, a._streamCleanupCallbacks)

    def test_RequestRequiringFactorySiteInConstructor(self):
        if False:
            while True:
                i = 10
        '\n        A custom L{Request} subclass that requires the site and factory in the\n        constructor is able to get them.\n        '
        d = defer.Deferred()

        class SuperRequest(DummyHTTPHandler):

            def __init__(self, *args, **kwargs):
                if False:
                    print('Hello World!')
                DummyHTTPHandler.__init__(self, *args, **kwargs)
                d.callback((self.channel.site, self.channel.factory))
        connection = H2Connection()
        httpFactory = http.HTTPFactory()
        connection.requestFactory = _makeRequestProxyFactory(SuperRequest)
        connection.factory = httpFactory
        connection.site = object()
        self.connectAndReceive(connection, self.getRequestHeaders, [])

        def validateFactoryAndSite(args):
            if False:
                i = 10
                return i + 15
            (site, factory) = args
            self.assertIs(site, connection.site)
            self.assertIs(factory, connection.factory)
        d.addCallback(validateFactoryAndSite)
        cleanupCallback = connection._streamCleanupCallbacks[1]
        return defer.gatherResults([d, cleanupCallback])

    def test_notifyOnCompleteRequest(self):
        if False:
            i = 10
            return i + 15
        '\n        A request sent to a HTTP/2 connection fires the\n        L{http.Request.notifyFinish} callback with a L{None} value.\n        '
        connection = H2Connection()
        connection.requestFactory = NotifyingRequestFactory(DummyHTTPHandler)
        (_, transport) = self.connectAndReceive(connection, self.getRequestHeaders, [])
        deferreds = connection.requestFactory.results
        self.assertEqual(len(deferreds), 1)

        def validate(result):
            if False:
                while True:
                    i = 10
            self.assertIsNone(result)
        d = deferreds[0]
        d.addCallback(validate)
        cleanupCallback = connection._streamCleanupCallbacks[1]
        return defer.gatherResults([d, cleanupCallback])

    def test_notifyOnResetStream(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A HTTP/2 reset stream fires the L{http.Request.notifyFinish} deferred\n        with L{ConnectionLost}.\n        '
        connection = H2Connection()
        connection.requestFactory = NotifyingRequestFactory(DelayedHTTPHandler)
        (frameFactory, transport) = self.connectAndReceive(connection, self.getRequestHeaders, [])
        deferreds = connection.requestFactory.results
        self.assertEqual(len(deferreds), 1)

        def callback(ign):
            if False:
                print('Hello World!')
            self.fail("Didn't errback, called back instead")

        def errback(reason):
            if False:
                i = 10
                return i + 15
            self.assertIsInstance(reason, failure.Failure)
            self.assertIs(reason.type, error.ConnectionLost)
            return None
        d = deferreds[0]
        d.addCallbacks(callback, errback)
        invalidData = frameFactory.buildRstStreamFrame(streamID=1).serialize()
        connection.dataReceived(invalidData)
        return d

    def test_failWithProtocolError(self):
        if False:
            print('Hello World!')
        '\n        A HTTP/2 protocol error triggers the L{http.Request.notifyFinish}\n        deferred for all outstanding requests with a Failure that contains the\n        underlying exception.\n        '
        connection = H2Connection()
        connection.requestFactory = NotifyingRequestFactory(DelayedHTTPHandler)
        (frameFactory, transport) = self.connectAndReceive(connection, self.getRequestHeaders, [])
        secondRequest = buildRequestBytes(self.getRequestHeaders, [], frameFactory=frameFactory, streamID=3)
        connection.dataReceived(secondRequest)
        deferreds = connection.requestFactory.results
        self.assertEqual(len(deferreds), 2)

        def callback(ign):
            if False:
                return 10
            self.fail("Didn't errback, called back instead")

        def errback(reason):
            if False:
                while True:
                    i = 10
            self.assertIsInstance(reason, failure.Failure)
            self.assertIsInstance(reason.value, h2.exceptions.ProtocolError)
            return None
        for d in deferreds:
            d.addCallbacks(callback, errback)
        invalidData = frameFactory.buildDataFrame(data=b'yo', streamID=240).serialize()
        connection.dataReceived(invalidData)
        return defer.gatherResults(deferreds)

    def test_failOnGoaway(self):
        if False:
            i = 10
            return i + 15
        '\n        A HTTP/2 GoAway triggers the L{http.Request.notifyFinish}\n        deferred for all outstanding requests with a Failure that contains a\n        RemoteGoAway error.\n        '
        connection = H2Connection()
        connection.requestFactory = NotifyingRequestFactory(DelayedHTTPHandler)
        (frameFactory, transport) = self.connectAndReceive(connection, self.getRequestHeaders, [])
        secondRequest = buildRequestBytes(self.getRequestHeaders, [], frameFactory=frameFactory, streamID=3)
        connection.dataReceived(secondRequest)
        deferreds = connection.requestFactory.results
        self.assertEqual(len(deferreds), 2)

        def callback(ign):
            if False:
                print('Hello World!')
            self.fail("Didn't errback, called back instead")

        def errback(reason):
            if False:
                i = 10
                return i + 15
            self.assertIsInstance(reason, failure.Failure)
            self.assertIs(reason.type, error.ConnectionLost)
            return None
        for d in deferreds:
            d.addCallbacks(callback, errback)
        invalidData = frameFactory.buildGoAwayFrame(lastStreamID=3).serialize()
        connection.dataReceived(invalidData)
        return defer.gatherResults(deferreds)

    def test_failOnStopProducing(self):
        if False:
            i = 10
            return i + 15
        '\n        The transport telling the HTTP/2 connection to stop producing will\n        fire all L{http.Request.notifyFinish} errbacks with L{error.}\n        '
        connection = H2Connection()
        connection.requestFactory = NotifyingRequestFactory(DelayedHTTPHandler)
        (frameFactory, transport) = self.connectAndReceive(connection, self.getRequestHeaders, [])
        secondRequest = buildRequestBytes(self.getRequestHeaders, [], frameFactory=frameFactory, streamID=3)
        connection.dataReceived(secondRequest)
        deferreds = connection.requestFactory.results
        self.assertEqual(len(deferreds), 2)

        def callback(ign):
            if False:
                while True:
                    i = 10
            self.fail("Didn't errback, called back instead")

        def errback(reason):
            if False:
                return 10
            self.assertIsInstance(reason, failure.Failure)
            self.assertIs(reason.type, error.ConnectionLost)
            return None
        for d in deferreds:
            d.addCallbacks(callback, errback)
        connection.stopProducing()
        return defer.gatherResults(deferreds)

    def test_notifyOnFast400(self):
        if False:
            print('Hello World!')
        '\n        A HTTP/2 stream that has had _respondToBadRequestAndDisconnect called\n        on it from a request handler calls the L{http.Request.notifyFinish}\n        errback with L{ConnectionLost}.\n        '
        connection = H2Connection()
        connection.requestFactory = NotifyingRequestFactory(DelayedHTTPHandler)
        (frameFactory, transport) = self.connectAndReceive(connection, self.getRequestHeaders, [])
        deferreds = connection.requestFactory.results
        self.assertEqual(len(deferreds), 1)

        def callback(ign):
            if False:
                i = 10
                return i + 15
            self.fail("Didn't errback, called back instead")

        def errback(reason):
            if False:
                while True:
                    i = 10
            self.assertIsInstance(reason, failure.Failure)
            self.assertIs(reason.type, error.ConnectionLost)
            return None
        d = deferreds[0]
        d.addCallbacks(callback, errback)
        stream = connection.streams[1]
        stream._respondToBadRequestAndDisconnect()
        return d

    def test_fast400WithCircuitBreaker(self):
        if False:
            i = 10
            return i + 15
        '\n        A HTTP/2 stream that has had _respondToBadRequestAndDisconnect\n        called on it does not write control frame data if its\n        transport is paused and its control frame limit has been\n        reached.\n        '
        memoryReactor = MemoryReactorClock()
        connection = H2Connection(memoryReactor)
        connection.callLater = memoryReactor.callLater
        connection.requestFactory = DelayedHTTPHandler
        streamID = 1
        frameFactory = FrameFactory()
        transport = StringTransport()
        clientConnectionPreface = frameFactory.clientConnectionPreface()
        connection.makeConnection(transport)
        connection.dataReceived(clientConnectionPreface)
        connection.dataReceived(buildRequestBytes(self.getRequestHeaders, [], frameFactory, streamID=streamID))
        connection.pauseProducing()
        connection._maxBufferedControlFrameBytes = 0
        connection._respondToBadRequestAndDisconnect(streamID)
        self.assertTrue(transport.disconnected)

    def test_bufferingAutomaticFrameData(self):
        if False:
            i = 10
            return i + 15
        '\n        If a the L{H2Connection} has been paused by the transport, it will\n        not write automatic frame data triggered by writes.\n        '
        connection = H2Connection()
        connection.requestFactory = DummyHTTPHandlerProxy
        frameFactory = FrameFactory()
        transport = StringTransport()
        clientConnectionPreface = frameFactory.clientConnectionPreface()
        connection.makeConnection(transport)
        connection.dataReceived(clientConnectionPreface)
        connection.pauseProducing()
        for _ in range(0, 100):
            connection.dataReceived(frameFactory.buildSettingsFrame({}).serialize())
        frames = framesFromBytes(transport.value())
        self.assertEqual(len(frames), 1)
        connection.resumeProducing()
        frames = framesFromBytes(transport.value())
        self.assertEqual(len(frames), 101)

    def test_bufferingAutomaticFrameDataWithCircuitBreaker(self):
        if False:
            i = 10
            return i + 15
        '\n        If the L{H2Connection} has been paused by the transport, it will\n        not write automatic frame data triggered by writes. If this buffer\n        gets too large, the connection will be dropped.\n        '
        connection = H2Connection()
        connection.requestFactory = DummyHTTPHandlerProxy
        frameFactory = FrameFactory()
        transport = StringTransport()
        clientConnectionPreface = frameFactory.clientConnectionPreface()
        connection.makeConnection(transport)
        connection.dataReceived(clientConnectionPreface)
        connection.pauseProducing()
        connection._maxBufferedControlFrameBytes = 100
        self.assertFalse(transport.disconnecting)
        for _ in range(0, 11):
            connection.dataReceived(frameFactory.buildSettingsFrame({}).serialize())
        self.assertFalse(transport.disconnecting)
        connection.dataReceived(frameFactory.buildSettingsFrame({}).serialize())
        self.assertTrue(transport.disconnected)

    def test_bufferingContinuesIfProducerIsPausedOnWrite(self):
        if False:
            return 10
        '\n        If the L{H2Connection} has buffered control frames, is unpaused, and then\n        paused while unbuffering, it persists the buffer and stops trying to write.\n        '

        class AutoPausingStringTransport(StringTransport):

            def write(self, *args, **kwargs):
                if False:
                    while True:
                        i = 10
                StringTransport.write(self, *args, **kwargs)
                self.producer.pauseProducing()
        connection = H2Connection()
        connection.requestFactory = DummyHTTPHandlerProxy
        frameFactory = FrameFactory()
        transport = AutoPausingStringTransport()
        transport.registerProducer(connection, True)
        clientConnectionPreface = frameFactory.clientConnectionPreface()
        connection.makeConnection(transport)
        connection.dataReceived(clientConnectionPreface)
        self.assertIsNotNone(connection._consumerBlocked)
        frames = framesFromBytes(transport.value())
        self.assertEqual(len(frames), 1)
        self.assertEqual(connection._bufferedControlFrameBytes, 0)
        for _ in range(0, 11):
            connection.dataReceived(frameFactory.buildSettingsFrame({}).serialize())
        frames = framesFromBytes(transport.value())
        self.assertEqual(len(frames), 1)
        self.assertEqual(connection._bufferedControlFrameBytes, 9 * 11)
        connection.resumeProducing()
        frames = framesFromBytes(transport.value())
        self.assertEqual(len(frames), 2)
        self.assertEqual(connection._bufferedControlFrameBytes, 9 * 10)

    def test_circuitBreakerAbortsAfterProtocolError(self):
        if False:
            return 10
        "\n        A client that triggers a L{h2.exceptions.ProtocolError} over a\n        paused connection that's reached its buffered control frame\n        limit causes that connection to be aborted.\n        "
        memoryReactor = MemoryReactorClock()
        connection = H2Connection(memoryReactor)
        connection.callLater = memoryReactor.callLater
        frameFactory = FrameFactory()
        transport = StringTransport()
        clientConnectionPreface = frameFactory.clientConnectionPreface()
        connection.makeConnection(transport)
        connection.dataReceived(clientConnectionPreface)
        connection.pauseProducing()
        connection._maxBufferedControlFrameBytes = 0
        invalidData = frameFactory.buildDataFrame(data=b'yo', streamID=240).serialize()
        connection.dataReceived(invalidData)
        self.assertTrue(transport.disconnected)

class H2FlowControlTests(unittest.TestCase, HTTP2TestHelpers):
    """
    Tests that ensure that we handle HTTP/2 flow control limits appropriately.
    """
    getRequestHeaders = [(b':method', b'GET'), (b':authority', b'localhost'), (b':path', b'/'), (b':scheme', b'https'), (b'user-agent', b'twisted-test-code')]
    getResponseData = b"'''\nNone\n'''\n"
    postRequestHeaders = [(b':method', b'POST'), (b':authority', b'localhost'), (b':path', b'/post_endpoint'), (b':scheme', b'https'), (b'user-agent', b'twisted-test-code'), (b'content-length', b'25')]
    postRequestData = [b'hello ', b'world, ', b"it's ", b'http/2!']
    postResponseData = b"'''\n25\nhello world, it's http/2!'''\n"

    def test_bufferExcessData(self):
        if False:
            print('Hello World!')
        '\n        When a L{Request} object is not using C{IProducer} to generate data and\n        so is not having backpressure exerted on it, the L{H2Stream} object\n        will buffer data until the flow control window is opened.\n        '
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyHTTPHandlerProxy
        requestBytes = f.clientConnectionPreface()
        requestBytes += f.buildSettingsFrame({h2.settings.SettingCodes.INITIAL_WINDOW_SIZE: 5}).serialize()
        requestBytes += buildRequestBytes(self.getRequestHeaders, [], f)
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
        bonusFrames = len(self.getResponseData) - 5
        for _ in range(bonusFrames):
            frame = f.buildWindowUpdateFrame(streamID=1, increment=1)
            a.dataReceived(frame.serialize())

        def validate(streamID):
            if False:
                i = 10
                return i + 15
            frames = framesFromBytes(b.value())
            self.assertTrue('END_STREAM' in frames[-1].flags)
            actualResponseData = b''.join((f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)))
            self.assertEqual(self.getResponseData, actualResponseData)
        return a._streamCleanupCallbacks[1].addCallback(validate)

    def test_producerBlockingUnblocking(self):
        if False:
            print('Hello World!')
        '\n        L{Request} objects that have registered producers get blocked and\n        unblocked according to HTTP/2 flow control.\n        '
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyProducerHandlerProxy
        requestBytes = f.clientConnectionPreface()
        requestBytes += f.buildSettingsFrame({h2.settings.SettingCodes.INITIAL_WINDOW_SIZE: 5}).serialize()
        requestBytes += buildRequestBytes(self.getRequestHeaders, [], f)
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
        stream = a.streams[1]
        request = stream._request.original
        self.assertTrue(stream._producerProducing)
        request.write(b'helloworld')
        self.assertFalse(stream._producerProducing)
        self.assertEqual(request.producer.events, ['pause'])
        a.dataReceived(f.buildWindowUpdateFrame(streamID=1, increment=5).serialize())
        self.assertFalse(stream._producerProducing)
        self.assertEqual(request.producer.events, ['pause'])
        a.dataReceived(f.buildWindowUpdateFrame(streamID=0, increment=5).serialize())
        self.assertFalse(stream._producerProducing)
        self.assertEqual(request.producer.events, ['pause'])
        a.dataReceived(f.buildWindowUpdateFrame(streamID=1, increment=5).serialize())
        self.assertTrue(stream._producerProducing)
        self.assertEqual(request.producer.events, ['pause', 'resume'])
        request.write(b'helloworld')
        self.assertFalse(stream._producerProducing)
        self.assertEqual(request.producer.events, ['pause', 'resume', 'pause'])
        a.dataReceived(f.buildWindowUpdateFrame(streamID=1, increment=50).serialize())
        self.assertTrue(stream._producerProducing)
        self.assertEqual(request.producer.events, ['pause', 'resume', 'pause', 'resume'])
        request.unregisterProducer()
        request.finish()

        def validate(streamID):
            if False:
                while True:
                    i = 10
            frames = framesFromBytes(b.value())
            self.assertTrue('END_STREAM' in frames[-1].flags)
            dataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
            self.assertEqual(dataChunks, [b'helloworld', b'helloworld', b''])
        return a._streamCleanupCallbacks[1].addCallback(validate)

    def test_flowControlExact(self):
        if False:
            print('Hello World!')
        '\n        Exactly filling the flow control window still blocks producers.\n        '
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyProducerHandlerProxy
        requestBytes = f.clientConnectionPreface()
        requestBytes += f.buildSettingsFrame({h2.settings.SettingCodes.INITIAL_WINDOW_SIZE: 5}).serialize()
        requestBytes += buildRequestBytes(self.getRequestHeaders, [], f)
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
        stream = a.streams[1]
        request = stream._request.original
        self.assertTrue(stream._producerProducing)
        request.write(b'helloworld')
        self.assertFalse(stream._producerProducing)
        self.assertEqual(request.producer.events, ['pause'])
        request.write(b'h')

        def window_open():
            if False:
                print('Hello World!')
            a.dataReceived(f.buildWindowUpdateFrame(streamID=1, increment=50).serialize())
            self.assertTrue(stream._producerProducing)
            self.assertEqual(request.producer.events, ['pause', 'resume'])
            request.unregisterProducer()
            request.finish()
        windowDefer = task.deferLater(reactor, 0, window_open)

        def validate(streamID):
            if False:
                return 10
            frames = framesFromBytes(b.value())
            self.assertTrue('END_STREAM' in frames[-1].flags)
            dataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
            self.assertEqual(dataChunks, [b'hello', b'world', b'h', b''])
        validateDefer = a._streamCleanupCallbacks[1].addCallback(validate)
        return defer.DeferredList([windowDefer, validateDefer])

    def test_endingBlockedStream(self):
        if False:
            print('Hello World!')
        '\n        L{Request} objects that end a stream that is currently blocked behind\n        flow control can still end the stream and get cleaned up.\n        '
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyProducerHandlerProxy
        requestBytes = f.clientConnectionPreface()
        requestBytes += f.buildSettingsFrame({h2.settings.SettingCodes.INITIAL_WINDOW_SIZE: 5}).serialize()
        requestBytes += buildRequestBytes(self.getRequestHeaders, [], f)
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
        stream = a.streams[1]
        request = stream._request.original
        self.assertTrue(stream._producerProducing)
        request.write(b'helloworld')
        request.unregisterProducer()
        request.finish()
        self.assertTrue(request.finished)
        reactor.callLater(0, a.dataReceived, f.buildWindowUpdateFrame(streamID=1, increment=50).serialize())

        def validate(streamID):
            if False:
                for i in range(10):
                    print('nop')
            frames = framesFromBytes(b.value())
            self.assertTrue('END_STREAM' in frames[-1].flags)
            dataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
            self.assertEqual(dataChunks, [b'hello', b'world', b''])
        return a._streamCleanupCallbacks[1].addCallback(validate)

    def test_responseWithoutBody(self):
        if False:
            return 10
        '\n        We safely handle responses without bodies.\n        '
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyProducerHandlerProxy
        requestBytes = f.clientConnectionPreface()
        requestBytes += buildRequestBytes(self.getRequestHeaders, [], f)
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
        stream = a.streams[1]
        request = stream._request.original
        cleanupCallback = a._streamCleanupCallbacks[1]
        request.unregisterProducer()
        request.finish()
        self.assertTrue(request.finished)

        def validate(streamID):
            if False:
                print('Hello World!')
            frames = framesFromBytes(b.value())
            self.assertEqual(len(frames), 3)
            self.assertTrue('END_STREAM' in frames[-1].flags)
            dataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
            self.assertEqual(dataChunks, [b''])
        return cleanupCallback.addCallback(validate)

    def test_windowUpdateForCompleteStream(self):
        if False:
            while True:
                i = 10
        "\n        WindowUpdate frames received after we've completed the stream are\n        safely handled.\n        "
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyProducerHandlerProxy
        requestBytes = f.clientConnectionPreface()
        requestBytes += buildRequestBytes(self.getRequestHeaders, [], f)
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
        stream = a.streams[1]
        request = stream._request.original
        cleanupCallback = a._streamCleanupCallbacks[1]
        request.unregisterProducer()
        request.finish()
        self.assertTrue(request.finished)
        a.dataReceived(f.buildWindowUpdateFrame(streamID=1, increment=50).serialize())

        def validate(streamID):
            if False:
                while True:
                    i = 10
            frames = framesFromBytes(b.value())
            self.assertEqual(len(frames), 3)
            self.assertTrue('END_STREAM' in frames[-1].flags)
            dataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
            self.assertEqual(dataChunks, [b''])
        return cleanupCallback.addCallback(validate)

    def test_producerUnblocked(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{Request} objects that have registered producers that are not blocked\n        behind flow control do not have their producer notified.\n        '
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyProducerHandlerProxy
        requestBytes = f.clientConnectionPreface()
        requestBytes += f.buildSettingsFrame({h2.settings.SettingCodes.INITIAL_WINDOW_SIZE: 5}).serialize()
        requestBytes += buildRequestBytes(self.getRequestHeaders, [], f)
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
        stream = a.streams[1]
        request = stream._request.original
        self.assertTrue(stream._producerProducing)
        request.write(b'word')
        self.assertTrue(stream._producerProducing)
        self.assertEqual(request.producer.events, [])
        a.dataReceived(f.buildWindowUpdateFrame(streamID=1, increment=5).serialize())
        self.assertTrue(stream._producerProducing)
        self.assertEqual(request.producer.events, [])
        request.unregisterProducer()
        request.finish()

        def validate(streamID):
            if False:
                for i in range(10):
                    print('nop')
            frames = framesFromBytes(b.value())
            self.assertTrue('END_STREAM' in frames[-1].flags)
            dataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
            self.assertEqual(dataChunks, [b'word', b''])
        return a._streamCleanupCallbacks[1].addCallback(validate)

    def test_unnecessaryWindowUpdate(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        When a WindowUpdate frame is received for the whole connection but no\n        data is currently waiting, nothing exciting happens.\n        '
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyHTTPHandlerProxy
        frames = buildRequestFrames(self.postRequestHeaders, self.postRequestData, f)
        frames.insert(1, f.buildWindowUpdateFrame(streamID=0, increment=5))
        requestBytes = f.clientConnectionPreface()
        requestBytes += b''.join((f.serialize() for f in frames))
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)

        def validate(streamID):
            if False:
                while True:
                    i = 10
            frames = framesFromBytes(b.value())
            self.assertTrue('END_STREAM' in frames[-1].flags)
            actualResponseData = b''.join((f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)))
            self.assertEqual(self.postResponseData, actualResponseData)
        return a._streamCleanupCallbacks[1].addCallback(validate)

    def test_unnecessaryWindowUpdateForStream(self):
        if False:
            print('Hello World!')
        '\n        When a WindowUpdate frame is received for a stream but no data is\n        currently waiting, that stream is not marked as unblocked and the\n        priority tree continues to assert that no stream can progress.\n        '
        f = FrameFactory()
        transport = StringTransport()
        conn = H2Connection()
        conn.requestFactory = DummyHTTPHandlerProxy
        frames = []
        frames.append(f.buildHeadersFrame(headers=self.postRequestHeaders, streamID=1))
        frames.append(f.buildWindowUpdateFrame(streamID=1, increment=5))
        data = f.clientConnectionPreface()
        data += b''.join((f.serialize() for f in frames))
        conn.makeConnection(transport)
        conn.dataReceived(data)
        self.assertAllStreamsBlocked(conn)

    def test_windowUpdateAfterTerminate(self):
        if False:
            i = 10
            return i + 15
        '\n        When a WindowUpdate frame is received for a stream that has been\n        aborted it is ignored.\n        '
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyHTTPHandlerProxy
        frames = buildRequestFrames(self.postRequestHeaders, self.postRequestData, f)
        requestBytes = f.clientConnectionPreface()
        requestBytes += b''.join((f.serialize() for f in frames))
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
        a.streams[1].abortConnection()
        windowUpdateFrame = f.buildWindowUpdateFrame(streamID=1, increment=5)
        a.dataReceived(windowUpdateFrame.serialize())
        frames = framesFromBytes(b.value())
        self.assertTrue(isinstance(frames[-1], hyperframe.frame.RstStreamFrame))

    def test_windowUpdateAfterComplete(self):
        if False:
            i = 10
            return i + 15
        '\n        When a WindowUpdate frame is received for a stream that has been\n        completed it is ignored.\n        '
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyHTTPHandlerProxy
        frames = buildRequestFrames(self.postRequestHeaders, self.postRequestData, f)
        requestBytes = f.clientConnectionPreface()
        requestBytes += b''.join((f.serialize() for f in frames))
        a.makeConnection(b)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)

        def update_window(*args):
            if False:
                i = 10
                return i + 15
            windowUpdateFrame = f.buildWindowUpdateFrame(streamID=1, increment=5)
            a.dataReceived(windowUpdateFrame.serialize())

        def validate(*args):
            if False:
                return 10
            frames = framesFromBytes(b.value())
            self.assertIn('END_STREAM', frames[-1].flags)
        d = a._streamCleanupCallbacks[1].addCallback(update_window)
        return d.addCallback(validate)

    def test_dataAndRstStream(self):
        if False:
            while True:
                i = 10
        '\n        When a DATA frame is received at the same time as RST_STREAM,\n        Twisted does not send WINDOW_UPDATE frames for the stream.\n        '
        frameFactory = FrameFactory()
        transport = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyHTTPHandlerProxy
        frameData = [b'\x00' * 2 ** 14] * 4
        bodyLength = f'{sum((len(data) for data in frameData))}'
        headers = self.postRequestHeaders[:-1] + [('content-length', bodyLength)]
        frames = buildRequestFrames(headers=headers, data=frameData, frameFactory=frameFactory)
        del frames[-1]
        frames.append(frameFactory.buildRstStreamFrame(streamID=1, errorCode=h2.errors.ErrorCodes.INTERNAL_ERROR))
        requestBytes = frameFactory.clientConnectionPreface()
        requestBytes += b''.join((f.serialize() for f in frames))
        a.makeConnection(transport)
        a.dataReceived(requestBytes)
        frames = framesFromBytes(transport.value())
        windowUpdateFrameIDs = [f.stream_id for f in frames if isinstance(f, hyperframe.frame.WindowUpdateFrame)]
        self.assertEqual([0], windowUpdateFrameIDs)
        headersFrames = [f for f in frames if isinstance(f, hyperframe.frame.HeadersFrame)]
        dataFrames = [f for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
        self.assertFalse(headersFrames)
        self.assertFalse(dataFrames)

    def test_abortRequestWithCircuitBreaker(self):
        if False:
            print('Hello World!')
        "\n        Aborting a request associated with a paused connection that's\n        reached its buffered control frame limit causes that\n        connection to be aborted.\n        "
        memoryReactor = MemoryReactorClock()
        connection = H2Connection(memoryReactor)
        connection.callLater = memoryReactor.callLater
        connection.requestFactory = DummyHTTPHandlerProxy
        frameFactory = FrameFactory()
        transport = StringTransport()
        clientConnectionPreface = frameFactory.clientConnectionPreface()
        connection.makeConnection(transport)
        connection.dataReceived(clientConnectionPreface)
        streamID = 1
        headersFrameData = frameFactory.buildHeadersFrame(headers=self.postRequestHeaders, streamID=streamID).serialize()
        connection.dataReceived(headersFrameData)
        connection.pauseProducing()
        connection._maxBufferedControlFrameBytes = 0
        transport.clear()
        connection.abortRequest(streamID)
        self.assertFalse(transport.value())
        self.assertTrue(transport.disconnected)

class HTTP2TransportChecking(unittest.TestCase, HTTP2TestHelpers):
    getRequestHeaders = [(b':method', b'GET'), (b':authority', b'localhost'), (b':path', b'/'), (b':scheme', b'https'), (b'user-agent', b'twisted-test-code'), (b'custom-header', b'1'), (b'custom-header', b'2')]

    def test_registerProducerWithTransport(self):
        if False:
            print('Hello World!')
        '\n        L{H2Connection} can be registered with the transport as a producer.\n        '
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyHTTPHandlerProxy
        b.registerProducer(a, True)
        self.assertTrue(b.producer is a)

    def test_pausingProducerPreventsDataSend(self):
        if False:
            return 10
        '\n        L{H2Connection} can be paused by its consumer. When paused it stops\n        sending data to the transport.\n        '
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyHTTPHandlerProxy
        frames = buildRequestFrames(self.getRequestHeaders, [], f)
        requestBytes = f.clientConnectionPreface()
        requestBytes += b''.join((f.serialize() for f in frames))
        a.makeConnection(b)
        b.registerProducer(a, True)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
        a.pauseProducing()
        cleanupCallback = a._streamCleanupCallbacks[1]

        def validateNotSent(*args):
            if False:
                return 10
            frames = framesFromBytes(b.value())
            self.assertEqual(len(frames), 2)
            self.assertFalse(isinstance(frames[-1], hyperframe.frame.DataFrame))
            a.resumeProducing()
            a.resumeProducing()
            a.resumeProducing()
            a.resumeProducing()
            a.resumeProducing()
            return cleanupCallback

        def validateComplete(*args):
            if False:
                for i in range(10):
                    print('nop')
            frames = framesFromBytes(b.value())
            self.assertEqual(len(frames), 4)
            self.assertTrue('END_STREAM' in frames[-1].flags)
        d = task.deferLater(reactor, 0.01, validateNotSent)
        d.addCallback(validateComplete)
        return d

    def test_stopProducing(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{H2Connection} can be stopped by its producer. That causes it to lose\n        its transport.\n        '
        f = FrameFactory()
        b = StringTransport()
        a = H2Connection()
        a.requestFactory = DummyHTTPHandlerProxy
        frames = buildRequestFrames(self.getRequestHeaders, [], f)
        requestBytes = f.clientConnectionPreface()
        requestBytes += b''.join((f.serialize() for f in frames))
        a.makeConnection(b)
        b.registerProducer(a, True)
        for byte in iterbytes(requestBytes):
            a.dataReceived(byte)
        a.stopProducing()
        frames = framesFromBytes(b.value())
        self.assertEqual(len(frames), 2)
        self.assertFalse(isinstance(frames[-1], hyperframe.frame.DataFrame))
        self.assertFalse(a._stillProducing)

    def test_passthroughHostAndPeer(self):
        if False:
            return 10
        '\n        A L{H2Stream} object correctly passes through host and peer information\n        from its L{H2Connection}.\n        '
        hostAddress = IPv4Address('TCP', '17.52.24.8', 443)
        peerAddress = IPv4Address('TCP', '17.188.0.12', 32008)
        frameFactory = FrameFactory()
        transport = StringTransport(hostAddress=hostAddress, peerAddress=peerAddress)
        connection = H2Connection()
        connection.requestFactory = DummyHTTPHandlerProxy
        connection.makeConnection(transport)
        frames = buildRequestFrames(self.getRequestHeaders, [], frameFactory)
        requestBytes = frameFactory.clientConnectionPreface()
        requestBytes += b''.join((frame.serialize() for frame in frames))
        for byte in iterbytes(requestBytes):
            connection.dataReceived(byte)
        stream = connection.streams[1]
        self.assertEqual(stream.getHost(), hostAddress)
        self.assertEqual(stream.getPeer(), peerAddress)
        cleanupCallback = connection._streamCleanupCallbacks[1]

        def validate(*args):
            if False:
                print('Hello World!')
            self.assertEqual(stream.getHost(), hostAddress)
            self.assertEqual(stream.getPeer(), peerAddress)
        return cleanupCallback.addCallback(validate)

class HTTP2SchedulingTests(unittest.TestCase, HTTP2TestHelpers):
    """
    The H2Connection object schedules certain events (mostly its data sending
    loop) using callbacks from the reactor. These tests validate that the calls
    are scheduled correctly.
    """

    def test_initiallySchedulesOneDataCall(self):
        if False:
            i = 10
            return i + 15
        '\n        When a H2Connection is established it schedules one call to be run as\n        soon as the reactor has time.\n        '
        reactor = task.Clock()
        a = H2Connection(reactor)
        calls = reactor.getDelayedCalls()
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertTrue(call.active())
        self.assertEqual(call.time, 0)
        self.assertEqual(call.func, a._sendPrioritisedData)
        self.assertEqual(call.args, ())
        self.assertEqual(call.kw, {})

class HTTP2TimeoutTests(unittest.TestCase, HTTP2TestHelpers):
    """
    The L{H2Connection} object times out idle connections.
    """
    getRequestHeaders = [(b':method', b'GET'), (b':authority', b'localhost'), (b':path', b'/'), (b':scheme', b'https'), (b'user-agent', b'twisted-test-code'), (b'custom-header', b'1'), (b'custom-header', b'2')]
    _DEFAULT = object()

    def patch_TimeoutMixin_clock(self, connection, reactor):
        if False:
            while True:
                i = 10
        '\n        Unfortunately, TimeoutMixin does not allow passing an explicit reactor\n        to test timeouts. For that reason, we need to monkeypatch the method\n        set up by the TimeoutMixin.\n\n        @param connection: The HTTP/2 connection object to patch.\n        @type connection: L{H2Connection}\n\n        @param reactor: The reactor whose callLater method we want.\n        @type reactor: An object implementing\n            L{twisted.internet.interfaces.IReactorTime}\n        '
        connection.callLater = reactor.callLater

    def initiateH2Connection(self, initialData, requestFactory):
        if False:
            i = 10
            return i + 15
        '\n        Performs test setup by building a HTTP/2 connection object, a transport\n        to back it, a reactor to run in, and sending in some initial data as\n        needed.\n\n        @param initialData: The initial HTTP/2 data to be fed into the\n            connection after setup.\n        @type initialData: L{bytes}\n\n        @param requestFactory: The L{Request} factory to use with the\n            connection.\n        '
        reactor = task.Clock()
        conn = H2Connection(reactor)
        conn.timeOut = 100
        self.patch_TimeoutMixin_clock(conn, reactor)
        transport = StringTransport()
        conn.requestFactory = _makeRequestProxyFactory(requestFactory)
        conn.makeConnection(transport)
        for byte in iterbytes(initialData):
            conn.dataReceived(byte)
        return (reactor, conn, transport)

    def assertTimedOut(self, data, frameCount, errorCode, lastStreamID):
        if False:
            print('Hello World!')
        '\n        Confirm that the data that was sent matches what we expect from a\n        timeout: namely, that it ends with a GOAWAY frame carrying an\n        appropriate error code and last stream ID.\n        '
        frames = framesFromBytes(data)
        self.assertEqual(len(frames), frameCount)
        self.assertTrue(isinstance(frames[-1], hyperframe.frame.GoAwayFrame))
        self.assertEqual(frames[-1].error_code, errorCode)
        self.assertEqual(frames[-1].last_stream_id, lastStreamID)

    def prepareAbortTest(self, abortTimeout=_DEFAULT):
        if False:
            print('Hello World!')
        '\n        Does the common setup for tests that want to test the aborting\n        functionality of the HTTP/2 stack.\n\n        @param abortTimeout: The value to use for the abortTimeout. Defaults to\n            whatever is set on L{H2Connection.abortTimeout}.\n        @type abortTimeout: L{int} or L{None}\n\n        @return: A tuple of the reactor being used for the connection, the\n            connection itself, and the transport.\n        '
        if abortTimeout is self._DEFAULT:
            abortTimeout = H2Connection.abortTimeout
        frameFactory = FrameFactory()
        initialData = frameFactory.clientConnectionPreface()
        (reactor, conn, transport) = self.initiateH2Connection(initialData, requestFactory=DummyHTTPHandler)
        conn.abortTimeout = abortTimeout
        reactor.advance(100)
        self.assertTimedOut(transport.value(), frameCount=2, errorCode=h2.errors.ErrorCodes.NO_ERROR, lastStreamID=0)
        self.assertTrue(transport.disconnecting)
        self.assertFalse(transport.disconnected)
        return (reactor, conn, transport)

    def test_timeoutAfterInactivity(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        When a L{H2Connection} does not receive any data for more than the\n        time out interval, it closes the connection cleanly.\n        '
        frameFactory = FrameFactory()
        initialData = frameFactory.clientConnectionPreface()
        (reactor, conn, transport) = self.initiateH2Connection(initialData, requestFactory=DummyHTTPHandler)
        preamble = transport.value()
        reactor.advance(99)
        self.assertEqual(preamble, transport.value())
        self.assertFalse(transport.disconnecting)
        reactor.advance(2)
        self.assertTimedOut(transport.value(), frameCount=2, errorCode=h2.errors.ErrorCodes.NO_ERROR, lastStreamID=0)
        self.assertTrue(transport.disconnecting)

    def test_timeoutResetByRequestData(self):
        if False:
            print('Hello World!')
        '\n        When a L{H2Connection} receives data, the timeout is reset.\n        '
        frameFactory = FrameFactory()
        initialData = b''
        (reactor, conn, transport) = self.initiateH2Connection(initialData, requestFactory=DummyHTTPHandler)
        for byte in iterbytes(frameFactory.clientConnectionPreface()):
            conn.dataReceived(byte)
            reactor.advance(99)
            self.assertFalse(transport.disconnecting)
        reactor.advance(2)
        self.assertTimedOut(transport.value(), frameCount=2, errorCode=h2.errors.ErrorCodes.NO_ERROR, lastStreamID=0)
        self.assertTrue(transport.disconnecting)

    def test_timeoutResetByResponseData(self):
        if False:
            while True:
                i = 10
        '\n        When a L{H2Connection} sends data, the timeout is reset.\n        '
        frameFactory = FrameFactory()
        initialData = b''
        requests = []
        frames = buildRequestFrames(self.getRequestHeaders, [], frameFactory)
        initialData = frameFactory.clientConnectionPreface()
        initialData += b''.join((f.serialize() for f in frames))

        def saveRequest(stream, queued):
            if False:
                i = 10
                return i + 15
            req = DelayedHTTPHandler(stream, queued=queued)
            requests.append(req)
            return req
        (reactor, conn, transport) = self.initiateH2Connection(initialData, requestFactory=saveRequest)
        conn.dataReceived(frameFactory.clientConnectionPreface())
        reactor.advance(99)
        self.assertEquals(len(requests), 1)
        for x in range(10):
            requests[0].write(b'some bytes')
            reactor.advance(99)
            self.assertFalse(transport.disconnecting)
        reactor.advance(2)
        self.assertTimedOut(transport.value(), frameCount=13, errorCode=h2.errors.ErrorCodes.PROTOCOL_ERROR, lastStreamID=1)

    def test_timeoutWithProtocolErrorIfStreamsOpen(self):
        if False:
            i = 10
            return i + 15
        '\n        When a L{H2Connection} times out with active streams, the error code\n        returned is L{h2.errors.ErrorCodes.PROTOCOL_ERROR}.\n        '
        frameFactory = FrameFactory()
        frames = buildRequestFrames(self.getRequestHeaders, [], frameFactory)
        initialData = frameFactory.clientConnectionPreface()
        initialData += b''.join((f.serialize() for f in frames))
        (reactor, conn, transport) = self.initiateH2Connection(initialData, requestFactory=DummyProducerHandler)
        reactor.advance(101)
        self.assertTimedOut(transport.value(), frameCount=2, errorCode=h2.errors.ErrorCodes.PROTOCOL_ERROR, lastStreamID=1)
        self.assertTrue(transport.disconnecting)

    def test_noTimeoutIfConnectionLost(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        When a L{H2Connection} loses its connection it cancels its timeout.\n        '
        frameFactory = FrameFactory()
        frames = buildRequestFrames(self.getRequestHeaders, [], frameFactory)
        initialData = frameFactory.clientConnectionPreface()
        initialData += b''.join((f.serialize() for f in frames))
        (reactor, conn, transport) = self.initiateH2Connection(initialData, requestFactory=DummyProducerHandler)
        sentData = transport.value()
        oldCallCount = len(reactor.getDelayedCalls())
        conn.connectionLost('reason')
        currentCallCount = len(reactor.getDelayedCalls())
        self.assertEqual(oldCallCount - 1, currentCallCount)
        reactor.advance(101)
        self.assertEqual(transport.value(), sentData)

    def test_timeoutEventuallyForcesConnectionClosed(self):
        if False:
            return 10
        "\n        When a L{H2Connection} has timed the connection out, and the transport\n        doesn't get torn down within 15 seconds, it gets forcibly closed.\n        "
        (reactor, conn, transport) = self.prepareAbortTest()
        reactor.advance(14)
        self.assertTrue(transport.disconnecting)
        self.assertFalse(transport.disconnected)
        reactor.advance(1)
        self.assertTrue(transport.disconnecting)
        self.assertTrue(transport.disconnected)

    def test_losingConnectionCancelsTheAbort(self):
        if False:
            print('Hello World!')
        '\n        When a L{H2Connection} has timed the connection out, getting\n        C{connectionLost} called on it cancels the forcible connection close.\n        '
        (reactor, conn, transport) = self.prepareAbortTest()
        reactor.advance(14)
        conn.connectionLost(None)
        reactor.advance(1)
        self.assertTrue(transport.disconnecting)
        self.assertFalse(transport.disconnected)

    def test_losingConnectionWithNoAbortTimeOut(self):
        if False:
            i = 10
            return i + 15
        '\n        When a L{H2Connection} has timed the connection out but the\n        C{abortTimeout} is set to L{None}, the connection is never aborted.\n        '
        (reactor, conn, transport) = self.prepareAbortTest(abortTimeout=None)
        reactor.advance(2 ** 32)
        self.assertTrue(transport.disconnecting)
        self.assertFalse(transport.disconnected)

    def test_connectionLostAfterForceClose(self):
        if False:
            i = 10
            return i + 15
        "\n        If a timed out transport doesn't close after 15 seconds, the\n        L{HTTPChannel} will forcibly close it.\n        "
        (reactor, conn, transport) = self.prepareAbortTest()
        reactor.advance(15)
        self.assertTrue(transport.disconnecting)
        self.assertTrue(transport.disconnected)
        conn.connectionLost(error.ConnectionDone)

    def test_timeOutClientThatSendsOnlyInvalidFrames(self):
        if False:
            while True:
                i = 10
        '\n        A client that sends only invalid frames is eventually timed out.\n        '
        memoryReactor = MemoryReactorClock()
        connection = H2Connection(memoryReactor)
        connection.callLater = memoryReactor.callLater
        connection.timeOut = 60
        frameFactory = FrameFactory()
        transport = StringTransport()
        clientConnectionPreface = frameFactory.clientConnectionPreface()
        connection.makeConnection(transport)
        connection.dataReceived(clientConnectionPreface)
        for _ in range(connection.timeOut + connection.abortTimeout):
            connection.dataReceived(frameFactory.buildRstStreamFrame(1).serialize())
            memoryReactor.advance(1)
        self.assertTrue(transport.disconnected)