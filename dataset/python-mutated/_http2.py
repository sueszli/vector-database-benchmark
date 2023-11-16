"""
HTTP2 Implementation

This is the basic server-side protocol implementation used by the Twisted
Web server for HTTP2.  This functionality is intended to be combined with the
HTTP/1.1 and HTTP/1.0 functionality in twisted.web.http to provide complete
protocol support for HTTP-type protocols.

This API is currently considered private because it's in early draft form. When
it has stabilised, it'll be made public.
"""
import io
from collections import deque
from typing import List
from zope.interface import implementer
import h2.config
import h2.connection
import h2.errors
import h2.events
import h2.exceptions
import priority
from twisted.internet._producer_helpers import _PullToPush
from twisted.internet.defer import Deferred
from twisted.internet.error import ConnectionLost
from twisted.internet.interfaces import IConsumer, IProtocol, IPushProducer, ISSLTransport, ITransport
from twisted.internet.protocol import Protocol
from twisted.logger import Logger
from twisted.protocols.policies import TimeoutMixin
from twisted.python.failure import Failure
from twisted.web.error import ExcessiveBufferingError
__all__: List[str] = []
_END_STREAM_SENTINEL = object()

@implementer(IProtocol, IPushProducer)
class H2Connection(Protocol, TimeoutMixin):
    """
    A class representing a single HTTP/2 connection.

    This implementation of L{IProtocol} works hand in hand with L{H2Stream}.
    This is because we have the requirement to register multiple producers for
    a single HTTP/2 connection, one for each stream. The standard Twisted
    interfaces don't really allow for this, so instead there's a custom
    interface between the two objects that allows them to work hand-in-hand here.

    @ivar conn: The HTTP/2 connection state machine.
    @type conn: L{h2.connection.H2Connection}

    @ivar streams: A mapping of stream IDs to L{H2Stream} objects, used to call
        specific methods on streams when events occur.
    @type streams: L{dict}, mapping L{int} stream IDs to L{H2Stream} objects.

    @ivar priority: A HTTP/2 priority tree used to ensure that responses are
        prioritised appropriately.
    @type priority: L{priority.PriorityTree}

    @ivar _consumerBlocked: A flag tracking whether or not the L{IConsumer}
        that is consuming this data has asked us to stop producing.
    @type _consumerBlocked: L{bool}

    @ivar _sendingDeferred: A L{Deferred} used to restart the data-sending loop
        when more response data has been produced. Will not be present if there
        is outstanding data still to send.
    @type _consumerBlocked: A L{twisted.internet.defer.Deferred}, or L{None}

    @ivar _outboundStreamQueues: A map of stream IDs to queues, used to store
        data blocks that are yet to be sent on the connection. These are used
        both to handle producers that do not respect L{IConsumer} but also to
        allow priority to multiplex data appropriately.
    @type _outboundStreamQueues: A L{dict} mapping L{int} stream IDs to
        L{collections.deque} queues, which contain either L{bytes} objects or
        C{_END_STREAM_SENTINEL}.

    @ivar _sender: A handle to the data-sending loop, allowing it to be
        terminated if needed.
    @type _sender: L{twisted.internet.task.LoopingCall}

    @ivar abortTimeout: The number of seconds to wait after we attempt to shut
        the transport down cleanly to give up and forcibly terminate it. This
        is only used when we time a connection out, to prevent errors causing
        the FD to get leaked. If this is L{None}, we will wait forever.
    @type abortTimeout: L{int}

    @ivar _abortingCall: The L{twisted.internet.base.DelayedCall} that will be
        used to forcibly close the transport if it doesn't close cleanly.
    @type _abortingCall: L{twisted.internet.base.DelayedCall}
    """
    factory = None
    site = None
    abortTimeout = 15
    _log = Logger()
    _abortingCall = None

    def __init__(self, reactor=None):
        if False:
            for i in range(10):
                print('nop')
        config = h2.config.H2Configuration(client_side=False, header_encoding=None)
        self.conn = h2.connection.H2Connection(config=config)
        self.streams = {}
        self.priority = priority.PriorityTree()
        self._consumerBlocked = None
        self._sendingDeferred = None
        self._outboundStreamQueues = {}
        self._streamCleanupCallbacks = {}
        self._stillProducing = True
        self._maxBufferedControlFrameBytes = 1024 * 17
        self._bufferedControlFrames = deque()
        self._bufferedControlFrameBytes = 0
        if reactor is None:
            from twisted.internet import reactor
        self._reactor = reactor
        self._reactor.callLater(0, self._sendPrioritisedData)

    def connectionMade(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called by the reactor when a connection is received. May also be called\n        by the L{twisted.web.http._GenericHTTPChannelProtocol} during upgrade\n        to HTTP/2.\n        '
        self.setTimeout(self.timeOut)
        self.conn.initiate_connection()
        self.transport.write(self.conn.data_to_send())

    def dataReceived(self, data):
        if False:
            i = 10
            return i + 15
        '\n        Called whenever a chunk of data is received from the transport.\n\n        @param data: The data received from the transport.\n        @type data: L{bytes}\n        '
        try:
            events = self.conn.receive_data(data)
        except h2.exceptions.ProtocolError:
            stillActive = self._tryToWriteControlData()
            if stillActive:
                self.transport.loseConnection()
                self.connectionLost(Failure(), _cancelTimeouts=False)
            return
        self.resetTimeout()
        for event in events:
            if isinstance(event, h2.events.RequestReceived):
                self._requestReceived(event)
            elif isinstance(event, h2.events.DataReceived):
                self._requestDataReceived(event)
            elif isinstance(event, h2.events.StreamEnded):
                self._requestEnded(event)
            elif isinstance(event, h2.events.StreamReset):
                self._requestAborted(event)
            elif isinstance(event, h2.events.WindowUpdated):
                self._handleWindowUpdate(event)
            elif isinstance(event, h2.events.PriorityUpdated):
                self._handlePriorityUpdate(event)
            elif isinstance(event, h2.events.ConnectionTerminated):
                self.transport.loseConnection()
                self.connectionLost(Failure(ConnectionLost('Remote peer sent GOAWAY')), _cancelTimeouts=False)
        self._tryToWriteControlData()

    def timeoutConnection(self):
        if False:
            return 10
        '\n        Called when the connection has been inactive for\n        L{self.timeOut<twisted.protocols.policies.TimeoutMixin.timeOut>}\n        seconds. Cleanly tears the connection down, attempting to notify the\n        peer if needed.\n\n        We override this method to add two extra bits of functionality:\n\n         - We want to log the timeout.\n         - We want to send a GOAWAY frame indicating that the connection is\n           being terminated, and whether it was clean or not. We have to do this\n           before the connection is torn down.\n        '
        self._log.info('Timing out client {client}', client=self.transport.getPeer())
        if self.conn.open_outbound_streams > 0 or self.conn.open_inbound_streams > 0:
            error_code = h2.errors.ErrorCodes.PROTOCOL_ERROR
        else:
            error_code = h2.errors.ErrorCodes.NO_ERROR
        self.conn.close_connection(error_code=error_code)
        self.transport.write(self.conn.data_to_send())
        if self.abortTimeout is not None:
            self._abortingCall = self.callLater(self.abortTimeout, self.forceAbortClient)
        self.transport.loseConnection()

    def forceAbortClient(self):
        if False:
            print('Hello World!')
        "\n        Called if C{abortTimeout} seconds have passed since the timeout fired,\n        and the connection still hasn't gone away. This can really only happen\n        on extremely bad connections or when clients are maliciously attempting\n        to keep connections open.\n        "
        self._log.info('Forcibly timing out client: {client}', client=self.transport.getPeer())
        self._abortingCall = None
        self.transport.abortConnection()

    def connectionLost(self, reason, _cancelTimeouts=True):
        if False:
            print('Hello World!')
        "\n        Called when the transport connection is lost.\n\n        Informs all outstanding response handlers that the connection\n        has been lost, and cleans up all internal state.\n\n        @param reason: See L{IProtocol.connectionLost}\n\n        @param _cancelTimeouts: Propagate the C{reason} to this\n            connection's streams but don't cancel any timers, so that\n            peers who never read the data we've written are eventually\n            timed out.\n        "
        self._stillProducing = False
        if _cancelTimeouts:
            self.setTimeout(None)
        for stream in self.streams.values():
            stream.connectionLost(reason)
        for streamID in list(self.streams.keys()):
            self._requestDone(streamID)
        if _cancelTimeouts and self._abortingCall is not None:
            self._abortingCall.cancel()
            self._abortingCall = None

    def stopProducing(self):
        if False:
            print('Hello World!')
        '\n        Stop producing data.\n\n        This tells the L{H2Connection} that its consumer has died, so it must\n        stop producing data for good.\n        '
        self.connectionLost(Failure(ConnectionLost('Producing stopped')))

    def pauseProducing(self):
        if False:
            while True:
                i = 10
        '\n        Pause producing data.\n\n        Tells the L{H2Connection} that it has produced too much data to process\n        for the time being, and to stop until resumeProducing() is called.\n        '
        self._consumerBlocked = Deferred()
        self._consumerBlocked.addCallback(self._flushBufferedControlData)

    def resumeProducing(self):
        if False:
            while True:
                i = 10
        '\n        Resume producing data.\n\n        This tells the L{H2Connection} to re-add itself to the main loop and\n        produce more data for the consumer.\n        '
        if self._consumerBlocked is not None:
            d = self._consumerBlocked
            self._consumerBlocked = None
            d.callback(None)

    def _sendPrioritisedData(self, *args):
        if False:
            i = 10
            return i + 15
        '\n        The data sending loop. This function repeatedly calls itself, either\n        from L{Deferred}s or from\n        L{reactor.callLater<twisted.internet.interfaces.IReactorTime.callLater>}\n\n        This function sends data on streams according to the rules of HTTP/2\n        priority. It ensures that the data from each stream is interleved\n        according to the priority signalled by the client, making sure that the\n        connection is used with maximal efficiency.\n\n        This function will execute if data is available: if all data is\n        exhausted, the function will place a deferred onto the L{H2Connection}\n        object and wait until it is called to resume executing.\n        '
        if not self._stillProducing:
            return
        stream = None
        while stream is None:
            try:
                stream = next(self.priority)
            except priority.DeadlockError:
                assert self._sendingDeferred is None
                self._sendingDeferred = Deferred()
                self._sendingDeferred.addCallback(self._sendPrioritisedData)
                return
        if self._consumerBlocked is not None:
            self._consumerBlocked.addCallback(self._sendPrioritisedData)
            return
        self.resetTimeout()
        remainingWindow = self.conn.local_flow_control_window(stream)
        frameData = self._outboundStreamQueues[stream].popleft()
        maxFrameSize = min(self.conn.max_outbound_frame_size, remainingWindow)
        if frameData is _END_STREAM_SENTINEL:
            self.conn.end_stream(stream)
            self.transport.write(self.conn.data_to_send())
            self._requestDone(stream)
        else:
            if len(frameData) > maxFrameSize:
                excessData = frameData[maxFrameSize:]
                frameData = frameData[:maxFrameSize]
                self._outboundStreamQueues[stream].appendleft(excessData)
            if frameData:
                self.conn.send_data(stream, frameData)
                self.transport.write(self.conn.data_to_send())
            if not self._outboundStreamQueues[stream]:
                self.priority.block(stream)
            if self.remainingOutboundWindow(stream) <= 0:
                self.streams[stream].flowControlBlocked()
        self._reactor.callLater(0, self._sendPrioritisedData)

    def _requestReceived(self, event):
        if False:
            for i in range(10):
                print('nop')
        '\n        Internal handler for when a request has been received.\n\n        @param event: The Hyper-h2 event that encodes information about the\n            received request.\n        @type event: L{h2.events.RequestReceived}\n        '
        stream = H2Stream(event.stream_id, self, event.headers, self.requestFactory, self.site, self.factory)
        self.streams[event.stream_id] = stream
        self._streamCleanupCallbacks[event.stream_id] = Deferred()
        self._outboundStreamQueues[event.stream_id] = deque()
        try:
            self.priority.insert_stream(event.stream_id)
        except priority.DuplicateStreamError:
            pass
        else:
            self.priority.block(event.stream_id)

    def _requestDataReceived(self, event):
        if False:
            i = 10
            return i + 15
        '\n        Internal handler for when a chunk of data is received for a given\n        request.\n\n        @param event: The Hyper-h2 event that encodes information about the\n            received data.\n        @type event: L{h2.events.DataReceived}\n        '
        stream = self.streams[event.stream_id]
        stream.receiveDataChunk(event.data, event.flow_controlled_length)

    def _requestEnded(self, event):
        if False:
            i = 10
            return i + 15
        '\n        Internal handler for when a request is complete, and we expect no\n        further data for that request.\n\n        @param event: The Hyper-h2 event that encodes information about the\n            completed stream.\n        @type event: L{h2.events.StreamEnded}\n        '
        stream = self.streams[event.stream_id]
        stream.requestComplete()

    def _requestAborted(self, event):
        if False:
            for i in range(10):
                print('nop')
        '\n        Internal handler for when a request is aborted by a remote peer.\n\n        @param event: The Hyper-h2 event that encodes information about the\n            reset stream.\n        @type event: L{h2.events.StreamReset}\n        '
        stream = self.streams[event.stream_id]
        stream.connectionLost(Failure(ConnectionLost('Stream reset with code %s' % event.error_code)))
        self._requestDone(event.stream_id)

    def _handlePriorityUpdate(self, event):
        if False:
            print('Hello World!')
        '\n        Internal handler for when a stream priority is updated.\n\n        @param event: The Hyper-h2 event that encodes information about the\n            stream reprioritization.\n        @type event: L{h2.events.PriorityUpdated}\n        '
        try:
            self.priority.reprioritize(stream_id=event.stream_id, depends_on=event.depends_on or None, weight=event.weight, exclusive=event.exclusive)
        except priority.MissingStreamError:
            self.priority.insert_stream(stream_id=event.stream_id, depends_on=event.depends_on or None, weight=event.weight, exclusive=event.exclusive)
            self.priority.block(event.stream_id)

    def writeHeaders(self, version, code, reason, headers, streamID):
        if False:
            print('Hello World!')
        '\n        Called by L{twisted.web.http.Request} objects to write a complete set\n        of HTTP headers to a stream.\n\n        @param version: The HTTP version in use. Unused in HTTP/2.\n        @type version: L{bytes}\n\n        @param code: The HTTP status code to write.\n        @type code: L{bytes}\n\n        @param reason: The HTTP reason phrase to write. Unused in HTTP/2.\n        @type reason: L{bytes}\n\n        @param headers: The headers to write to the stream.\n        @type headers: L{twisted.web.http_headers.Headers}\n\n        @param streamID: The ID of the stream to write the headers to.\n        @type streamID: L{int}\n        '
        headers.insert(0, (b':status', code))
        try:
            self.conn.send_headers(streamID, headers)
        except h2.exceptions.StreamClosedError:
            return
        else:
            self._tryToWriteControlData()

    def writeDataToStream(self, streamID, data):
        if False:
            for i in range(10):
                print('nop')
        '\n        May be called by L{H2Stream} objects to write response data to a given\n        stream. Writes a single data frame.\n\n        @param streamID: The ID of the stream to write the data to.\n        @type streamID: L{int}\n\n        @param data: The data chunk to write to the stream.\n        @type data: L{bytes}\n        '
        self._outboundStreamQueues[streamID].append(data)
        if self.conn.local_flow_control_window(streamID) > 0:
            self.priority.unblock(streamID)
            if self._sendingDeferred is not None:
                d = self._sendingDeferred
                self._sendingDeferred = None
                d.callback(streamID)
        if self.remainingOutboundWindow(streamID) <= 0:
            self.streams[streamID].flowControlBlocked()

    def endRequest(self, streamID):
        if False:
            print('Hello World!')
        '\n        Called by L{H2Stream} objects to signal completion of a response.\n\n        @param streamID: The ID of the stream to write the data to.\n        @type streamID: L{int}\n        '
        self._outboundStreamQueues[streamID].append(_END_STREAM_SENTINEL)
        self.priority.unblock(streamID)
        if self._sendingDeferred is not None:
            d = self._sendingDeferred
            self._sendingDeferred = None
            d.callback(streamID)

    def abortRequest(self, streamID):
        if False:
            return 10
        '\n        Called by L{H2Stream} objects to request early termination of a stream.\n        This emits a RstStream frame and then removes all stream state.\n\n        @param streamID: The ID of the stream to write the data to.\n        @type streamID: L{int}\n        '
        self.conn.reset_stream(streamID)
        stillActive = self._tryToWriteControlData()
        if stillActive:
            self._requestDone(streamID)

    def _requestDone(self, streamID):
        if False:
            return 10
        '\n        Called internally by the data sending loop to clean up state that was\n        being used for the stream. Called when the stream is complete.\n\n        @param streamID: The ID of the stream to clean up state for.\n        @type streamID: L{int}\n        '
        del self._outboundStreamQueues[streamID]
        self.priority.remove_stream(streamID)
        del self.streams[streamID]
        cleanupCallback = self._streamCleanupCallbacks.pop(streamID)
        cleanupCallback.callback(streamID)

    def remainingOutboundWindow(self, streamID):
        if False:
            for i in range(10):
                print('nop')
        "\n        Called to determine how much room is left in the send window for a\n        given stream. Allows us to handle blocking and unblocking producers.\n\n        @param streamID: The ID of the stream whose flow control window we'll\n            check.\n        @type streamID: L{int}\n\n        @return: The amount of room remaining in the send window for the given\n            stream, including the data queued to be sent.\n        @rtype: L{int}\n        "
        windowSize = self.conn.local_flow_control_window(streamID)
        sendQueue = self._outboundStreamQueues[streamID]
        alreadyConsumed = sum((len(chunk) for chunk in sendQueue if chunk is not _END_STREAM_SENTINEL))
        return windowSize - alreadyConsumed

    def _handleWindowUpdate(self, event):
        if False:
            i = 10
            return i + 15
        '\n        Manage flow control windows.\n\n        Streams that are blocked on flow control will register themselves with\n        the connection. This will fire deferreds that wake those streams up and\n        allow them to continue processing.\n\n        @param event: The Hyper-h2 event that encodes information about the\n            flow control window change.\n        @type event: L{h2.events.WindowUpdated}\n        '
        streamID = event.stream_id
        if streamID:
            if not self._streamIsActive(streamID):
                return
            if self._outboundStreamQueues.get(streamID):
                self.priority.unblock(streamID)
            self.streams[streamID].windowUpdated()
        else:
            for stream in self.streams.values():
                stream.windowUpdated()
                if self._outboundStreamQueues.get(stream.streamID):
                    self.priority.unblock(stream.streamID)

    def getPeer(self):
        if False:
            print('Hello World!')
        '\n        Get the remote address of this connection.\n\n        Treat this method with caution.  It is the unfortunate result of the\n        CGI and Jabber standards, but should not be considered reliable for\n        the usual host of reasons; port forwarding, proxying, firewalls, IP\n        masquerading, etc.\n\n        @return: An L{IAddress} provider.\n        '
        return self.transport.getPeer()

    def getHost(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Similar to getPeer, but returns an address describing this side of the\n        connection.\n\n        @return: An L{IAddress} provider.\n        '
        return self.transport.getHost()

    def openStreamWindow(self, streamID, increment):
        if False:
            while True:
                i = 10
        '\n        Open the stream window by a given increment.\n\n        @param streamID: The ID of the stream whose window needs to be opened.\n        @type streamID: L{int}\n\n        @param increment: The amount by which the stream window must be\n        incremented.\n        @type increment: L{int}\n        '
        self.conn.acknowledge_received_data(increment, streamID)
        self._tryToWriteControlData()

    def _isSecure(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns L{True} if this channel is using a secure transport.\n\n        @returns: L{True} if this channel is secure.\n        @rtype: L{bool}\n        '
        return ISSLTransport(self.transport, None) is not None

    def _send100Continue(self, streamID):
        if False:
            while True:
                i = 10
        '\n        Sends a 100 Continue response, used to signal to clients that further\n        processing will be performed.\n\n        @param streamID: The ID of the stream that needs the 100 Continue\n        response\n        @type streamID: L{int}\n        '
        headers = [(b':status', b'100')]
        self.conn.send_headers(headers=headers, stream_id=streamID)
        self._tryToWriteControlData()

    def _respondToBadRequestAndDisconnect(self, streamID):
        if False:
            print('Hello World!')
        "\n        This is a quick and dirty way of responding to bad requests.\n\n        As described by HTTP standard we should be patient and accept the\n        whole request from the client before sending a polite bad request\n        response, even in the case when clients send tons of data.\n\n        Unlike in the HTTP/1.1 case, this does not actually disconnect the\n        underlying transport: there's no need. This instead just sends a 400\n        response and terminates the stream.\n\n        @param streamID: The ID of the stream that needs the 100 Continue\n        response\n        @type streamID: L{int}\n        "
        headers = [(b':status', b'400')]
        self.conn.send_headers(headers=headers, stream_id=streamID, end_stream=True)
        stillActive = self._tryToWriteControlData()
        if stillActive:
            stream = self.streams[streamID]
            stream.connectionLost(Failure(ConnectionLost('Invalid request')))
            self._requestDone(streamID)

    def _streamIsActive(self, streamID):
        if False:
            i = 10
            return i + 15
        '\n        Checks whether Twisted has still got state for a given stream and so\n        can process events for that stream.\n\n        @param streamID: The ID of the stream that needs processing.\n        @type streamID: L{int}\n\n        @return: Whether the stream still has state allocated.\n        @rtype: L{bool}\n        '
        return streamID in self.streams

    def _tryToWriteControlData(self):
        if False:
            i = 10
            return i + 15
        "\n        Checks whether the connection is blocked on flow control and,\n        if it isn't, writes any buffered control data.\n\n        @return: L{True} if the connection is still active and\n            L{False} if it was aborted because too many bytes have\n            been written but not consumed by the other end.\n        "
        bufferedBytes = self.conn.data_to_send()
        if not bufferedBytes:
            return True
        if self._consumerBlocked is None and (not self._bufferedControlFrames):
            self.transport.write(bufferedBytes)
            return True
        else:
            self._bufferedControlFrames.append(bufferedBytes)
            self._bufferedControlFrameBytes += len(bufferedBytes)
            if self._bufferedControlFrameBytes >= self._maxBufferedControlFrameBytes:
                maxBuffCtrlFrameBytes = self._maxBufferedControlFrameBytes
                self._log.error('Maximum number of control frame bytes buffered: {bufferedControlFrameBytes} > = {maxBufferedControlFrameBytes}. Aborting connection to client: {client} ', bufferedControlFrameBytes=self._bufferedControlFrameBytes, maxBufferedControlFrameBytes=maxBuffCtrlFrameBytes, client=self.transport.getPeer())
                self.transport.abortConnection()
                self.connectionLost(Failure(ExcessiveBufferingError()))
                return False
            return True

    def _flushBufferedControlData(self, *args):
        if False:
            i = 10
            return i + 15
        '\n        Called when the connection is marked writable again after being marked unwritable.\n        Attempts to flush buffered control data if there is any.\n        '
        while self._consumerBlocked is None and self._bufferedControlFrames:
            nextWrite = self._bufferedControlFrames.popleft()
            self._bufferedControlFrameBytes -= len(nextWrite)
            self.transport.write(nextWrite)

@implementer(ITransport, IConsumer, IPushProducer)
class H2Stream:
    """
    A class representing a single HTTP/2 stream.

    This class works hand-in-hand with L{H2Connection}. It acts to provide an
    implementation of L{ITransport}, L{IConsumer}, and L{IProducer} that work
    for a single HTTP/2 connection, while tightly cleaving to the interface
    provided by those interfaces. It does this by having a tight coupling to
    L{H2Connection}, which allows associating many of the functions of
    L{ITransport}, L{IConsumer}, and L{IProducer} to objects on a
    stream-specific level.

    @ivar streamID: The numerical stream ID that this object corresponds to.
    @type streamID: L{int}

    @ivar producing: Whether this stream is currently allowed to produce data
        to its consumer.
    @type producing: L{bool}

    @ivar command: The HTTP verb used on the request.
    @type command: L{unicode}

    @ivar path: The HTTP path used on the request.
    @type path: L{unicode}

    @ivar producer: The object producing the response, if any.
    @type producer: L{IProducer}

    @ivar site: The L{twisted.web.server.Site} object this stream belongs to,
        if any.
    @type site: L{twisted.web.server.Site}

    @ivar factory: The L{twisted.web.http.HTTPFactory} object that constructed
        this stream's parent connection.
    @type factory: L{twisted.web.http.HTTPFactory}

    @ivar _producerProducing: Whether the producer stored in producer is
        currently producing data.
    @type _producerProducing: L{bool}

    @ivar _inboundDataBuffer: Any data that has been received from the network
        but has not yet been received by the consumer.
    @type _inboundDataBuffer: A L{collections.deque} containing L{bytes}

    @ivar _conn: A reference to the connection this stream belongs to.
    @type _conn: L{H2Connection}

    @ivar _request: A request object that this stream corresponds to.
    @type _request: L{twisted.web.iweb.IRequest}

    @ivar _buffer: A buffer containing data produced by the producer that could
        not be sent on the network at this time.
    @type _buffer: L{io.BytesIO}
    """
    transport = None

    def __init__(self, streamID, connection, headers, requestFactory, site, factory):
        if False:
            while True:
                i = 10
        "\n        Initialize this HTTP/2 stream.\n\n        @param streamID: The numerical stream ID that this object corresponds\n            to.\n        @type streamID: L{int}\n\n        @param connection: The HTTP/2 connection this stream belongs to.\n        @type connection: L{H2Connection}\n\n        @param headers: The HTTP/2 request headers.\n        @type headers: A L{list} of L{tuple}s of header name and header value,\n            both as L{bytes}.\n\n        @param requestFactory: A function that builds appropriate request\n            request objects.\n        @type requestFactory: A callable that returns a\n            L{twisted.web.iweb.IRequest}.\n\n        @param site: The L{twisted.web.server.Site} object this stream belongs\n            to, if any.\n        @type site: L{twisted.web.server.Site}\n\n        @param factory: The L{twisted.web.http.HTTPFactory} object that\n            constructed this stream's parent connection.\n        @type factory: L{twisted.web.http.HTTPFactory}\n        "
        self.streamID = streamID
        self.site = site
        self.factory = factory
        self.producing = True
        self.command = None
        self.path = None
        self.producer = None
        self._producerProducing = False
        self._hasStreamingProducer = None
        self._inboundDataBuffer = deque()
        self._conn = connection
        self._request = requestFactory(self, queued=False)
        self._buffer = io.BytesIO()
        self._convertHeaders(headers)

    def _convertHeaders(self, headers):
        if False:
            print('Hello World!')
        "\n        This method converts the HTTP/2 header set into something that looks\n        like HTTP/1.1. In particular, it strips the 'special' headers and adds\n        a Host: header.\n\n        @param headers: The HTTP/2 header set.\n        @type headers: A L{list} of L{tuple}s of header name and header value,\n            both as L{bytes}.\n        "
        gotLength = False
        for header in headers:
            if not header[0].startswith(b':'):
                gotLength = _addHeaderToRequest(self._request, header) or gotLength
            elif header[0] == b':method':
                self.command = header[1]
            elif header[0] == b':path':
                self.path = header[1]
            elif header[0] == b':authority':
                _addHeaderToRequest(self._request, (b'host', header[1]))
        if not gotLength:
            if self.command in (b'GET', b'HEAD'):
                self._request.gotLength(0)
            else:
                self._request.gotLength(None)
        self._request.parseCookies()
        expectContinue = self._request.requestHeaders.getRawHeaders(b'expect')
        if expectContinue and expectContinue[0].lower() == b'100-continue':
            self._send100Continue()

    def receiveDataChunk(self, data, flowControlledLength):
        if False:
            print('Hello World!')
        '\n        Called when the connection has received a chunk of data from the\n        underlying transport. If the stream has been registered with a\n        consumer, and is currently able to push data, immediately passes it\n        through. Otherwise, buffers the chunk until we can start producing.\n\n        @param data: The chunk of data that was received.\n        @type data: L{bytes}\n\n        @param flowControlledLength: The total flow controlled length of this\n            chunk, which is used when we want to re-open the window. May be\n            different to C{len(data)}.\n        @type flowControlledLength: L{int}\n        '
        if not self.producing:
            self._inboundDataBuffer.append((data, flowControlledLength))
        else:
            self._request.handleContentChunk(data)
            self._conn.openStreamWindow(self.streamID, flowControlledLength)

    def requestComplete(self):
        if False:
            print('Hello World!')
        '\n        Called by the L{H2Connection} when the all data for a request has been\n        received. Currently, with the legacy L{twisted.web.http.Request}\n        object, just calls requestReceived unless the producer wants us to be\n        quiet.\n        '
        if self.producing:
            self._request.requestReceived(self.command, self.path, b'HTTP/2')
        else:
            self._inboundDataBuffer.append((_END_STREAM_SENTINEL, None))

    def connectionLost(self, reason):
        if False:
            i = 10
            return i + 15
        '\n        Called by the L{H2Connection} when a connection is lost or a stream is\n        reset.\n\n        @param reason: The reason the connection was lost.\n        @type reason: L{str}\n        '
        self._request.connectionLost(reason)

    def windowUpdated(self):
        if False:
            return 10
        "\n        Called by the L{H2Connection} when this stream's flow control window\n        has been opened.\n        "
        if not self.producer:
            return
        if self._producerProducing:
            return
        remainingWindow = self._conn.remainingOutboundWindow(self.streamID)
        if not remainingWindow > 0:
            return
        self._producerProducing = True
        self.producer.resumeProducing()

    def flowControlBlocked(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Called by the L{H2Connection} when this stream's flow control window\n        has been exhausted.\n        "
        if not self.producer:
            return
        if self._producerProducing:
            self.producer.pauseProducing()
            self._producerProducing = False

    def writeHeaders(self, version, code, reason, headers):
        if False:
            for i in range(10):
                print('nop')
        '\n        Called by the consumer to write headers to the stream.\n\n        @param version: The HTTP version.\n        @type version: L{bytes}\n\n        @param code: The status code.\n        @type code: L{int}\n\n        @param reason: The reason phrase. Ignored in HTTP/2.\n        @type reason: L{bytes}\n\n        @param headers: The HTTP response headers.\n        @type headers: Any iterable of two-tuples of L{bytes}, representing header\n            names and header values.\n        '
        self._conn.writeHeaders(version, code, reason, headers, self.streamID)

    def requestDone(self, request):
        if False:
            return 10
        '\n        Called by a consumer to clean up whatever permanent state is in use.\n\n        @param request: The request calling the method.\n        @type request: L{twisted.web.iweb.IRequest}\n        '
        self._conn.endRequest(self.streamID)

    def _send100Continue(self):
        if False:
            while True:
                i = 10
        '\n        Sends a 100 Continue response, used to signal to clients that further\n        processing will be performed.\n        '
        self._conn._send100Continue(self.streamID)

    def _respondToBadRequestAndDisconnect(self):
        if False:
            while True:
                i = 10
        "\n        This is a quick and dirty way of responding to bad requests.\n\n        As described by HTTP standard we should be patient and accept the\n        whole request from the client before sending a polite bad request\n        response, even in the case when clients send tons of data.\n\n        Unlike in the HTTP/1.1 case, this does not actually disconnect the\n        underlying transport: there's no need. This instead just sends a 400\n        response and terminates the stream.\n        "
        self._conn._respondToBadRequestAndDisconnect(self.streamID)

    def write(self, data):
        if False:
            while True:
                i = 10
        '\n        Write a single chunk of data into a data frame.\n\n        @param data: The data chunk to send.\n        @type data: L{bytes}\n        '
        self._conn.writeDataToStream(self.streamID, data)
        return

    def writeSequence(self, iovec):
        if False:
            i = 10
            return i + 15
        '\n        Write a sequence of chunks of data into data frames.\n\n        @param iovec: A sequence of chunks to send.\n        @type iovec: An iterable of L{bytes} chunks.\n        '
        for chunk in iovec:
            self.write(chunk)

    def loseConnection(self):
        if False:
            return 10
        '\n        Close the connection after writing all pending data.\n        '
        self._conn.endRequest(self.streamID)

    def abortConnection(self):
        if False:
            i = 10
            return i + 15
        '\n        Forcefully abort the connection by sending a RstStream frame.\n        '
        self._conn.abortRequest(self.streamID)

    def getPeer(self):
        if False:
            while True:
                i = 10
        '\n        Get information about the peer.\n        '
        return self._conn.getPeer()

    def getHost(self):
        if False:
            i = 10
            return i + 15
        '\n        Similar to getPeer, but for this side of the connection.\n        '
        return self._conn.getHost()

    def isSecure(self):
        if False:
            while True:
                i = 10
        '\n        Returns L{True} if this channel is using a secure transport.\n\n        @returns: L{True} if this channel is secure.\n        @rtype: L{bool}\n        '
        return self._conn._isSecure()

    def registerProducer(self, producer, streaming):
        if False:
            return 10
        '\n        Register to receive data from a producer.\n\n        This sets self to be a consumer for a producer.  When this object runs\n        out of data (as when a send(2) call on a socket succeeds in moving the\n        last data from a userspace buffer into a kernelspace buffer), it will\n        ask the producer to resumeProducing().\n\n        For L{IPullProducer} providers, C{resumeProducing} will be called once\n        each time data is required.\n\n        For L{IPushProducer} providers, C{pauseProducing} will be called\n        whenever the write buffer fills up and C{resumeProducing} will only be\n        called when it empties.\n\n        @param producer: The producer to register.\n        @type producer: L{IProducer} provider\n\n        @param streaming: L{True} if C{producer} provides L{IPushProducer},\n        L{False} if C{producer} provides L{IPullProducer}.\n        @type streaming: L{bool}\n\n        @raise RuntimeError: If a producer is already registered.\n\n        @return: L{None}\n        '
        if self.producer:
            raise ValueError('registering producer %s before previous one (%s) was unregistered' % (producer, self.producer))
        if not streaming:
            self.hasStreamingProducer = False
            producer = _PullToPush(producer, self)
            producer.startStreaming()
        else:
            self.hasStreamingProducer = True
        self.producer = producer
        self._producerProducing = True

    def unregisterProducer(self):
        if False:
            while True:
                i = 10
        '\n        @see: L{IConsumer.unregisterProducer}\n        '
        if self.producer is not None and (not self.hasStreamingProducer):
            self.producer.stopStreaming()
        self._producerProducing = False
        self.producer = None
        self.hasStreamingProducer = None

    def stopProducing(self):
        if False:
            return 10
        '\n        @see: L{IProducer.stopProducing}\n        '
        self.producing = False
        self.abortConnection()

    def pauseProducing(self):
        if False:
            i = 10
            return i + 15
        '\n        @see: L{IPushProducer.pauseProducing}\n        '
        self.producing = False

    def resumeProducing(self):
        if False:
            i = 10
            return i + 15
        '\n        @see: L{IPushProducer.resumeProducing}\n        '
        self.producing = True
        consumedLength = 0
        while self.producing and self._inboundDataBuffer:
            (chunk, flowControlledLength) = self._inboundDataBuffer.popleft()
            if chunk is _END_STREAM_SENTINEL:
                self.requestComplete()
            else:
                consumedLength += flowControlledLength
                self._request.handleContentChunk(chunk)
        self._conn.openStreamWindow(self.streamID, consumedLength)

def _addHeaderToRequest(request, header):
    if False:
        print('Hello World!')
    '\n    Add a header tuple to a request header object.\n\n    @param request: The request to add the header tuple to.\n    @type request: L{twisted.web.http.Request}\n\n    @param header: The header tuple to add to the request.\n    @type header: A L{tuple} with two elements, the header name and header\n        value, both as L{bytes}.\n\n    @return: If the header being added was the C{Content-Length} header.\n    @rtype: L{bool}\n    '
    requestHeaders = request.requestHeaders
    (name, value) = header
    values = requestHeaders.getRawHeaders(name)
    if values is not None:
        values.append(value)
    else:
        requestHeaders.setRawHeaders(name, [value])
    if name == b'content-length':
        request.gotLength(int(value))
        return True
    return False