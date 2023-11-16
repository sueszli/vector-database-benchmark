"""
Hierarchical Token Bucket traffic shaping.

Patterned after U{Martin Devera's Hierarchical Token Bucket traffic
shaper for the Linux kernel<http://luxik.cdi.cz/~devik/qos/htb/>}.

@seealso: U{HTB Linux queuing discipline manual - user guide
  <http://luxik.cdi.cz/~devik/qos/htb/manual/userg.htm>}
@seealso: U{Token Bucket Filter in Linux Advanced Routing & Traffic Control
    HOWTO<http://lartc.org/howto/lartc.qdisc.classless.html#AEN682>}
"""
from time import time
from typing import Optional
from zope.interface import Interface, implementer
from twisted.protocols import pcp

class Bucket:
    """
    Implementation of a Token bucket.

    A bucket can hold a certain number of tokens and it drains over time.

    @cvar maxburst: The maximum number of tokens that the bucket can
        hold at any given time. If this is L{None}, the bucket has
        an infinite size.
    @type maxburst: C{int}
    @cvar rate: The rate at which the bucket drains, in number
        of tokens per second. If the rate is L{None}, the bucket
        drains instantaneously.
    @type rate: C{int}
    """
    maxburst: Optional[int] = None
    rate: Optional[int] = None
    _refcount = 0

    def __init__(self, parentBucket=None):
        if False:
            i = 10
            return i + 15
        '\n        Create a L{Bucket} that may have a parent L{Bucket}.\n\n        @param parentBucket: If a parent Bucket is specified,\n            all L{add} and L{drip} operations on this L{Bucket}\n            will be applied on the parent L{Bucket} as well.\n        @type parentBucket: L{Bucket}\n        '
        self.content = 0
        self.parentBucket = parentBucket
        self.lastDrip = time()

    def add(self, amount):
        if False:
            i = 10
            return i + 15
        '\n        Adds tokens to the L{Bucket} and its C{parentBucket}.\n\n        This will add as many of the C{amount} tokens as will fit into both\n        this L{Bucket} and its C{parentBucket}.\n\n        @param amount: The number of tokens to try to add.\n        @type amount: C{int}\n\n        @returns: The number of tokens that actually fit.\n        @returntype: C{int}\n        '
        self.drip()
        if self.maxburst is None:
            allowable = amount
        else:
            allowable = min(amount, self.maxburst - self.content)
        if self.parentBucket is not None:
            allowable = self.parentBucket.add(allowable)
        self.content += allowable
        return allowable

    def drip(self):
        if False:
            while True:
                i = 10
        '\n        Let some of the bucket drain.\n\n        The L{Bucket} drains at the rate specified by the class\n        variable C{rate}.\n\n        @returns: C{True} if the bucket is empty after this drip.\n        @returntype: C{bool}\n        '
        if self.parentBucket is not None:
            self.parentBucket.drip()
        if self.rate is None:
            self.content = 0
        else:
            now = time()
            deltaTime = now - self.lastDrip
            deltaTokens = deltaTime * self.rate
            self.content = max(0, self.content - deltaTokens)
            self.lastDrip = now
        return self.content == 0

class IBucketFilter(Interface):

    def getBucketFor(*somethings, **some_kw):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a L{Bucket} corresponding to the provided parameters.\n\n        @returntype: L{Bucket}\n        '

@implementer(IBucketFilter)
class HierarchicalBucketFilter:
    """
    Filter things into buckets that can be nested.

    @cvar bucketFactory: Class of buckets to make.
    @type bucketFactory: L{Bucket}
    @cvar sweepInterval: Seconds between sweeping out the bucket cache.
    @type sweepInterval: C{int}
    """
    bucketFactory = Bucket
    sweepInterval: Optional[int] = None

    def __init__(self, parentFilter=None):
        if False:
            i = 10
            return i + 15
        self.buckets = {}
        self.parentFilter = parentFilter
        self.lastSweep = time()

    def getBucketFor(self, *a, **kw):
        if False:
            i = 10
            return i + 15
        '\n        Find or create a L{Bucket} corresponding to the provided parameters.\n\n        Any parameters are passed on to L{getBucketKey}, from them it\n        decides which bucket you get.\n\n        @returntype: L{Bucket}\n        '
        if self.sweepInterval is not None and time() - self.lastSweep > self.sweepInterval:
            self.sweep()
        if self.parentFilter:
            parentBucket = self.parentFilter.getBucketFor(self, *a, **kw)
        else:
            parentBucket = None
        key = self.getBucketKey(*a, **kw)
        bucket = self.buckets.get(key)
        if bucket is None:
            bucket = self.bucketFactory(parentBucket)
            self.buckets[key] = bucket
        return bucket

    def getBucketKey(self, *a, **kw):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a key based on the input parameters to choose a L{Bucket}.\n\n        The default implementation returns the same key for all\n        arguments. Override this method to provide L{Bucket} selection.\n\n        @returns: Something to be used as a key in the bucket cache.\n        '
        return None

    def sweep(self):
        if False:
            return 10
        '\n        Remove empty buckets.\n        '
        for (key, bucket) in self.buckets.items():
            bucket_is_empty = bucket.drip()
            if bucket._refcount == 0 and bucket_is_empty:
                del self.buckets[key]
        self.lastSweep = time()

class FilterByHost(HierarchicalBucketFilter):
    """
    A Hierarchical Bucket filter with a L{Bucket} for each host.
    """
    sweepInterval = 60 * 20

    def getBucketKey(self, transport):
        if False:
            for i in range(10):
                print('nop')
        return transport.getPeer()[1]

class FilterByServer(HierarchicalBucketFilter):
    """
    A Hierarchical Bucket filter with a L{Bucket} for each service.
    """
    sweepInterval = None

    def getBucketKey(self, transport):
        if False:
            while True:
                i = 10
        return transport.getHost()[2]

class ShapedConsumer(pcp.ProducerConsumerProxy):
    """
    Wraps a C{Consumer} and shapes the rate at which it receives data.
    """
    iAmStreaming = False

    def __init__(self, consumer, bucket):
        if False:
            for i in range(10):
                print('nop')
        pcp.ProducerConsumerProxy.__init__(self, consumer)
        self.bucket = bucket
        self.bucket._refcount += 1

    def _writeSomeData(self, data):
        if False:
            i = 10
            return i + 15
        amount = self.bucket.add(len(data))
        return pcp.ProducerConsumerProxy._writeSomeData(self, data[:amount])

    def stopProducing(self):
        if False:
            return 10
        pcp.ProducerConsumerProxy.stopProducing(self)
        self.bucket._refcount -= 1

class ShapedTransport(ShapedConsumer):
    """
    Wraps a C{Transport} and shapes the rate at which it receives data.

    This is a L{ShapedConsumer} with a little bit of magic to provide for
    the case where the consumer it wraps is also a C{Transport} and people
    will be attempting to access attributes this does not proxy as a
    C{Consumer} (e.g. C{loseConnection}).
    """
    iAmStreaming = False

    def __getattr__(self, name):
        if False:
            print('Hello World!')
        return getattr(self.consumer, name)

class ShapedProtocolFactory:
    """
    Dispense C{Protocols} with traffic shaping on their transports.

    Usage::

        myserver = SomeFactory()
        myserver.protocol = ShapedProtocolFactory(myserver.protocol,
                                                  bucketFilter)

    Where C{SomeServerFactory} is a L{twisted.internet.protocol.Factory}, and
    C{bucketFilter} is an instance of L{HierarchicalBucketFilter}.
    """

    def __init__(self, protoClass, bucketFilter):
        if False:
            return 10
        '\n        Tell me what to wrap and where to get buckets.\n\n        @param protoClass: The class of C{Protocol} this will generate\n          wrapped instances of.\n        @type protoClass: L{Protocol<twisted.internet.interfaces.IProtocol>}\n          class\n        @param bucketFilter: The filter which will determine how\n          traffic is shaped.\n        @type bucketFilter: L{HierarchicalBucketFilter}.\n        '
        self.protocol = protoClass
        self.bucketFilter = bucketFilter

    def __call__(self, *a, **kw):
        if False:
            i = 10
            return i + 15
        "\n        Make a C{Protocol} instance with a shaped transport.\n\n        Any parameters will be passed on to the protocol's initializer.\n\n        @returns: A C{Protocol} instance with a L{ShapedTransport}.\n        "
        proto = self.protocol(*a, **kw)
        origMakeConnection = proto.makeConnection

        def makeConnection(transport):
            if False:
                while True:
                    i = 10
            bucket = self.bucketFilter.getBucketFor(transport)
            shapedTransport = ShapedTransport(transport, bucket)
            return origMakeConnection(shapedTransport)
        proto.makeConnection = makeConnection
        return proto