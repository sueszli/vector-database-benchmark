from apache_beam.io.gcp.pubsublite.external import _ReadExternal
from apache_beam.io.gcp.pubsublite.external import _WriteExternal
from apache_beam.transforms import Map
from apache_beam.transforms import PTransform
try:
    from google.cloud import pubsublite
except ImportError:
    pubsublite = None

class ReadFromPubSubLite(PTransform):
    """
  A ``PTransform`` for reading from Pub/Sub Lite.

  Produces a PCollection of google.cloud.pubsublite.SequencedMessage

  Experimental; no backwards-compatibility guarantees.
  """

    def __init__(self, subscription_path, deduplicate=None, expansion_service=None):
        if False:
            for i in range(10):
                print('nop')
        "Initializes ``ReadFromPubSubLite``.\n\n    Args:\n      subscription_path: Pub/Sub Lite Subscription in the form\n          projects/<project>/locations/<location>/subscriptions/<subscription>\n      deduplicate: Whether to deduplicate messages based on the value of\n          the 'x-goog-pubsublite-dataflow-uuid' attribute. Defaults to False.\n    "
        super().__init__()
        self._source = _ReadExternal(subscription_path=subscription_path, deduplicate=deduplicate, expansion_service=expansion_service)

    def expand(self, pvalue):
        if False:
            print('Hello World!')
        pcoll = pvalue.pipeline | self._source
        pcoll.element_type = bytes
        pcoll = pcoll | Map(pubsublite.SequencedMessage.deserialize)
        pcoll.element_type = pubsublite.SequencedMessage
        return pcoll

class WriteToPubSubLite(PTransform):
    """
  A ``PTransform`` for writing to Pub/Sub Lite.

  Consumes a PCollection of google.cloud.pubsublite.PubSubMessage

  Experimental; no backwards-compatibility guarantees.
  """

    def __init__(self, topic_path, add_uuids=None, expansion_service=None):
        if False:
            i = 10
            return i + 15
        "Initializes ``WriteToPubSubLite``.\n\n    Args:\n      topic_path: A Pub/Sub Lite Topic path.\n      add_uuids: Whether to add uuids to the 'x-goog-pubsublite-dataflow-uuid'\n          uuid attribute. Defaults to False.\n    "
        super().__init__()
        self._source = _WriteExternal(topic_path=topic_path, add_uuids=add_uuids, expansion_service=expansion_service)

    @staticmethod
    def _message_to_proto_str(element: pubsublite.PubSubMessage):
        if False:
            print('Hello World!')
        if not isinstance(element, pubsublite.PubSubMessage):
            raise TypeError('Unexpected element. Type: %s (expected: PubSubMessage), value: %r' % (type(element), element))
        return pubsublite.PubSubMessage.serialize(element)

    def expand(self, pcoll):
        if False:
            i = 10
            return i + 15
        pcoll = pcoll | Map(WriteToPubSubLite._message_to_proto_str)
        pcoll.element_type = bytes
        pcoll = pcoll | self._source
        return pcoll