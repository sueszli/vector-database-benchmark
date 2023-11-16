"""TPU hardware feature info."""
import enum
from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.util.tf_export import tf_export

@tf_export('tpu.experimental.HardwareFeature')
class HardwareFeature(object):
    """class holds all the feature info about the TPU."""

    def __init__(self, tpu_hardware_feature_proto):
        if False:
            return 10
        'Store TPU hardware feature info.\n\n    Args:\n      tpu_hardware_feature_proto: protobuf which describe the tpu hardware\n        feature.\n    '
        self.tpu_hardware_feature_proto = tpu_hardware_feature_proto

    class EmbeddingFeature(enum.Enum):
        """Embedding feature flag strings.

    UNSUPPORTED: No embedding lookup accelerator available on the tpu.
    V1: Embedding lookup accelerator V1. The embedding lookup operation can only
        be placed at the beginning of computation. Only one instance of
        embedding
        lookup layer is allowed.
    V2: Embedding lookup accelerator V2. The embedding lookup operation can be
        placed anywhere of the computation. Multiple instances of embedding
        lookup layer is allowed.
    """
        UNSUPPORTED = 'UNSUPPORTED'
        V1 = 'V1'
        V2 = 'V2'

    @classmethod
    def _embedding_feature_proto_to_string(cls, embedding_feature_proto):
        if False:
            while True:
                i = 10
        'Convert the embedding feature proto to enum string.'
        embedding_feature_proto_to_string_map = {topology_pb2.TPUHardwareFeature.EmbeddingFeature.UNSUPPORTED: HardwareFeature.EmbeddingFeature.UNSUPPORTED, topology_pb2.TPUHardwareFeature.EmbeddingFeature.V1: HardwareFeature.EmbeddingFeature.V1, topology_pb2.TPUHardwareFeature.EmbeddingFeature.V2: HardwareFeature.EmbeddingFeature.V2}
        return embedding_feature_proto_to_string_map.get(embedding_feature_proto, HardwareFeature.EmbeddingFeature.UNSUPPORTED)

    @property
    def embedding_feature(self):
        if False:
            print('Hello World!')
        'TPU embedding feature.\n\n    Returns:\n      An EmbeddingFeature enum.\n    '
        return HardwareFeature._embedding_feature_proto_to_string(self.tpu_hardware_feature_proto.embedding_feature)

    @property
    def num_embedding_devices_per_chip(self):
        if False:
            for i in range(10):
                print('nop')
        'Number of embedding accelerator devices per chip.\n\n    Returns:\n      Number of embedding devices per chip.\n    '
        return self.tpu_hardware_feature_proto.num_embedding_devices_per_chip