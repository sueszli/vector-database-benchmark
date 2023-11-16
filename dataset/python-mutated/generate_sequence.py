from apache_beam.transforms.external import ExternalTransform
from apache_beam.transforms.external import ImplicitSchemaPayloadBuilder

class GenerateSequence(ExternalTransform):
    """
    An external PTransform which provides a bounded or unbounded stream of
    integers.

    Note: To use this transform, you need to start the Java expansion service.
    Please refer to the portability documentation on how to do that. The
    expansion service address has to be provided when instantiating this
    transform. During pipeline translation this transform will be replaced by
    the Java SDK's GenerateSequence.

    If you start Flink's job server, the expansion service will be started on
    port 8097. This is also the configured default for this transform. For a
    different address, please set the expansion_service parameter.

    For more information see:
    - https://beam.apache.org/documentation/runners/flink/
    - https://beam.apache.org/roadmap/portability/

    Note: Runners need to support translating Read operations in order to use
    this source. At the moment only the Flink Runner supports this.

    Experimental; no backwards compatibility guarantees.
  """
    URN = 'beam:external:java:generate_sequence:v1'

    def __init__(self, start, stop=None, elements_per_period=None, max_read_time=None, expansion_service=None):
        if False:
            while True:
                i = 10
        super().__init__(self.URN, ImplicitSchemaPayloadBuilder({'start': start, 'stop': stop, 'elements_per_period': elements_per_period, 'max_read_time': max_read_time}), expansion_service)