from pyflink.datastream.connectors import Source
from pyflink.java_gateway import get_gateway
__all__ = ['NumberSequenceSource']

class NumberSequenceSource(Source):
    """
    A data source that produces a sequence of numbers (longs). This source is useful for testing and
    for cases that just need a stream of N events of any kind.

    The source splits the sequence into as many parallel sub-sequences as there are parallel
    source readers. Each sub-sequence will be produced in order. Consequently, if the parallelism is
    limited to one, this will produce one sequence in order.

    This source is always bounded. For very long sequences (for example over the entire domain of
    long integer values), user may want to consider executing the application in a streaming manner,
    because, despite the fact that the produced stream is bounded, the end bound is pretty far away.
    """

    def __init__(self, start: int, end: int):
        if False:
            while True:
                i = 10
        '\n        Creates a new NumberSequenceSource that produces parallel sequences covering the\n        range start to end (both boundaries are inclusive).\n        '
        JNumberSequenceSource = get_gateway().jvm.org.apache.flink.api.connector.source.lib.NumberSequenceSource
        j_seq_source = JNumberSequenceSource(start, end)
        super(NumberSequenceSource, self).__init__(source=j_seq_source)