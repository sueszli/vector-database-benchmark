"""Unit tests for flink_streaming_impulse_source."""
import logging
import unittest
import apache_beam as beam
from apache_beam.io.flink.flink_streaming_impulse_source import FlinkStreamingImpulseSource

class FlinkStreamingImpulseSourceTest(unittest.TestCase):

    def test_serialization(self):
        if False:
            print('Hello World!')
        p = beam.Pipeline()
        p | FlinkStreamingImpulseSource()
        beam.Pipeline.from_runner_api(p.to_runner_api(), p.runner, p._options)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()