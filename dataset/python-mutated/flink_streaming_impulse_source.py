"""
A PTransform that provides an unbounded, streaming source of empty byte arrays.

This can only be used with the flink runner.
"""
import json
from typing import Any
from typing import Dict
from apache_beam import PTransform
from apache_beam import Windowing
from apache_beam import pvalue
from apache_beam.transforms.window import GlobalWindows

class FlinkStreamingImpulseSource(PTransform):
    URN = 'flink:transform:streaming_impulse:v1'
    config = {}

    def expand(self, pbegin):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(pbegin, pvalue.PBegin), 'Input to transform must be a PBegin but found %s' % pbegin
        return pvalue.PCollection(pbegin.pipeline, is_bounded=False)

    def get_windowing(self, unused_inputs):
        if False:
            for i in range(10):
                print('nop')
        return Windowing(GlobalWindows())

    def infer_output_type(self, unused_input_type):
        if False:
            i = 10
            return i + 15
        return bytes

    def to_runner_api_parameter(self, context):
        if False:
            return 10
        assert isinstance(self, FlinkStreamingImpulseSource), 'expected instance of StreamingImpulseSource, but got %s' % self.__class__
        return (self.URN, json.dumps(self.config))

    def set_interval_ms(self, interval_ms):
        if False:
            i = 10
            return i + 15
        'Sets the interval (in milliseconds) between messages in the stream.\n    '
        self.config['interval_ms'] = interval_ms
        return self

    def set_message_count(self, message_count):
        if False:
            print('Hello World!')
        'If non-zero, the stream will produce only this many total messages.\n    Otherwise produces an unbounded number of messages.\n    '
        self.config['message_count'] = message_count
        return self

    @staticmethod
    @PTransform.register_urn(URN, None)
    def from_runner_api_parameter(_ptransform, spec_parameter, _context):
        if False:
            while True:
                i = 10
        if isinstance(spec_parameter, bytes):
            spec_parameter = spec_parameter.decode('utf-8')
        config = json.loads(spec_parameter)
        instance = FlinkStreamingImpulseSource()
        if 'interval_ms' in config:
            instance.set_interval_ms(config['interval_ms'])
        if 'message_count' in config:
            instance.set_message_count(config['message_count'])
        return instance