from typing import Any
from apache_beam.coders.coder_impl import StreamCoderImpl, create_InputStream, create_OutputStream
from pyflink.fn_execution.stream_slow import OutputStream
from pyflink.fn_execution.beam.beam_stream_slow import BeamInputStream, BeamTimeBasedOutputStream

class PassThroughLengthPrefixCoderImpl(StreamCoderImpl):

    def __init__(self, value_coder):
        if False:
            while True:
                i = 10
        self._value_coder = value_coder

    def encode_to_stream(self, value, out: create_OutputStream, nested: bool) -> Any:
        if False:
            while True:
                i = 10
        self._value_coder.encode_to_stream(value, out, nested)

    def decode_from_stream(self, in_stream: create_InputStream, nested: bool) -> Any:
        if False:
            while True:
                i = 10
        return self._value_coder.decode_from_stream(in_stream, nested)

    def get_estimated_size_and_observables(self, value: Any, nested=False):
        if False:
            while True:
                i = 10
        return (0, [])

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'PassThroughLengthPrefixCoderImpl[%s]' % self._value_coder

class FlinkFieldCoderBeamWrapper(StreamCoderImpl):
    """
    Bridge between Beam coder and Flink coder for the low-level FieldCoder.
    """

    def __init__(self, value_coder):
        if False:
            print('Hello World!')
        self._value_coder = value_coder
        self._data_output_stream = OutputStream()

    def encode_to_stream(self, value, out_stream: create_OutputStream, nested):
        if False:
            return 10
        self._value_coder.encode_to_stream(value, self._data_output_stream)
        out_stream.write(self._data_output_stream.get())
        self._data_output_stream.clear()

    def decode_from_stream(self, in_stream: create_InputStream, nested):
        if False:
            i = 10
            return i + 15
        data_input_stream = BeamInputStream(in_stream)
        return self._value_coder.decode_from_stream(data_input_stream)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'FlinkFieldCoderBeamWrapper[%s]' % self._value_coder

class FlinkLengthPrefixCoderBeamWrapper(FlinkFieldCoderBeamWrapper):
    """
    Bridge between Beam coder and Flink coder for the top-level LengthPrefixCoder.
    """

    def __init__(self, value_coder):
        if False:
            while True:
                i = 10
        super(FlinkLengthPrefixCoderBeamWrapper, self).__init__(value_coder)
        self._output_stream = BeamTimeBasedOutputStream()

    def encode_to_stream(self, value, out_stream: create_OutputStream, nested):
        if False:
            i = 10
            return i + 15
        self._output_stream.reset_output_stream(out_stream)
        self._value_coder.encode_to_stream(value, self._data_output_stream)
        self._output_stream.write(self._data_output_stream.get())
        self._data_output_stream.clear()

    def __repr__(self):
        if False:
            return 10
        return 'FlinkLengthPrefixCoderBeamWrapper[%s]' % self._value_coder