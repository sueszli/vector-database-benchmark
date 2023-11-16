"""
helpers
~~~~~~~

This module contains helpers for the h2 tests.
"""
from hpack.hpack import Encoder
from hyperframe.frame import AltSvcFrame
from hyperframe.frame import ContinuationFrame
from hyperframe.frame import DataFrame
from hyperframe.frame import GoAwayFrame
from hyperframe.frame import HeadersFrame
from hyperframe.frame import PingFrame
from hyperframe.frame import PriorityFrame
from hyperframe.frame import PushPromiseFrame
from hyperframe.frame import RstStreamFrame
from hyperframe.frame import SettingsFrame
from hyperframe.frame import WindowUpdateFrame
SAMPLE_SETTINGS = {SettingsFrame.HEADER_TABLE_SIZE: 4096, SettingsFrame.ENABLE_PUSH: 1, SettingsFrame.MAX_CONCURRENT_STREAMS: 2}

class FrameFactory:
    """
    A class containing lots of helper methods and state to build frames. This
    allows test cases to easily build correct HTTP/2 frames to feed to
    hyper-h2.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.encoder = Encoder()

    def refresh_encoder(self):
        if False:
            return 10
        self.encoder = Encoder()

    def preamble(self):
        if False:
            while True:
                i = 10
        return b'PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n'

    def build_headers_frame(self, headers, flags=[], stream_id=1, **priority_kwargs):
        if False:
            return 10
        '\n        Builds a single valid headers frame out of the contained headers.\n        '
        f = HeadersFrame(stream_id)
        f.data = self.encoder.encode(headers)
        f.flags.add('END_HEADERS')
        for flag in flags:
            f.flags.add(flag)
        for (k, v) in priority_kwargs.items():
            setattr(f, k, v)
        return f

    def build_continuation_frame(self, header_block, flags=[], stream_id=1):
        if False:
            while True:
                i = 10
        '\n        Builds a single continuation frame out of the binary header block.\n        '
        f = ContinuationFrame(stream_id)
        f.data = header_block
        f.flags = set(flags)
        return f

    def build_data_frame(self, data, flags=None, stream_id=1, padding_len=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Builds a single data frame out of a chunk of data.\n        '
        flags = set(flags) if flags is not None else set()
        f = DataFrame(stream_id)
        f.data = data
        f.flags = flags
        if padding_len:
            flags.add('PADDED')
            f.pad_length = padding_len
        return f

    def build_settings_frame(self, settings, ack=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Builds a single settings frame.\n        '
        f = SettingsFrame(0)
        if ack:
            f.flags.add('ACK')
        f.settings = settings
        return f

    def build_window_update_frame(self, stream_id, increment):
        if False:
            i = 10
            return i + 15
        '\n        Builds a single WindowUpdate frame.\n        '
        f = WindowUpdateFrame(stream_id)
        f.window_increment = increment
        return f

    def build_ping_frame(self, ping_data, flags=None):
        if False:
            return 10
        '\n        Builds a single Ping frame.\n        '
        f = PingFrame(0)
        f.opaque_data = ping_data
        if flags:
            f.flags = set(flags)
        return f

    def build_goaway_frame(self, last_stream_id, error_code=0, additional_data=b''):
        if False:
            print('Hello World!')
        '\n        Builds a single GOAWAY frame.\n        '
        f = GoAwayFrame(0)
        f.error_code = error_code
        f.last_stream_id = last_stream_id
        f.additional_data = additional_data
        return f

    def build_rst_stream_frame(self, stream_id, error_code=0):
        if False:
            i = 10
            return i + 15
        '\n        Builds a single RST_STREAM frame.\n        '
        f = RstStreamFrame(stream_id)
        f.error_code = error_code
        return f

    def build_push_promise_frame(self, stream_id, promised_stream_id, headers, flags=[]):
        if False:
            return 10
        '\n        Builds a single PUSH_PROMISE frame.\n        '
        f = PushPromiseFrame(stream_id)
        f.promised_stream_id = promised_stream_id
        f.data = self.encoder.encode(headers)
        f.flags = set(flags)
        f.flags.add('END_HEADERS')
        return f

    def build_priority_frame(self, stream_id, weight, depends_on=0, exclusive=False):
        if False:
            return 10
        '\n        Builds a single priority frame.\n        '
        f = PriorityFrame(stream_id)
        f.depends_on = depends_on
        f.stream_weight = weight
        f.exclusive = exclusive
        return f

    def build_alt_svc_frame(self, stream_id, origin, field):
        if False:
            for i in range(10):
                print('nop')
        '\n        Builds a single ALTSVC frame.\n        '
        f = AltSvcFrame(stream_id)
        f.origin = origin
        f.field = field
        return f

    def change_table_size(self, new_size):
        if False:
            i = 10
            return i + 15
        '\n        Causes the encoder to send a dynamic size update in the next header\n        block it sends.\n        '
        self.encoder.header_table_size = new_size