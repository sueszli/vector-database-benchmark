"""
This function decodes a Stream, applying the filters specified in the Filter entry
of its stream dictionary
"""
import typing
from decimal import Decimal
from borb.io.filter.ascii85_decode import ASCII85Decode
from borb.io.filter.flate_decode import FlateDecode
from borb.io.filter.lzw_decode import LZWDecode
from borb.io.filter.run_length_decode import RunLengthDecode
from borb.io.read.types import Dictionary
from borb.io.read.types import List
from borb.io.read.types import Name
from borb.io.read.types import Stream

def decode_stream(s: Stream) -> Stream:
    if False:
        return 10
    '\n    This function decodes a Stream, applying the filters specified in the Filter entry\n    of its stream dictionary\n    '
    assert isinstance(s, Stream), 'decode_stream only works on Stream objects'
    assert 'Bytes' in s, 'decode_stream only works on Stream objects with a `Bytes` key.'
    filters: typing.List[str] = []
    if 'Filter' in s:
        if isinstance(s['Filter'], List):
            filters = s['Filter']
        else:
            filters = [s['Filter']]
    decode_params: typing.List[Dictionary] = []
    if 'DecodeParms' in s:
        if isinstance(s['DecodeParms'], List):
            decode_params = s['DecodeParms']
            decode_params = [x or Dictionary() for x in decode_params]
        else:
            assert s['DecodeParms'] is not None
            assert isinstance(s['DecodeParms'], Dictionary)
            decode_params = [s['DecodeParms']]
    else:
        decode_params = [Dictionary() for x in range(0, len(filters))]
    transformed_bytes = s['Bytes']
    for (filter_index, filter_name) in enumerate(filters):
        if filter_name in ['FlateDecode', 'Fl']:
            transformed_bytes = FlateDecode.decode(bytes_in=transformed_bytes, columns=int(decode_params[filter_index].get('Columns', Decimal(1))), predictor=int(decode_params[filter_index].get('Predictor', Decimal(1))), bits_per_component=int(decode_params[filter_index].get('BitsPerComponent', Decimal(8))))
            continue
        if filter_name in ['ASCII85Decode']:
            transformed_bytes = ASCII85Decode.decode(transformed_bytes)
            continue
        if filter_name in ['LZWDecode']:
            transformed_bytes = LZWDecode().decode(transformed_bytes)
            continue
        if filter_name in ['RunLengthDecode']:
            transformed_bytes = RunLengthDecode.decode(transformed_bytes)
            continue
        assert False, 'Unknown /Filter %s' % filter_name
    s[Name('DecodedBytes')] = transformed_bytes
    return s