import pytest
from werkzeug.datastructures import Headers
from werkzeug.sansio.multipart import Data
from werkzeug.sansio.multipart import Epilogue
from werkzeug.sansio.multipart import Field
from werkzeug.sansio.multipart import File
from werkzeug.sansio.multipart import MultipartDecoder
from werkzeug.sansio.multipart import MultipartEncoder
from werkzeug.sansio.multipart import NeedData
from werkzeug.sansio.multipart import Preamble

def test_decoder_simple() -> None:
    if False:
        i = 10
        return i + 15
    boundary = b'---------------------------9704338192090380615194531385$'
    decoder = MultipartDecoder(boundary)
    data = '\n-----------------------------9704338192090380615194531385$\nContent-Disposition: form-data; name="fname"\n\nß∑œß∂ƒå∂\n-----------------------------9704338192090380615194531385$\nContent-Disposition: form-data; name="lname"; filename="bob"\n\nasdasd\n-----------------------------9704338192090380615194531385$--\n    '.replace('\n', '\r\n').encode()
    decoder.receive_data(data)
    decoder.receive_data(None)
    events = [decoder.next_event()]
    while not isinstance(events[-1], Epilogue):
        events.append(decoder.next_event())
    assert events == [Preamble(data=b''), Field(name='fname', headers=Headers([('Content-Disposition', 'form-data; name="fname"')])), Data(data='ß∑œß∂ƒå∂'.encode(), more_data=False), File(name='lname', filename='bob', headers=Headers([('Content-Disposition', 'form-data; name="lname"; filename="bob"')])), Data(data=b'asdasd', more_data=False), Epilogue(data=b'    ')]
    encoder = MultipartEncoder(boundary)
    result = b''
    for event in events:
        result += encoder.send_event(event)
    assert data == result

@pytest.mark.parametrize('data_start', [b'A', b'\n', b'\r', b'\r\n', b'\n\r', b'A\n', b'A\r', b'A\r\n', b'A\n\r'])
@pytest.mark.parametrize('data_end', [b'', b'\r\n--foo'])
def test_decoder_data_start_with_different_newline_positions(data_start: bytes, data_end: bytes) -> None:
    if False:
        while True:
            i = 10
    boundary = b'foo'
    data = b'\r\n--foo\r\nContent-Disposition: form-data; name="test"; filename="testfile"\r\nContent-Type: application/octet-stream\r\n\r\n' + data_start + b'\r\nBCDE' + data_end
    decoder = MultipartDecoder(boundary)
    decoder.receive_data(data)
    events = [decoder.next_event()]
    while not isinstance(events[-1], Data):
        events.append(decoder.next_event())
    expected = data_start if data_end == b'' else data_start + b'\r\nBCDE'
    assert events == [Preamble(data=b''), File(name='test', filename='testfile', headers=Headers([('Content-Disposition', 'form-data; name="test"; filename="testfile"'), ('Content-Type', 'application/octet-stream')])), Data(data=expected, more_data=True)]

def test_chunked_boundaries() -> None:
    if False:
        return 10
    boundary = b'--boundary'
    decoder = MultipartDecoder(boundary)
    decoder.receive_data(b'--')
    assert isinstance(decoder.next_event(), NeedData)
    decoder.receive_data(b'--boundary\r\n')
    assert isinstance(decoder.next_event(), Preamble)
    decoder.receive_data(b'Content-Disposition: form-data;')
    assert isinstance(decoder.next_event(), NeedData)
    decoder.receive_data(b'name="fname"\r\n\r\n')
    assert isinstance(decoder.next_event(), Field)
    decoder.receive_data(b'longer than the boundary')
    assert isinstance(decoder.next_event(), Data)
    decoder.receive_data(b'also longer, but includes a linebreak\r\n--')
    assert isinstance(decoder.next_event(), Data)
    assert isinstance(decoder.next_event(), NeedData)
    decoder.receive_data(b'--boundary--\r\n')
    event = decoder.next_event()
    assert isinstance(event, Data)
    assert not event.more_data
    decoder.receive_data(None)
    assert isinstance(decoder.next_event(), Epilogue)

def test_empty_field() -> None:
    if False:
        while True:
            i = 10
    boundary = b'foo'
    decoder = MultipartDecoder(boundary)
    data = '\n--foo\nContent-Disposition: form-data; name="text"\nContent-Type: text/plain; charset="UTF-8"\n\nSome Text\n--foo\nContent-Disposition: form-data; name="empty"\nContent-Type: text/plain; charset="UTF-8"\n\n--foo--\n    '.replace('\n', '\r\n').encode()
    decoder.receive_data(data)
    decoder.receive_data(None)
    events = [decoder.next_event()]
    while not isinstance(events[-1], Epilogue):
        events.append(decoder.next_event())
    assert events == [Preamble(data=b''), Field(name='text', headers=Headers([('Content-Disposition', 'form-data; name="text"'), ('Content-Type', 'text/plain; charset="UTF-8"')])), Data(data=b'Some Text', more_data=False), Field(name='empty', headers=Headers([('Content-Disposition', 'form-data; name="empty"'), ('Content-Type', 'text/plain; charset="UTF-8"')])), Data(data=b'', more_data=False), Epilogue(data=b'    ')]
    encoder = MultipartEncoder(boundary)
    result = b''
    for event in events:
        result += encoder.send_event(event)
    assert data == result