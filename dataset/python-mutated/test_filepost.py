from __future__ import annotations
import pytest
from urllib3.fields import RequestField
from urllib3.filepost import _TYPE_FIELDS, encode_multipart_formdata
BOUNDARY = '!! test boundary !!'
BOUNDARY_BYTES = BOUNDARY.encode()

class TestMultipartEncoding:

    @pytest.mark.parametrize('fields', [dict(k='v', k2='v2'), [('k', 'v'), ('k2', 'v2')]])
    def test_input_datastructures(self, fields: _TYPE_FIELDS) -> None:
        if False:
            for i in range(10):
                print('nop')
        (encoded, _) = encode_multipart_formdata(fields, boundary=BOUNDARY)
        assert encoded.count(BOUNDARY_BYTES) == 3

    @pytest.mark.parametrize('fields', [[('k', 'v'), ('k2', 'v2')], [('k', b'v'), ('k2', b'v2')], [('k', b'v'), ('k2', 'v2')]])
    def test_field_encoding(self, fields: _TYPE_FIELDS) -> None:
        if False:
            i = 10
            return i + 15
        (encoded, content_type) = encode_multipart_formdata(fields, boundary=BOUNDARY)
        expected = b'--' + BOUNDARY_BYTES + b'\r\nContent-Disposition: form-data; name="k"\r\n\r\nv\r\n--' + BOUNDARY_BYTES + b'\r\nContent-Disposition: form-data; name="k2"\r\n\r\nv2\r\n--' + BOUNDARY_BYTES + b'--\r\n'
        assert encoded == expected
        assert content_type == 'multipart/form-data; boundary=' + str(BOUNDARY)

    def test_filename(self) -> None:
        if False:
            return 10
        fields = [('k', ('somename', b'v'))]
        (encoded, content_type) = encode_multipart_formdata(fields, boundary=BOUNDARY)
        expected = b'--' + BOUNDARY_BYTES + b'\r\nContent-Disposition: form-data; name="k"; filename="somename"\r\nContent-Type: application/octet-stream\r\n\r\nv\r\n--' + BOUNDARY_BYTES + b'--\r\n'
        assert encoded == expected
        assert content_type == 'multipart/form-data; boundary=' + str(BOUNDARY)

    def test_textplain(self) -> None:
        if False:
            while True:
                i = 10
        fields = [('k', ('somefile.txt', b'v'))]
        (encoded, content_type) = encode_multipart_formdata(fields, boundary=BOUNDARY)
        expected = b'--' + BOUNDARY_BYTES + b'\r\nContent-Disposition: form-data; name="k"; filename="somefile.txt"\r\nContent-Type: text/plain\r\n\r\nv\r\n--' + BOUNDARY_BYTES + b'--\r\n'
        assert encoded == expected
        assert content_type == 'multipart/form-data; boundary=' + str(BOUNDARY)

    def test_explicit(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        fields = [('k', ('somefile.txt', b'v', 'image/jpeg'))]
        (encoded, content_type) = encode_multipart_formdata(fields, boundary=BOUNDARY)
        expected = b'--' + BOUNDARY_BYTES + b'\r\nContent-Disposition: form-data; name="k"; filename="somefile.txt"\r\nContent-Type: image/jpeg\r\n\r\nv\r\n--' + BOUNDARY_BYTES + b'--\r\n'
        assert encoded == expected
        assert content_type == 'multipart/form-data; boundary=' + str(BOUNDARY)

    def test_request_fields(self) -> None:
        if False:
            print('Hello World!')
        fields = [RequestField('k', b'v', filename='somefile.txt', headers={'Content-Type': 'image/jpeg'})]
        (encoded, content_type) = encode_multipart_formdata(fields, boundary=BOUNDARY)
        expected = b'--' + BOUNDARY_BYTES + b'\r\nContent-Type: image/jpeg\r\n\r\nv\r\n--' + BOUNDARY_BYTES + b'--\r\n'
        assert encoded == expected