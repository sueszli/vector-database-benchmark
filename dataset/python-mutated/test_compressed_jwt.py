import json
import zlib
from jwt import PyJWT

class CompressedPyJWT(PyJWT):

    def _decode_payload(self, decoded):
        if False:
            while True:
                i = 10
        return json.loads(zlib.decompress(decoded['payload'], wbits=-15).decode('utf-8'))

def test_decodes_complete_valid_jwt_with_compressed_payload():
    if False:
        for i in range(10):
            print('nop')
    example_payload = {'hello': 'world'}
    example_secret = 'secret'
    example_jwt = b'eyJhbGciOiAiSFMyNTYiLCAidHlwIjogIkpXVCJ9.q1bKSM3JyVeyUlAqzy/KSVGqBQA=.08wHYeuh1rJXmcBcMrz6NxmbxAnCQp2rGTKfRNIkxiw='
    decoded = CompressedPyJWT().decode_complete(example_jwt, example_secret, algorithms=['HS256'])
    assert decoded == {'header': {'alg': 'HS256', 'typ': 'JWT'}, 'payload': example_payload, 'signature': b'\xd3\xcc\x07a\xeb\xa1\xd6\xb2W\x99\xc0\\2\xbc\xfa7\x19\x9b\xc4\t\xc2B\x9d\xab\x192\x9fD\xd2$\xc6,'}