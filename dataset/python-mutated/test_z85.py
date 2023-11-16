"""Test Z85 encoding

confirm values and roundtrip with test values from the reference implementation.
"""
from unittest import TestCase
from zmq.utils import z85

class TestZ85(TestCase):

    def test_client_public(self):
        if False:
            for i in range(10):
                print('nop')
        client_public = b'\xbb\x88G\x1de\xe2e\x9b0\xc5ZS!\xce\xbbZ\xab+p\xa3\x98d\\&\xdc\xa2\xb2\xfc\xb4?\xc5\x18'
        encoded = z85.encode(client_public)
        assert encoded == b'Yne@$w-vo<fVvi]a<NY6T1ed:M$fCG*[IaLV{hID'
        decoded = z85.decode(encoded)
        assert decoded == client_public

    def test_client_secret(self):
        if False:
            for i in range(10):
                print('nop')
        client_secret = b'{\xb8d\xb4\x89\xaf\xa3g\x1f\xbei\x10\x1f\x94\xb3\x89r\xf2H\x16\xdf\xb0\x1bQek?\xec\x8d\xfd\x08\x88'
        encoded = z85.encode(client_secret)
        assert encoded == b'D:)Q[IlAW!ahhC2ac:9*A}h:p?([4%wOTJ%JR%cs'
        decoded = z85.decode(encoded)
        assert decoded == client_secret

    def test_server_public(self):
        if False:
            print('Hello World!')
        server_public = b"T\xfc\xba$\xe92I\x96\x93\x16\xfba|\x87+\xb0\xc1\xd1\xff\x14\x80\x04'\xc5\x94\xcb\xfa\xcf\x1b\xc2\xd6R"
        encoded = z85.encode(server_public)
        assert encoded == b'rq:rM>}U?@Lns47E1%kR.o@n%FcmmsL/@{H8]yf7'
        decoded = z85.decode(encoded)
        assert decoded == server_public

    def test_server_secret(self):
        if False:
            print('Hello World!')
        server_secret = b'\x8e\x0b\xddiv(\xb9\x1d\x8f$U\x87\xee\x95\xc5\xb0MH\x96?y%\x98w\xb4\x9c\xd9\x06:\xea\xd3\xb7'
        encoded = z85.encode(server_secret)
        assert encoded == b'JTKVSB%%)wK0E.X)V>+}o?pNmC{O&4W4b!Ni{Lh6'
        decoded = z85.decode(encoded)
        assert decoded == server_secret