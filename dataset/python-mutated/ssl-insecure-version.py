import ssl
from pyOpenSSL import SSL
ssl.wrap_socket(ssl_version=ssl.PROTOCOL_SSLv2)
SSL.Context(method=SSL.SSLv2_METHOD)
SSL.Context(method=SSL.SSLv23_METHOD)
herp_derp(ssl_version=ssl.PROTOCOL_SSLv2)
herp_derp(method=SSL.SSLv2_METHOD)
herp_derp(method=SSL.SSLv23_METHOD)
ssl.wrap_socket(ssl_version=ssl.PROTOCOL_SSLv3)
ssl.wrap_socket(ssl_version=ssl.PROTOCOL_TLSv1)
SSL.Context(method=SSL.SSLv3_METHOD)
SSL.Context(method=SSL.TLSv1_METHOD)
herp_derp(ssl_version=ssl.PROTOCOL_SSLv3)
herp_derp(ssl_version=ssl.PROTOCOL_TLSv1)
herp_derp(method=SSL.SSLv3_METHOD)
herp_derp(method=SSL.TLSv1_METHOD)
ssl.wrap_socket(ssl_version=ssl.PROTOCOL_TLSv1_1)
SSL.Context(method=SSL.TLSv1_1_METHOD)
herp_derp(ssl_version=ssl.PROTOCOL_TLSv1_1)
herp_derp(method=SSL.TLSv1_1_METHOD)
ssl.wrap_socket()

def open_ssl_socket(version=ssl.PROTOCOL_SSLv2):
    if False:
        return 10
    pass

def open_ssl_socket(version=SSL.SSLv2_METHOD):
    if False:
        i = 10
        return i + 15
    pass

def open_ssl_socket(version=SSL.SSLv23_METHOD):
    if False:
        for i in range(10):
            print('nop')
    pass

def open_ssl_socket(version=SSL.TLSv1_1_METHOD):
    if False:
        while True:
            i = 10
    pass

def open_ssl_socket(version=SSL.TLSv1_2_METHOD):
    if False:
        print('Hello World!')
    pass