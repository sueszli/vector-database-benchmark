from __future__ import absolute_import, division, print_function, with_statement
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from ssshare.shadowsocks.crypto import rc4_md5
from ssshare.shadowsocks.crypto import openssl
from ssshare.shadowsocks.crypto import sodium
from ssshare.shadowsocks.crypto import table

def run(func):
    if False:
        while True:
            i = 10
    try:
        func()
    except:
        pass

def run_n(func, name):
    if False:
        i = 10
        return i + 15
    try:
        func(name)
    except:
        pass

def main():
    if False:
        return 10
    print('\nrc4_md5')
    rc4_md5.test()
    print('\naes-256-cfb')
    openssl.test_aes_256_cfb()
    print('\naes-128-cfb')
    openssl.test_aes_128_cfb()
    print('\nbf-cfb')
    run(openssl.test_bf_cfb)
    print('\ncamellia-128-cfb')
    run_n(openssl.run_method, 'camellia-128-cfb')
    print('\ncast5-cfb')
    run_n(openssl.run_method, 'cast5-cfb')
    print('\nidea-cfb')
    run_n(openssl.run_method, 'idea-cfb')
    print('\nseed-cfb')
    run_n(openssl.run_method, 'seed-cfb')
    print('\nsalsa20')
    run(sodium.test_salsa20)
    print('\nchacha20')
    run(sodium.test_chacha20)
if __name__ == '__main__':
    main()