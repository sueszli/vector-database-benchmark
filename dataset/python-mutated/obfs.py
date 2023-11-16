from __future__ import absolute_import, division, print_function, with_statement
import os
import sys
import hashlib
import logging
from ssshare.shadowsocks import common
from ssshare.shadowsocks.obfsplugin import plain, http_simple, obfs_tls, verify, auth, auth_chain
method_supported = {}
method_supported.update(plain.obfs_map)
method_supported.update(http_simple.obfs_map)
method_supported.update(obfs_tls.obfs_map)
method_supported.update(verify.obfs_map)
method_supported.update(auth.obfs_map)
method_supported.update(auth_chain.obfs_map)

def mu_protocol():
    if False:
        while True:
            i = 10
    return ['auth_aes128_md5', 'auth_aes128_sha1', 'auth_chain_a']

class server_info(object):

    def __init__(self, data):
        if False:
            while True:
                i = 10
        self.data = data

class obfs(object):

    def __init__(self, method):
        if False:
            print('Hello World!')
        method = common.to_str(method)
        self.method = method
        self._method_info = self.get_method_info(method)
        if self._method_info:
            self.obfs = self.get_obfs(method)
        else:
            raise Exception('obfs plugin [%s] not supported' % method)

    def init_data(self):
        if False:
            while True:
                i = 10
        return self.obfs.init_data()

    def set_server_info(self, server_info):
        if False:
            for i in range(10):
                print('nop')
        return self.obfs.set_server_info(server_info)

    def get_server_info(self):
        if False:
            return 10
        return self.obfs.get_server_info()

    def get_method_info(self, method):
        if False:
            while True:
                i = 10
        method = method.lower()
        m = method_supported.get(method)
        return m

    def get_obfs(self, method):
        if False:
            print('Hello World!')
        m = self._method_info
        return m[0](method)

    def get_overhead(self, direction):
        if False:
            print('Hello World!')
        return self.obfs.get_overhead(direction)

    def client_pre_encrypt(self, buf):
        if False:
            i = 10
            return i + 15
        return self.obfs.client_pre_encrypt(buf)

    def client_encode(self, buf):
        if False:
            while True:
                i = 10
        return self.obfs.client_encode(buf)

    def client_decode(self, buf):
        if False:
            for i in range(10):
                print('nop')
        return self.obfs.client_decode(buf)

    def client_post_decrypt(self, buf):
        if False:
            while True:
                i = 10
        return self.obfs.client_post_decrypt(buf)

    def server_pre_encrypt(self, buf):
        if False:
            for i in range(10):
                print('nop')
        return self.obfs.server_pre_encrypt(buf)

    def server_encode(self, buf):
        if False:
            i = 10
            return i + 15
        return self.obfs.server_encode(buf)

    def server_decode(self, buf):
        if False:
            for i in range(10):
                print('nop')
        return self.obfs.server_decode(buf)

    def server_post_decrypt(self, buf):
        if False:
            print('Hello World!')
        return self.obfs.server_post_decrypt(buf)

    def client_udp_pre_encrypt(self, buf):
        if False:
            i = 10
            return i + 15
        return self.obfs.client_udp_pre_encrypt(buf)

    def client_udp_post_decrypt(self, buf):
        if False:
            while True:
                i = 10
        return self.obfs.client_udp_post_decrypt(buf)

    def server_udp_pre_encrypt(self, buf, uid):
        if False:
            print('Hello World!')
        return self.obfs.server_udp_pre_encrypt(buf, uid)

    def server_udp_post_decrypt(self, buf):
        if False:
            while True:
                i = 10
        return self.obfs.server_udp_post_decrypt(buf)

    def dispose(self):
        if False:
            for i in range(10):
                print('nop')
        self.obfs.dispose()
        del self.obfs