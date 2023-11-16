"""This module contains needed classes, functions and constants
to implement dh-ietf1024-sha256-aes128-cbc-pkcs7 secret encryption
algorithm."""
import hmac
import os
from hashlib import sha256
from secretstorage.util import int_from_bytes, int_to_bytes
DH_PRIME_1024_BYTES = (255, 255, 255, 255, 255, 255, 255, 255, 201, 15, 218, 162, 33, 104, 194, 52, 196, 198, 98, 139, 128, 220, 28, 209, 41, 2, 78, 8, 138, 103, 204, 116, 2, 11, 190, 166, 59, 19, 155, 34, 81, 74, 8, 121, 142, 52, 4, 221, 239, 149, 25, 179, 205, 58, 67, 27, 48, 43, 10, 109, 242, 95, 20, 55, 79, 225, 53, 109, 109, 81, 194, 69, 228, 133, 181, 118, 98, 94, 126, 198, 244, 76, 66, 233, 166, 55, 237, 107, 11, 255, 92, 182, 244, 6, 183, 237, 238, 56, 107, 251, 90, 137, 159, 165, 174, 159, 36, 17, 124, 75, 31, 230, 73, 40, 102, 81, 236, 230, 83, 129, 255, 255, 255, 255, 255, 255, 255, 255)
DH_PRIME_1024 = int_from_bytes(bytearray(DH_PRIME_1024_BYTES), 'big')

class Session(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.object_path = None
        self.server_public_key = None
        self.aes_key = None
        self.encrypted = True
        self.my_private_key = int_from_bytes(os.urandom(128), 'big')
        self.my_public_key = pow(2, self.my_private_key, DH_PRIME_1024)

    def set_server_public_key(self, server_public_key):
        if False:
            return 10
        common_secret = pow(server_public_key, self.my_private_key, DH_PRIME_1024)
        common_secret = int_to_bytes(common_secret)
        common_secret = b'\x00' * (128 - len(common_secret)) + common_secret
        salt = b'\x00' * 32
        pseudo_random_key = hmac.new(salt, common_secret, sha256).digest()
        output_block = hmac.new(pseudo_random_key, b'\x01', sha256).digest()
        self.aes_key = output_block[:16]