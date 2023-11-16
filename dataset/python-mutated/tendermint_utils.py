import base64
import hashlib
import json
from binascii import hexlify
try:
    from hashlib import sha3_256
except ImportError:
    from sha3 import sha3_256

def encode_transaction(value):
    if False:
        print('Hello World!')
    'Encode a transaction (dict) to Base64.'
    return base64.b64encode(json.dumps(value).encode('utf8')).decode('utf8')

def decode_transaction(raw):
    if False:
        i = 10
        return i + 15
    'Decode a transaction from bytes to a dict.'
    return json.loads(raw.decode('utf8'))

def decode_transaction_base64(value):
    if False:
        print('Hello World!')
    'Decode a transaction from Base64.'
    return json.loads(base64.b64decode(value.encode('utf8')).decode('utf8'))

def calculate_hash(key_list):
    if False:
        for i in range(10):
            print('nop')
    if not key_list:
        return ''
    full_hash = sha3_256()
    for key in key_list:
        full_hash.update(key.encode('utf8'))
    return full_hash.hexdigest()

def merkleroot(hashes):
    if False:
        i = 10
        return i + 15
    'Computes the merkle root for a given list.\n\n    Args:\n        hashes (:obj:`list` of :obj:`bytes`): The leaves of the tree.\n\n    Returns:\n        str: Merkle root in hexadecimal form.\n\n    '
    if not hashes:
        return sha3_256(b'').hexdigest()
    if len(hashes) == 1:
        return hexlify(hashes[0]).decode()
    if len(hashes) % 2 == 1:
        hashes.append(hashes[-1])
    parent_hashes = [sha3_256(hashes[i] + hashes[i + 1]).digest() for i in range(0, len(hashes) - 1, 2)]
    return merkleroot(parent_hashes)

def public_key64_to_address(base64_public_key):
    if False:
        print('Hello World!')
    'Note this only compatible with Tendermint 0.19.x'
    ed25519_public_key = public_key_from_base64(base64_public_key)
    encoded_public_key = amino_encoded_public_key(ed25519_public_key)
    return hashlib.new('ripemd160', encoded_public_key).hexdigest().upper()

def public_key_from_base64(base64_public_key):
    if False:
        while True:
            i = 10
    return key_from_base64(base64_public_key)

def key_from_base64(base64_key):
    if False:
        return 10
    return base64.b64decode(base64_key).hex().upper()

def public_key_to_base64(ed25519_public_key):
    if False:
        return 10
    return key_to_base64(ed25519_public_key)

def key_to_base64(ed25519_key):
    if False:
        return 10
    ed25519_key = bytes.fromhex(ed25519_key)
    return base64.b64encode(ed25519_key).decode('utf-8')

def amino_encoded_public_key(ed25519_public_key):
    if False:
        for i in range(10):
            print('nop')
    return bytes.fromhex('1624DE6220{}'.format(ed25519_public_key))