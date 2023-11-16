"""A place to store TSIG keys."""
import base64
from typing import Any, Dict
import dns.name
import dns.tsig

def from_text(textring: Dict[str, Any]) -> Dict[dns.name.Name, dns.tsig.Key]:
    if False:
        i = 10
        return i + 15
    'Convert a dictionary containing (textual DNS name, base64 secret)\n    pairs into a binary keyring which has (dns.name.Name, bytes) pairs, or\n    a dictionary containing (textual DNS name, (algorithm, base64 secret))\n    pairs into a binary keyring which has (dns.name.Name, dns.tsig.Key) pairs.\n    @rtype: dict'
    keyring = {}
    for (name, value) in textring.items():
        kname = dns.name.from_text(name)
        if isinstance(value, str):
            keyring[kname] = dns.tsig.Key(kname, value).secret
        else:
            (algorithm, secret) = value
            keyring[kname] = dns.tsig.Key(kname, secret, algorithm)
    return keyring

def to_text(keyring: Dict[dns.name.Name, Any]) -> Dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    'Convert a dictionary containing (dns.name.Name, dns.tsig.Key) pairs\n    into a text keyring which has (textual DNS name, (textual algorithm,\n    base64 secret)) pairs, or a dictionary containing (dns.name.Name, bytes)\n    pairs into a text keyring which has (textual DNS name, base64 secret) pairs.\n    @rtype: dict'
    textring = {}

    def b64encode(secret):
        if False:
            while True:
                i = 10
        return base64.encodebytes(secret).decode().rstrip()
    for (name, key) in keyring.items():
        tname = name.to_text()
        if isinstance(key, bytes):
            textring[tname] = b64encode(key)
        else:
            if isinstance(key.secret, bytes):
                text_secret = b64encode(key.secret)
            else:
                text_secret = str(key.secret)
            textring[tname] = (key.algorithm.to_text(), text_secret)
    return textring