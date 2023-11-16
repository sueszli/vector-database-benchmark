import base64
import hashlib
from typing import Any, Optional

def hash(value: Any, hash_type: str='md5', salt: Optional[str]=None) -> str:
    if False:
        print('Hello World!')
    '\n      Implementation of a custom Jinja2 hash filter\n      Hash type defaults to \'md5\' if one is not specified.\n\n      If you are using this has function for GDPR compliance, then\n      you should probably also pass in a salt as discussed in:\n      https://security.stackexchange.com/questions/202022/hashing-email-addresses-for-gdpr-compliance\n\n      This can be used in a low code connector definition under the AddFields transformation.\n      For example:\n\n    rates_stream:\n      $ref: "#/definitions/base_stream"\n      $parameters:\n        name: "rates"\n        primary_key: "date"\n        path: "/exchangerates_data/latest"\n      transformations:\n        - type: AddFields\n          fields:\n            - path: ["some_new_path"]\n              value: "{{ record[\'rates\'][\'CAD\'] | hash(\'md5\', \'mysalt\')  }}"\n\n\n\n      :param value: value to be hashed\n      :param hash_type: valid hash type\n      :param salt: a salt that will be combined with the value to ensure that the hash created for a given value on this system\n                   is different from the hash created for that value on other systems.\n      :return: computed hash as a hexadecimal string\n    '
    hash_func = getattr(hashlib, hash_type, None)
    if hash_func:
        hash_obj = hash_func()
        hash_obj.update(str(value).encode('utf-8'))
        if salt:
            hash_obj.update(str(salt).encode('utf-8'))
        computed_hash: str = hash_obj.hexdigest()
    else:
        raise AttributeError('No hashing function named {hname}'.format(hname=hash_type))
    return computed_hash

def base64encode(value: str) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Implementation of a custom Jinja2 base64encode filter\n\n    For example:\n\n      OAuthAuthenticator:\n        $ref: "#/definitions/OAuthAuthenticator"\n        $parameters:\n          name: "client_id"\n          value: "{{ config[\'client_id\'] | base64encode }}"\n\n    :param value: value to be encoded in base64\n    :return: base64 encoded string\n    '
    return base64.b64encode(value.encode('utf-8')).decode()

def base64decode(value: str) -> str:
    if False:
        return 10
    '\n    Implementation of a custom Jinja2 base64decode filter\n\n    For example:\n\n      OAuthAuthenticator:\n        $ref: "#/definitions/OAuthAuthenticator"\n        $parameters:\n          name: "client_id"\n          value: "{{ config[\'client_id\'] | base64decode }}"\n\n    :param value: value to be decoded from base64\n    :return: base64 decoded string\n    '
    return base64.b64decode(value.encode('utf-8')).decode()
_filters_list = [hash, base64encode, base64decode]
filters = {f.__name__: f for f in _filters_list}