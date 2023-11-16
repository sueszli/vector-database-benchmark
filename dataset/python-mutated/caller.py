import requests
import json
import base64
import time
import random
import hmac
from hashlib import sha256
import os
from collections import OrderedDict
import urllib.parse
from requests.packages import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
appid = os.getenv('APPID')
apikey = os.getenv('APIKEY')
headers = {'Content-Type': 'application/json'}
use_secure_cert = False

def request_params(payload):
    if False:
        for i in range(10):
            print('nop')
    params = OrderedDict()
    params['appid'] = appid
    ord_payload = OrderedDict(sorted(payload.items(), key=lambda k: k[0]))
    params['payload'] = urllib.parse.unquote(urllib.parse.urlencode(ord_payload))
    params['timestamp'] = str(int(time.time() * 1000))
    sign_string = urllib.parse.unquote(urllib.parse.urlencode(params))
    sign = str(base64.b64encode(hmac.new(apikey.encode('utf-8'), sign_string.encode('utf-8'), digestmod=sha256).digest()), 'utf-8')
    params['payload'] = ord_payload
    params['sign'] = sign
    return params

def construct_url(ip, port, action):
    if False:
        i = 10
        return i + 15
    return 'https://' + ip + ':' + port + '/ehsm?Action=' + action

def post_request(ip, port, action, payload):
    if False:
        print('Hello World!')
    url = construct_url(ip, port, action)
    params = request_params(payload)
    create_resp = requests.post(url=url, data=json.dumps(params), headers=headers, timeout=100, verify=use_secure_cert)
    result = json.loads(create_resp.text)['result']
    return result

def request_primary_key_cipher_text(ip, port):
    if False:
        while True:
            i = 10
    action = 'CreateKey'
    payload = {'keyspec': 'EH_AES_GCM_128', 'origin': 'EH_INTERNAL_KEY'}
    primary_key_cipher_text = post_request(ip, port, action, payload)['keyid']
    return primary_key_cipher_text

def request_data_key_cipher_text(ip, port, encrypted_primary_key, data_key_length):
    if False:
        i = 10
        return i + 15
    action = 'GenerateDataKeyWithoutPlaintext'
    payload = {'keyid': encrypted_primary_key, 'keylen': data_key_length, 'aad': 'test'}
    data_key_cipher_text = post_request(ip, port, action, payload)['ciphertext']
    return data_key_cipher_text

def request_data_key_plaintext(ip, port, encrypted_primary_key, encrypted_data_key):
    if False:
        for i in range(10):
            print('nop')
    action = 'Decrypt'
    payload = {'keyid': encrypted_primary_key, 'ciphertext': encrypted_data_key, 'aad': 'test'}
    data_key_plaintext = post_request(ip, port, action, payload)['plaintext']
    return data_key_plaintext