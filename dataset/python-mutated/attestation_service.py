import requests
import json
import base64
from collections import OrderedDict
use_secure_cert = False

def bigdl_attestation_service(base_url, app_id, api_key, quote, policy_id):
    if False:
        return 10
    headers = {'Content-Type': 'application/json'}
    payload = OrderedDict()
    payload['appID'] = app_id
    payload['apiKey'] = api_key
    payload['quote'] = base64.b64encode(quote).decode()
    if len(policy_id) > 0:
        payload['policyID'] = policy_id
    try:
        resp = requests.post(url='https://' + base_url + '/verifyQuote', data=json.dumps(payload), headers=headers, verify=use_secure_cert)
        resp_dict = json.loads(resp.text)
        result = resp_dict['result']
    except (json.JSONDecodeError, KeyError):
        result = -1
    return result

def bigdl_attestation_service_register(base_url, app_id, api_key, policy_type, mr_enclave, mr_signer):
    if False:
        for i in range(10):
            print('nop')
    headers = {'Content-Type': 'application/json'}
    payload = OrderedDict()
    payload['appID'] = app_id
    payload['apiKey'] = api_key
    payload['policyType'] = policy_type
    payload['mrEnclave'] = mr_enclave
    payload['mrSigner'] = mr_signer
    try:
        resp = requests.post(url='https://' + base_url + '/registerPolicy', data=json.dumps(payload), headers=headers, verify=use_secure_cert)
        resp_dict = json.loads(resp.text)
        result = resp_dict['policyID']
    except (json.JSONDecodeError, KeyError):
        result = -1
    return result

def amber(base_url, api_key, quote, policy_id, proxies):
    if False:
        i = 10
        return i + 15
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json', 'x-api-key': api_key}
    payload = OrderedDict()
    payload['quote'] = base64.b64encode(quote).decode()
    if len(policy_id) > 0:
        payload['policy_ids'] = policy_id
    try:
        resp = requests.post(url=base_url + '/appraisal/v1/attest', data=json.dumps(payload), headers=headers, verify=use_secure_cert, proxies=proxies)
        resp_dict = json.loads(resp.text)
        if len(resp_dict['token']) > 0:
            result = 0
        else:
            result = -1
    except (json.JSONDecodeError, KeyError):
        result = -1
    return result