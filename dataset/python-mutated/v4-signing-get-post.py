"""
Important

The AWS SDKs sign API requests for you using the access key that you specify when you
configure the SDK. When you use an SDK, you don’t need to learn how to sign API requests.
We recommend that you use the AWS SDKs to send API requests, instead of writing your own code.

The following example is a reference to help you get started if you have a need to write
your own code to send and sign requests. The example is for reference only and is not
maintained as functional code.
"""
import sys, os, base64, datetime, hashlib, hmac
import requests
method = 'POST'
service = 'dynamodb'
host = 'dynamodb.us-west-2.amazonaws.com'
region = 'us-west-2'
endpoint = 'https://dynamodb.us-west-2.amazonaws.com/'
content_type = 'application/x-amz-json-1.0'
amz_target = 'DynamoDB_20120810.CreateTable'
request_parameters = '{'
request_parameters += '"KeySchema": [{"KeyType": "HASH","AttributeName": "Id"}],'
request_parameters += '"TableName": "TestTable","AttributeDefinitions": [{"AttributeName": "Id","AttributeType": "S"}],'
request_parameters += '"ProvisionedThroughput": {"WriteCapacityUnits": 5,"ReadCapacityUnits": 5}'
request_parameters += '}'

def sign(key, msg):
    if False:
        i = 10
        return i + 15
    return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()

def getSignatureKey(key, date_stamp, regionName, serviceName):
    if False:
        for i in range(10):
            print('nop')
    kDate = sign(('AWS4' + key).encode('utf-8'), date_stamp)
    kRegion = sign(kDate, regionName)
    kService = sign(kRegion, serviceName)
    kSigning = sign(kService, 'aws4_request')
    return kSigning
access_key = os.environ.get('AWS_ACCESS_KEY_ID')
secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
if access_key is None or secret_key is None:
    print('No access key is available.')
    sys.exit()
t = datetime.datetime.utcnow()
amz_date = t.strftime('%Y%m%dT%H%M%SZ')
date_stamp = t.strftime('%Y%m%d')
canonical_uri = '/'
canonical_querystring = ''
canonical_headers = 'content-type:' + content_type + '\n' + 'host:' + host + '\n' + 'x-amz-date:' + amz_date + '\n' + 'x-amz-target:' + amz_target + '\n'
signed_headers = 'content-type;host;x-amz-date;x-amz-target'
payload_hash = hashlib.sha256(request_parameters.encode('utf-8')).hexdigest()
canonical_request = method + '\n' + canonical_uri + '\n' + canonical_querystring + '\n' + canonical_headers + '\n' + signed_headers + '\n' + payload_hash
algorithm = 'AWS4-HMAC-SHA256'
credential_scope = date_stamp + '/' + region + '/' + service + '/' + 'aws4_request'
string_to_sign = algorithm + '\n' + amz_date + '\n' + credential_scope + '\n' + hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()
signing_key = getSignatureKey(secret_key, date_stamp, region, service)
signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
authorization_header = algorithm + ' ' + 'Credential=' + access_key + '/' + credential_scope + ', ' + 'SignedHeaders=' + signed_headers + ', ' + 'Signature=' + signature
headers = {'Content-Type': content_type, 'X-Amz-Date': amz_date, 'X-Amz-Target': amz_target, 'Authorization': authorization_header}
print('\nBEGIN REQUEST++++++++++++++++++++++++++++++++++++')
print('Request URL = ' + endpoint)
r = requests.post(endpoint, data=request_parameters, headers=headers)
print('\nRESPONSE++++++++++++++++++++++++++++++++++++')
print('Response code: %d\n' % r.status_code)
print(r.text)