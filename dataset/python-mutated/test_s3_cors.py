import os
import pytest
import requests
import xmltodict
from botocore.exceptions import ClientError
from localstack import config
from localstack.aws.handlers.cors import ALLOWED_CORS_ORIGINS
from localstack.config import S3_VIRTUAL_HOSTNAME
from localstack.constants import AWS_REGION_US_EAST_1, LOCALHOST_HOSTNAME, TEST_AWS_ACCESS_KEY_ID, TEST_AWS_REGION_NAME
from localstack.testing.pytest import markers
from localstack.utils.aws import aws_stack
from localstack.utils.strings import short_uid

def _bucket_url_vhost(bucket_name: str, region: str='', localstack_host: str=None) -> str:
    if False:
        while True:
            i = 10
    if not region:
        region = AWS_REGION_US_EAST_1
    if os.environ.get('TEST_TARGET') == 'AWS_CLOUD':
        if region == 'us-east-1':
            return f'https://{bucket_name}.s3.amazonaws.com'
        else:
            return f'https://{bucket_name}.s3.{region}.amazonaws.com'
    host = localstack_host or (f's3.{region}.{LOCALHOST_HOSTNAME}' if region != 'us-east-1' else S3_VIRTUAL_HOSTNAME)
    s3_edge_url = config.external_service_url(host=host)
    return s3_edge_url.replace(f'://{host}', f'://{bucket_name}.{host}')

@pytest.fixture
def snapshot_headers(snapshot):
    if False:
        print('Hello World!')
    snapshot.add_transformer([snapshot.transform.key_value('x-amz-id-2'), snapshot.transform.key_value('x-amz-request-id'), snapshot.transform.key_value('date', reference_replacement=False), snapshot.transform.key_value('Last-Modified', reference_replacement=False), snapshot.transform.key_value('server')])

@pytest.fixture
def match_headers(snapshot, snapshot_headers):
    if False:
        for i in range(10):
            print('nop')

    def _match(key: str, response: requests.Response):
        if False:
            return 10
        lower_case_headers = {'Date', 'Server', 'Accept-Ranges'}
        headers = {k if k not in lower_case_headers else k.lower(): v for (k, v) in dict(response.headers).items()}
        match_object = {'StatusCode': response.status_code, 'Headers': headers}
        if response.headers.get('Content-Type') in ('application/xml', 'text/xml') and response.content:
            match_object['Body'] = xmltodict.parse(response.content)
        else:
            match_object['Body'] = response.text
        snapshot.match(key, match_object)
    return _match

@pytest.fixture(autouse=True)
def allow_bucket_acl(s3_bucket, aws_client):
    if False:
        for i in range(10):
            print('nop')
    '\n    # Since April 2023, AWS will by default block setting ACL to your bucket and object. You need to manually disable\n    # the BucketOwnershipControls and PublicAccessBlock to make your objects public.\n    # See https://aws.amazon.com/about-aws/whats-new/2022/12/amazon-s3-automatically-enable-block-public-access-disable-access-control-lists-buckets-april-2023/\n    '
    aws_client.s3.delete_bucket_ownership_controls(Bucket=s3_bucket)
    aws_client.s3.delete_public_access_block(Bucket=s3_bucket)

@markers.snapshot.skip_snapshot_verify(paths=['$..x-amz-id-2'])
class TestS3Cors:

    @markers.aws.validated
    def test_cors_http_options_no_config(self, s3_bucket, snapshot, aws_client, allow_bucket_acl):
        if False:
            return 10
        snapshot.add_transformer([snapshot.transform.key_value('HostId', reference_replacement=False), snapshot.transform.key_value('RequestId')])
        key = 'test-cors-options-no-config'
        body = 'cors-test'
        response = aws_client.s3.put_object(Bucket=s3_bucket, Key=key, Body=body, ACL='public-read')
        assert response['ResponseMetadata']['HTTPStatusCode'] == 200
        key_url = f'{_bucket_url_vhost(bucket_name=s3_bucket)}/{key}'
        response = requests.options(key_url)
        assert response.status_code == 400
        parsed_response = xmltodict.parse(response.content)
        snapshot.match('options-no-origin', parsed_response)
        response = requests.options(key_url, headers={'Origin': 'whatever', 'Access-Control-Request-Method': 'PUT'})
        assert response.status_code == 403
        parsed_response = xmltodict.parse(response.content)
        snapshot.match('options-with-origin-and-method', parsed_response)
        response = requests.options(key_url, headers={'Origin': 'whatever'})
        assert response.status_code == 403
        parsed_response = xmltodict.parse(response.content)
        snapshot.match('options-with-origin-no-method', parsed_response)

    @markers.aws.validated
    def test_cors_http_get_no_config(self, s3_bucket, snapshot, aws_client):
        if False:
            i = 10
            return i + 15
        snapshot.add_transformer([snapshot.transform.key_value('HostId', reference_replacement=False), snapshot.transform.key_value('RequestId')])
        key = 'test-cors-get-no-config'
        body = 'cors-test'
        response = aws_client.s3.put_object(Bucket=s3_bucket, Key=key, Body=body, ACL='public-read')
        assert response['ResponseMetadata']['HTTPStatusCode'] == 200
        key_url = f'{_bucket_url_vhost(bucket_name=s3_bucket)}/{key}'
        response = requests.get(key_url)
        assert response.status_code == 200
        assert response.text == body
        assert not any(('access-control' in header.lower() for header in response.headers))
        response = requests.get(key_url, headers={'Origin': 'whatever'})
        assert response.status_code == 200
        assert response.text == body
        assert not any(('access-control' in header.lower() for header in response.headers))

    @markers.aws.only_localstack
    def test_cors_no_config_localstack_allowed(self, s3_bucket, aws_client):
        if False:
            while True:
                i = 10
        key = 'test-cors-get-no-config'
        body = 'cors-test'
        response = aws_client.s3.put_object(Bucket=s3_bucket, Key=key, Body=body, ACL='public-read')
        assert response['ResponseMetadata']['HTTPStatusCode'] == 200
        key_url = f'{_bucket_url_vhost(bucket_name=s3_bucket)}/{key}'
        origin = ALLOWED_CORS_ORIGINS[0]
        response = requests.options(key_url, headers={'Origin': origin, 'Access-Control-Request-Method': 'PUT'})
        assert response.ok
        assert response.headers['Access-Control-Allow-Origin'] == origin
        response = requests.get(key_url, headers={'Origin': origin})
        assert response.status_code == 200
        assert response.text == body
        assert response.headers['Access-Control-Allow-Origin'] == origin

    @markers.aws.only_localstack
    def test_cors_list_buckets(self):
        if False:
            return 10
        url = f'{config.internal_service_url()}/'
        origin = ALLOWED_CORS_ORIGINS[0]
        headers = aws_stack.mock_aws_request_headers('s3', aws_access_key_id=TEST_AWS_ACCESS_KEY_ID, region_name=TEST_AWS_REGION_NAME)
        headers['Origin'] = origin
        response = requests.options(url, headers={**headers, 'Access-Control-Request-Method': 'GET'})
        assert response.ok
        assert response.headers['Access-Control-Allow-Origin'] == origin
        response = requests.get(url, headers=headers)
        assert response.status_code == 200
        assert response.headers['Access-Control-Allow-Origin'] == origin
        assert b'<ListAllMyBuckets' in response.content

    @markers.aws.validated
    def test_cors_http_options_non_existent_bucket(self, s3_bucket, snapshot):
        if False:
            return 10
        snapshot.add_transformer([snapshot.transform.key_value('HostId', reference_replacement=False), snapshot.transform.key_value('RequestId')])
        key = 'test-cors-options-no-bucket'
        key_url = f"{_bucket_url_vhost(bucket_name=f'fake-bucket-{short_uid()}-{short_uid()}')}/{key}"
        response = requests.options(key_url)
        assert response.status_code == 400
        parsed_response = xmltodict.parse(response.content)
        snapshot.match('options-no-origin', parsed_response)
        response = requests.options(key_url, headers={'Origin': 'whatever'})
        assert response.status_code == 403
        parsed_response = xmltodict.parse(response.content)
        snapshot.match('options-with-origin', parsed_response)

    @markers.aws.only_localstack
    def test_cors_http_options_non_existent_bucket_ls_allowed(self, s3_bucket):
        if False:
            while True:
                i = 10
        key = 'test-cors-options-no-bucket'
        key_url = f"{_bucket_url_vhost(bucket_name=f'fake-bucket-{short_uid()}')}/{key}"
        origin = ALLOWED_CORS_ORIGINS[0]
        response = requests.options(key_url, headers={'Origin': origin})
        assert response.ok
        assert response.headers['Access-Control-Allow-Origin'] == origin

    @markers.aws.validated
    @markers.snapshot.skip_snapshot_verify(paths=['$..Body.Error.HostId', '$..Body.Error.RequestId', '$..Headers.Connection', '$..Headers.Content-Length', '$..Headers.Transfer-Encoding'])
    @markers.snapshot.skip_snapshot_verify(condition=lambda : config.LEGACY_V2_S3_PROVIDER, paths=['$..Headers.x-amz-server-side-encryption'])
    def test_cors_match_origins(self, s3_bucket, match_headers, aws_client, allow_bucket_acl):
        if False:
            return 10
        bucket_cors_config = {'CORSRules': [{'AllowedOrigins': ['https://localhost:4200'], 'AllowedMethods': ['GET', 'PUT'], 'MaxAgeSeconds': 3000, 'AllowedHeaders': ['*']}]}
        object_key = 'test-cors-123'
        response = aws_client.s3.put_object(Bucket=s3_bucket, Key=object_key, Body='test-cors', ACL='public-read')
        assert response['ResponseMetadata']['HTTPStatusCode'] == 200
        aws_client.s3.put_bucket_cors(Bucket=s3_bucket, CORSConfiguration=bucket_cors_config)
        key_url = f'{_bucket_url_vhost(bucket_name=s3_bucket)}/{object_key}'
        opt_req = requests.options(key_url)
        match_headers('opt-no-origin', opt_req)
        get_req = requests.get(key_url)
        match_headers('get-no-origin', get_req)
        opt_req = requests.options(key_url, headers={'referer': 'https://localhost:4200', 'Access-Control-Request-Method': 'PUT'})
        match_headers('opt-referer', opt_req)
        get_req = requests.get(key_url, headers={'referer': 'https://localhost:4200'})
        match_headers('get-referer', get_req)
        opt_req = requests.options(key_url, headers={'Origin': 'https://localhost:4200', 'Access-Control-Request-Method': 'PUT'})
        match_headers('opt-right-origin', opt_req)
        get_req = requests.get(key_url, headers={'Origin': 'https://localhost:4200'})
        match_headers('get-right-origin', get_req)
        opt_req = requests.options(key_url, headers={'Origin': 'http://localhost:4200', 'Access-Control-Request-Method': 'PUT'})
        match_headers('opt-wrong-origin', opt_req)
        get_req = requests.get(key_url, headers={'Origin': 'http://localhost:4200'})
        match_headers('get-wrong-origin', get_req)
        bucket_cors_config = {'CORSRules': [{'AllowedOrigins': ['*'], 'AllowedMethods': ['GET', 'PUT'], 'MaxAgeSeconds': 3000, 'AllowedHeaders': ['*']}]}
        aws_client.s3.put_bucket_cors(Bucket=s3_bucket, CORSConfiguration=bucket_cors_config)
        opt_req = requests.options(key_url, headers={'Origin': 'http://random:1234', 'Access-Control-Request-Method': 'PUT'})
        match_headers('opt-random-wildcard-origin', opt_req)
        get_req = requests.get(key_url, headers={'Origin': 'http://random:1234'})
        match_headers('get-random-wildcard-origin', get_req)

    @markers.aws.validated
    @markers.snapshot.skip_snapshot_verify(paths=['$..Body.Error.HostId', '$..Body.Error.RequestId', '$..Headers.Connection', '$..Headers.Content-Length', '$..Headers.Transfer-Encoding', '$.put-op.Body', '$.put-op.Headers.Content-Type'])
    @markers.snapshot.skip_snapshot_verify(condition=lambda : config.LEGACY_V2_S3_PROVIDER, paths=['$..Headers.x-amz-server-side-encryption'])
    def test_cors_match_methods(self, s3_bucket, match_headers, aws_client, allow_bucket_acl):
        if False:
            i = 10
            return i + 15
        origin = 'https://localhost:4200'
        bucket_cors_config = {'CORSRules': [{'AllowedOrigins': [origin], 'AllowedMethods': ['GET'], 'MaxAgeSeconds': 3000, 'AllowedHeaders': ['*']}]}
        object_key = 'test-cors-method'
        aws_client.s3.put_bucket_acl(Bucket=s3_bucket, ACL='public-read-write')
        response = aws_client.s3.put_object(Bucket=s3_bucket, Key=object_key, Body='test-cors', ACL='public-read')
        assert response['ResponseMetadata']['HTTPStatusCode'] == 200
        aws_client.s3.put_bucket_cors(Bucket=s3_bucket, CORSConfiguration=bucket_cors_config)
        key_url = f'{_bucket_url_vhost(bucket_name=s3_bucket)}/{object_key}'
        opt_req = requests.options(key_url, headers={'Origin': origin, 'Access-Control-Request-Method': 'GET'})
        match_headers('opt-get', opt_req)
        get_req = requests.get(key_url, headers={'Origin': origin, 'Access-Control-Request-Method': 'PUT'})
        match_headers('get-wrong-op', get_req)
        get_req = requests.get(key_url, headers={'Origin': origin})
        match_headers('get-op', get_req)
        new_key_url = f'{_bucket_url_vhost(bucket_name=s3_bucket)}/{object_key}new'
        opt_req = requests.options(new_key_url, headers={'Origin': origin, 'Access-Control-Request-Method': 'PUT'})
        match_headers('opt-put', opt_req)
        get_req = requests.put(new_key_url, headers={'Origin': origin})
        match_headers('put-op', get_req)

    @markers.aws.validated
    @markers.snapshot.skip_snapshot_verify(paths=['$..Body.Error.HostId', '$..Body.Error.RequestId', '$..Headers.Connection', '$..Headers.Content-Length', '$..Headers.Transfer-Encoding', '$.put-op.Body', '$.put-op.Headers.Content-Type'])
    @markers.snapshot.skip_snapshot_verify(condition=lambda : config.LEGACY_V2_S3_PROVIDER, paths=['$..Headers.x-amz-server-side-encryption'])
    def test_cors_match_headers(self, s3_bucket, match_headers, aws_client, allow_bucket_acl):
        if False:
            print('Hello World!')
        origin = 'https://localhost:4200'
        bucket_cors_config = {'CORSRules': [{'AllowedOrigins': [origin], 'AllowedMethods': ['GET'], 'MaxAgeSeconds': 3000, 'AllowedHeaders': ['*']}]}
        aws_client.s3.put_bucket_acl(Bucket=s3_bucket, ACL='public-read-write')
        object_key = 'test-cors-method'
        response = aws_client.s3.put_object(Bucket=s3_bucket, Key=object_key, Body='test-cors', ACL='public-read')
        assert response['ResponseMetadata']['HTTPStatusCode'] == 200
        aws_client.s3.put_bucket_cors(Bucket=s3_bucket, CORSConfiguration=bucket_cors_config)
        key_url = f'{_bucket_url_vhost(bucket_name=s3_bucket)}/{object_key}'
        opt_req = requests.options(key_url, headers={'Origin': origin, 'Access-Control-Request-Method': 'GET', 'Access-Control-Request-Headers': 'x-amz-request-payer'})
        match_headers('opt-get', opt_req)
        opt_req = requests.options(key_url, headers={'Origin': origin, 'Access-Control-Request-Method': 'GET', 'Access-Control-Request-Headers': 'x-amz-request-payer, x-amz-expected-bucket-owner'})
        match_headers('opt-get-two', opt_req)
        get_req = requests.get(key_url, headers={'Origin': origin, 'x-amz-request-payer': 'requester'})
        match_headers('get-op', get_req)
        bucket_cors_config = {'CORSRules': [{'AllowedOrigins': [origin], 'AllowedMethods': ['GET'], 'MaxAgeSeconds': 3000, 'AllowedHeaders': ['x-amz-expected-bucket-owner', 'x-amz-server-side-encryption-customer-algorithm']}]}
        aws_client.s3.put_bucket_cors(Bucket=s3_bucket, CORSConfiguration=bucket_cors_config)
        opt_req = requests.options(key_url, headers={'Origin': origin, 'Access-Control-Request-Method': 'GET', 'Access-Control-Request-Headers': 'x-amz-request-payer'})
        match_headers('opt-get-non-allowed', opt_req)
        assert opt_req.status_code == 403
        opt_req = requests.options(key_url, headers={'Origin': origin, 'Access-Control-Request-Method': 'GET', 'Access-Control-Request-Headers': 'x-amz-expected-bucket-owner'})
        match_headers('opt-get-allowed', opt_req)
        assert opt_req.ok
        get_req = requests.get(key_url, headers={'Origin': origin, 'Access-Control-Request-Headers': 'x-amz-request-payer'})
        match_headers('get-non-allowed-with-acl', get_req)
        get_req = requests.get(key_url, headers={'Origin': origin, 'x-amz-request-payer': 'requester'})
        match_headers('get-non-allowed', get_req)

    @markers.aws.validated
    @markers.snapshot.skip_snapshot_verify(paths=['$.opt-get.Headers.Content-Type'])
    def test_cors_expose_headers(self, s3_bucket, match_headers, aws_client, allow_bucket_acl):
        if False:
            i = 10
            return i + 15
        object_key = 'test-cors-expose'
        aws_client.s3.put_bucket_acl(Bucket=s3_bucket, ACL='public-read-write')
        response = aws_client.s3.put_object(Bucket=s3_bucket, Key=object_key, Body='test-cors', ACL='public-read')
        assert response['ResponseMetadata']['HTTPStatusCode'] == 200
        bucket_cors_config = {'CORSRules': [{'AllowedOrigins': ['*'], 'AllowedMethods': ['GET'], 'ExposeHeaders': ['x-amz-id-2', 'x-amz-request-id', 'x-amz-request-payer']}]}
        aws_client.s3.put_bucket_cors(Bucket=s3_bucket, CORSConfiguration=bucket_cors_config)
        key_url = f'{_bucket_url_vhost(bucket_name=s3_bucket)}/{object_key}'
        opt_req = requests.options(key_url, headers={'Origin': 'localhost:4566', 'Access-Control-Request-Method': 'GET'})
        match_headers('opt-get', opt_req)

    @markers.aws.validated
    def test_get_cors(self, s3_bucket, snapshot, aws_client):
        if False:
            print('Hello World!')
        snapshot.add_transformer(snapshot.transform.key_value('BucketName'))
        with pytest.raises(ClientError) as e:
            aws_client.s3.get_bucket_cors(Bucket=s3_bucket)
        snapshot.match('get-cors-no-set', e.value.response)
        bucket_cors_config = {'CORSRules': [{'AllowedOrigins': ['*'], 'AllowedMethods': ['GET']}]}
        aws_client.s3.put_bucket_cors(Bucket=s3_bucket, CORSConfiguration=bucket_cors_config)
        response = aws_client.s3.get_bucket_cors(Bucket=s3_bucket)
        snapshot.match('get-cors-after-set', response)

    @markers.aws.validated
    def test_put_cors(self, s3_bucket, snapshot, aws_client):
        if False:
            i = 10
            return i + 15
        bucket_cors_config = {'CORSRules': [{'AllowedOrigins': ['https://test.com', 'https://app.test.com', 'http://test.com:80'], 'AllowedMethods': ['GET', 'PUT', 'HEAD'], 'MaxAgeSeconds': 3000, 'AllowedHeaders': ['x-amz-expected-bucket-owner', 'x-amz-server-side-encryption-customer-algorithm']}]}
        put_response = aws_client.s3.put_bucket_cors(Bucket=s3_bucket, CORSConfiguration=bucket_cors_config)
        snapshot.match('put-cors', put_response)
        response = aws_client.s3.get_bucket_cors(Bucket=s3_bucket)
        snapshot.match('get-cors', response)

    @markers.aws.validated
    @markers.snapshot.skip_snapshot_verify(paths=['$..Body.Error.HostId', '$..Body.Error.RequestId', '$..Headers.Content-Length', '$..Headers.Transfer-Encoding'])
    def test_put_cors_default_values(self, s3_bucket, match_headers, aws_client, allow_bucket_acl):
        if False:
            i = 10
            return i + 15
        aws_client.s3.put_bucket_acl(Bucket=s3_bucket, ACL='public-read-write')
        object_key = 'test-cors-default'
        response = aws_client.s3.put_object(Bucket=s3_bucket, Key=object_key, Body='test-cors', ACL='public-read')
        assert response['ResponseMetadata']['HTTPStatusCode'] == 200
        bucket_cors_config = {'CORSRules': [{'AllowedOrigins': ['*'], 'AllowedMethods': ['GET']}]}
        aws_client.s3.put_bucket_cors(Bucket=s3_bucket, CORSConfiguration=bucket_cors_config)
        key_url = f'{_bucket_url_vhost(bucket_name=s3_bucket)}/{object_key}'
        opt_req = requests.options(key_url, headers={'Origin': 'localhost:4566', 'Access-Control-Request-Method': 'GET'})
        match_headers('opt-get', opt_req)
        opt_req = requests.options(key_url, headers={'Origin': 'localhost:4566', 'Access-Control-Request-Method': 'GET', 'Access-Control-Request-Headers': 'x-amz-request-payer'})
        match_headers('opt-get-headers', opt_req)

    @markers.aws.validated
    def test_put_cors_invalid_rules(self, s3_bucket, snapshot, aws_client):
        if False:
            print('Hello World!')
        bucket_cors_config = {'CORSRules': [{'AllowedOrigins': ['*', 'https://test.com'], 'AllowedMethods': ['GET', 'PUT', 'HEAD', 'MYMETHOD']}]}
        with pytest.raises(ClientError) as e:
            aws_client.s3.put_bucket_cors(Bucket=s3_bucket, CORSConfiguration=bucket_cors_config)
        snapshot.match('put-cors-exc', e.value.response)
        with pytest.raises(ClientError) as e:
            aws_client.s3.put_bucket_cors(Bucket=s3_bucket, CORSConfiguration={'CORSRules': []})
        snapshot.match('put-cors-exc-empty', e.value.response)

    @markers.aws.validated
    def test_put_cors_empty_origin(self, s3_bucket, snapshot, aws_client):
        if False:
            for i in range(10):
                print('nop')
        bucket_cors_config = {'CORSRules': [{'AllowedOrigins': [''], 'AllowedMethods': ['GET', 'PUT', 'HEAD']}]}
        aws_client.s3.put_bucket_cors(Bucket=s3_bucket, CORSConfiguration=bucket_cors_config)
        response = aws_client.s3.get_bucket_cors(Bucket=s3_bucket)
        snapshot.match('get-cors-empty', response)

    @markers.aws.validated
    def test_delete_cors(self, s3_bucket, snapshot, aws_client):
        if False:
            for i in range(10):
                print('nop')
        snapshot.add_transformer(snapshot.transform.key_value('BucketName'))
        response = aws_client.s3.delete_bucket_cors(Bucket=s3_bucket)
        snapshot.match('delete-cors-before-set', response)
        bucket_cors_config = {'CORSRules': [{'AllowedOrigins': ['*'], 'AllowedMethods': ['GET']}]}
        put_response = aws_client.s3.put_bucket_cors(Bucket=s3_bucket, CORSConfiguration=bucket_cors_config)
        snapshot.match('put-cors', put_response)
        response = aws_client.s3.get_bucket_cors(Bucket=s3_bucket)
        snapshot.match('get-cors', response)
        response = aws_client.s3.delete_bucket_cors(Bucket=s3_bucket)
        snapshot.match('delete-cors', response)
        with pytest.raises(ClientError) as e:
            aws_client.s3.get_bucket_cors(Bucket=s3_bucket)
        snapshot.match('get-cors-deleted', e.value.response)