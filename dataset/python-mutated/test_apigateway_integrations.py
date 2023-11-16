import contextlib
import json
import textwrap
from urllib.parse import urlparse
import pytest
import requests
from botocore.exceptions import ClientError
from pytest_httpserver import HTTPServer
from werkzeug import Request, Response
from localstack import config
from localstack.constants import APPLICATION_JSON, TEST_AWS_ACCOUNT_ID
from localstack.services.apigateway.helpers import path_based_url
from localstack.services.lambda_.networking import get_main_endpoint_from_container
from localstack.testing.aws.util import is_aws_cloud
from localstack.testing.pytest import markers
from localstack.testing.pytest.fixtures import PUBLIC_HTTP_ECHO_SERVER_URL
from localstack.utils.strings import short_uid, to_bytes, to_str
from localstack.utils.sync import retry
from tests.aws.services.apigateway.apigateway_fixtures import api_invoke_url, create_rest_api_deployment
from tests.aws.services.apigateway.conftest import DEFAULT_STAGE_NAME
from tests.aws.services.lambda_.test_lambda import TEST_LAMBDA_LIBS

@markers.aws.unknown
def test_http_integration(create_rest_apigw, aws_client, echo_http_server):
    if False:
        for i in range(10):
            print('nop')
    (api_id, _, root_id) = create_rest_apigw(name='my_api', description='this is my api')
    aws_client.apigateway.put_method(restApiId=api_id, resourceId=root_id, httpMethod='GET', authorizationType='none')
    aws_client.apigateway.put_method_response(restApiId=api_id, resourceId=root_id, httpMethod='GET', statusCode='200')
    aws_client.apigateway.put_integration(restApiId=api_id, resourceId=root_id, httpMethod='GET', type='HTTP', uri=echo_http_server, integrationHttpMethod='GET')
    stage_name = 'staging'
    aws_client.apigateway.create_deployment(restApiId=api_id, stageName=stage_name)
    url = path_based_url(api_id=api_id, stage_name=stage_name, path='/')
    response = requests.get(url)
    assert response.status_code == 200

@pytest.fixture
def status_code_http_server(httpserver: HTTPServer):
    if False:
        i = 10
        return i + 15
    'Spins up a local HTTP echo server and returns the endpoint URL'
    if is_aws_cloud():
        return f'{PUBLIC_HTTP_ECHO_SERVER_URL}/'

    def _echo(request: Request) -> Response:
        if False:
            i = 10
            return i + 15
        result = {'data': request.data or '{}', 'headers': dict(request.headers), 'url': request.url, 'method': request.method}
        status_code = request.url.rpartition('/')[2]
        response_body = json.dumps(result)
        return Response(response_body, status=int(status_code))
    httpserver.expect_request('').respond_with_handler(_echo)
    http_endpoint = httpserver.url_for('/')
    return http_endpoint

@markers.aws.validated
def test_http_integration_status_code_selection(create_rest_apigw, aws_client, status_code_http_server):
    if False:
        while True:
            i = 10
    (api_id, _, root_id) = create_rest_apigw(name='my_api', description='this is my api')
    resource_id = aws_client.apigateway.create_resource(restApiId=api_id, parentId=root_id, pathPart='{status}')['id']
    aws_client.apigateway.put_method(restApiId=api_id, resourceId=resource_id, httpMethod='GET', authorizationType='none', requestParameters={'method.request.path.status': True})
    aws_client.apigateway.put_integration(restApiId=api_id, resourceId=resource_id, httpMethod='GET', type='HTTP', uri=f'{status_code_http_server}status/{{status}}', requestParameters={'integration.request.path.status': 'method.request.path.status'}, integrationHttpMethod='GET')
    aws_client.apigateway.put_method_response(restApiId=api_id, resourceId=resource_id, statusCode='200', httpMethod='GET')
    aws_client.apigateway.put_integration_response(restApiId=api_id, resourceId=resource_id, statusCode='200', httpMethod='GET')
    aws_client.apigateway.put_method_response(restApiId=api_id, resourceId=resource_id, statusCode='400', httpMethod='GET')
    aws_client.apigateway.put_integration_response(restApiId=api_id, resourceId=resource_id, statusCode='400', httpMethod='GET', selectionPattern='4\\d{2}')
    stage_name = 'test'
    aws_client.apigateway.create_deployment(restApiId=api_id, stageName=stage_name)
    invocation_url = api_invoke_url(api_id=api_id, stage=stage_name, path='/')

    def invoke_api(url, requested_response_code: int, expected_response_code: int):
        if False:
            return 10
        apigw_response = requests.get(f'{url}{requested_response_code}', headers={'User-Agent': 'python-requests/testing'}, verify=False)
        assert expected_response_code == apigw_response.status_code
        return apigw_response
    retry(invoke_api, sleep=2, retries=10, url=invocation_url, expected_response_code=400, requested_response_code=404)
    retry(invoke_api, sleep=2, retries=10, url=invocation_url, expected_response_code=200, requested_response_code=201)

@markers.aws.validated
def test_put_integration_responses(create_rest_apigw, aws_client, echo_http_server_post, snapshot):
    if False:
        return 10
    snapshot.add_transformers_list([snapshot.transform.key_value('cacheNamespace'), snapshot.transform.key_value('uri'), snapshot.transform.key_value('id')])
    (api_id, _, root_id) = create_rest_apigw(name='my_api', description='this is my api')
    response = aws_client.apigateway.put_method(restApiId=api_id, resourceId=root_id, httpMethod='GET', authorizationType='NONE')
    snapshot.match('put-method-get', response)
    response = aws_client.apigateway.put_method_response(restApiId=api_id, resourceId=root_id, httpMethod='GET', statusCode='200')
    snapshot.match('put-method-response-get', response)
    response = aws_client.apigateway.put_integration(restApiId=api_id, resourceId=root_id, httpMethod='GET', type='HTTP', uri=echo_http_server_post, integrationHttpMethod='POST')
    snapshot.match('put-integration-get', response)
    response = aws_client.apigateway.put_integration_response(restApiId=api_id, resourceId=root_id, httpMethod='GET', statusCode='200', selectionPattern='2\\d{2}', responseTemplates={})
    snapshot.match('put-integration-response-get', response)
    response = aws_client.apigateway.get_integration_response(restApiId=api_id, resourceId=root_id, httpMethod='GET', statusCode='200')
    snapshot.match('get-integration-response-get', response)
    response = aws_client.apigateway.get_method(restApiId=api_id, resourceId=root_id, httpMethod='GET')
    snapshot.match('get-method-get', response)
    stage_name = 'local'
    response = aws_client.apigateway.create_deployment(restApiId=api_id, stageName=stage_name)
    snapshot.match('deploy', response)
    url = api_invoke_url(api_id, stage=stage_name, path='/')
    response = requests.get(url)
    assert response.ok
    response = aws_client.apigateway.delete_integration_response(restApiId=api_id, resourceId=root_id, httpMethod='GET', statusCode='200')
    snapshot.match('delete-integration-response-get', response)
    response = aws_client.apigateway.get_method(restApiId=api_id, resourceId=root_id, httpMethod='GET')
    snapshot.match('get-method-get-after-int-resp-delete', response)
    response = aws_client.apigateway.put_method(restApiId=api_id, resourceId=root_id, httpMethod='PUT', authorizationType='none')
    snapshot.match('put-method-put', response)
    response = aws_client.apigateway.put_method_response(restApiId=api_id, resourceId=root_id, httpMethod='PUT', statusCode='200')
    snapshot.match('put-method-response-put', response)
    response = aws_client.apigateway.put_integration(restApiId=api_id, resourceId=root_id, httpMethod='PUT', type='HTTP', uri=echo_http_server_post, integrationHttpMethod='POST')
    snapshot.match('put-integration-put', response)
    response = aws_client.apigateway.put_integration_response(restApiId=api_id, resourceId=root_id, httpMethod='PUT', statusCode='200', selectionPattern='2\\d{2}', contentHandling='CONVERT_TO_BINARY')
    snapshot.match('put-integration-response-put', response)
    response = aws_client.apigateway.get_integration_response(restApiId=api_id, resourceId=root_id, httpMethod='PUT', statusCode='200')
    snapshot.match('get-integration-response-put', response)

@markers.aws.unknown
def test_put_integration_response_with_response_template(aws_client, echo_http_server_post):
    if False:
        print('Hello World!')
    response = aws_client.apigateway.create_rest_api(name='my_api', description='this is my api')
    api_id = response['id']
    resources = aws_client.apigateway.get_resources(restApiId=api_id)
    root_id = [resource for resource in resources['items'] if resource['path'] == '/'][0]['id']
    aws_client.apigateway.put_method(restApiId=api_id, resourceId=root_id, httpMethod='GET', authorizationType='NONE')
    aws_client.apigateway.put_method_response(restApiId=api_id, resourceId=root_id, httpMethod='GET', statusCode='200')
    aws_client.apigateway.put_integration(restApiId=api_id, resourceId=root_id, httpMethod='GET', type='HTTP', uri=echo_http_server_post, integrationHttpMethod='POST')
    aws_client.apigateway.put_integration_response(restApiId=api_id, resourceId=root_id, httpMethod='GET', statusCode='200', selectionPattern='foobar', responseTemplates={'application/json': json.dumps({'data': 'test'})})
    response = aws_client.apigateway.get_integration_response(restApiId=api_id, resourceId=root_id, httpMethod='GET', statusCode='200')
    response['ResponseMetadata'].pop('HTTPHeaders', None)
    response['ResponseMetadata'].pop('RetryAttempts', None)
    response['ResponseMetadata'].pop('RequestId', None)
    assert response == {'statusCode': '200', 'selectionPattern': 'foobar', 'ResponseMetadata': {'HTTPStatusCode': 200}, 'responseTemplates': {'application/json': json.dumps({'data': 'test'})}}

@markers.aws.unknown
def test_put_integration_validation(aws_client, echo_http_server, echo_http_server_post):
    if False:
        i = 10
        return i + 15
    response = aws_client.apigateway.create_rest_api(name='my_api', description='this is my api')
    api_id = response['id']
    resources = aws_client.apigateway.get_resources(restApiId=api_id)
    root_id = [resource for resource in resources['items'] if resource['path'] == '/'][0]['id']
    aws_client.apigateway.put_method(restApiId=api_id, resourceId=root_id, httpMethod='GET', authorizationType='NONE')
    aws_client.apigateway.put_method_response(restApiId=api_id, resourceId=root_id, httpMethod='GET', statusCode='200')
    http_types = ['HTTP', 'HTTP_PROXY']
    aws_types = ['AWS', 'AWS_PROXY']
    types_requiring_integration_method = http_types + ['AWS']
    types_not_requiring_integration_method = ['MOCK']
    for _type in types_requiring_integration_method:
        with pytest.raises(ClientError) as ex:
            aws_client.apigateway.put_integration(restApiId=api_id, resourceId=root_id, httpMethod='GET', type=_type, uri=echo_http_server)
        assert ex.value.response['Error']['Code'] == 'BadRequestException'
        assert ex.value.response['Error']['Message'] == 'Enumeration value for HttpMethod must be non-empty'
    for _type in types_not_requiring_integration_method:
        aws_client.apigateway.put_integration(restApiId=api_id, resourceId=root_id, httpMethod='GET', type=_type, uri=echo_http_server)
    for _type in http_types:
        aws_client.apigateway.put_integration(restApiId=api_id, resourceId=root_id, httpMethod='GET', type=_type, uri=echo_http_server_post, integrationHttpMethod='POST')
    for _type in ['AWS']:
        aws_client.apigateway.put_integration(restApiId=api_id, resourceId=root_id, credentials='arn:aws:iam::{}:role/service-role/testfunction-role-oe783psq'.format(TEST_AWS_ACCOUNT_ID), httpMethod='GET', type=_type, uri='arn:aws:apigateway:us-west-2:s3:path/b/k', integrationHttpMethod='POST')
    for _type in aws_types:
        aws_client.apigateway.put_integration(restApiId=api_id, resourceId=root_id, httpMethod='GET', type=_type, uri='arn:aws:apigateway:eu-west-1:lambda:path/2015-03-31/functions/arn:aws:lambda:eu-west-1:012345678901:function:MyLambda/invocations', integrationHttpMethod='POST')
    for _type in ['AWS_PROXY']:
        with pytest.raises(ClientError) as ex:
            aws_client.apigateway.put_integration(restApiId=api_id, resourceId=root_id, credentials='arn:aws:iam::{}:role/service-role/testfunction-role-oe783psq'.format(TEST_AWS_ACCOUNT_ID), httpMethod='GET', type=_type, uri='arn:aws:apigateway:us-west-2:s3:path/b/k', integrationHttpMethod='POST')
        assert ex.value.response['Error']['Code'] == 'BadRequestException'
        assert ex.value.response['Error']['Message'] == "Integrations of type 'AWS_PROXY' currently only supports Lambda function and Firehose stream invocations."
    for _type in http_types:
        with pytest.raises(ClientError) as ex:
            aws_client.apigateway.put_integration(restApiId=api_id, resourceId=root_id, httpMethod='GET', type=_type, uri='non-valid-http', integrationHttpMethod='POST')
        assert ex.value.response['Error']['Code'] == 'BadRequestException'
        assert ex.value.response['Error']['Message'] == 'Invalid HTTP endpoint specified for URI'
    with pytest.raises(ClientError) as ex:
        aws_client.apigateway.put_integration(restApiId=api_id, resourceId=root_id, httpMethod='GET', type='AWS', uri='non-valid-arn', integrationHttpMethod='POST')
    assert ex.value.response['Error']['Code'] == 'BadRequestException'
    assert ex.value.response['Error']['Message'] == 'Invalid ARN specified in the request'
    with pytest.raises(ClientError) as ex:
        aws_client.apigateway.put_integration(restApiId=api_id, resourceId=root_id, httpMethod='GET', type='AWS', uri='arn:aws:iam::0000000000:role/service-role/asdf', integrationHttpMethod='POST')
    assert ex.value.response['Error']['Code'] == 'BadRequestException'
    assert ex.value.response['Error']['Message'] == 'AWS ARN for integration must contain path or action'

@pytest.fixture
def default_vpc(aws_client):
    if False:
        for i in range(10):
            print('nop')
    vpcs = aws_client.ec2.describe_vpcs()
    for vpc in vpcs['Vpcs']:
        if vpc.get('IsDefault'):
            return vpc
    raise Exception('Default VPC not found')

@pytest.fixture
def create_vpc_endpoint(default_vpc, aws_client):
    if False:
        print('Hello World!')
    endpoints = []

    def _create(**kwargs):
        if False:
            while True:
                i = 10
        kwargs.setdefault('VpcId', default_vpc['VpcId'])
        result = aws_client.ec2.create_vpc_endpoint(**kwargs)
        endpoints.append(result['VpcEndpoint']['VpcEndpointId'])
        return result['VpcEndpoint']
    yield _create
    for endpoint in endpoints:
        with contextlib.suppress(Exception):
            aws_client.ec2.delete_vpc_endpoints(VpcEndpointIds=[endpoint])

@markers.snapshot.skip_snapshot_verify(paths=['$..endpointConfiguration.types', '$..policy.Statement..Resource'])
@markers.aws.validated
def test_create_execute_api_vpc_endpoint(create_rest_api_with_integration, dynamodb_create_table, create_vpc_endpoint, default_vpc, create_lambda_function, ec2_create_security_group, snapshot, aws_client):
    if False:
        return 10
    poll_sleep = 5 if is_aws_cloud() else 1
    snapshot.add_transformer(snapshot.transform.key_value('DnsName'))
    snapshot.add_transformer(snapshot.transform.key_value('GroupId'))
    snapshot.add_transformer(snapshot.transform.key_value('GroupName'))
    snapshot.add_transformer(snapshot.transform.key_value('SubnetIds'))
    snapshot.add_transformer(snapshot.transform.key_value('VpcId'))
    snapshot.add_transformer(snapshot.transform.key_value('VpcEndpointId'))
    snapshot.add_transformer(snapshot.transform.key_value('HostedZoneId'))
    snapshot.add_transformer(snapshot.transform.key_value('id'))
    snapshot.add_transformer(snapshot.transform.key_value('name'))
    table = dynamodb_create_table()['TableDescription']
    table_name = table['TableName']
    item_ids = ('test', 'test2', 'test 3')
    for item_id in item_ids:
        aws_client.dynamodb.put_item(TableName=table_name, Item={'id': {'S': item_id}})
    request_templates = {APPLICATION_JSON: json.dumps({'TableName': table_name})}
    region_name = aws_client.apigateway.meta.region_name
    integration_uri = f'arn:aws:apigateway:{region_name}:dynamodb:action/Scan'
    api_id = create_rest_api_with_integration(integration_uri=integration_uri, req_templates=request_templates, integration_type='AWS')
    service_name = f'com.amazonaws.{region_name}.execute-api'
    service_names = aws_client.ec2.describe_vpc_endpoint_services()['ServiceNames']
    assert service_name in service_names
    vpc_id = default_vpc['VpcId']
    security_group = ec2_create_security_group(VpcId=vpc_id, Description='Test SG for API GW', ports=[443])
    security_group = security_group['GroupId']
    subnets = aws_client.ec2.describe_subnets(Filters=[{'Name': 'vpc-id', 'Values': [vpc_id]}])
    subnets = [sub['SubnetId'] for sub in subnets['Subnets']]
    endpoints = aws_client.ec2.describe_vpc_endpoints(MaxResults=1000)['VpcEndpoints']
    matching = [ep for ep in endpoints if ep['ServiceName'] == service_name]
    if matching:
        endpoint_id = matching[0]['VpcEndpointId']
    else:
        result = create_vpc_endpoint(ServiceName=service_name, VpcEndpointType='Interface', SubnetIds=subnets, SecurityGroupIds=[security_group])
        endpoint_id = result['VpcEndpointId']

    def _check_available():
        if False:
            return 10
        result = aws_client.ec2.describe_vpc_endpoints(VpcEndpointIds=[endpoint_id])
        endpoint_details = result['VpcEndpoints'][0]
        endpoint_details['DnsEntries'] = endpoint_details['DnsEntries'][:1]
        endpoint_details.pop('SubnetIds', None)
        endpoint_details.pop('NetworkInterfaceIds', None)
        assert endpoint_details['State'] == 'available'
        snapshot.match('endpoint-details', endpoint_details)
    retry(_check_available, retries=30, sleep=poll_sleep)
    patches = [{'op': 'replace', 'path': '/endpointConfiguration/types/EDGE', 'value': 'PRIVATE'}, {'op': 'add', 'path': '/endpointConfiguration/vpcEndpointIds', 'value': endpoint_id}]
    aws_client.apigateway.update_rest_api(restApiId=api_id, patchOperations=patches)
    subdomain = f'{api_id}-{endpoint_id}'
    endpoint = api_invoke_url(subdomain, stage=DEFAULT_STAGE_NAME, path='/test')
    host_header = urlparse(endpoint).netloc
    if not is_aws_cloud():
        api_host = get_main_endpoint_from_container()
        endpoint = endpoint.replace(host_header, f'{api_host}:{config.GATEWAY_LISTEN[0].port}')
    lambda_code = textwrap.dedent(f'\n    def handler(event, context):\n        import requests\n        headers = {{"content-type": "application/json", "host": "{host_header}"}}\n        result = requests.post("{endpoint}", headers=headers)\n        return {{"content": result.content.decode("utf-8"), "code": result.status_code}}\n    ')
    func_name = f'test-{short_uid()}'
    vpc_config = {'SubnetIds': subnets, 'SecurityGroupIds': [security_group]}
    create_lambda_function(func_name=func_name, handler_file=lambda_code, libs=TEST_LAMBDA_LIBS, timeout=10, VpcConfig=vpc_config)
    statement = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Principal': '*', 'Action': 'execute-api:Invoke', 'Resource': ['execute-api:/*']}]}
    patches = [{'op': 'replace', 'path': '/policy', 'value': json.dumps(statement)}]
    result = aws_client.apigateway.update_rest_api(restApiId=api_id, patchOperations=patches)
    result['policy'] = json.loads(to_bytes(result['policy']).decode('unicode_escape'))
    snapshot.match('api-details', result)
    create_rest_api_deployment(aws_client.apigateway, restApiId=api_id, stageName=DEFAULT_STAGE_NAME)

    def _invoke_api():
        if False:
            while True:
                i = 10
        result = aws_client.lambda_.invoke(FunctionName=func_name, Payload='{}')
        result = json.loads(to_str(result['Payload'].read()))
        items = json.loads(result['content'])['Items']
        assert len(items) == len(item_ids)
    retry(_invoke_api, retries=15, sleep=poll_sleep)