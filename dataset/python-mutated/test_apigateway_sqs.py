import base64
import json
import re
import requests
from localstack.constants import TEST_AWS_ACCOUNT_ID, TEST_AWS_REGION_NAME
from localstack.services.apigateway.helpers import connect_api_gateway_to_sqs, path_based_url
from localstack.testing.pytest import markers
from localstack.utils.aws import queries
from localstack.utils.strings import short_uid, to_str
from localstack.utils.sync import retry
from localstack.utils.xml import is_valid_xml
from tests.aws.services.apigateway.apigateway_fixtures import api_invoke_url
from tests.aws.services.apigateway.conftest import APIGATEWAY_ASSUME_ROLE_POLICY
from tests.aws.services.apigateway.test_apigateway_basic import TEST_STAGE_NAME

@markers.aws.unknown
def test_api_gateway_sqs_integration(aws_client, sqs_create_queue, sqs_get_queue_arn):
    if False:
        while True:
            i = 10
    queue_name = f'queue-{short_uid()}'
    sqs_create_queue(QueueName=queue_name)
    result = connect_api_gateway_to_sqs('test_gateway4', stage_name=TEST_STAGE_NAME, queue_arn=queue_name, path='/data', account_id=TEST_AWS_ACCOUNT_ID, region_name=TEST_AWS_REGION_NAME)
    test_data = {'spam': 'eggs'}
    url = path_based_url(api_id=result['id'], stage_name=TEST_STAGE_NAME, path='/data')
    result = requests.post(url, data=json.dumps(test_data))
    assert 200 == result.status_code
    queue_arn = sqs_get_queue_arn(queue_name)
    messages = queries.sqs_receive_message(queue_arn)['Messages']
    assert 1 == len(messages)
    assert test_data == json.loads(base64.b64decode(messages[0]['Body']))

@markers.aws.validated
def test_sqs_aws_integration(create_rest_apigw, sqs_create_queue, aws_client, create_role_with_policy, region, account_id, snapshot):
    if False:
        i = 10
        return i + 15
    queue_name = f'queue-{short_uid()}'
    sqs_create_queue(QueueName=queue_name)
    (_, role_arn) = create_role_with_policy('Allow', 'sqs:SendMessage', json.dumps(APIGATEWAY_ASSUME_ROLE_POLICY), '*')
    (api_id, _, root) = create_rest_apigw(name=f'test-api-${short_uid()}', description='Test Integration with SQS')
    resource_id = aws_client.apigateway.create_resource(restApiId=api_id, parentId=root, pathPart='sqs')['id']
    aws_client.apigateway.put_method(restApiId=api_id, resourceId=resource_id, httpMethod='POST', authorizationType='NONE')
    aws_client.apigateway.put_integration(restApiId=api_id, resourceId=resource_id, httpMethod='POST', type='AWS', integrationHttpMethod='POST', uri=f'arn:aws:apigateway:{region}:sqs:path/{account_id}/{queue_name}', credentials=role_arn, requestParameters={'integration.request.header.Content-Type': "'application/x-www-form-urlencoded'"}, requestTemplates={'application/json': 'Action=SendMessage&MessageBody=$input.body'}, passthroughBehavior='NEVER')
    aws_client.apigateway.put_method_response(restApiId=api_id, resourceId=resource_id, httpMethod='POST', statusCode='200', responseModels={'application/json': 'Empty'})
    aws_client.apigateway.put_integration_response(restApiId=api_id, resourceId=resource_id, httpMethod='POST', statusCode='200', responseTemplates={'application/json': '{"message": "great success!"}'})
    response = aws_client.apigateway.create_deployment(restApiId=api_id)
    deployment_id = response['id']
    aws_client.apigateway.create_stage(restApiId=api_id, stageName=TEST_STAGE_NAME, deploymentId=deployment_id)
    invocation_url = api_invoke_url(api_id=api_id, stage=TEST_STAGE_NAME, path='/sqs')

    def invoke_api(url):
        if False:
            return 10
        _response = requests.post(url, verify=False)
        assert _response.ok
        content = _response.json()
        assert content == {'message': 'great success!'}
        return content
    response_data = retry(invoke_api, sleep=2, retries=10, url=invocation_url)
    snapshot.match('sqs-aws-integration', response_data)

@markers.aws.validated
def test_sqs_request_and_response_xml_templates_integration(create_rest_apigw, sqs_create_queue, aws_client, create_role_with_policy, region, account_id, snapshot):
    if False:
        while True:
            i = 10
    queue_name = f'queue-{short_uid()}'
    sqs_create_queue(QueueName=queue_name)
    (_, role_arn) = create_role_with_policy('Allow', 'sqs:SendMessage', json.dumps(APIGATEWAY_ASSUME_ROLE_POLICY), '*')
    (api_id, _, root) = create_rest_apigw(name=f'test-api-${short_uid()}')
    resource_id = aws_client.apigateway.create_resource(restApiId=api_id, parentId=root, pathPart='sqs')['id']
    aws_client.apigateway.put_method(restApiId=api_id, resourceId=resource_id, httpMethod='POST', authorizationType='NONE')
    aws_client.apigateway.put_integration(restApiId=api_id, resourceId=resource_id, httpMethod='POST', type='AWS', integrationHttpMethod='POST', uri=f'arn:aws:apigateway:{region}:sqs:path/{account_id}/{queue_name}', credentials=role_arn, requestParameters={'integration.request.header.Content-Type': "'application/x-www-form-urlencoded'"}, requestTemplates={'application/json': 'Action=SendMessage&MessageBody=$input.body'}, passthroughBehavior='NEVER')
    aws_client.apigateway.put_method_response(restApiId=api_id, resourceId=resource_id, httpMethod='POST', statusCode='200', responseModels={'application/json': 'Empty'})
    aws_client.apigateway.put_integration_response(restApiId=api_id, resourceId=resource_id, httpMethod='POST', statusCode='200', responseTemplates={'application/json': '\n            #set($responseBody = $input.path(\'$.SendMessageResponse\'))\n            #set($requestId = $input.path(\'$.SendMessageResponse.ResponseMetadata.RequestId\'))\n            #set($messageId = $responseBody.SendMessageResult.MessageId)\n            {\n            "requestId": "$requestId",\n            "messageId": "$messageId"\n            }\n            '})
    response = aws_client.apigateway.create_deployment(restApiId=api_id)
    deployment_id = response['id']
    aws_client.apigateway.create_stage(restApiId=api_id, stageName=TEST_STAGE_NAME, deploymentId=deployment_id)
    invocation_url = api_invoke_url(api_id=api_id, stage=TEST_STAGE_NAME, path='/sqs')

    def invoke_api(url, is_valid_xml=None):
        if False:
            i = 10
            return i + 15
        _response = requests.post(url, data='<xml>Hello World</xml>', verify=False)
        if is_valid_xml:
            assert is_valid_xml(_response.content.decode('utf-8'))
            return _response
        assert _response.ok
        return _response
    response_data = retry(invoke_api, sleep=2, retries=10, url=invocation_url)
    snapshot.match('sqs-json-response', response_data.json())
    aws_client.apigateway.update_integration(restApiId=api_id, resourceId=resource_id, httpMethod='POST', patchOperations=[{'op': 'add', 'path': '/requestParameters/integration.request.header.Accept', 'value': "'application/xml'"}])
    aws_client.apigateway.update_integration_response(restApiId=api_id, resourceId=resource_id, httpMethod='POST', statusCode='200', patchOperations=[{'op': 'remove', 'path': '/responseTemplates/application~1json', 'value': 'application/json'}])
    deployment = aws_client.apigateway.create_deployment(restApiId=api_id)
    aws_client.apigateway.update_stage(restApiId=api_id, stageName=TEST_STAGE_NAME, patchOperations=[{'op': 'replace', 'path': '/deploymentId', 'value': deployment['id']}])
    response = retry(invoke_api, sleep=2, retries=10, url=invocation_url, is_valid_xml=is_valid_xml)
    xml_body = to_str(response.content)
    assert re.search('<MessageId>.*</MessageId>', xml_body)
    assert re.search('<MD5OfMessageBody>.*</MD5OfMessageBody>', xml_body)
    assert re.search('<RequestId>.*</RequestId>', xml_body)