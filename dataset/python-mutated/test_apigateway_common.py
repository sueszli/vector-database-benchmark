import json
import pytest
import requests
from botocore.exceptions import ClientError
from localstack.aws.api.lambda_ import Runtime
from localstack.testing.aws.util import is_aws_cloud
from localstack.testing.pytest import markers
from localstack.utils.aws.arns import parse_arn
from localstack.utils.strings import short_uid
from localstack.utils.sync import retry
from tests.aws.services.apigateway.apigateway_fixtures import api_invoke_url, create_rest_api_deployment, create_rest_api_integration, create_rest_api_stage, create_rest_resource_method
from tests.aws.services.lambda_.test_lambda import TEST_LAMBDA_AWS_PROXY

class TestApiGatewayCommon:
    """
    In this class we won't test individual CRUD API calls but how those will affect the integrations and
    requests/responses from the API.
    """

    @markers.aws.validated
    @markers.snapshot.skip_snapshot_verify(paths=['$.invalid-request-body.Type'])
    def test_api_gateway_request_validator(self, create_lambda_function, create_rest_apigw, apigw_redeploy_api, snapshot, aws_client):
        if False:
            for i in range(10):
                print('nop')
        snapshot.add_transformers_list([snapshot.transform.key_value('requestValidatorId'), snapshot.transform.key_value('cacheNamespace'), snapshot.transform.key_value('id'), snapshot.transform.key_value('fn_name'), snapshot.transform.key_value('fn_arn')])
        fn_name = f'test-{short_uid()}'
        create_lambda_function(func_name=fn_name, handler_file=TEST_LAMBDA_AWS_PROXY, runtime=Runtime.python3_9)
        lambda_arn = aws_client.lambda_.get_function(FunctionName=fn_name)['Configuration']['FunctionArn']
        snapshot.match('register-lambda', {'fn_name': fn_name, 'fn_arn': lambda_arn})
        parsed_arn = parse_arn(lambda_arn)
        region = parsed_arn['region']
        account_id = parsed_arn['account']
        (api_id, _, root) = create_rest_apigw(name='aws lambda api')
        resource_1 = aws_client.apigateway.create_resource(restApiId=api_id, parentId=root, pathPart='test')['id']
        resource_id = aws_client.apigateway.create_resource(restApiId=api_id, parentId=resource_1, pathPart='{test}')['id']
        validator_id = aws_client.apigateway.create_request_validator(restApiId=api_id, name='test-validator', validateRequestParameters=True, validateRequestBody=True)['id']
        for http_method in ('GET', 'POST'):
            aws_client.apigateway.put_method(restApiId=api_id, resourceId=resource_id, httpMethod=http_method, authorizationType='NONE', requestValidatorId=validator_id, requestParameters={'method.request.path.test': True})
            aws_client.apigateway.put_integration(restApiId=api_id, resourceId=resource_id, httpMethod=http_method, integrationHttpMethod='POST', type='AWS_PROXY', uri=f'arn:aws:apigateway:{region}:lambda:path//2015-03-31/functions/{lambda_arn}/invocations')
            aws_client.apigateway.put_method_response(restApiId=api_id, resourceId=resource_id, httpMethod=http_method, statusCode='200')
            aws_client.apigateway.put_integration_response(restApiId=api_id, resourceId=resource_id, httpMethod=http_method, statusCode='200')
        stage_name = 'local'
        deploy_1 = aws_client.apigateway.create_deployment(restApiId=api_id, stageName=stage_name)
        snapshot.match('deploy-1', deploy_1)
        source_arn = f'arn:aws:execute-api:{region}:{account_id}:{api_id}/*/*/test/*'
        aws_client.lambda_.add_permission(FunctionName=lambda_arn, StatementId=str(short_uid()), Action='lambda:InvokeFunction', Principal='apigateway.amazonaws.com', SourceArn=source_arn)
        url = api_invoke_url(api_id, stage=stage_name, path='/test/value')
        response = requests.post(url, json={'test': 'test'})
        assert response.ok
        assert json.loads(response.json()['body']) == {'test': 'test'}
        response_get = requests.get(url)
        assert response_get.ok
        response = aws_client.apigateway.update_method(restApiId=api_id, resourceId=resource_id, httpMethod='POST', patchOperations=[{'op': 'add', 'path': '/requestParameters/method.request.path.issuer', 'value': 'true'}, {'op': 'remove', 'path': '/requestParameters/method.request.path.test', 'value': 'true'}])
        snapshot.match('change-request-path-names', response)
        apigw_redeploy_api(rest_api_id=api_id, stage_name=stage_name)
        response = requests.post(url, json={'test': 'test'})
        if response.status_code == 400:
            snapshot.match('missing-required-request-params', response.json())
        aws_client.apigateway.create_model(restApiId=api_id, name='testSchema', contentType='application/json', schema=json.dumps({'title': 'testSchema', 'type': 'object', 'properties': {'a': {'type': 'number'}, 'b': {'type': 'number'}}, 'required': ['a', 'b']}))
        for http_method in ('GET', 'POST'):
            response = aws_client.apigateway.update_method(restApiId=api_id, resourceId=resource_id, httpMethod=http_method, patchOperations=[{'op': 'add', 'path': '/requestModels/application~1json', 'value': 'testSchema'}])
            snapshot.match(f'add-schema-{http_method}', response)
        response = aws_client.apigateway.update_method(restApiId=api_id, resourceId=resource_id, httpMethod='POST', patchOperations=[{'op': 'add', 'path': '/requestParameters/method.request.path.test', 'value': 'true'}, {'op': 'remove', 'path': '/requestParameters/method.request.path.issuer', 'value': 'true'}])
        snapshot.match('revert-request-path-names', response)
        apigw_redeploy_api(rest_api_id=api_id, stage_name=stage_name)
        response = requests.post(url, json={'test': 'test'})
        assert response.status_code == 400
        snapshot.match('invalid-request-body', response.json())
        response_get = requests.get(url)
        assert response_get.status_code == 400
        response_get = requests.get(url, headers={'Content-Type': 'application/json'})
        assert response_get.status_code == 400
        for http_method in ('GET', 'POST'):
            response = aws_client.apigateway.update_method(restApiId=api_id, resourceId=resource_id, httpMethod=http_method, patchOperations=[{'op': 'replace', 'path': '/requestValidatorId', 'value': ''}])
            snapshot.match(f'remove-validator-{http_method}', response)
        apigw_redeploy_api(rest_api_id=api_id, stage_name=stage_name)
        response = requests.post(url, json={'test': 'test'})
        assert response.ok
        assert json.loads(response.json()['body']) == {'test': 'test'}
        response_get = requests.get(url)
        assert response_get.ok

class TestUsagePlans:

    @markers.aws.validated
    def test_api_key_required_for_methods(self, aws_client, snapshot, create_rest_apigw, apigw_redeploy_api):
        if False:
            i = 10
            return i + 15
        snapshot.add_transformer(snapshot.transform.apigateway_api())
        snapshot.add_transformers_list([snapshot.transform.key_value('apiId'), snapshot.transform.key_value('value')])
        (api_id, _, root_id) = create_rest_apigw(name='test API key', apiKeySource='HEADER')
        resource = aws_client.apigateway.create_resource(restApiId=api_id, parentId=root_id, pathPart='test')
        resource_id = resource['id']
        aws_client.apigateway.put_method(restApiId=api_id, resourceId=resource_id, httpMethod='GET', authorizationType='NONE', apiKeyRequired=True)
        aws_client.apigateway.put_method_response(restApiId=api_id, resourceId=resource_id, httpMethod='GET', statusCode='200')
        aws_client.apigateway.put_integration(restApiId=api_id, resourceId=resource_id, httpMethod='GET', integrationHttpMethod='GET', type='MOCK', requestTemplates={'application/json': '{"statusCode": 200}'})
        aws_client.apigateway.put_integration_response(restApiId=api_id, resourceId=resource_id, httpMethod='GET', statusCode='200', selectionPattern='')
        stage_name = 'dev'
        aws_client.apigateway.create_deployment(restApiId=api_id, stageName=stage_name)
        usage_plan_response = aws_client.apigateway.create_usage_plan(name=f'test-plan-{short_uid()}', description='Test Usage Plan for API key', quota={'limit': 10, 'period': 'DAY', 'offset': 0}, throttle={'rateLimit': 2, 'burstLimit': 1}, apiStages=[{'apiId': api_id, 'stage': stage_name}], tags={'tag_key': 'tag_value'})
        snapshot.match('create-usage-plan', usage_plan_response)
        usage_plan_id = usage_plan_response['id']
        key_name = f'testApiKey-{short_uid()}'
        api_key_response = aws_client.apigateway.create_api_key(name=key_name, enabled=True)
        snapshot.match('create-api-key', api_key_response)
        api_key_id = api_key_response['id']
        create_usage_plan_key_resp = aws_client.apigateway.create_usage_plan_key(usagePlanId=usage_plan_id, keyId=api_key_id, keyType='API_KEY')
        snapshot.match('create-usage-plan-key', create_usage_plan_key_resp)
        url = api_invoke_url(api_id=api_id, stage=stage_name, path='/test')
        response = requests.get(url)
        assert response.status_code == 403

        def _assert_with_key(expected_status_code: int):
            if False:
                while True:
                    i = 10
            _response = requests.get(url, headers={'x-api-key': api_key_response['value']})
            assert _response.status_code == expected_status_code
        retries = 10 if is_aws_cloud() else 3
        sleep = 12 if is_aws_cloud() else 1
        retry(_assert_with_key, retries=retries, sleep=sleep, expected_status_code=200)
        patch_operations = [{'op': 'replace', 'path': '/enabled', 'value': 'false'}]
        response = aws_client.apigateway.update_api_key(apiKey=api_key_id, patchOperations=patch_operations)
        snapshot.match('update-api-key-disabled', response)
        retry(_assert_with_key, retries=retries, sleep=sleep, expected_status_code=403)

    @markers.aws.validated
    def test_usage_plan_crud(self, create_rest_apigw, snapshot, aws_client, echo_http_server_post):
        if False:
            i = 10
            return i + 15
        snapshot.add_transformer(snapshot.transform.key_value('id', reference_replacement=True))
        snapshot.add_transformer(snapshot.transform.key_value('name'))
        snapshot.add_transformer(snapshot.transform.key_value('description'))
        snapshot.add_transformer(snapshot.transform.key_value('apiId', reference_replacement=True))
        old_usage_plans = aws_client.apigateway.get_usage_plans().get('items', [])
        for usage_plan in old_usage_plans:
            aws_client.apigateway.delete_usage_plan(usagePlanId=usage_plan['id'])
        (api_id, _, root) = create_rest_apigw(name=f'test-api-{short_uid()}', description='this is my api')
        create_rest_resource_method(aws_client.apigateway, restApiId=api_id, resourceId=root, httpMethod='GET', authorizationType='none')
        create_rest_api_integration(aws_client.apigateway, restApiId=api_id, resourceId=root, httpMethod='GET', integrationHttpMethod='POST', type='HTTP', uri=echo_http_server_post)
        (deployment_id, _) = create_rest_api_deployment(aws_client.apigateway, restApiId=api_id)
        stage = create_rest_api_stage(aws_client.apigateway, restApiId=api_id, stageName='dev', deploymentId=deployment_id)
        response = aws_client.apigateway.create_usage_plan(name=f'test-usage-plan-{short_uid()}', description='this is my usage plan', apiStages=[{'apiId': api_id, 'stage': stage}])
        snapshot.match('create-usage-plan', response)
        usage_plan_id = response['id']
        response = aws_client.apigateway.get_usage_plan(usagePlanId=usage_plan_id)
        snapshot.match('get-usage-plan', response)
        response = aws_client.apigateway.get_usage_plans()
        snapshot.match('get-usage-plans', response)
        response = aws_client.apigateway.update_usage_plan(usagePlanId=usage_plan_id, patchOperations=[{'op': 'replace', 'path': '/throttle/burstLimit', 'value': '100'}, {'op': 'replace', 'path': '/throttle/rateLimit', 'value': '200'}])
        snapshot.match('update-usage-plan', response)

class TestDocumentations:

    @markers.aws.validated
    def test_documentation_parts_and_versions(self, aws_client, create_rest_apigw, apigw_add_transformers, snapshot):
        if False:
            return 10
        client = aws_client.apigateway
        (api_id, api_name, root_id) = create_rest_apigw()
        response = client.create_documentation_part(restApiId=api_id, location={'type': 'API'}, properties=json.dumps({'foo': 'bar'}))
        snapshot.match('create-part-response', response)
        response = client.get_documentation_parts(restApiId=api_id)
        snapshot.match('get-parts-response', response)
        response = client.create_documentation_version(restApiId=api_id, documentationVersion='v123')
        snapshot.match('create-version-response', response)
        response = client.update_documentation_version(restApiId=api_id, documentationVersion='v123', patchOperations=[{'op': 'replace', 'path': '/description', 'value': 'doc version new'}])
        snapshot.match('update-version-response', response)
        response = client.get_documentation_version(restApiId=api_id, documentationVersion='v123')
        snapshot.match('get-version-response', response)

class TestStages:

    @markers.aws.validated
    @markers.snapshot.skip_snapshot_verify(paths=['$..createdDate', '$..lastUpdatedDate'])
    def test_create_update_stages(self, aws_client, create_rest_apigw, apigw_add_transformers, snapshot):
        if False:
            return 10
        client = aws_client.apigateway
        (api_id, api_name, root_id) = create_rest_apigw()
        client.put_method(restApiId=api_id, resourceId=root_id, httpMethod='GET', authorizationType='NONE')
        client.put_integration(restApiId=api_id, resourceId=root_id, httpMethod='GET', type='MOCK')
        response = client.create_deployment(restApiId=api_id)
        deployment_id = response['id']
        client.create_documentation_part(restApiId=api_id, location={'type': 'API'}, properties=json.dumps({'foo': 'bar'}))
        client.create_documentation_version(restApiId=api_id, documentationVersion='v123')
        response = client.create_stage(restApiId=api_id, stageName='s1', deploymentId=deployment_id, description='my stage', documentationVersion='v123')
        snapshot.match('create-stage', response)
        with pytest.raises(ClientError) as ctx:
            client.update_stage(restApiId=api_id, stageName='s1', patchOperations=[{'op': 'replace', 'path': '/documentation_version', 'value': '123'}])
        snapshot.match('error-update-doc-version', ctx.value.response)
        with pytest.raises(ClientError) as ctx:
            client.update_stage(restApiId=api_id, stageName='s1', patchOperations=[{'op': 'replace', 'path': '/tags/tag1', 'value': 'value1'}])
        snapshot.match('error-update-tags', ctx.value.response)
        response = client.update_stage(restApiId=api_id, stageName='s1', patchOperations=[{'op': 'replace', 'path': '/description', 'value': 'stage new'}, {'op': 'replace', 'path': '/variables/var1', 'value': 'test'}, {'op': 'replace', 'path': '/variables/var2', 'value': 'test2'}, {'op': 'replace', 'path': '/*/*/throttling/burstLimit', 'value': '123'}, {'op': 'replace', 'path': '/*/*/caching/enabled', 'value': 'true'}, {'op': 'replace', 'path': '/tracingEnabled', 'value': 'true'}, {'op': 'replace', 'path': '/test/GET/throttling/burstLimit', 'value': '124'}])
        snapshot.match('update-stage', response)
        response = client.get_stage(restApiId=api_id, stageName='s1')
        snapshot.match('get-stage', response)
        response = client.update_stage(restApiId=api_id, stageName='s1', patchOperations=[{'op': 'replace', 'path': '/*/*/throttling/burstLimit', 'value': '100'}])
        snapshot.match('update-stage-override', response)

class TestDeployments:

    @markers.aws.validated
    @markers.snapshot.skip_snapshot_verify(paths=['$..createdDate', '$..lastUpdatedDate'])
    @pytest.mark.parametrize('create_stage_manually', [True, False])
    def test_create_delete_deployments(self, create_stage_manually, aws_client, create_rest_apigw, apigw_add_transformers, snapshot):
        if False:
            for i in range(10):
                print('nop')
        snapshot.add_transformer(snapshot.transform.apigateway_api())
        client = aws_client.apigateway
        (api_id, _, root_id) = create_rest_apigw()
        client.put_method(restApiId=api_id, resourceId=root_id, httpMethod='GET', authorizationType='NONE')
        client.put_integration(restApiId=api_id, resourceId=root_id, httpMethod='GET', type='MOCK')
        kwargs = {} if create_stage_manually else {'stageName': 's1'}
        response = client.create_deployment(restApiId=api_id, **kwargs)
        deployment_id = response['id']
        if create_stage_manually:
            client.create_stage(restApiId=api_id, stageName='s1', deploymentId=deployment_id)
        response = client.get_deployment(restApiId=api_id, deploymentId=deployment_id)
        snapshot.match('get-deployment', response)
        response = client.get_stages(restApiId=api_id)
        snapshot.match('get-stages', response)
        for i in range(3):
            with pytest.raises(ClientError) as ctx:
                client.delete_deployment(restApiId=api_id, deploymentId=deployment_id)
            snapshot.match(f'delete-deployment-error-{i}', ctx.value.response)
            client.delete_stage(restApiId=api_id, stageName='s1')
            client.delete_deployment(restApiId=api_id, deploymentId=deployment_id)
            response = client.create_deployment(restApiId=api_id, **kwargs)
            deployment_id = response['id']
            if create_stage_manually:
                client.create_stage(restApiId=api_id, stageName='s1', deploymentId=deployment_id)
            response = client.get_deployments(restApiId=api_id)
            snapshot.match(f'get-deployments-{i}', response)
            response = client.get_stages(restApiId=api_id)
            snapshot.match(f'get-stages-{i}', response)

    @markers.aws.validated
    @markers.snapshot.skip_snapshot_verify(paths=['$..createdDate', '$..lastUpdatedDate'])
    def test_create_update_deployments(self, aws_client, create_rest_apigw, apigw_add_transformers, snapshot):
        if False:
            print('Hello World!')
        snapshot.add_transformer(snapshot.transform.apigateway_api())
        client = aws_client.apigateway
        (api_id, _, root_id) = create_rest_apigw()
        client.put_method(restApiId=api_id, resourceId=root_id, httpMethod='GET', authorizationType='NONE')
        client.put_integration(restApiId=api_id, resourceId=root_id, httpMethod='GET', type='MOCK')
        response = client.create_deployment(restApiId=api_id)
        deployment_id_1 = response['id']
        client.create_stage(restApiId=api_id, stageName='s1', deploymentId=deployment_id_1)
        response = client.get_deployment(restApiId=api_id, deploymentId=deployment_id_1)
        snapshot.match('get-deployment-1', response)
        response = client.get_stages(restApiId=api_id)
        snapshot.match('get-stages', response)
        with pytest.raises(ClientError) as ctx:
            client.delete_deployment(restApiId=api_id, deploymentId=deployment_id_1)
        snapshot.match('delete-deployment-error', ctx.value.response)
        response = client.create_deployment(restApiId=api_id, stageName='s1')
        deployment_id_2 = response['id']
        response = client.get_deployment(restApiId=api_id, deploymentId=deployment_id_1)
        snapshot.match('get-deployment-1-after-update', response)
        response = client.get_deployment(restApiId=api_id, deploymentId=deployment_id_2)
        snapshot.match('get-deployment-2', response)
        response = client.get_stages(restApiId=api_id)
        snapshot.match('get-stages-after-update', response)
        response = client.delete_deployment(restApiId=api_id, deploymentId=deployment_id_1)
        snapshot.match('delete-deployment-1', response)
        with pytest.raises(ClientError) as ctx:
            client.delete_deployment(restApiId=api_id, deploymentId=deployment_id_2)
        snapshot.match('delete-deployment-2-error', ctx.value.response)

class TestProxyRouting:

    @markers.aws.validated
    def test_proxy_routing_with_hardcoded_resource_sibling(self, aws_client, create_rest_apigw, apigw_redeploy_api):
        if False:
            for i in range(10):
                print('nop')
        (api_id, _, root_id) = create_rest_apigw(name='test proxy routing')

        def _create_mock_integration_with_200_response_template(resource_id: str, http_method: str, response_template: dict):
            if False:
                print('Hello World!')
            aws_client.apigateway.put_method(restApiId=api_id, resourceId=resource_id, httpMethod=http_method, authorizationType='NONE')
            aws_client.apigateway.put_method_response(restApiId=api_id, resourceId=resource_id, httpMethod=http_method, statusCode='200')
            aws_client.apigateway.put_integration(restApiId=api_id, resourceId=resource_id, httpMethod=http_method, type='MOCK', requestTemplates={'application/json': '{"statusCode": 200}'})
            aws_client.apigateway.put_integration_response(restApiId=api_id, resourceId=resource_id, httpMethod=http_method, statusCode='200', selectionPattern='', responseTemplates={'application/json': json.dumps(response_template)})
        resource = aws_client.apigateway.create_resource(restApiId=api_id, parentId=root_id, pathPart='test')
        hardcoded_resource_id = resource['id']
        response_template_post = {'statusCode': 200, 'message': 'POST request'}
        _create_mock_integration_with_200_response_template(hardcoded_resource_id, 'POST', response_template_post)
        resource = aws_client.apigateway.create_resource(restApiId=api_id, parentId=hardcoded_resource_id, pathPart='any')
        any_resource_id = resource['id']
        response_template_any = {'statusCode': 200, 'message': 'ANY request'}
        _create_mock_integration_with_200_response_template(any_resource_id, 'ANY', response_template_any)
        resource = aws_client.apigateway.create_resource(restApiId=api_id, parentId=root_id, pathPart='{proxy+}')
        proxy_resource_id = resource['id']
        response_template_options = {'statusCode': 200, 'message': 'OPTIONS request'}
        _create_mock_integration_with_200_response_template(proxy_resource_id, 'OPTIONS', response_template_options)
        stage_name = 'dev'
        aws_client.apigateway.create_deployment(restApiId=api_id, stageName=stage_name)
        url = api_invoke_url(api_id=api_id, stage=stage_name, path='/test')

        def _invoke_api(req_url: str, http_method: str, expected_type: str):
            if False:
                for i in range(10):
                    print('nop')
            _response = requests.request(http_method.upper(), req_url)
            assert _response.ok
            assert _response.json()['message'] == f'{expected_type} request'
        retries = 10 if is_aws_cloud() else 3
        sleep = 3 if is_aws_cloud() else 1
        retry(_invoke_api, retries=retries, sleep=sleep, req_url=url, http_method='OPTIONS', expected_type='OPTIONS')
        retry(_invoke_api, retries=retries, sleep=sleep, req_url=url, http_method='POST', expected_type='POST')
        any_url = api_invoke_url(api_id=api_id, stage=stage_name, path='/test/any')
        retry(_invoke_api, retries=retries, sleep=sleep, req_url=any_url, http_method='OPTIONS', expected_type='ANY')
        retry(_invoke_api, retries=retries, sleep=sleep, req_url=any_url, http_method='GET', expected_type='ANY')