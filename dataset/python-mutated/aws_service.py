"""
Purpose

Shows how to use the AWS SDK for Python (Boto3) with Amazon API Gateway to
create a REST API that uses Amazon DynamoDB to store user profiles.
"""
import argparse
import json
import logging
from pprint import pprint
import boto3
from botocore.exceptions import ClientError
import requests
logger = logging.getLogger(__name__)

class ApiGatewayToService:
    """
    Encapsulates Amazon API Gateway functions that are used to create a REST API that
    integrates with another AWS service.
    """

    def __init__(self, apig_client):
        if False:
            return 10
        '\n        :param apig_client: A Boto3 API Gateway client.\n        '
        self.apig_client = apig_client
        self.api_id = None
        self.root_id = None
        self.stage = None

    def create_rest_api(self, api_name):
        if False:
            i = 10
            return i + 15
        '\n        Creates a REST API on API Gateway. The default API has only a root resource\n        and no HTTP methods.\n\n        :param api_name: The name of the API. This descriptive name is not used in\n                         the API path.\n        :return: The ID of the newly created API.\n        '
        try:
            result = self.apig_client.create_rest_api(name=api_name)
            self.api_id = result['id']
            logger.info('Created REST API %s with ID %s.', api_name, self.api_id)
        except ClientError:
            logger.exception("Couldn't create REST API %s.", api_name)
            raise
        try:
            result = self.apig_client.get_resources(restApiId=self.api_id)
            self.root_id = next((item for item in result['items'] if item['path'] == '/'))['id']
        except ClientError:
            logger.exception("Couldn't get resources for API %s.", self.api_id)
            raise
        except StopIteration as err:
            logger.exception('No root resource found in API %s.', self.api_id)
            raise ValueError from err
        return self.api_id

    def add_rest_resource(self, parent_id, resource_path):
        if False:
            print('Hello World!')
        '\n        Adds a resource to a REST API.\n\n        :param parent_id: The ID of the parent resource.\n        :param resource_path: The path of the new resource, relative to the parent.\n        :return: The ID of the new resource.\n        '
        try:
            result = self.apig_client.create_resource(restApiId=self.api_id, parentId=parent_id, pathPart=resource_path)
            resource_id = result['id']
            logger.info('Created resource %s.', resource_path)
        except ClientError:
            logger.exception("Couldn't create resource %s.", resource_path)
            raise
        else:
            return resource_id

    def add_integration_method(self, resource_id, rest_method, service_endpoint_prefix, service_action, service_method, role_arn, mapping_template):
        if False:
            return 10
        "\n        Adds an integration method to a REST API. An integration method is a REST\n        resource, such as '/users', and an HTTP verb, such as GET. The integration\n        method is backed by an AWS service, such as Amazon DynamoDB.\n\n        :param resource_id: The ID of the REST resource.\n        :param rest_method: The HTTP verb used with the REST resource.\n        :param service_endpoint_prefix: The service endpoint that is integrated with\n                                        this method, such as 'dynamodb'.\n        :param service_action: The action that is called on the service, such as\n                               'GetItem'.\n        :param service_method: The HTTP method of the service request, such as POST.\n        :param role_arn: The Amazon Resource Name (ARN) of a role that grants API\n                         Gateway permission to use the specified action with the\n                         service.\n        :param mapping_template: A mapping template that is used to translate REST\n                                 elements, such as query parameters, to the request\n                                 body format required by the service.\n        "
        service_uri = f'arn:aws:apigateway:{self.apig_client.meta.region_name}:{service_endpoint_prefix}:action/{service_action}'
        try:
            self.apig_client.put_method(restApiId=self.api_id, resourceId=resource_id, httpMethod=rest_method, authorizationType='NONE')
            self.apig_client.put_method_response(restApiId=self.api_id, resourceId=resource_id, httpMethod=rest_method, statusCode='200', responseModels={'application/json': 'Empty'})
            logger.info('Created %s method for resource %s.', rest_method, resource_id)
        except ClientError:
            logger.exception("Couldn't create %s method for resource %s.", rest_method, resource_id)
            raise
        try:
            self.apig_client.put_integration(restApiId=self.api_id, resourceId=resource_id, httpMethod=rest_method, type='AWS', integrationHttpMethod=service_method, credentials=role_arn, requestTemplates={'application/json': json.dumps(mapping_template)}, uri=service_uri, passthroughBehavior='WHEN_NO_TEMPLATES')
            self.apig_client.put_integration_response(restApiId=self.api_id, resourceId=resource_id, httpMethod=rest_method, statusCode='200', responseTemplates={'application/json': ''})
            logger.info('Created integration for resource %s to service URI %s.', resource_id, service_uri)
        except ClientError:
            logger.exception("Couldn't create integration for resource %s to service URI %s.", resource_id, service_uri)
            raise

    def deploy_api(self, stage_name):
        if False:
            while True:
                i = 10
        "\n        Deploys a REST API. After a REST API is deployed, it can be called from any\n        REST client, such as the Python Requests package or Postman.\n\n        :param stage_name: The stage of the API to deploy, such as 'test'.\n        :return: The base URL of the deployed REST API.\n        "
        try:
            self.apig_client.create_deployment(restApiId=self.api_id, stageName=stage_name)
            self.stage = stage_name
            logger.info('Deployed stage %s.', stage_name)
        except ClientError:
            logger.exception("Couldn't deploy stage %s.", stage_name)
            raise
        else:
            return self.api_url()

    def api_url(self, resource=None):
        if False:
            print('Hello World!')
        '\n        Builds the REST API URL from its parts.\n\n        :param resource: The resource path to append to the base URL.\n        :return: The REST URL to the specified resource.\n        '
        url = f'https://{self.api_id}.execute-api.{self.apig_client.meta.region_name}.amazonaws.com/{self.stage}'
        if resource is not None:
            url = f'{url}/{resource}'
        return url

    def get_rest_api_id(self, api_name):
        if False:
            while True:
                i = 10
        '\n        Gets the ID of a REST API from its name by searching the list of REST APIs\n        for the current account. Because names need not be unique, this returns only\n        the first API with the specified name.\n\n        :param api_name: The name of the API to look up.\n        :return: The ID of the specified API.\n        '
        try:
            rest_api = None
            paginator = self.apig_client.get_paginator('get_rest_apis')
            for page in paginator.paginate():
                rest_api = next((item for item in page['items'] if item['name'] == api_name), None)
                if rest_api is not None:
                    break
            self.api_id = rest_api['id']
            logger.info('Found ID %s for API %s.', rest_api['id'], api_name)
        except ClientError:
            logger.exception("Couldn't find ID for API %s.", api_name)
            raise
        else:
            return rest_api['id']

    def delete_rest_api(self):
        if False:
            i = 10
            return i + 15
        '\n        Deletes a REST API, including all of its resources and configuration.\n        '
        try:
            self.apig_client.delete_rest_api(restApiId=self.api_id)
            logger.info('Deleted REST API %s.', self.api_id)
            self.api_id = None
        except ClientError:
            logger.exception("Couldn't delete REST API %s.", self.api_id)
            raise

def deploy(stack_name, cf_resource):
    if False:
        return 10
    '\n    Deploys prerequisite resources used by the `usage_demo` script. The resources are\n    defined in the associated `setup.yaml` AWS CloudFormation script and are deployed\n    as a CloudFormation stack so they can be easily managed and destroyed.\n\n    :param stack_name: The name of the CloudFormation stack.\n    :param cf_resource: A Boto3 CloudFormation resource.\n    '
    with open('setup.yaml') as setup_file:
        setup_template = setup_file.read()
    print(f'Creating {stack_name}.')
    stack = cf_resource.create_stack(StackName=stack_name, TemplateBody=setup_template, Capabilities=['CAPABILITY_NAMED_IAM'])
    print('Waiting for stack to deploy.')
    waiter = cf_resource.meta.client.get_waiter('stack_create_complete')
    waiter.wait(StackName=stack.name)
    stack.load()
    print(f'Stack status: {stack.stack_status}')
    print('Created resources:')
    for resource in stack.resource_summaries.all():
        print(f'\t{resource.resource_type}, {resource.physical_resource_id}')

def usage_demo(table_name, role_name, rest_api_name):
    if False:
        while True:
            i = 10
    '\n    Demonstrates how to used API Gateway to create and deploy a REST API, and how\n    to use the Requests package to call it.\n\n    :param table_name: The name of the demo DynamoDB table.\n    :param role_name: The name of the demo role that grants API Gateway permission to\n                      call DynamoDB.\n    :param rest_api_name: The name of the demo REST API created by the demo.\n    '
    gateway = ApiGatewayToService(boto3.client('apigateway'))
    role = boto3.resource('iam').Role(role_name)
    print('Creating REST API in API Gateway.')
    gateway.create_rest_api(rest_api_name)
    print('Adding resources to the REST API.')
    profiles_id = gateway.add_rest_resource(gateway.root_id, 'profiles')
    username_id = gateway.add_rest_resource(profiles_id, '{username}')
    print('Adding integration methods to read and write profiles in Amazon DynamoDB.')
    gateway.add_integration_method(profiles_id, 'GET', 'dynamodb', 'Scan', 'POST', role.arn, {'TableName': table_name})
    gateway.add_integration_method(profiles_id, 'POST', 'dynamodb', 'PutItem', 'POST', role.arn, {'TableName': table_name, 'Item': {'username': {'S': "$input.path('$.username')"}, 'name': {'S': "$input.path('$.name')"}, 'title': {'S': "$input.path('$.title')"}}})
    gateway.add_integration_method(username_id, 'GET', 'dynamodb', 'GetItem', 'POST', role.arn, {'TableName': table_name, 'Key': {'username': {'S': '$method.request.path.username'}}})
    stage = 'test'
    print(f'Deploying the {stage} stage.')
    gateway.deploy_api(stage)
    profiles_url = gateway.api_url('profiles')
    print(f'Using the Requests package to post some people to the profiles REST API at {profiles_url}.')
    requests.post(profiles_url, json={'username': 'will', 'name': 'William Shakespeare', 'title': 'playwright'})
    requests.post(profiles_url, json={'username': 'ludwig', 'name': 'Ludwig van Beethoven', 'title': 'composer'})
    requests.post(profiles_url, json={'username': 'jane', 'name': 'Jane Austen', 'title': 'author'})
    print('Getting the list of profiles from the REST API.')
    profiles = requests.get(profiles_url).json()
    pprint(profiles)
    print(f"Getting just the profile for username 'jane' (URL: {profiles_url}/jane).")
    jane = requests.get(f'{profiles_url}/jane').json()
    pprint(jane)

def destroy(rest_api_name, stack, cf_resource):
    if False:
        print('Hello World!')
    '\n    Destroys the REST API created by the demo, the resources managed by the\n    CloudFormation stack, and the CloudFormation stack itself.\n\n    :param rest_api_name: The name of the demo REST API.\n    :param stack: The CloudFormation stack that manages the demo resources.\n    :param cf_resource: A Boto3 CloudFormation resource.\n    '
    print(f'Deleting REST API {rest_api_name}.')
    gateway = ApiGatewayToService(boto3.client('apigateway'))
    gateway.get_rest_api_id(rest_api_name)
    gateway.delete_rest_api()
    print(f'Deleting {stack.name}.')
    stack.delete()
    print('Waiting for stack removal.')
    waiter = cf_resource.meta.client.get_waiter('stack_delete_complete')
    waiter.wait(StackName=stack.name)
    print('Stack delete complete.')

def main():
    if False:
        print('Hello World!')
    parser = argparse.ArgumentParser(description="Runs the Amazon API Gateway demo. Run this script with the 'deploy' flag to deploy prerequisite resources, then with the 'demo' flag to see example usage. Run with the 'destroy' flag to clean up all resources.")
    parser.add_argument('action', choices=['deploy', 'demo', 'destroy'], help='Indicates the action the script performs.')
    parser.print_usage = parser.print_help
    args = parser.parse_args()
    print('-' * 88)
    print('Welcome to the Amazon API Gateway AWS service demo!')
    print('-' * 88)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    cf_resource = boto3.resource('cloudformation')
    rest_api_name = 'doc-example-apigateway-dynamodb-profiles'
    stack = cf_resource.Stack('python-example-code-apigateway-dynamodb-profiles')
    if args.action == 'deploy':
        print('Deploying prerequisite resources for the demo.')
        deploy(stack.name, cf_resource)
        print("To see example usage, run the script again with the 'demo' flag.")
    elif args.action == 'demo':
        print('Demonstrating how to use API Gateway to set up a REST API and call it with the Python Requests package.')
        table_name = None
        role_name = None
        for resource in stack.resource_summaries.all():
            if resource.resource_type == 'AWS::DynamoDB::Table':
                table_name = resource.physical_resource_id
            elif resource.resource_type == 'AWS::IAM::Role':
                role_name = resource.physical_resource_id
        usage_demo(table_name, role_name, rest_api_name)
        print("To clean up all AWS resources created for the demo, run this script again with the 'destroy' flag.")
    elif args.action == 'destroy':
        print('Destroying AWS resources created for the demo.')
        destroy(rest_api_name, stack, cf_resource)
    print('-' * 88)
if __name__ == '__main__':
    main()