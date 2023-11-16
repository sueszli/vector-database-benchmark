import argparse
import os

def create_resource_attribute_definition(project_id: str, location: str, dataset_id: str, consent_store_id: str, resource_attribute_definition_id: str):
    if False:
        while True:
            i = 10
    'Creates a RESOURCE attribute definition. A RESOURCE attribute is an attribute whose value is\n    determined by the properties of the data or action.\n\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/consent\n    before running the sample.'
    from googleapiclient import discovery
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    consent_store_parent = 'projects/{}/locations/{}/datasets/{}/consentStores/{}'.format(project_id, location, dataset_id, consent_store_id)
    body = {'description': 'whether the data is identifiable', 'category': 'RESOURCE', 'allowed_values': ['identifiable', 'de-identified']}
    request = client.projects().locations().datasets().consentStores().attributeDefinitions().create(parent=consent_store_parent, body=body, attributeDefinitionId=resource_attribute_definition_id)
    response = request.execute()
    print(f'Created RESOURCE attribute definition: {response}')
    return response

def create_request_attribute_definition(project_id: str, location: str, dataset_id: str, consent_store_id: str, request_attribute_definition_id: str):
    if False:
        for i in range(10):
            print('nop')
    "Creates a REQUEST attribute definition. A REQUEST attribute is an attribute whose value is determined\n    by the requester's identity or purpose.\n\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/consent\n    before running the sample."
    from googleapiclient import discovery
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    consent_store_parent = 'projects/{}/locations/{}/datasets/{}/consentStores/{}'.format(project_id, location, dataset_id, consent_store_id)
    body = {'description': 'what groups are consented for access', 'category': 'REQUEST', 'allowed_values': ['internal-researcher', 'external-researcher', 'clinical-admin']}
    request = client.projects().locations().datasets().consentStores().attributeDefinitions().create(parent=consent_store_parent, body=body, attributeDefinitionId=request_attribute_definition_id)
    response = request.execute()
    print(f'Created REQUEST attribute definition: {response}')
    return response

def get_attribute_definition(project_id: str, location: str, dataset_id: str, consent_store_id: str, attribute_definition_id: str):
    if False:
        i = 10
        return i + 15
    'Gets the specified attribute definition.\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/consent\n    before running the sample.'
    from googleapiclient import discovery
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    consent_store_parent = 'projects/{}/locations/{}/datasets/{}/consentStores/{}'.format(project_id, location, dataset_id, consent_store_id)
    attribute_definition_name = '{}/attributeDefinitions/{}'.format(consent_store_parent, attribute_definition_id)
    request = client.projects().locations().datasets().consentStores().attributeDefinitions().get(name=attribute_definition_name)
    response = request.execute()
    print(f'Got attribute definition: {attribute_definition_id}')
    return response

def list_attribute_definitions(project_id: str, location: str, dataset_id: str, consent_store_id: str):
    if False:
        i = 10
        return i + 15
    'Lists the attribute definitions in the given consent store.\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/consent\n    before running the sample.'
    from googleapiclient import discovery
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    attribute_definition_parent = 'projects/{}/locations/{}/datasets/{}/consentStores/{}'.format(project_id, location, dataset_id, consent_store_id)
    attribute_definitions = client.projects().locations().datasets().consentStores().attributeDefinitions().list(parent=attribute_definition_parent).execute().get('attributeDefinitions', [])
    for attribute_definition in attribute_definitions:
        print(attribute_definition)
    return attribute_definitions

def patch_attribute_definition(project_id: str, location: str, dataset_id: str, consent_store_id: str, attribute_definition_id: str, description: str):
    if False:
        i = 10
        return i + 15
    'Updates the attribute definition.\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/consent\n    before running the sample.'
    from googleapiclient import discovery
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    attribute_definition_parent = 'projects/{}/locations/{}/datasets/{}/consentStores/{}'.format(project_id, location, dataset_id, consent_store_id)
    attribute_definition_name = '{}/attributeDefinitions/{}'.format(attribute_definition_parent, attribute_definition_id)
    patch = {'description': description}
    request = client.projects().locations().datasets().consentStores().attributeDefinitions().patch(name=attribute_definition_name, updateMask='description', body=patch)
    response = request.execute()
    print('Patched attribute definition {} with new description: {}'.format(attribute_definition_id, description))
    return response

def delete_attribute_definition(project_id: str, location: str, dataset_id: str, consent_store_id: str, attribute_definition_id: str):
    if False:
        print('Hello World!')
    'Deletes the specified attribute definition.\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/consent\n    before running the sample.'
    from googleapiclient import discovery
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    consent_store_parent = 'projects/{}/locations/{}/datasets/{}/consentStores/{}'.format(project_id, location, dataset_id, consent_store_id)
    attribute_definition_name = '{}/attributeDefinitions/{}'.format(consent_store_parent, attribute_definition_id)
    request = client.projects().locations().datasets().consentStores().attributeDefinitions().delete(name=attribute_definition_name)
    response = request.execute()
    print(f'Deleted attribute definition: {attribute_definition_id}')
    return response

def parse_command_line_args():
    if False:
        for i in range(10):
            print('nop')
    'Parses command line arguments.'
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project_id', default=os.environ.get('GOOGLE_CLOUD_PROJECT'), help='GCP cloud project name')
    parser.add_argument('--location', default='us-central1', help='GCP location')
    parser.add_argument('--dataset_id', default=None, help='ID of dataset')
    parser.add_argument('--consent_store_id', default=None, help='ID of consent store')
    parser.add_argument('--resource_attribute_definition_id', default=None, help='ID of a RESOURCE attribute definition')
    parser.add_argument('--request_attribute_definition_id', default=None, help='ID of a REQUEST attribute definition')
    parser.add_argument('--attribute_definition_id', default=None, help='ID of an attribute definition')
    parser.add_argument('--description', default=None, help='A description of an attribute')
    command = parser.add_subparsers(dest='command')
    command.add_parser('create-resource-attribute-definition', help=create_resource_attribute_definition.__doc__)
    command.add_parser('create-request-attribute-definition', help=create_request_attribute_definition.__doc__)
    command.add_parser('get-attribute-definition', help=get_attribute_definition.__doc__)
    command.add_parser('list-attribute-definitions', help=list_attribute_definitions.__doc__)
    command.add_parser('patch-attribute-definition', help=patch_attribute_definition.__doc__)
    command.add_parser('delete-attribute-definition', help=delete_attribute_definition.__doc__)
    return parser.parse_args()

def run_command(args):
    if False:
        for i in range(10):
            print('nop')
    'Calls the program using the specified command.'
    if args.project_id is None:
        print('You must specify a project ID or set the "GOOGLE_CLOUD_PROJECT" environment variable.')
        return
    elif args.command == 'create-resource-attribute-definition':
        create_resource_attribute_definition(args.project_id, args.location, args.dataset_id, args.consent_store_id, args.resource_attribute_definition_id)
    elif args.command == 'create-request-attribute-definition':
        create_request_attribute_definition(args.project_id, args.location, args.dataset_id, args.consent_store_id, args.request_attribute_definition_id)
    elif args.command == 'get-attribute-definition':
        get_attribute_definition(args.project_id, args.location, args.dataset_id, args.consent_store_id, args.attribute_definition_id)
    elif args.command == 'list-attribute-definitions':
        list_attribute_definitions(args.project_id, args.location, args.dataset_id, args.consent_store_id)
    elif args.command == 'patch-attribute-definition':
        patch_attribute_definition(args.project_id, args.location, args.dataset_id, args.consent_store_id, args.attribute_definition_id, args.description)
    elif args.command == 'delete-attribute-definition':
        delete_attribute_definition(args.project_id, args.location, args.dataset_id, args.consent_store_id, args.attribute_definition_id)

def main():
    if False:
        while True:
            i = 10
    args = parse_command_line_args()
    run_command(args)
if __name__ == '__main__':
    main()