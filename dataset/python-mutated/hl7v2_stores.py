import argparse
import os

def create_hl7v2_store(project_id, location, dataset_id, hl7v2_store_id):
    if False:
        return 10
    'Creates a new HL7v2 store within the parent dataset.\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/hl7v2\n    before running the sample.'
    from googleapiclient import discovery
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    hl7v2_store_parent = 'projects/{}/locations/{}/datasets/{}'.format(project_id, location, dataset_id)
    body = {'parserConfig': {'version': 'V3'}}
    request = client.projects().locations().datasets().hl7V2Stores().create(parent=hl7v2_store_parent, body=body, hl7V2StoreId=hl7v2_store_id)
    response = request.execute()
    print(f'Created HL7v2 store: {hl7v2_store_id}')
    return response

def delete_hl7v2_store(project_id, location, dataset_id, hl7v2_store_id):
    if False:
        print('Hello World!')
    'Deletes the specified HL7v2 store.\n\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/hl7v2\n    before running the sample.'
    from googleapiclient import discovery
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    hl7v2_store_parent = 'projects/{}/locations/{}/datasets/{}'.format(project_id, location, dataset_id)
    hl7v2_store_name = f'{hl7v2_store_parent}/hl7V2Stores/{hl7v2_store_id}'
    request = client.projects().locations().datasets().hl7V2Stores().delete(name=hl7v2_store_name)
    response = request.execute()
    print(f'Deleted HL7v2 store: {hl7v2_store_id}')
    return response

def get_hl7v2_store(project_id, location, dataset_id, hl7v2_store_id):
    if False:
        for i in range(10):
            print('nop')
    'Gets the specified HL7v2 store.\n\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/hl7v2\n    before running the sample.'
    from googleapiclient import discovery
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    hl7v2_store_parent = 'projects/{}/locations/{}/datasets/{}'.format(project_id, location, dataset_id)
    hl7v2_store_name = f'{hl7v2_store_parent}/hl7V2Stores/{hl7v2_store_id}'
    hl7v2_stores = client.projects().locations().datasets().hl7V2Stores()
    hl7v2_store = hl7v2_stores.get(name=hl7v2_store_name).execute()
    print('Name: {}'.format(hl7v2_store.get('name')))
    if hl7v2_store.get('notificationConfigs') is not None:
        print('Notification configs:')
        for notification_config in hl7v2_store.get('notificationConfigs'):
            print('\tPub/Sub topic: {}'.format(notification_config.get('pubsubTopic')), '\tFilter: {}'.format(notification_config.get('filter')))
    return hl7v2_store

def list_hl7v2_stores(project_id, location, dataset_id):
    if False:
        while True:
            i = 10
    'Lists the HL7v2 stores in the given dataset.\n\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/hl7v2\n    before running the sample.'
    from googleapiclient import discovery
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    hl7v2_store_parent = 'projects/{}/locations/{}/datasets/{}'.format(project_id, location, dataset_id)
    hl7v2_stores = client.projects().locations().datasets().hl7V2Stores().list(parent=hl7v2_store_parent).execute().get('hl7V2Stores', [])
    for hl7v2_store in hl7v2_stores:
        print('HL7v2 store:\nName: {}'.format(hl7v2_store.get('name')))
        if hl7v2_store.get('notificationConfigs') is not None:
            print('Notification configs:')
            for notification_config in hl7v2_store.get('notificationConfigs'):
                print('\tPub/Sub topic: {}'.format(notification_config.get('pubsubTopic')), '\tFilter: {}'.format(notification_config.get('filter')))
    return hl7v2_stores

def patch_hl7v2_store(project_id, location, dataset_id, hl7v2_store_id):
    if False:
        i = 10
        return i + 15
    'Updates the HL7v2 store.\n\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/hl7v2\n    before running the sample.'
    from googleapiclient import discovery
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    hl7v2_store_parent = 'projects/{}/locations/{}/datasets/{}'.format(project_id, location, dataset_id)
    hl7v2_store_name = f'{hl7v2_store_parent}/hl7V2Stores/{hl7v2_store_id}'
    patch = {'notificationConfigs': None}
    request = client.projects().locations().datasets().hl7V2Stores().patch(name=hl7v2_store_name, updateMask='notificationConfigs', body=patch)
    response = request.execute()
    print(f'Patched HL7v2 store {hl7v2_store_id} with Cloud Pub/Sub topic: None')
    return response

def get_hl7v2_store_iam_policy(project_id, location, dataset_id, hl7v2_store_id):
    if False:
        while True:
            i = 10
    'Gets the IAM policy for the specified HL7v2 store.\n\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/hl7v2\n    before running the sample.'
    from googleapiclient import discovery
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    hl7v2_store_parent = 'projects/{}/locations/{}/datasets/{}'.format(project_id, location, dataset_id)
    hl7v2_store_name = f'{hl7v2_store_parent}/hl7V2Stores/{hl7v2_store_id}'
    request = client.projects().locations().datasets().hl7V2Stores().getIamPolicy(resource=hl7v2_store_name)
    response = request.execute()
    print('etag: {}'.format(response.get('name')))
    return response

def set_hl7v2_store_iam_policy(project_id, location, dataset_id, hl7v2_store_id, member, role, etag=None):
    if False:
        print('Hello World!')
    "Sets the IAM policy for the specified HL7v2 store.\n        A single member will be assigned a single role. A member can be any of:\n        - allUsers, that is, anyone\n        - allAuthenticatedUsers, anyone authenticated with a Google account\n        - user:email, as in 'user:somebody@example.com'\n        - group:email, as in 'group:admins@example.com'\n        - domain:domainname, as in 'domain:example.com'\n        - serviceAccount:email,\n            as in 'serviceAccount:my-other-app@appspot.gserviceaccount.com'\n        A role can be any IAM role, such as 'roles/viewer', 'roles/owner',\n        or 'roles/editor'.\n\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/hl7v2\n    before running the sample."
    from googleapiclient import discovery
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    hl7v2_store_parent = 'projects/{}/locations/{}/datasets/{}'.format(project_id, location, dataset_id)
    hl7v2_store_name = f'{hl7v2_store_parent}/hl7V2Stores/{hl7v2_store_id}'
    policy = {'bindings': [{'role': role, 'members': [member]}]}
    if etag is not None:
        policy['etag'] = etag
    request = client.projects().locations().datasets().hl7V2Stores().setIamPolicy(resource=hl7v2_store_name, body={'policy': policy})
    response = request.execute()
    print('etag: {}'.format(response.get('name')))
    print('bindings: {}'.format(response.get('bindings')))
    return response

def parse_command_line_args():
    if False:
        while True:
            i = 10
    'Parses command line arguments.'
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project_id', default=os.environ.get('GOOGLE_CLOUD_PROJECT'), help='GCP project name')
    parser.add_argument('--location', default='us-central1', help='GCP location')
    parser.add_argument('--dataset_id', default=None, help='Name of dataset')
    parser.add_argument('--hl7v2_store_id', default=None, help='Name of HL7v2 store')
    parser.add_argument('--pubsub_topic', default=None, help='The Cloud Pub/Sub topic where notifications of changes are published')
    parser.add_argument('--member', default=None, help='Member to add to IAM policy (e.g. "domain:example.com")')
    parser.add_argument('--role', default=None, help='IAM Role to give to member (e.g. "roles/viewer")')
    command = parser.add_subparsers(dest='command')
    command.add_parser('create-hl7v2-store', help=create_hl7v2_store.__doc__)
    command.add_parser('delete-hl7v2-store', help=delete_hl7v2_store.__doc__)
    command.add_parser('get-hl7v2-store', help=get_hl7v2_store.__doc__)
    command.add_parser('list-hl7v2-stores', help=list_hl7v2_stores.__doc__)
    command.add_parser('patch-hl7v2-store', help=patch_hl7v2_store.__doc__)
    command.add_parser('get_iam_policy', help=get_hl7v2_store_iam_policy.__doc__)
    command.add_parser('set_iam_policy', help=set_hl7v2_store_iam_policy.__doc__)
    return parser.parse_args()

def run_command(args):
    if False:
        return 10
    'Calls the program using the specified command.'
    if args.project_id is None:
        print('You must specify a project ID or set the "GOOGLE_CLOUD_PROJECT" environment variable.')
        return
    elif args.command == 'create-hl7v2-store':
        create_hl7v2_store(args.project_id, args.location, args.dataset_id, args.hl7v2_store_id)
    elif args.command == 'delete-hl7v2-store':
        delete_hl7v2_store(args.project_id, args.location, args.dataset_id, args.hl7v2_store_id)
    elif args.command == 'get-hl7v2-store':
        get_hl7v2_store(args.project_id, args.location, args.dataset_id, args.hl7v2_store_id)
    elif args.command == 'list-hl7v2-stores':
        list_hl7v2_stores(args.project_id, args.location, args.dataset_id)
    elif args.command == 'patch-hl7v2-store':
        patch_hl7v2_store(args.project_id, args.location, args.dataset_id, args.hl7v2_store_id, args.pubsub_topic)
    elif args.command == 'get_hl7v2_store_iam_policy':
        get_hl7v2_store_iam_policy(args.project_id, args.location, args.dataset_id, args.hl7v2_store_id)
    elif args.command == 'set_hl7v2_store_iam_policy':
        set_hl7v2_store_iam_policy(args.project_id, args.location, args.dataset_id, args.hl7v2_store_id, args.member, args.role)

def main():
    if False:
        i = 10
        return i + 15
    args = parse_command_line_args()
    run_command(args)
if __name__ == '__main__':
    main()