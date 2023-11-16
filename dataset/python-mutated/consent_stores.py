import argparse
import os

def create_consent_store(project_id: str, location: str, dataset_id: str, consent_store_id: str):
    if False:
        while True:
            i = 10
    'Creates a new consent store within the parent dataset.\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/consent\n    before running the sample.'
    from googleapiclient import discovery
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    consent_store_parent = f'projects/{project_id}/locations/{location}/datasets/{dataset_id}'
    request = client.projects().locations().datasets().consentStores().create(parent=consent_store_parent, body={}, consentStoreId=consent_store_id)
    response = request.execute()
    print(f'Created consent store: {consent_store_id}')
    return response

def delete_consent_store(project_id: str, location: str, dataset_id: str, consent_store_id: str):
    if False:
        while True:
            i = 10
    'Deletes the specified consent store.\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/consent\n    before running the sample.'
    from googleapiclient import discovery
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    consent_store_parent = 'projects/{}/locations/{}/datasets/{}'.format(project_id, location, dataset_id)
    consent_store_name = '{}/consentStores/{}'.format(consent_store_parent, consent_store_id)
    request = client.projects().locations().datasets().consentStores().delete(name=consent_store_name)
    response = request.execute()
    print(f'Deleted consent store: {consent_store_id}')
    return response

def get_consent_store(project_id: str, location: str, dataset_id: str, consent_store_id: str):
    if False:
        for i in range(10):
            print('nop')
    'Gets the specified consent store.\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/consent\n    before running the sample.'
    from googleapiclient import discovery
    import json
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    consent_store_parent = 'projects/{}/locations/{}/datasets/{}'.format(project_id, location, dataset_id)
    consent_store_name = '{}/consentStores/{}'.format(consent_store_parent, consent_store_id)
    consent_stores = client.projects().locations().datasets().consentStores()
    consent_store = consent_stores.get(name=consent_store_name).execute()
    print(json.dumps(consent_store, indent=2))
    return consent_store

def list_consent_stores(project_id, location, dataset_id):
    if False:
        for i in range(10):
            print('nop')
    'Lists the consent stores in the given dataset.\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/consent\n    before running the sample.'
    from googleapiclient import discovery
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    consent_store_parent = 'projects/{}/locations/{}/datasets/{}'.format(project_id, location, dataset_id)
    consent_stores = client.projects().locations().datasets().consentStores().list(parent=consent_store_parent).execute().get('consentStores', [])
    for consent_store in consent_stores:
        print(consent_store)
    return consent_stores

def patch_consent_store(project_id: str, location: str, dataset_id: str, consent_store_id: str, default_consent_ttl):
    if False:
        while True:
            i = 10
    'Updates the consent store.\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/consent\n    before running the sample.'
    from googleapiclient import discovery
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    consent_store_parent = 'projects/{}/locations/{}/datasets/{}'.format(project_id, location, dataset_id)
    consent_store_name = '{}/consentStores/{}'.format(consent_store_parent, consent_store_id)
    patch = {'defaultConsentTtl': default_consent_ttl}
    request = client.projects().locations().datasets().consentStores().patch(name=consent_store_name, updateMask='defaultConsentTtl', body=patch)
    response = request.execute()
    print('Patched consent store {} with new default consent TTL: {}'.format(consent_store_id, default_consent_ttl))
    return response

def get_consent_store_iam_policy(project_id: str, location: str, dataset_id: str, consent_store_id: str):
    if False:
        return 10
    'Gets the IAM policy for the specified consent store.\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/consent\n    before running the sample.'
    from googleapiclient import discovery
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    consent_store_parent = 'projects/{}/locations/{}/datasets/{}'.format(project_id, location, dataset_id)
    consent_store_name = '{}/consentStores/{}'.format(consent_store_parent, consent_store_id)
    request = client.projects().locations().datasets().consentStores().getIamPolicy(resource=consent_store_name)
    response = request.execute()
    print('etag: {}'.format(response.get('name')))
    return response

def set_consent_store_iam_policy(project_id: str, location: str, dataset_id: str, consent_store_id: str, member, role, etag=None):
    if False:
        while True:
            i = 10
    "Sets the IAM policy for the specified consent store.\n    A single member will be assigned a single role. A member can be any of:\n    - allUsers, that is, anyone\n    - allAuthenticatedUsers, anyone authenticated with a Google account\n    - user:email, as in 'user:somebody@example.com'\n    - group:email, as in 'group:admins@example.com'\n    - domain:domainname, as in 'domain:example.com'\n    - serviceAccount:email,\n        as in 'serviceAccount:my-other-app@appspot.gserviceaccount.com'\n    A role can be any IAM role, such as 'roles/viewer', 'roles/owner',\n    or 'roles/editor'\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/consent\n    before running the sample."
    from googleapiclient import discovery
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    consent_store_parent = 'projects/{}/locations/{}/datasets/{}'.format(project_id, location, dataset_id)
    consent_store_name = '{}/consentStores/{}'.format(consent_store_parent, consent_store_id)
    policy = {'bindings': [{'role': role, 'members': [member]}]}
    if etag is not None:
        policy['etag'] = etag
    request = client.projects().locations().datasets().consentStores().setIamPolicy(resource=consent_store_name, body={'policy': policy})
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
    parser.add_argument('--consent_store_id', default=None, help='Name of consent store')
    parser.add_argument('--default_consent_ttl', default=None, help='Default time-to-live (TTL) of consents in the consent store.')
    parser.add_argument('--export_format', choices=['FORMAT_UNSPECIFIED', 'consent', 'JSON_BIGQUERY_IMPORT'], default='consent', help='Specifies the output format. If the format is unspecified, thedefault functionality is to export to consent.')
    parser.add_argument('--member', default=None, help='Member to add to IAM policy (e.g. "domain:example.com")')
    parser.add_argument('--role', default=None, help='IAM Role to give to member (e.g. "roles/viewer")')
    command = parser.add_subparsers(dest='command')
    command.add_parser('create-consent-store', help=create_consent_store.__doc__)
    command.add_parser('delete-consent-store', help=delete_consent_store.__doc__)
    command.add_parser('get-consent-store', help=get_consent_store.__doc__)
    command.add_parser('list-consent-stores', help=list_consent_stores.__doc__)
    command.add_parser('patch-consent-store', help=patch_consent_store.__doc__)
    command.add_parser('get_iam_policy', help=get_consent_store_iam_policy.__doc__)
    command.add_parser('set_iam_policy', help=set_consent_store_iam_policy.__doc__)
    return parser.parse_args()

def run_command(args):
    if False:
        i = 10
        return i + 15
    'Calls the program using the specified command.'
    if args.project_id is None:
        print('You must specify a project ID or set the "GOOGLE_CLOUD_PROJECT" environment variable.')
        return
    elif args.command == 'create-consent-store':
        create_consent_store(args.project_id, args.location, args.dataset_id, args.consent_store_id)
    elif args.command == 'delete-consent-store':
        delete_consent_store(args.project_id, args.location, args.dataset_id, args.consent_store_id)
    elif args.command == 'get-consent-store':
        get_consent_store(args.project_id, args.location, args.dataset_id, args.consent_store_id)
    elif args.command == 'list-consent-stores':
        list_consent_stores(args.project_id, args.location, args.dataset_id)
    elif args.command == 'patch-consent-store':
        patch_consent_store(args.project_id, args.location, args.dataset_id, args.consent_store_id, args.default_consent_ttl)
    elif args.command == 'get_iam_policy':
        get_consent_store_iam_policy(args.project_id, args.location, args.dataset_id, args.consent_store_id)
    elif args.command == 'set_iam_policy':
        set_consent_store_iam_policy(args.project_id, args.location, args.dataset_id, args.consent_store_id, args.member, args.role)

def main():
    if False:
        while True:
            i = 10
    args = parse_command_line_args()
    run_command(args)
if __name__ == '__main__':
    main()