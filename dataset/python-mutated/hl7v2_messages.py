import argparse
import os

def create_hl7v2_message(project_id, location, dataset_id, hl7v2_store_id, hl7v2_message_file):
    if False:
        for i in range(10):
            print('nop')
    'Creates an HL7v2 message and sends a notification to the\n    Cloud Pub/Sub topic.\n\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/hl7v2\n    before running the sample.'
    from googleapiclient import discovery
    import json
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    hl7v2_parent = f'projects/{project_id}/locations/{location}'
    hl7v2_store_name = '{}/datasets/{}/hl7V2Stores/{}'.format(hl7v2_parent, dataset_id, hl7v2_store_id)
    with open(hl7v2_message_file) as hl7v2_message:
        hl7v2_message_content = json.load(hl7v2_message)
    request = client.projects().locations().datasets().hl7V2Stores().messages().create(parent=hl7v2_store_name, body=hl7v2_message_content)
    response = request.execute()
    print(f'Created HL7v2 message from file: {hl7v2_message_file}')
    return response

def delete_hl7v2_message(project_id, location, dataset_id, hl7v2_store_id, hl7v2_message_id):
    if False:
        i = 10
        return i + 15
    'Deletes an HL7v2 message.\n\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/hl7v2\n    before running the sample.'
    from googleapiclient import discovery
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    hl7v2_parent = f'projects/{project_id}/locations/{location}'
    hl7v2_message = '{}/datasets/{}/hl7V2Stores/{}/messages/{}'.format(hl7v2_parent, dataset_id, hl7v2_store_id, hl7v2_message_id)
    request = client.projects().locations().datasets().hl7V2Stores().messages().delete(name=hl7v2_message)
    response = request.execute()
    print(f'Deleted HL7v2 message with ID: {hl7v2_message_id}')
    return response

def get_hl7v2_message(project_id, location, dataset_id, hl7v2_store_id, hl7v2_message_id):
    if False:
        return 10
    'Gets an HL7v2 message.\n\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/hl7v2\n    before running the sample.'
    from googleapiclient import discovery
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    hl7v2_parent = f'projects/{project_id}/locations/{location}'
    hl7v2_message_name = '{}/datasets/{}/hl7V2Stores/{}/messages/{}'.format(hl7v2_parent, dataset_id, hl7v2_store_id, hl7v2_message_id)
    msgs = client.projects().locations().datasets().hl7V2Stores().messages()
    message = msgs.get(name=hl7v2_message_name).execute()
    print('Name: {}'.format(message.get('name')))
    print('Data: {}'.format(message.get('data')))
    print('Creation time: {}'.format(message.get('createTime')))
    print('Sending facility: {}'.format(message.get('sendFacility')))
    print('Time sent: {}'.format(message.get('sendTime')))
    print('Message type: {}'.format(message.get('messageType')))
    print('Patient IDs:')
    patient_ids = message.get('patientIds')
    for patient_id in patient_ids:
        print('\tPatient value: {}'.format(patient_id.get('value')))
        print('\tPatient type: {}'.format(patient_id.get('type')))
    print('Labels: {}'.format(message.get('labels')))
    print(message)
    return message

def ingest_hl7v2_message(project_id, location, dataset_id, hl7v2_store_id, hl7v2_message_file):
    if False:
        while True:
            i = 10
    'Ingests a new HL7v2 message from the hospital and sends a notification\n    to the Cloud Pub/Sub topic. Return is an HL7v2 ACK message if the message\n    was successfully stored.\n\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/hl7v2\n    before running the sample.'
    from googleapiclient import discovery
    import json
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    hl7v2_parent = f'projects/{project_id}/locations/{location}'
    hl7v2_store_name = '{}/datasets/{}/hl7V2Stores/{}'.format(hl7v2_parent, dataset_id, hl7v2_store_id)
    with open(hl7v2_message_file) as hl7v2_message:
        hl7v2_message_content = json.load(hl7v2_message)
    request = client.projects().locations().datasets().hl7V2Stores().messages().ingest(parent=hl7v2_store_name, body=hl7v2_message_content)
    response = request.execute()
    print(f'Ingested HL7v2 message from file: {hl7v2_message_file}')
    return response

def list_hl7v2_messages(project_id, location, dataset_id, hl7v2_store_id):
    if False:
        while True:
            i = 10
    'Lists all the messages in the given HL7v2 store with support for\n    filtering.\n\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/hl7v2\n    before running the sample.'
    from googleapiclient import discovery
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    hl7v2_messages_parent = 'projects/{}/locations/{}/datasets/{}'.format(project_id, location, dataset_id)
    hl7v2_message_path = '{}/hl7V2Stores/{}'.format(hl7v2_messages_parent, hl7v2_store_id)
    hl7v2_messages = client.projects().locations().datasets().hl7V2Stores().messages().list(parent=hl7v2_message_path).execute().get('hl7V2Messages', [])
    for hl7v2_message in hl7v2_messages:
        print(hl7v2_message)
    return hl7v2_messages

def patch_hl7v2_message(project_id, location, dataset_id, hl7v2_store_id, hl7v2_message_id, label_key, label_value):
    if False:
        return 10
    'Updates the message.\n\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/hl7v2\n    before running the sample.'
    from googleapiclient import discovery
    api_version = 'v1'
    service_name = 'healthcare'
    client = discovery.build(service_name, api_version)
    hl7v2_message_parent = f'projects/{project_id}/locations/{location}'
    hl7v2_message_name = '{}/datasets/{}/hl7V2Stores/{}/messages/{}'.format(hl7v2_message_parent, dataset_id, hl7v2_store_id, hl7v2_message_id)
    patch = {'labels': {label_key: label_value}}
    request = client.projects().locations().datasets().hl7V2Stores().messages().patch(name=hl7v2_message_name, updateMask='labels', body=patch)
    response = request.execute()
    print('Patched HL7v2 message {} with labels:\n\t{}: {}'.format(hl7v2_message_id, label_key, label_value))
    return response

def parse_command_line_args():
    if False:
        print('Hello World!')
    'Parses command line arguments.'
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--project_id', default=os.environ.get('GOOGLE_CLOUD_PROJECT'), help='GCP project name')
    parser.add_argument('--location', default='us-central1', help='GCP location')
    parser.add_argument('--dataset_id', default=None, help='Name of dataset')
    parser.add_argument('--hl7v2_store_id', default=None, help='Name of HL7v2 store')
    parser.add_argument('--hl7v2_message_file', default=None, help='A file containing a base64-encoded HL7v2 message')
    parser.add_argument('--hl7v2_message_id', default=None, help='The identifier for the message returned by the server')
    parser.add_argument('--label_key', default=None, help='Arbitrary label key to apply to the message')
    parser.add_argument('--label_value', default=None, help='Arbitrary label value to apply to the message')
    command = parser.add_subparsers(dest='command')
    command.add_parser('create-hl7v2-message', help=create_hl7v2_message.__doc__)
    command.add_parser('delete-hl7v2-message', help=delete_hl7v2_message.__doc__)
    command.add_parser('get-hl7v2-message', help=get_hl7v2_message.__doc__)
    command.add_parser('ingest-hl7v2-message', help=ingest_hl7v2_message.__doc__)
    command.add_parser('list-hl7v2-messages', help=list_hl7v2_messages.__doc__)
    command.add_parser('patch-hl7v2-message', help=patch_hl7v2_message.__doc__)
    return parser.parse_args()

def run_command(args):
    if False:
        return 10
    'Calls the program using the specified command.'
    if args.project_id is None:
        print('You must specify a project ID or set the "GOOGLE_CLOUD_PROJECT" environment variable.')
        return
    elif args.command == 'create-hl7v2-message':
        create_hl7v2_message(args.project_id, args.location, args.dataset_id, args.hl7v2_store_id, args.hl7v2_message_file)
    elif args.command == 'delete-hl7v2-message':
        delete_hl7v2_message(args.project_id, args.location, args.dataset_id, args.hl7v2_store_id, args.hl7v2_message_id)
    elif args.command == 'get-hl7v2-message':
        get_hl7v2_message(args.project_id, args.location, args.dataset_id, args.hl7v2_store_id, args.hl7v2_message_id)
    elif args.command == 'ingest-hl7v2-message':
        ingest_hl7v2_message(args.project_id, args.location, args.dataset_id, args.hl7v2_store_id, args.hl7v2_message_file)
    elif args.command == 'list-hl7v2-messages':
        list_hl7v2_messages(args.project_id, args.location, args.dataset_id, args.hl7v2_store_id)
    elif args.command == 'patch-hl7v2-message':
        patch_hl7v2_message(args.project_id, args.location, args.dataset_id, args.hl7v2_store_id, args.hl7v2_message_id, args.label_key, args.label_value)

def main():
    if False:
        i = 10
        return i + 15
    args = parse_command_line_args()
    run_command(args)
if __name__ == '__main__':
    main()