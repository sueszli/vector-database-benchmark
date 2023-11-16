import argparse
import json
import os

def dicomweb_store_instance(project_id, location, dataset_id, dicom_store_id, dcm_file):
    if False:
        i = 10
        return i + 15
    'Handles the POST requests specified in the DICOMweb standard.\n\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/dicom\n    before running the sample.'
    import os
    from google.auth.transport import requests
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_file(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform'])
    session = requests.AuthorizedSession(scoped_credentials)
    base_url = 'https://healthcare.googleapis.com/v1'
    url = f'{base_url}/projects/{project_id}/locations/{location}'
    dicomweb_path = '{}/datasets/{}/dicomStores/{}/dicomWeb/studies'.format(url, dataset_id, dicom_store_id)
    with open(dcm_file, 'rb') as dcm:
        dcm_content = dcm.read()
    headers = {'Content-Type': 'application/dicom'}
    response = session.post(dicomweb_path, data=dcm_content, headers=headers)
    response.raise_for_status()
    print('Stored DICOM instance:')
    print(response.text)
    return response

def dicomweb_search_instance(project_id, location, dataset_id, dicom_store_id):
    if False:
        return 10
    'Handles the GET requests specified in DICOMweb standard.\n\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/dicom\n    before running the sample.'
    import os
    from google.auth.transport import requests
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_file(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform'])
    session = requests.AuthorizedSession(scoped_credentials)
    base_url = 'https://healthcare.googleapis.com/v1'
    url = f'{base_url}/projects/{project_id}/locations/{location}'
    dicomweb_path = '{}/datasets/{}/dicomStores/{}/dicomWeb/instances'.format(url, dataset_id, dicom_store_id)
    headers = {'Content-Type': 'application/dicom+json; charset=utf-8'}
    response = session.get(dicomweb_path, headers=headers)
    response.raise_for_status()
    instances = response.json()
    print('Instances:')
    print(json.dumps(instances, indent=2))
    return instances

def dicomweb_retrieve_study(project_id, location, dataset_id, dicom_store_id, study_uid):
    if False:
        for i in range(10):
            print('nop')
    'Handles the GET requests specified in the DICOMweb standard.\n\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/dicom\n    before running the sample.'
    import os
    from google.auth.transport import requests
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_file(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform'])
    session = requests.AuthorizedSession(scoped_credentials)
    base_url = 'https://healthcare.googleapis.com/v1'
    url = f'{base_url}/projects/{project_id}/locations/{location}'
    dicomweb_path = '{}/datasets/{}/dicomStores/{}/dicomWeb/studies/{}'.format(url, dataset_id, dicom_store_id, study_uid)
    file_name = 'study.multipart'
    response = session.get(dicomweb_path)
    response.raise_for_status()
    with open(file_name, 'wb') as f:
        f.write(response.content)
        print(f'Retrieved study and saved to {file_name} in current directory')
    return response

def dicomweb_search_studies(project_id, location, dataset_id, dicom_store_id):
    if False:
        print('Hello World!')
    'Handles the GET requests specified in the DICOMweb standard.\n\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/dicom\n    before running the sample.'
    import os
    from google.auth.transport import requests
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_file(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform'])
    session = requests.AuthorizedSession(scoped_credentials)
    base_url = 'https://healthcare.googleapis.com/v1'
    url = f'{base_url}/projects/{project_id}/locations/{location}'
    dicomweb_path = '{}/datasets/{}/dicomStores/{}/dicomWeb/studies'.format(url, dataset_id, dicom_store_id)
    params = {'PatientName': 'Sally Zhang'}
    response = session.get(dicomweb_path, params=params)
    response.raise_for_status()
    print(f'Studies found: response is {response}')

def dicomweb_retrieve_instance(project_id, location, dataset_id, dicom_store_id, study_uid, series_uid, instance_uid):
    if False:
        print('Hello World!')
    'Handles the GET requests specified in the DICOMweb standard.\n\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/dicom\n    before running the sample.'
    import os
    from google.auth.transport import requests
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_file(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform'])
    session = requests.AuthorizedSession(scoped_credentials)
    base_url = 'https://healthcare.googleapis.com/v1'
    url = f'{base_url}/projects/{project_id}/locations/{location}'
    dicom_store_path = '{}/datasets/{}/dicomStores/{}'.format(url, dataset_id, dicom_store_id)
    dicomweb_path = '{}/dicomWeb/studies/{}/series/{}/instances/{}'.format(dicom_store_path, study_uid, series_uid, instance_uid)
    file_name = 'instance.dcm'
    headers = {'Accept': 'application/dicom; transfer-syntax=*'}
    response = session.get(dicomweb_path, headers=headers)
    response.raise_for_status()
    with open(file_name, 'wb') as f:
        f.write(response.content)
        print('Retrieved DICOM instance and saved to {} in current directory'.format(file_name))
    return response

def dicomweb_retrieve_rendered(project_id, location, dataset_id, dicom_store_id, study_uid, series_uid, instance_uid):
    if False:
        while True:
            i = 10
    'Handles the GET requests specified in the DICOMweb standard.\n\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/dicom\n    before running the sample.'
    import os
    from google.auth.transport import requests
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_file(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform'])
    session = requests.AuthorizedSession(scoped_credentials)
    base_url = 'https://healthcare.googleapis.com/v1'
    url = f'{base_url}/projects/{project_id}/locations/{location}'
    dicom_store_path = '{}/datasets/{}/dicomStores/{}'.format(url, dataset_id, dicom_store_id)
    dicomweb_path = '{}/dicomWeb/studies/{}/series/{}/instances/{}/rendered'.format(dicom_store_path, study_uid, series_uid, instance_uid)
    file_name = 'rendered_image.png'
    headers = {'Accept': 'image/png'}
    response = session.get(dicomweb_path, headers=headers)
    response.raise_for_status()
    with open(file_name, 'wb') as f:
        f.write(response.content)
        print('Retrieved rendered image and saved to {} in current directory'.format(file_name))
    return response

def dicomweb_delete_study(project_id, location, dataset_id, dicom_store_id, study_uid):
    if False:
        return 10
    'Handles DELETE requests equivalent to the GET requests specified in\n    the WADO-RS standard.\n\n    See https://github.com/GoogleCloudPlatform/python-docs-samples/tree/main/healthcare/api-client/v1/dicom\n    before running the sample.'
    import os
    from google.auth.transport import requests
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_file(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform'])
    session = requests.AuthorizedSession(scoped_credentials)
    base_url = 'https://healthcare.googleapis.com/v1'
    url = f'{base_url}/projects/{project_id}/locations/{location}'
    dicomweb_path = '{}/datasets/{}/dicomStores/{}/dicomWeb/studies/{}'.format(url, dataset_id, dicom_store_id, study_uid)
    headers = {'Content-Type': 'application/dicom+json; charset=utf-8'}
    response = session.delete(dicomweb_path, headers=headers)
    response.raise_for_status()
    print('Deleted study.')
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
    parser.add_argument('--dicom_store_id', default=None, help='Name of DICOM store')
    parser.add_argument('--dcm_file', default=None, help='File name for DCM file to store.')
    parser.add_argument('--study_uid', default=None, help='Unique identifier for a study.')
    parser.add_argument('--series_uid', default=None, help='Unique identifier for a series.')
    parser.add_argument('--instance_uid', default=None, help='Unique identifier for an instance.')
    command = parser.add_subparsers(dest='command')
    command.add_parser('dicomweb-store-instance', help=dicomweb_store_instance.__doc__)
    command.add_parser('dicomweb-search-instance', help=dicomweb_search_instance.__doc__)
    command.add_parser('dicomweb-retrieve-study', help=dicomweb_retrieve_study.__doc__)
    command.add_parser('dicomweb-search-studies', help=dicomweb_search_studies.__doc__)
    command.add_parser('dicomweb-retrieve-instance', help=dicomweb_retrieve_instance.__doc__)
    command.add_parser('dicomweb-retrieve-rendered', help=dicomweb_retrieve_rendered.__doc__)
    command.add_parser('dicomweb-delete-study', help=dicomweb_delete_study.__doc__)
    return parser.parse_args()

def run_command(args):
    if False:
        while True:
            i = 10
    'Calls the program using the specified command.'
    if args.project_id is None:
        print('You must specify a project ID or set the "GOOGLE_CLOUD_PROJECT" environment variable.')
        return
    elif args.command == 'dicomweb-store-instance':
        dicomweb_store_instance(args.project_id, args.location, args.dataset_id, args.dicom_store_id, args.dcm_file)
    elif args.command == 'dicomweb-search-instance':
        dicomweb_search_instance(args.project_id, args.location, args.dataset_id, args.dicom_store_id)
    elif args.command == 'dicomweb-retrieve-study':
        dicomweb_retrieve_study(args.project_id, args.location, args.dataset_id, args.dicom_store_id, args.study_uid)
    elif args.command == 'dicomweb-retrieve-instance':
        dicomweb_retrieve_instance(args.project_id, args.location, args.dataset_id, args.dicom_store_id, args.study_uid, args.series_uid, args.instance_uid)
    elif args.command == 'dicomweb-search-studies':
        dicomweb_search_studies(args.project_id, args.location, args.dataset_id, args.dicom_store_id)
    elif args.command == 'dicomweb-retrieve-rendered':
        dicomweb_retrieve_rendered(args.project_id, args.location, args.dataset_id, args.dicom_store_id, args.study_uid, args.series_uid, args.instance_uid)
    elif args.command == 'dicomweb-delete-study':
        dicomweb_delete_study(args.project_id, args.location, args.dataset_id, args.dicom_store_id, args.study_uid)

def main():
    if False:
        for i in range(10):
            print('nop')
    args = parse_command_line_args()
    run_command(args)
if __name__ == '__main__':
    main()