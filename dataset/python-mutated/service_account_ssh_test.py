import base64
import json
import os
import random
from subprocess import CalledProcessError
import time
import backoff
from google.auth.exceptions import RefreshError
from google.oauth2 import service_account
import googleapiclient.discovery
import pytest
from service_account_ssh import main
'\nThe service account that runs this test must have the following roles:\n- roles/compute.instanceAdmin.v1\n- roles/compute.securityAdmin\n- roles/iam.serviceAccountAdmin\n- roles/iam.serviceAccountKeyAdmin\n- roles/iam.serviceAccountUser\nThe Project Editor legacy role is not sufficient because it does not grant\nseveral necessary permissions.\n'

def test_main():
    if False:
        i = 10
        return i + 15
    pytest.skip('We are disabling this test, as it will be replaced.')
    cmd = 'uname -a'
    project = os.environ['GOOGLE_CLOUD_PROJECT']
    test_id = f'oslogin-test-{str(random.randint(0, 1000000))}'
    zone = 'us-east1-d'
    image_family = 'projects/debian-cloud/global/images/family/debian-11'
    machine_type = f'zones/{zone}/machineTypes/f1-micro'
    account_email = '{test_id}@{project}.iam.gserviceaccount.com'.format(test_id=test_id, project=project)
    iam = googleapiclient.discovery.build('iam', 'v1', cache_discovery=False)
    compute = googleapiclient.discovery.build('compute', 'v1', cache_discovery=False)
    try:
        print('Creating test resources.')
        service_account_key = setup_resources(compute, iam, project, test_id, zone, image_family, machine_type, account_email)
    except Exception:
        print('Cleaning up partially created test resources.')
        cleanup_resources(compute, iam, project, test_id, zone, account_email)
        raise Exception('Could not set up the necessary test resources.')
    hostname = compute.instances().get(project=project, zone=zone, instance=test_id, fields='networkInterfaces/accessConfigs/natIP').execute()['networkInterfaces'][0]['accessConfigs'][0]['natIP']
    credentials = service_account.Credentials.from_service_account_info(json.loads(base64.b64decode(service_account_key['privateKeyData']).decode('utf-8')))
    oslogin = googleapiclient.discovery.build('oslogin', 'v1', cache_discovery=False, credentials=credentials)
    account = 'users/' + account_email

    @backoff.on_exception(backoff.expo, (CalledProcessError, RefreshError), max_tries=5)
    def ssh_login():
        if False:
            for i in range(10):
                print('nop')
        response = main(cmd, project, test_id, zone, oslogin, account, hostname)
        response = ' '.join(response)
        assert_value = f'{test_id}'
        assert assert_value in response
    ssh_login()
    cleanup_resources(compute, iam, project, test_id, zone, account_email)

def setup_resources(compute, iam, project, test_id, zone, image_family, machine_type, account_email):
    if False:
        print('Hello World!')
    iam.projects().serviceAccounts().create(name='projects/' + project, body={'accountId': test_id}).execute()
    time.sleep(5)
    iam.projects().serviceAccounts().setIamPolicy(resource='projects/' + project + '/serviceAccounts/' + account_email, body={'policy': {'bindings': [{'members': ['serviceAccount:' + account_email], 'role': 'roles/iam.serviceAccountUser'}]}}).execute()
    service_account_key = iam.projects().serviceAccounts().keys().create(name='projects/' + project + '/serviceAccounts/' + account_email, body={}).execute()
    firewall_config = {'name': test_id, 'network': '/global/networks/default', 'targetServiceAccounts': [account_email], 'sourceRanges': ['0.0.0.0/0'], 'allowed': [{'IPProtocol': 'tcp', 'ports': ['22']}]}
    compute.firewalls().insert(project=project, body=firewall_config).execute()
    instance_config = {'name': test_id, 'machineType': machine_type, 'disks': [{'boot': True, 'autoDelete': True, 'initializeParams': {'sourceImage': image_family}}], 'networkInterfaces': [{'network': 'global/networks/default', 'accessConfigs': [{'type': 'ONE_TO_ONE_NAT', 'name': 'External NAT'}]}], 'serviceAccounts': [{'email': account_email, 'scopes': ['https://www.googleapis.com/auth/cloud-platform']}], 'metadata': {'items': [{'key': 'enable-oslogin', 'value': 'TRUE'}]}}
    operation = compute.instances().insert(project=project, zone=zone, body=instance_config).execute()
    while compute.zoneOperations().get(project=project, zone=zone, operation=operation['name']).execute()['status'] != 'DONE':
        time.sleep(5)
    time.sleep(10)
    compute.instances().setIamPolicy(project=project, zone=zone, resource=test_id, body={'bindings': [{'members': ['serviceAccount:' + account_email], 'role': 'roles/compute.osLogin'}]}).execute()
    while compute.instances().getIamPolicy(project=project, zone=zone, resource=test_id, fields='bindings/role').execute()['bindings'][0]['role'] != 'roles/compute.osLogin':
        time.sleep(5)
    return service_account_key

def cleanup_resources(compute, iam, project, test_id, zone, account_email):
    if False:
        print('Hello World!')
    try:
        compute.firewalls().delete(project=project, firewall=test_id).execute()
    except Exception:
        pass
    try:
        delete = compute.instances().delete(project=project, zone=zone, instance=test_id).execute()
        while compute.zoneOperations().get(project=project, zone=zone, operation=delete['name']).execute()['status'] != 'DONE':
            time.sleep(5)
    except Exception:
        pass
    try:
        iam.projects().serviceAccounts().delete(name='projects/' + project + '/serviceAccounts/' + account_email).execute()
    except Exception:
        pass