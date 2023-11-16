"""Example of using the OS Login API to apply public SSH keys for a service
account, and use that service account to execute commands on a remote
instance over SSH. This example uses zonal DNS names to address instances
on the same internal VPC network.
"""
import argparse
import logging
import subprocess
import time
from typing import List, Optional
import uuid
from google.auth.exceptions import RefreshError
import googleapiclient.discovery
import requests
SERVICE_ACCOUNT_METADATA_URL = 'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email'
HEADERS = {'Metadata-Flavor': 'Google'}

def execute(cmd: List[str], cwd: Optional[str]=None, capture_output: bool=False, env: Optional[dict]=None, raise_errors: bool=True) -> (int, str):
    if False:
        while True:
            i = 10
    '\n    Execute an external command (wrapper for Python subprocess).\n\n    Args:\n        cmd: command to be executed, presented as list of strings.\n        cwd: directory where you want to execute the command.\n        capture_output: do you want to capture the commands output?\n        env: environmental variables to be used for command execution.\n        raise_errors: should errors of the executed command be raised as exception?\n\n    Returns:\n        A tuple containing the return code of the command and its output.\n    '
    logging.info(f'Executing command: {str(cmd)}')
    stdout = subprocess.PIPE if capture_output else None
    process = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=stdout)
    output = process.communicate()[0]
    returncode = process.returncode
    if returncode:
        if raise_errors:
            raise subprocess.CalledProcessError(returncode, cmd)
        else:
            logging.info('Command returned error status %s', returncode)
    if output:
        logging.info(output)
    return (returncode, output)

def create_ssh_key(oslogin: googleapiclient.discovery.Resource, account: str, private_key_file: Optional[str]=None, expire_time: int=300) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Generate an SSH key pair and apply it to the specified account.\n\n    Args:\n        oslogin: the OSLogin resource object, needed to communicate with API.\n        account: name of the account to be used.\n        private_key_file: path at which the private key file will be stored.\n        expire_time: expiration time of the SSH key (is seconds).\n\n    Returns:\n        Path to the private SSH key file.mypy\n    '
    private_key_file = private_key_file or '/tmp/key-' + str(uuid.uuid4())
    execute(['ssh-keygen', '-t', 'rsa', '-N', '', '-f', private_key_file])
    with open(private_key_file + '.pub') as original:
        public_key = original.read().strip()
    expiration = int((time.time() + expire_time) * 1000000)
    body = {'key': public_key, 'expirationTimeUsec': expiration}
    print(f'Creating key {account} and {body}')
    for attempt_no in range(1, 4):
        try:
            oslogin.users().importSshPublicKey(parent=account, body=body).execute()
        except RefreshError as err:
            if attempt_no == 3:
                raise err
            time.sleep(attempt_no)
        else:
            break
    return private_key_file

def run_ssh(cmd: str, private_key_file: str, username: str, hostname: str) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Run a command on a remote system.\n\n    Args:\n        cmd: the command to be run on remote system.\n        private_key_file: private SSH key to use for authentication.\n        username: username on the remote system.\n        hostname: name of the remote system.\n\n    Returns:\n        A list of strings representing the commands output.\n    '
    ssh_command = ['ssh', '-i', private_key_file, '-o', 'StrictHostKeyChecking=no', f'{username}@{hostname}', cmd]
    ssh = subprocess.Popen(ssh_command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    result = ssh.stdout.readlines()
    return result if result else ssh.stderr.readlines()

def main(cmd: str, project: str, instance: Optional[str]=None, zone: Optional[str]=None, oslogin: Optional[googleapiclient.discovery.Resource]=None, account: Optional[str]=None, hostname: Optional[str]=None) -> List[str]:
    if False:
        while True:
            i = 10
    '\n    Run a command on a remote system.\n\n    This method will first create a new SSH key and then use it to\n    execute a specified command over SSH on remote machine.\n\n    The generated SSH key will be safely deleted at the end.\n\n    Args:\n        cmd: command to execute on remote host.\n        project: name of the project that the remote host resides in.\n        instance: name of the remote host.\n        zone: zone in which the remote host can be found.\n        oslogin: the OSLogin client to be used. New one will be created if left as None.\n        account: name of the account to be used\n        hostname: hostname of the remote system.\n\n    Returns:\n        Output of the executed command.\n    '
    oslogin = oslogin or googleapiclient.discovery.build('oslogin', 'v1')
    account = account or requests.get(SERVICE_ACCOUNT_METADATA_URL, headers=HEADERS).text
    if not account.startswith('users/'):
        account = 'users/' + account
    private_key_file = create_ssh_key(oslogin, account)
    for attempt_no in range(1, 4):
        try:
            profile = oslogin.users().getLoginProfile(name=account).execute()
        except RefreshError as err:
            if attempt_no == 3:
                raise err
            time.sleep(attempt_no)
        else:
            username = profile.get('posixAccounts')[0].get('username')
            break
    hostname = hostname or '{instance}.{zone}.c.{project}.internal'.format(instance=instance, zone=zone, project=project)
    result = run_ssh(cmd, private_key_file, username, hostname)
    for line in result:
        print(line.rstrip('\n\r'))
    execute(['shred', private_key_file])
    execute(['rm', private_key_file])
    execute(['rm', private_key_file + '.pub'])
    return result
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cmd', default='uname -a', help='The command to run on the remote instance.')
    parser.add_argument('--project', help='Your Google Cloud project ID.')
    parser.add_argument('--zone', help='The zone where the target instance is located.')
    parser.add_argument('--instance', help='The target instance for the ssh command.')
    parser.add_argument('--account', help='The service account email.')
    parser.add_argument('--hostname', help='The external IP address or hostname for the target instance.')
    args = parser.parse_args()
    main(args.cmd, args.project, instance=args.instance, zone=args.zone, account=args.account, hostname=args.hostname)