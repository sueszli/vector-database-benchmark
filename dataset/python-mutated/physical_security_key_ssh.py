import argparse
import os
import subprocess
from typing import Optional
import googleapiclient.discovery

def write_ssh_key_files(security_keys: list[dict], directory: str) -> list[str]:
    if False:
        print('Hello World!')
    '\n    Store the SSH key files.\n\n    Saves the SSH keys into files inside specified directory. Using the naming\n    template of `google_sk_{i}`.\n\n    Args:\n        security_keys: list of dictionaries representing security keys retrieved\n            from the OSLogin API.\n        directory: path to directory in which the security keys will be stored.\n\n    Returns:\n        List of paths to the saved keys.\n    '
    key_files = []
    for (index, key) in enumerate(security_keys):
        key_file = os.path.join(directory, f'google_sk_{index}')
        with open(key_file, 'w') as f:
            f.write(key.get('privateKey'))
            os.chmod(key_file, 384)
            key_files.append(key_file)
    return key_files

def ssh_command(key_files: list[str], username: str, ip_address: str) -> list[str]:
    if False:
        i = 10
        return i + 15
    '\n    Construct the SSH command for a given IP address and key files.\n\n    Args:\n        key_files: SSH keys to be used for authentication.\n        username: username used to authenticate.\n        ip_address: the IP address or hostname of the remote system.\n\n    Returns:\n        SSH command as a list of strings.\n    '
    command = ['ssh']
    for key_file in key_files:
        command.extend(['-i', key_file])
    command.append(f'{username}@{ip_address}')
    return command

def main(user_key: str, ip_address: str, dryrun: bool, directory: Optional[str]=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Configure SSH key files and print SSH command.\n\n    Args:\n        user_key: name of the user you want to authenticate as. Usually an email address.\n        ip_address: the IP address of the machine you want to connect to.\n        dryrun: bool flag to do dry run, without connecting to the remote machine.\n        directory: the directory to store SSH private keys.\n    '
    directory = directory or os.path.join(os.path.expanduser('~'), '.ssh')
    oslogin = googleapiclient.discovery.build('oslogin', 'v1beta')
    profile = oslogin.users().getLoginProfile(name=f'users/{user_key}', view='SECURITY_KEY').execute()
    if 'posixAccounts' not in profile:
        print("You don't have a POSIX account configured.")
        print('Please make sure that you have enabled OS Login for your VM.')
        return
    username = profile.get('posixAccounts')[0].get('username')
    security_keys = profile.get('securityKeys')
    if security_keys is None:
        print('The account you are using to authenticate does not have any security keys assigned to it.')
        print('Please check your Application Default Credentials (https://cloud.google.com/docs/authentication/application-default-credentials).')
        print('More info about using security keys: https://cloud.google.com/compute/docs/oslogin/security-keys')
        return
    key_files = write_ssh_key_files(security_keys, directory)
    command = ssh_command(key_files, username, ip_address)
    if dryrun:
        print(' '.join(command))
    else:
        subprocess.call(command)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--user_key', help='Your primary email address.')
    parser.add_argument('--ip_address', help='The external IP address of the VM you want to connect to.')
    parser.add_argument('--directory', help='The directory to store SSH private keys.')
    parser.add_argument('--dryrun', dest='dryrun', default=False, action='store_true', help='Turn off dryrun mode to execute the SSH command')
    args = parser.parse_args()
    main(args.user_key, args.ip_address, args.dryrun, args.directory)