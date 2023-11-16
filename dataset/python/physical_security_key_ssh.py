#!/usr/bin/env python
#  Copyright 2022 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# [START compute_oslogin_physical_sk_script]
import argparse
import os
import subprocess
from typing import Optional

import googleapiclient.discovery


def write_ssh_key_files(security_keys: list[dict], directory: str) -> list[str]:
    """
    Store the SSH key files.

    Saves the SSH keys into files inside specified directory. Using the naming
    template of `google_sk_{i}`.

    Args:
        security_keys: list of dictionaries representing security keys retrieved
            from the OSLogin API.
        directory: path to directory in which the security keys will be stored.

    Returns:
        List of paths to the saved keys.
    """
    key_files = []
    for index, key in enumerate(security_keys):
        key_file = os.path.join(directory, f"google_sk_{index}")
        with open(key_file, "w") as f:
            f.write(key.get("privateKey"))
            os.chmod(key_file, 0o600)
            key_files.append(key_file)
    return key_files


def ssh_command(key_files: list[str], username: str, ip_address: str) -> list[str]:
    """
    Construct the SSH command for a given IP address and key files.

    Args:
        key_files: SSH keys to be used for authentication.
        username: username used to authenticate.
        ip_address: the IP address or hostname of the remote system.

    Returns:
        SSH command as a list of strings.
    """
    command = ["ssh"]
    for key_file in key_files:
        command.extend(["-i", key_file])
    command.append(f"{username}@{ip_address}")
    return command


def main(
    user_key: str, ip_address: str, dryrun: bool, directory: Optional[str] = None
) -> None:
    """
    Configure SSH key files and print SSH command.

    Args:
        user_key: name of the user you want to authenticate as. Usually an email address.
        ip_address: the IP address of the machine you want to connect to.
        dryrun: bool flag to do dry run, without connecting to the remote machine.
        directory: the directory to store SSH private keys.
    """
    directory = directory or os.path.join(os.path.expanduser("~"), ".ssh")

    # Create the OS Login API object.
    oslogin = googleapiclient.discovery.build("oslogin", "v1beta")

    # Retrieve security keys and OS Login username from a user's Google account.
    profile = (
        oslogin.users()
        .getLoginProfile(name=f"users/{user_key}", view="SECURITY_KEY")
        .execute()
    )

    if "posixAccounts" not in profile:
        print("You don't have a POSIX account configured.")
        print("Please make sure that you have enabled OS Login for your VM.")
        return

    username = profile.get("posixAccounts")[0].get("username")

    # Write the SSH private key files.
    security_keys = profile.get("securityKeys")

    if security_keys is None:
        print(
            "The account you are using to authenticate does not have any security keys assigned to it."
        )
        print(
            "Please check your Application Default Credentials "
            "(https://cloud.google.com/docs/authentication/application-default-credentials)."
        )
        print(
            "More info about using security keys: https://cloud.google.com/compute/docs/oslogin/security-keys"
        )
        return

    key_files = write_ssh_key_files(security_keys, directory)

    # Compose the SSH command.
    command = ssh_command(key_files, username, ip_address)

    if dryrun:
        # Print the SSH command.
        print(" ".join(command))
    else:
        # Connect to the IP address over SSH.
        subprocess.call(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--user_key", help="Your primary email address.")
    parser.add_argument(
        "--ip_address", help="The external IP address of the VM you want to connect to."
    )
    parser.add_argument("--directory", help="The directory to store SSH private keys.")
    parser.add_argument(
        "--dryrun",
        dest="dryrun",
        default=False,
        action="store_true",
        help="Turn off dryrun mode to execute the SSH command",
    )
    args = parser.parse_args()

    main(args.user_key, args.ip_address, args.dryrun, args.directory)
# [END compute_oslogin_physical_sk_script]
