#  Copyright 2022 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


# [START compute_certificate_create_regional]
from pathlib import Path
from pprint import pprint
from typing import Union

from googleapiclient import discovery


def create_regional_certificate(
    project_id: str,
    region: str,
    certificate_file: Union[str, Path],
    private_key_file: Union[str, Path],
    certificate_name: str,
    description: str = "Certificate created from a code sample.",
) -> dict:
    """
    Create a regional SSL self-signed certificate within your Google Cloud project.

    Args:
        project_id: project ID or project number of the Cloud project you want to use.
        region: name of the region you want to use.
        certificate_file: path to the file with the certificate you want to create in your project.
        private_key_file: path to the private key you used to sign the certificate with.
        certificate_name: name for the certificate once it's created in your project.
        description: description of the certificate.

        Returns:
        Dictionary with information about the new regional SSL self-signed certificate.
    """
    service = discovery.build("compute", "v1")

    # Read the cert into memory
    with open(certificate_file) as f:
        _temp_cert = f.read()

    # Read the private_key into memory
    with open(private_key_file) as f:
        _temp_key = f.read()

    # Now that the certificate and private key are in memory, you can create the
    # certificate resource
    ssl_certificate_body = {
        "name": certificate_name,
        "description": description,
        "certificate": _temp_cert,
        "privateKey": _temp_key,
    }
    request = service.regionSslCertificates().insert(
        project=project_id, region=region, body=ssl_certificate_body
    )
    response = request.execute()
    pprint(response)

    return response


# [END compute_certificate_create_regional]
