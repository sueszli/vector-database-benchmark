"""
FILE: sample_set_get_image.py

DESCRIPTION:
    This sample demonstrates setting and getting OCI and non-OCI images to a repository.

USAGE:
    python sample_set_get_image.py

    Set the environment variables with your own values before running the sample:
    1) CONTAINERREGISTRY_ENDPOINT - The URL of your Container Registry account
"""
import os
import json
from io import BytesIO
from dotenv import find_dotenv, load_dotenv
from azure.containerregistry import ContainerRegistryClient, DigestValidationError
from utilities import get_authority, get_credential

class SetGetImage(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        load_dotenv(find_dotenv())
        self.endpoint = os.environ['CONTAINERREGISTRY_ENDPOINT']
        self.authority = get_authority(self.endpoint)
        self.credential = get_credential(self.authority)

    def set_oci_image(self):
        if False:
            print('Hello World!')
        self.repository_name = 'sample-oci-image'
        layer = BytesIO(b'Sample layer')
        config = BytesIO(json.dumps({'sample config': 'content'}).encode())
        with ContainerRegistryClient(self.endpoint, self.credential) as client:
            (layer_digest, layer_size) = client.upload_blob(self.repository_name, layer)
            print(f'Uploaded layer: digest - {layer_digest}, size - {layer_size}')
            (config_digest, config_size) = client.upload_blob(self.repository_name, config)
            print(f'Uploaded config: digest - {config_digest}, size - {config_size}')
            oci_manifest = {'config': {'mediaType': 'application/vnd.oci.image.config.v1+json', 'digest': config_digest, 'sizeInBytes': config_size}, 'schemaVersion': 2, 'layers': [{'mediaType': 'application/vnd.oci.image.layer.v1.tar', 'digest': layer_digest, 'size': layer_size, 'annotations': {'org.opencontainers.image.ref.name': 'artifact.txt'}}]}
            manifest_digest = client.set_manifest(self.repository_name, oci_manifest, tag='latest')
            print(f'Uploaded manifest: digest - {manifest_digest}')

    def get_oci_image(self):
        if False:
            return 10
        with ContainerRegistryClient(self.endpoint, self.credential) as client:
            get_manifest_result = client.get_manifest(self.repository_name, 'latest')
            received_manifest = get_manifest_result.manifest
            print(f'Got manifest:\n{received_manifest}')
            for layer in received_manifest['layers']:
                layer_file_name = layer['digest'].split(':')[1]
                try:
                    stream = client.download_blob(self.repository_name, layer['digest'])
                    with open(layer_file_name, 'wb') as layer_file:
                        for chunk in stream:
                            layer_file.write(chunk)
                except DigestValidationError:
                    print(f'Downloaded layer digest value did not match. Deleting file {layer_file_name}.')
                    os.remove(layer_file_name)
                print(f'Got layer: {layer_file_name}')
            config_file_name = 'config.json'
            try:
                stream = client.download_blob(self.repository_name, received_manifest['config']['digest'])
                with open(config_file_name, 'wb') as config_file:
                    for chunk in stream:
                        config_file.write(chunk)
            except DigestValidationError:
                print(f'Downloaded config digest value did not match. Deleting file {config_file_name}.')
                os.remove(config_file_name)
            print(f'Got config: {config_file_name}')

    def delete_blob(self):
        if False:
            i = 10
            return i + 15
        with ContainerRegistryClient(self.endpoint, self.credential) as client:
            get_manifest_result = client.get_manifest(self.repository_name, 'latest')
            received_manifest = get_manifest_result.manifest
            for layer in received_manifest['layers']:
                client.delete_blob(self.repository_name, layer['digest'])
            client.delete_blob(self.repository_name, received_manifest['config']['digest'])

    def delete_oci_image(self):
        if False:
            print('Hello World!')
        with ContainerRegistryClient(self.endpoint, self.credential) as client:
            get_manifest_result = client.get_manifest(self.repository_name, 'latest')
            client.delete_manifest(self.repository_name, get_manifest_result.digest)

    def set_get_oci_image(self):
        if False:
            return 10
        self.set_oci_image()
        self.get_oci_image()
        self.delete_blob()
        self.delete_oci_image()

    def set_get_docker_image(self):
        if False:
            return 10
        repository_name = 'sample-docker-image'
        with ContainerRegistryClient(self.endpoint, self.credential) as client:
            layer = BytesIO(b'Sample layer')
            (layer_digest, layer_size) = client.upload_blob(repository_name, layer)
            print(f'Uploaded layer: digest - {layer_digest}, size - {layer_size}')
            config = BytesIO(json.dumps({'sample config': 'content'}).encode())
            (config_digest, config_size) = client.upload_blob(repository_name, config)
            print(f'Uploaded config: digest - {config_digest}, size - {config_size}')
            docker_manifest = {'config': {'digest': config_digest, 'mediaType': 'application/vnd.docker.container.image.v1+json', 'size': config_size}, 'layers': [{'digest': layer_digest, 'mediaType': 'application/vnd.docker.image.rootfs.diff.tar.gzip', 'size': layer_size}], 'mediaType': 'application/vnd.docker.distribution.manifest.v2+json', 'schemaVersion': 2}
            client.set_manifest(repository_name, docker_manifest, tag='sample', media_type=str(docker_manifest['mediaType']))
            get_manifest_result = client.get_manifest(repository_name, 'sample')
            received_manifest = get_manifest_result.manifest
            print(received_manifest)
            received_manifest_media_type = get_manifest_result.media_type
            print(received_manifest_media_type)
            client.delete_manifest(repository_name, get_manifest_result.digest)
if __name__ == '__main__':
    sample = SetGetImage()
    sample.set_get_oci_image()
    sample.set_get_docker_image()