"""
FILE: sample_set_image_properties.py

DESCRIPTION:
    This sample demonstrates setting an image's properties on the tag so it can't be overwritten during a lengthy
    deployment.

USAGE:
    python sample_set_image_properties.py

    Set the environment variables with your own values before running the sample:
    1) CONTAINERREGISTRY_ENDPOINT - The URL of your Container Registry account
    
    This sample assumes your registry has a repository "library/hello-world" with image tagged "v1",
    run load_registry() if you don't have.
    Set the environment variables with your own values before running load_registry():
    1) CONTAINERREGISTRY_ENDPOINT - The URL of your Container Registry account
    2) CONTAINERREGISTRY_TENANT_ID - The service principal's tenant ID
    3) CONTAINERREGISTRY_CLIENT_ID - The service principal's client ID
    4) CONTAINERREGISTRY_CLIENT_SECRET - The service principal's client secret
"""
import os
from dotenv import find_dotenv, load_dotenv
from azure.containerregistry import ContainerRegistryClient
from utilities import load_registry, get_authority, get_credential

class SetImageProperties(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        load_dotenv(find_dotenv())
        self.endpoint = os.environ['CONTAINERREGISTRY_ENDPOINT']
        self.authority = get_authority(self.endpoint)
        self.credential = get_credential(self.authority)

    def set_image_properties(self):
        if False:
            while True:
                i = 10
        load_registry(self.endpoint)
        with ContainerRegistryClient(self.endpoint, self.credential) as client:
            client.update_manifest_properties('library/hello-world', 'v1', can_write=False, can_delete=False)
if __name__ == '__main__':
    sample = SetImageProperties()
    sample.set_image_properties()