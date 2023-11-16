from azure.identity import DefaultAzureCredential
from azure.mgmt.compute import ComputeManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-compute\n# USAGE\n    python image_create_data_disk_from_ablob_included.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        i = 10
        return i + 15
    client = ComputeManagementClient(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.images.begin_create_or_update(resource_group_name='myResourceGroup', image_name='myImage', parameters={'location': 'West US', 'properties': {'storageProfile': {'dataDisks': [{'blobUri': 'https://mystorageaccount.blob.core.windows.net/dataimages/dataimage.vhd', 'lun': 1}], 'osDisk': {'blobUri': 'https://mystorageaccount.blob.core.windows.net/osimages/osimage.vhd', 'osState': 'Generalized', 'osType': 'Linux'}, 'zoneResilient': False}}}).result()
    print(response)
if __name__ == '__main__':
    main()