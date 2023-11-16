from azure.identity import DefaultAzureCredential
from azure.mgmt.imagebuilder import ImageBuilderClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-imagebuilder\n# USAGE\n    python cancel_image_build.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        print('Hello World!')
    client = ImageBuilderClient(credential=DefaultAzureCredential(), subscription_id='{subscription-id}')
    response = client.virtual_machine_image_templates.begin_cancel(resource_group_name='myResourceGroup', image_template_name='myImageTemplate').result()
    print(response)
if __name__ == '__main__':
    main()