from azure.identity import DefaultAzureCredential
from azure.mgmt.databoxedge import DataBoxEdgeManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-databoxedge\n# USAGE\n    python trigger_support_package.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = DataBoxEdgeManagementClient(credential=DefaultAzureCredential(), subscription_id='4385cf00-2d3a-425a-832f-f4285b1c9dce')
    response = client.support_packages.begin_trigger_support_package(device_name='testedgedevice', resource_group_name='GroupForEdgeAutomation', trigger_support_package_request={'properties': {'include': 'DefaultWithDumps', 'maximumTimeStamp': '2018-12-18T02:19:51.4270267Z', 'minimumTimeStamp': '2018-12-18T02:18:51.4270267Z'}}).result()
    print(response)
if __name__ == '__main__':
    main()