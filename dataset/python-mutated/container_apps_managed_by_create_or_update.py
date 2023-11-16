from azure.identity import DefaultAzureCredential
from azure.mgmt.appcontainers import ContainerAppsAPIClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-appcontainers\n# USAGE\n    python container_apps_managed_by_create_or_update.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        while True:
            i = 10
    client = ContainerAppsAPIClient(credential=DefaultAzureCredential(), subscription_id='34adfa4f-cedf-4dc0-ba29-b6d1a69ab345')
    response = client.container_apps.begin_create_or_update(resource_group_name='rg', container_app_name='testcontainerAppManagedBy', container_app_envelope={'location': 'East US', 'managedBy': '/subscriptions/34adfa4f-cedf-4dc0-ba29-b6d1a69ab345/resourceGroups/rg/providers/Microsoft.AppPlatform/Spring/springapp', 'properties': {'configuration': {'ingress': {'exposedPort': 4000, 'external': True, 'targetPort': 3000, 'traffic': [{'revisionName': 'testcontainerAppManagedBy-ab1234', 'weight': 100}], 'transport': 'tcp'}}, 'environmentId': '/subscriptions/34adfa4f-cedf-4dc0-ba29-b6d1a69ab345/resourceGroups/rg/providers/Microsoft.App/managedEnvironments/demokube', 'template': {'containers': [{'image': 'repo/testcontainerAppManagedBy:v1', 'name': 'testcontainerAppManagedBy', 'probes': [{'initialDelaySeconds': 3, 'periodSeconds': 3, 'tcpSocket': {'port': 8080}, 'type': 'Liveness'}]}], 'scale': {'maxReplicas': 5, 'minReplicas': 1, 'rules': [{'name': 'tcpscalingrule', 'tcp': {'metadata': {'concurrentConnections': '50'}}}]}}}}).result()
    print(response)
if __name__ == '__main__':
    main()