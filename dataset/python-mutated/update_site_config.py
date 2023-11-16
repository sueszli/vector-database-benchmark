from azure.identity import DefaultAzureCredential
from azure.mgmt.web import WebSiteManagementClient
'\n# PREREQUISITES\n    pip install azure-identity\n    pip install azure-mgmt-web\n# USAGE\n    python update_site_config.py\n\n    Before run the sample, please set the values of the client ID, tenant ID and client secret\n    of the AAD application as environment variables: AZURE_CLIENT_ID, AZURE_TENANT_ID,\n    AZURE_CLIENT_SECRET. For more info about how to get the value, please see:\n    https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal\n'

def main():
    if False:
        for i in range(10):
            print('nop')
    client = WebSiteManagementClient(credential=DefaultAzureCredential(), subscription_id='34adfa4f-cedf-4dc0-ba29-b6d1a69ab345')
    response = client.web_apps.create_or_update_configuration(resource_group_name='testrg123', name='sitef6141', site_config={'properties': {'acrUseManagedIdentityCreds': False, 'alwaysOn': False, 'appCommandLine': '', 'autoHealEnabled': False, 'azureStorageAccounts': {}, 'defaultDocuments': ['Default.htm', 'Default.html', 'Default.asp', 'index.htm', 'index.html', 'iisstart.htm', 'default.aspx', 'index.php', 'hostingstart.html'], 'detailedErrorLoggingEnabled': False, 'ftpsState': 'AllAllowed', 'functionAppScaleLimit': 0, 'functionsRuntimeScaleMonitoringEnabled': False, 'http20Enabled': False, 'httpLoggingEnabled': False, 'linuxFxVersion': '', 'loadBalancing': 'LeastRequests', 'logsDirectorySizeLimit': 35, 'managedPipelineMode': 'Integrated', 'minTlsVersion': '1.2', 'minimumElasticInstanceCount': 0, 'netFrameworkVersion': 'v4.0', 'nodeVersion': '', 'numberOfWorkers': 1, 'phpVersion': '5.6', 'powerShellVersion': '', 'pythonVersion': '', 'remoteDebuggingEnabled': False, 'requestTracingEnabled': False, 'scmMinTlsVersion': '1.2', 'use32BitWorkerProcess': True, 'virtualApplications': [{'physicalPath': 'site\\wwwroot', 'preloadEnabled': False, 'virtualPath': '/'}], 'vnetName': '', 'vnetPrivatePortsCount': 0, 'vnetRouteAllEnabled': False, 'webSocketsEnabled': False}})
    print(response)
if __name__ == '__main__':
    main()