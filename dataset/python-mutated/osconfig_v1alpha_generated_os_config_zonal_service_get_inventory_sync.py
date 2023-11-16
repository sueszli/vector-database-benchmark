from google.cloud import osconfig_v1alpha

def sample_get_inventory():
    if False:
        return 10
    client = osconfig_v1alpha.OsConfigZonalServiceClient()
    request = osconfig_v1alpha.GetInventoryRequest(name='name_value')
    response = client.get_inventory(request=request)
    print(response)