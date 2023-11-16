from google.cloud import kms_inventory_v1

def sample_get_protected_resources_summary():
    if False:
        return 10
    client = kms_inventory_v1.KeyTrackingServiceClient()
    request = kms_inventory_v1.GetProtectedResourcesSummaryRequest(name='name_value')
    response = client.get_protected_resources_summary(request=request)
    print(response)