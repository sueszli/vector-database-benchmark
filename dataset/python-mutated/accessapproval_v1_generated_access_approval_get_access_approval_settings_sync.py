from google.cloud import accessapproval_v1

def sample_get_access_approval_settings():
    if False:
        i = 10
        return i + 15
    client = accessapproval_v1.AccessApprovalClient()
    request = accessapproval_v1.GetAccessApprovalSettingsMessage()
    response = client.get_access_approval_settings(request=request)
    print(response)