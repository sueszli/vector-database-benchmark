from google.cloud import accessapproval_v1

def sample_update_access_approval_settings():
    if False:
        while True:
            i = 10
    client = accessapproval_v1.AccessApprovalClient()
    request = accessapproval_v1.UpdateAccessApprovalSettingsMessage()
    response = client.update_access_approval_settings(request=request)
    print(response)