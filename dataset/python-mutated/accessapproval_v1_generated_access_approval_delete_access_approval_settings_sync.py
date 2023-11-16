from google.cloud import accessapproval_v1

def sample_delete_access_approval_settings():
    if False:
        for i in range(10):
            print('nop')
    client = accessapproval_v1.AccessApprovalClient()
    request = accessapproval_v1.DeleteAccessApprovalSettingsMessage()
    client.delete_access_approval_settings(request=request)