from google.cloud import securitycenter_v1beta1

def sample_create_finding():
    if False:
        return 10
    client = securitycenter_v1beta1.SecurityCenterClient()
    request = securitycenter_v1beta1.CreateFindingRequest(parent='parent_value', finding_id='finding_id_value')
    response = client.create_finding(request=request)
    print(response)