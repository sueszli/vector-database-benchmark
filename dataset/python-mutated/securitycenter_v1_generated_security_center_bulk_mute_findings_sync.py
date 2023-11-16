from google.cloud import securitycenter_v1

def sample_bulk_mute_findings():
    if False:
        return 10
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.BulkMuteFindingsRequest(parent='parent_value')
    operation = client.bulk_mute_findings(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)