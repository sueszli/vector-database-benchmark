from google.cloud import securitycenter_v1

def sample_run_asset_discovery():
    if False:
        while True:
            i = 10
    client = securitycenter_v1.SecurityCenterClient()
    request = securitycenter_v1.RunAssetDiscoveryRequest(parent='parent_value')
    operation = client.run_asset_discovery(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)