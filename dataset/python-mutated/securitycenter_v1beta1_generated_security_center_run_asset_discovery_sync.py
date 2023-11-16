from google.cloud import securitycenter_v1beta1

def sample_run_asset_discovery():
    if False:
        for i in range(10):
            print('nop')
    client = securitycenter_v1beta1.SecurityCenterClient()
    request = securitycenter_v1beta1.RunAssetDiscoveryRequest(parent='parent_value')
    operation = client.run_asset_discovery(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)